# custom_llm_bridge.py
import json
import os
import random
from pathlib import Path
from threading import Thread
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any

# External libs (same as original)
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama

# --- Helper functions (ported with small safety for device) -----------------

def load_llm(model_path: str, tokenizer_path: str, device: str = "cuda"):
    print(f"Loading tokenizer from {tokenizer_path}.")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading pretrained model from {model_path}.")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def prepare_chat_log(
    prompt: str, initial_system_message: Optional[str], chat_log: Optional[List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """
    Build/append the chat log for a user prompt and optional initial system message.
    """
    if chat_log is None:
        if initial_system_message is None:
            chat_log = []
        else:
            chat_log = [{"role": "system", "content": initial_system_message}]
    chat_log.append({"role": "user", "content": prompt})
    return chat_log


def ollama_api_call(model_name: str, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    """Make API call to Ollama using the official library"""
    actual_model_name = model_name.replace("ollama_", "")

    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    try:
        response = ollama.chat(model=actual_model_name, messages=messages, options=options if options else None)
        return response["message"]["content"]
    except Exception as e:
        raise Exception(f"Ollama API error: {e}")


def pipeline(
    model_name: str,
    model,
    tokenizer,
    query: str,
    chat_log: List[Dict[str, str]],
    initial_system_message: Optional[str],
    max_output_tokens: int,
    temperature: Optional[float] = None,
    do_sample: bool = False,
    top_p: Optional[float] = None,
    boxed: bool = False,
    device: str = "cuda",
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    Single-input pipeline for local transformer models.
    Returns (answer, updated_chat_log, full_answer)
    """
    if "r1_distill" in model_name:
        initial_system_message = None
        if boxed:
            query += " Put your final answer within \\boxed{}."

    chat_log = prepare_chat_log(query, initial_system_message, chat_log=chat_log)

    # Build inputs using tokenizer's chat template API if available
    inputs = tokenizer.apply_chat_template(chat_log, add_generation_prompt=True, return_tensors="pt").to(device)
    if inputs.size(1) > tokenizer.model_max_length:
        answer = "The input sequence is too long. Aborting."
        chat_log.append({"role": "assistant", "content": answer})
        return answer, chat_log, answer

    attention_mask = torch.ones(1, inputs.size(1)).to(device)
    output = model.generate(
        inputs,
        max_length=inputs.size(1) + max_output_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=do_sample,
        top_p=top_p,
    )

    if (output[0][-1] != tokenizer.eos_token_id and len(output[0]) - inputs.size(1) == max_output_tokens):
        print(f"WARNING: Max. token length ({max_output_tokens}) exceeded.")

    answer = tokenizer.decode(output[0][inputs.size(1):], skip_special_tokens=True).strip()
    full_answer = answer

    if "r1_distill" in model_name:
        # apply the original post-processing rules
        answer = answer.split("</think>")[-1]
        if "boxed{" in answer and answer.rfind("}") != -1:
            answer = answer[:answer.rfind("}")].split("boxed{")[-1]
        answer = answer.split("Answer:**")[-1]
        answer = answer.replace("\_", "_").replace("\\'", "'").replace("'", "'")
        if len(answer.split("**")) == 3:
            answer = answer.split("**")[-2]
        answer = answer.strip()

    chat_log.append({"role": "assistant", "content": answer})
    return answer, chat_log, full_answer


def pipeline_batch(
    model_name: str,
    model,
    tokenizer,
    queries: List[str],
    system_message: Optional[str],
    max_output_tokens: int,
    chat_logs: Optional[List[List[Dict[str, str]]]],
    temperature: Optional[float] = None,
    do_sample: bool = False,
    top_p: Optional[float] = None,
    boxed: bool = False,
    device: str = "cuda",
) -> Tuple[List[str], List[List[Dict[str, str]]], List[str]]:
    """
    Batch pipeline for local transformer models.
    Returns (answers, updated_chat_logs, full_answers)
    """
    if "r1_distill" in model_name:
        if boxed:
            for i in range(len(queries)):
                queries[i] += " Put your final answer within \\boxed{}."

    if chat_logs is None:
        chat_logs = [None for _ in range(len(queries))]

    for i in range(len(queries)):
        chat_logs[i] = prepare_chat_log(queries[i], system_message, chat_log=chat_logs[i])

    outputs = []
    prompts = tokenizer.apply_chat_template(chat_logs, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=False, tokenize=False)
    tokenized_input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)

    num_input_tokens = tokenized_input.input_ids.size(1)
    # conservative batch sizing similar to original
    batch_size = int(tokenizer.model_max_length / 2 / num_input_tokens) if num_input_tokens > 0 else 1
    if batch_size < 1 and num_input_tokens <= tokenizer.model_max_length:
        batch_size = 1

    input_batches = torch.split(tokenized_input.input_ids, batch_size)
    attention_batches = torch.split(tokenized_input.attention_mask, batch_size)

    for i in range(len(input_batches)):
        batch_output = model.generate(
            input_batches[i],
            max_length=num_input_tokens + max_output_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_batches[i],
            do_sample=do_sample,
            top_p=top_p,
        )
        outputs.extend(batch_output)

    # Trim prompt tokens from each response
    for i in range(len(outputs)):
        outputs[i] = outputs[i][num_input_tokens:]

    full_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers: List[str] = []
    for i, answer in enumerate(full_answers):
        if "r1_distill" in model_name:
            answer = answer.split("</think>")[-1]
            if "boxed{" in answer and answer.rfind("}") != -1:
                answer = answer[:answer.rfind("}")].split("boxed{")[-1]
            answer = answer.split("Answer:**")[-1]
            answer = answer.replace("\_", "_").replace("\\'", "'").replace("'", "'")
            if len(answer.split("**")) == 3:
                answer = answer.split("**")[-2]
        answer = answer.strip()
        chat_logs[i].append({"role": "assistant", "content": answer})
        answers.append(answer)

    return answers, chat_logs, full_answers


def load_configs_from_file(file_path: Optional[str | Path] = None) -> Dict[str, Any]:
    if file_path is None:
        file_path = Path(__file__).parent / "llm_configs.json"
    else:
        file_path = Path(file_path)
    with open(file_path) as json_file:
        configs = json.load(json_file)
    if "llm" in configs:
        return configs["llm"]
    return configs

# -------------------------- Class: LlmBridge ---------------------------------

class LlmBridge:
    def __init__(self, model_name: str, configs_path: Optional[str | Path] = None, verbose: bool = True):
        """
        Standalone LlmBridge (framework-free).
        - model_name: string identifying model (same semantics as original, e.g., "gpt-4", "ollama_xxx", "deepseek-...")
        - configs_path: path to llm_configs.json (defaults to module folder / llm_configs.json)
        - verbose: use print() debugging when True
        """
        self.verbose = verbose
        self.model_name = model_name
        configs = load_configs_from_file(configs_path)
        # read commonly used config keys; if missing, supply reasonable defaults
        self.temperature = configs.get("llm_temperature", None)
        self.seed = configs.get("llm_seed", None)
        self.parallelization_mode = configs.get("llm_parallelization_mode", "sequential")
        self.initial_system_message = configs.get("llm_default_system_message", None)
        self.do_sample = configs.get("llm_do_sample", False)
        self.top_p = configs.get("llm_top_p", None)
        self.max_output_tokens = configs.get("llm_max_output_tokens", 256)

        # Device handling: keep original behavior where possible but fallback to CPU if CUDA not available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"[LlmBridge] Initializing model: {self.model_name} (device={self.device})")

        # prepare API clients and local models depending on model_name
        self.model = None
        self.tokenizer = None
        self.clients = []
        self.client = None

        # Ensure environment variables loaded when needed
        load_dotenv()

        # Branching logic mirroring the original LlmBridge
        try:
            if "ollama_" in self.model_name:
                if self.verbose:
                    print(f"[LlmBridge] Using Ollama model: {self.model_name.replace('ollama_', '')}")
                # Ollama uses no local model or tokenizer
                self.model = None
                self.tokenizer = None

            elif "oss" in self.model_name or "meta" in self.model_name or "qwen" in self.model_name or "gemma" in self.model_name.lower() or "mistral" in self.model_name.lower() or "nvidia" in self.model_name:
                # NVIDIA / OpenAI-like client (original used OpenAI with a base_url)
                if self.verbose:
                    print(f"[LlmBridge] Using NVIDIA/meta-style model: {self.model_name}")
                keys = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
                if not keys: keys = [""]
                self.clients = [OpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in keys]

            elif "gpt" in self.model_name:
                if self.verbose:
                    print(f"[LlmBridge] Using OpenAI model: {self.model_name}")
                keys = [k.strip() for k in os.environ.get("OPENAI_API_KEY", "").split(",") if k.strip()]
                if not keys: keys = [""]
                self.clients = [OpenAI(api_key=k) for k in keys]

            elif "deepseek" in self.model_name:
                if self.verbose:
                    print(f"[LlmBridge] Using DeepSeek model: {self.model_name}")
                keys = [k.strip() for k in os.environ.get("DEEP_SEEK_API_KEY", "").split(",") if k.strip()]
                if not keys: keys = [""]
                self.clients = [OpenAI(api_key=k, base_url="https://api.deepseek.com/v1") for k in keys]

            elif "gemini" in self.model_name:
                if self.verbose:
                    print(f"[LlmBridge] Using Gemini model: {self.model_name}")
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                self.model = genai.GenerativeModel(self.model_name)

            else:
                # Local model (transformers) handling
                if "r1_distill" in self.model_name:
                    self.max_output_tokens = configs.get("llm_max_output_tokens_reasoning_model", self.max_output_tokens)

                # Expect model path keys in configs like "<model_name>_path"
                model_key = f"{self.model_name}_path"
                tokenizer_key = model_key  # original used the same key for tokenizer_path
                model_path = configs.get(model_key)
                tokenizer_path = configs.get(tokenizer_key)
                if not model_path or not tokenizer_path:
                    raise KeyError(f"Missing model path for local model '{self.model_name}' in configs (key '{model_key}').")
                if self.verbose:
                    print(f"[LlmBridge] Loading local model: {self.model_name} from {model_path}")
                self.model, self.tokenizer = load_llm(model_path, tokenizer_path, device=self.device)
        except Exception as e:
            # Surface initialization issues but don't crash silently
            print(f"[LlmBridge] Initialization error for model '{self.model_name}': {e}")
            raise

    def get_client(self):
        if self.clients:
            return random.choice(self.clients)
        return self.client

    # --- forwarding helper methods (for threaded batch processing) -------------
    def forward_to_openai(self, idx: int, answers: List[Optional[str]], chat_log: List[Dict[str, str]]) -> None:
        temperature = 0.0 if self.temperature is None else self.temperature
        
        # GEMMA STRICT FIX: Map system dict strings entirely into user roles.
        if "gemma" in self.model_name.lower():
            safe_chat_log = []
            for msg in chat_log:
                if msg["role"] == "system":
                    safe_chat_log.append({"role": "user", "content": "SYSTEM INSTRUCTION: " + msg["content"]})
                else:
                    safe_chat_log.append(msg)
            chat_log = safe_chat_log

        # create chat completion (OpenAI client wrapper used in original)
        try:
            client = self.get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=chat_log,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p,
                temperature=temperature,
                seed=self.seed,
                timeout=120,
            )
            answers[idx] = response.choices[0].message.content
        except Exception as e:
            answers[idx] = f"OpenAI API error: {e}"
            if self.verbose:
                print(f"[LlmBridge] forward_to_openai error idx={idx}: {e}")

    def forward_to_ollama(self, idx: int, answers: List[Optional[str]], chat_log: List[Dict[str, str]]) -> None:
        try:
            answer = ollama_api_call(self.model_name, chat_log, self.temperature, self.max_output_tokens)
            answers[idx] = answer
        except Exception as e:
            error_message = f"Error processing Ollama request for index {idx}: {e}"
            if self.verbose:
                print(f"[LlmBridge] {error_message}")
            answers[idx] = error_message

    def forward_to_gemini(self, idx: int, answers: List[Optional[str]], chat_log: List[Dict[str, str]]) -> None:
        # Convert chat log to Gemini format
        gemini_history = []
        for msg in chat_log:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        generation_config = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature if self.temperature is not None else 0.9,
            "top_p": self.top_p,
        }

        try:
            response = self.model.generate_content(gemini_history, generation_config=generation_config)
            answers[idx] = response.text
        except Exception as e:
            error_message = f"Error processing Gemini request for index {idx}: {e}"
            if self.verbose:
                print(f"[LlmBridge] {error_message}")
            answers[idx] = error_message

    # --- public API: single and batch queries ---------------------------------
    def ask_llm(self, question: str, chat_log: Optional[List[Dict[str, str]]] = None, log: bool = True):
        """
        Ask a single question to the configured LLM.
        Returns (answer, updated_chat_log, full_answer)
        """
        if "ollama_" in self.model_name:
            chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
            answer = ollama_api_call(self.model_name, chat_log, self.temperature, self.max_output_tokens)
            chat_log.append({"role": "assistant", "content": answer})
            full_answer = answer

        elif "gpt" in self.model_name or "deepseek" in self.model_name or "nvidia" in self.model_name or "meta" in self.model_name or "oss" in self.model_name or "qwen" in self.model_name or "gemma" in self.model_name.lower() or "mistral" in self.model_name.lower():
            chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
            client = self.get_client()
            response = client.chat.completions.create(
                model=self.model_name, messages=chat_log, temperature=self.temperature, seed=self.seed, timeout=120
            )
            answer = response.choices[0].message.content
            chat_log.append({"role": "assistant", "content": answer})
            full_answer = answer

        elif "gemini" in self.model_name:
            chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
            gemini_history = []
            for msg in chat_log:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [msg["content"]]})
            generation_config = {
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature if self.temperature is not None else 0.9,
                "top_p": self.top_p,
            }
            response = self.model.generate_content(gemini_history, generation_config=generation_config)
            answer = response.text
            chat_log.append({"role": "assistant", "content": answer})
            full_answer = answer

        else:
            # Local model pipeline
            answer, chat_log, full_answer = pipeline(
                self.model_name,
                self.model,
                self.tokenizer,
                question,
                chat_log,
                self.initial_system_message,
                self.max_output_tokens,
                self.temperature,
                self.do_sample,
                self.top_p,
                boxed=True,
                device=self.device,
            )

        if log and self.verbose:
            print(f"\n[Ask Question]: {question}\n\n[{self.model_name} Full Answer]: {full_answer}\n\n[{self.model_name} Shortened Answer]: {answer}\n")

        return answer, chat_log, full_answer

    def ask_llm_batch(self, questions: List[str], chat_logs: Optional[List[List[Dict[str, str]]]] = None):
        """
        Ask multiple questions (batch). Preserves behaviour and parallelization rules of the original.
        Returns (answers, updated_chat_logs)
        """
        # Validate parallelization compatibility like original
        api_models = ("gpt", "meta", "deepseek", "gemini", "ollama_", "nvidia", "qwen", "gemma", "mistral")
        if any(k in self.model_name.lower() for k in api_models) and self.parallelization_mode == "batch_processing":
            raise ValueError("Batch processing is not supported for API-based models (GPT, DeepSeek, Gemini, Ollama). "
                             "Please use parallelization_mode 'multiprocessing' instead.")
        if not any(k in self.model_name for k in api_models) and self.parallelization_mode == "multiprocessing":
            raise ValueError("Multi-threading is not supported for local models. Use parallelization mode 'batch_processing' instead.")

        if self.verbose:
            print(f"[LlmBridge] ask_llm_batch: first query (if exists): {questions[0] if len(questions)>0 else None}")

        if self.parallelization_mode == "sequential":
            if chat_logs is None:
                chat_logs = [None for _ in range(len(questions))]
            answers: List[str] = []
            full_answers: List[str] = []
            for i in range(len(questions)):
                answer, chat_log, full_answer = self.ask_llm(questions[i], chat_logs[i], log=False)
                answers.append(answer)
                full_answers.append(full_answer)
                chat_logs[i] = chat_log
            if self.verbose:
                print(f"[LlmBridge] {self.model_name} Full Answer (first): {full_answers[0] if full_answers else None}")
        else:
            # parallelized mode
            if any(k in self.model_name.lower() for k in api_models):
                # Use threads to call appropriate API-forwarding functions
                if chat_logs is None:
                    chat_logs = [None for _ in range(len(questions))]
                procs: List[Thread] = []
                answers: List[Optional[str]] = [None for _ in range(len(questions))]
                for idx, question in enumerate(questions):
                    chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_logs[idx])
                    chat_logs[idx] = chat_log

                    if "gemini" in self.model_name:
                        target_func = self.forward_to_gemini
                    elif "ollama_" in self.model_name:
                        target_func = self.forward_to_ollama
                    else:
                        target_func = self.forward_to_openai

                    p = Thread(target=target_func, args=(idx, answers, chat_log))
                    procs.append(p)
                    p.start()

                for p in procs:
                    p.join()

                for i, answer in enumerate(answers):
                    chat_logs[i].append({"role": "assistant", "content": answer})
                full_answers = answers  # for API models full=shortened
            else:
                # Local model batch pipeline
                answers, chat_logs, full_answers = pipeline_batch(
                    self.model_name,
                    self.model,
                    self.tokenizer,
                    questions,
                    self.initial_system_message,
                    self.max_output_tokens,
                    chat_logs,
                    self.temperature,
                    self.do_sample,
                    self.top_p,
                    boxed=True,
                    device=self.device,
                )
                if self.verbose:
                    print(f"[LlmBridge] {self.model_name} Full Answer (first): {full_answers[0] if full_answers else None}")

        if self.verbose:
            print(f"[LlmBridge] {self.model_name} Shortened Answer (first): {answers[0] if answers else None}")

        return answers, chat_logs

# End of custom_llm_bridge.py


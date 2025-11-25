import os.path as osp
import sys
from threading import Thread, Semaphore, Lock
import openai
import torch
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
sys.path.append('.')
import re
from collections import deque
from datetime import datetime, timedelta
import dotenv

dotenv.load_dotenv(osp.join(osp.dirname(osp.abspath(__file__)), '.env'))
# Global semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 50
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
retry_lock = Lock()

# Token tracking for rate limiting
class TokenRateLimiter:
    def __init__(self, tokens_per_minute=100000, safety_margin=0.9):
        self.tokens_per_minute = int(tokens_per_minute * safety_margin)  # Use 90% of limit for safety
        self.token_usage = deque()  # Store (timestamp, token_count) tuples
        self.lock = Lock()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using 1 token ≈ 4 characters rule"""
        return len(text) // 4 + 1  # Add 1 to avoid zero tokens
    
    def wait_if_needed(self, estimated_tokens: int):
        """Wait if adding these tokens would exceed rate limit"""
        with self.lock:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Remove old entries (older than 1 minute)
            while self.token_usage and self.token_usage[0][0] < one_minute_ago:
                self.token_usage.popleft()
            
            # Calculate current token usage in the last minute
            current_usage = sum(tokens for _, tokens in self.token_usage)
            
            # If adding new tokens would exceed limit, wait
            if current_usage + estimated_tokens > self.tokens_per_minute:
                # Calculate how long to wait
                if self.token_usage:
                    oldest_timestamp = self.token_usage[0][0]
                    wait_time = (oldest_timestamp + timedelta(minutes=1) - now).total_seconds()
                    wait_time = max(wait_time, 0.1)  # Wait at least 0.1 seconds
                else:
                    wait_time = 1.0
                
                print(f'Token rate limit approaching: {current_usage}/{self.tokens_per_minute} TPM used. '
                      f'Waiting {wait_time:.2f}s for quota refresh...')
                time.sleep(wait_time)
                
                # Recursively check again after waiting
                return self.wait_if_needed(estimated_tokens)
            
            # Record this token usage
            self.token_usage.append((now, estimated_tokens))
    
    def get_current_usage(self) -> tuple:
        """Get current token usage for monitoring"""
        with self.lock:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Remove old entries
            while self.token_usage and self.token_usage[0][0] < one_minute_ago:
                self.token_usage.popleft()
            
            current_usage = sum(tokens for _, tokens in self.token_usage)
            return current_usage, self.tokens_per_minute

# Global rate limiter instance
token_limiter = TokenRateLimiter()

def parse_retry_time(error_message: str) -> float:
    """
    Parse the retry time from OpenAI error message.
    Args:
        error_message (str): The error message from OpenAI API
    Returns:
        float: Time to wait in seconds
    """
    # Try to find "try again in Xms" or "try again in Xs"
    match = re.search(r'try again in (\d+(?:\.\d+)?)(ms|s)', error_message)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == 'ms':
            return value / 1000.0
        else:
            return value
    return 1.0  # Default wait time

def get_openai_embedding(emb_client,
                         idx,
                         answers, 
                         text: str,
                         model: str,
                         max_retry: int = 10,
                         base_sleep_time: float = 10.0) -> None:
    """
    Get the OpenAI embedding for a given text with rate limiting.
    Args:
        emb_client: OpenAI client
        idx: Index for storing result
        answers: Shared list to store results
        text (str): The input text to be embedded.
        model (str): The model to use for embedding.
        max_retry (int): Maximum number of retries in case of an error. Default is 10.
        base_sleep_time (float): Base sleep time between retries in seconds. Default is 1.0.
    """
    assert isinstance(text, str), f'text must be str, but got {type(text)}'
    assert len(text) > 0, 'text to be embedded should be non-empty'
    
    retry_count = 0
    
    while retry_count < max_retry:
        try:
            # Estimate tokens and wait if needed to stay within rate limit
            estimated_tokens = token_limiter.estimate_tokens(text)
            token_limiter.wait_if_needed(estimated_tokens)
            
            # Acquire semaphore to limit concurrent requests
            with request_semaphore:
                emb = emb_client.embeddings.create(input=[text], model=model)
                answers[idx] = torch.FloatTensor(emb.data[0].embedding).view(1, -1)
                return
                
        except openai.BadRequestError as e:
            error_str = str(e)
            print(f'BadRequestError at idx {idx}: {error_str}')
            
            ori_length = len(text.split(' '))
            match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', error_str)
            
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    
                    try:
                        # Check rate limit for shortened text too
                        estimated_tokens = token_limiter.estimate_tokens(shorten_text)
                        token_limiter.wait_if_needed(estimated_tokens)
                        
                        with request_semaphore:
                            emb = emb_client.embeddings.create(input=[shorten_text], model=model)
                            print(f'idx {idx}: length={length} works! reduce_rate={0.1 * reduce_rate}.')
                            answers[idx] = torch.FloatTensor(emb.data[0].embedding).view(1, -1)
                            return
                    except openai.RateLimitError:
                        # If we hit rate limit while shortening, break and handle below
                        break
                    except Exception as shorten_error:
                        continue
                        
            # If we couldn't handle the bad request, raise it
            raise RuntimeError(f"Could not process text at idx {idx} even after shortening")
            
        except openai.RateLimitError as e:
            error_str = str(e)
            retry_time = parse_retry_time(error_str)
            
            # Extract actual usage from error message if available
            usage_match = re.search(r'Used (\d+)', error_str)
            if usage_match:
                print(f'API reported usage: {usage_match.group(1)} tokens')
            
            # Use exponential backoff with the parsed retry time
            sleep_time = max(retry_time, base_sleep_time * (2 ** retry_count))
            
            with retry_lock:
                current_usage, limit = token_limiter.get_current_usage()
                print(f'RateLimitError at idx {idx} (attempt {retry_count + 1}/{max_retry})')
                print(f'Local tracking: {current_usage}/{limit} TPM | Error: {error_str}')
                print(f'Sleeping for {sleep_time:.2f} seconds...')
            
            time.sleep(sleep_time)
            retry_count += 1
            
        except openai.APITimeoutError as e:
            sleep_time = base_sleep_time * (2 ** retry_count)
            
            with retry_lock:
                print(f'APITimeoutError at idx {idx} (attempt {retry_count + 1}/{max_retry}): {e}')
                print(f'Sleeping for {sleep_time:.2f} seconds...')
            
            time.sleep(sleep_time)
            retry_count += 1
            
        except Exception as e:
            sleep_time = base_sleep_time * (2 ** retry_count)
            
            with retry_lock:
                print(f'APITimeoutError at idx {idx} (attempt {retry_count + 1}/{max_retry}): {e}')
                print(f'Sleeping for {sleep_time:.2f} seconds...')
            
            time.sleep(sleep_time)
            retry_count += 1
    
    raise RuntimeError(f"Failed to get embedding for idx {idx} after {max_retry} retries")


def get_openai_embeddings(texts: list,
                          emb_client: openai.OpenAI,
                          emb_model: str,
                          answers: list,
                          n_max_nodes: int = 50,
                          delay_between_batches: float = 0.1,
                          tokens_per_minute: int = 1000000) -> torch.FloatTensor:
    """
    Get embeddings for a list of texts using OpenAI's embedding model.
    Args:
        texts (list): List of input texts to be embedded.
        emb_client: OpenAI client
        emb_model (str): The model to use for embedding.
        answers (list): Pre-allocated list to store results
        n_max_nodes (int): Maximum number of parallel processes. Default is 50.
        delay_between_batches (float): Delay between starting threads to avoid burst. Default is 0.1s.
        tokens_per_minute (int): Rate limit for tokens per minute. Default is 1000000.
    Returns:
        torch.FloatTensor: A tensor containing embeddings for all input texts.
    """
    assert isinstance(texts, list), f'texts must be list, but got {type(texts)}'
    assert all([len(s) > 0 for s in texts]), 'every string in the `texts` list to be embedded should be non-empty'
    
    # Update global rate limiter with specified TPM
    global token_limiter
    token_limiter = TokenRateLimiter(tokens_per_minute=tokens_per_minute)
    
    # Initialize answers list if not already done
    if len(answers) != len(texts):
        answers.clear()
        answers.extend([None] * len(texts))
    
    # Estimate total tokens
    total_tokens = sum(token_limiter.estimate_tokens(text) for text in texts)
    print(f'Estimated total tokens: {total_tokens:,} (~{total_tokens/tokens_per_minute:.2f} minutes at max rate)')
    
    procs = []
    
    # Start threads with a small delay to avoid bursting
    for idx, text in enumerate(tqdm(texts, desc="Starting embedding threads")):
        p = Thread(target=get_openai_embedding, args=(emb_client, idx, answers, text, emb_model))
        procs.append(p)
        p.start()
        
        # Small delay to avoid bursting all requests at once
        if (idx + 1) % MAX_CONCURRENT_REQUESTS == 0:
            time.sleep(delay_between_batches)
    
    # Wait for all threads to complete
    for p in tqdm(procs, desc="Waiting for threads to complete"):
        p.join()
    
    # Check if all embeddings were retrieved
    if None in answers:
        failed_indices = [i for i, ans in enumerate(answers) if ans is None]
        print(f"Failed to get embeddings for indices: {failed_indices}")
    
    results = torch.cat(answers, dim=0)
    return results

from openai import OpenAI
import os
from time import sleep
import sys
from stark_qa import load_skb

dataset_name = 'mag'
kb = load_skb(dataset_name, download_processed=True)

# Suppress all print statements
# sys.stdout = open('logs_output.txt', 'w')
# sys.stderr = open('logs_error.txt', 'w')
emb_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
print(os.environ.get("OPENAI_API_KEY"))
emb_model = "text-embedding-ada-002"
file_name_prefix = dataset_name + "_openai_emb_"

step_size = 500
batch_no = 1727

vector_keys = []
texts_to_embed = []

print("Preparing texts to embed...")
for candidate_idx in range(1172723):
    text_to_embed = kb.get_doc_info(candidate_idx)
    vector_key = f"{candidate_idx}"
    vector_keys.append(vector_key)
    texts_to_embed.append(text_to_embed)

print(f"Total texts to embed: {len(texts_to_embed)}")
for i in range(batch_no*step_size, len(texts_to_embed), step_size):
    print(f"Processing batch {i//step_size}...")
    batch_texts = texts_to_embed[i:i+step_size]
    batch_vector_keys = vector_keys[i:i+step_size]
    answers = [None] * len(batch_texts)
    embeddings = get_openai_embeddings(batch_texts, emb_client, emb_model, answers)
    # Save embeddings and vector keys
    batch_emb_dict = {}
    for j, vector_key in enumerate(batch_vector_keys):
        batch_emb_dict[vector_key] = embeddings[j]
    torch.save(batch_emb_dict, f"{file_name_prefix}{i//step_size}.pt")
    print(f"Saved batch {(i + 5000)//step_size} with {len(batch_vector_keys)} embeddings.")
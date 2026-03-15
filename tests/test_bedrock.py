"""
Amazon Bedrock API test
Model  : meta.llama3-3-70b-instruct-v1:0
Auth   : ABSK API key (Bearer token — no IAM/AWS-CLI credentials needed)
Tests  : 1) single response  2) concurrent batch response

ABSK keys are Amazon Bedrock "API Keys" created in the Bedrock console.
They work with direct HTTP `Authorization: Bearer <ABSK_KEY>` — no SigV4 needed.
"""

import os, json, time, base64, concurrent.futures
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
BEDROCK_API_KEY = os.environ["BEDROCK_API_KEY"]   # ABSK... key from .env
MODEL_ID        = "us.meta.llama3-3-70b-instruct-v1:0"  # cross-region inference profile
AWS_REGION      = "us-east-1"        # change if your account is in another region
MAX_TOKENS      = 512

# Base runtime endpoint (same as boto3 bedrock-runtime uses)
BASE_URL = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"

# ── Display decoded key identity (not the secret) ─────────────────────────────
_decoded  = base64.b64decode(BEDROCK_API_KEY[4:]).decode()
_identity, _ = _decoded.split(":", 1)
print("=" * 62)
print(f"  Key identity : {_identity}")
print(f"  Model        : {MODEL_ID}")
print(f"  Region       : {AWS_REGION}")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llama3_payload(prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    """Native Meta Llama-3 payload format for Bedrock InvokeModel."""
    return {
        "prompt": (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        ),
        "max_gen_len": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def _session() -> requests.Session:
    """Shared Session with ABSK bearer token pre-configured."""
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "Accept":        "application/json",
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
    })
    return s


SESSION = _session()   # one session shared across all calls


def invoke(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Send a single InvokeModel request to Bedrock.
    Uses Bearer auth — no AWS credentials required.
    """
    url  = f"{BASE_URL}/model/{MODEL_ID}/invoke"
    body = _llama3_payload(prompt, max_tokens)
    resp = SESSION.post(url, json=body, timeout=120)

    if not resp.ok:
        raise RuntimeError(
            f"HTTP {resp.status_code}  {resp.reason}\n"
            f"Body: {resp.text[:500]}"
        )

    data = resp.json()
    # Bedrock Llama-3 returns {"generation": "...", ...}
    return data.get("generation", str(data))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — single / normal response
# ─────────────────────────────────────────────────────────────────────────────

def test_single():
    prompt = "What is the capital of France? Answer in one sentence."
    print(f"\n{'─'*62}")
    print("TEST 1 — Single response")
    print(f"  Prompt : {prompt}")
    print(f"{'─'*62}")

    t0 = time.time()
    try:
        answer = invoke(prompt)
        elapsed = time.time() - t0
        print(f"  ✓  {elapsed:.2f}s")
        print(f"  Response: {answer.strip()}")
        return True
    except Exception as exc:
        print(f"  ✗  {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — concurrent batch response  (thread-pool; no S3 needed)
# ─────────────────────────────────────────────────────────────────────────────

BATCH_PROMPTS = [
    "Name three early symptoms of Type-2 diabetes.",
    "What is the primary role of mitochondria in a cell?",
    "Summarise the CRISPR-Cas9 mechanism in two sentences.",
    "What is the main structural difference between RNA and DNA?",
    "Which gene is most commonly mutated in cystic fibrosis?",
]


def test_batch():
    n = len(BATCH_PROMPTS)
    print(f"\n{'─'*62}")
    print(f"TEST 2 — Concurrent batch ({n} prompts in parallel)")
    print(f"{'─'*62}")

    results: dict = {}
    errors:  dict = {}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(invoke, p, 256): i for i, p in enumerate(BATCH_PROMPTS)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                errors[idx] = str(exc)
    elapsed = time.time() - t0

    ok  = len(results)
    err = len(errors)
    print(f"\n  Finished {n} requests in {elapsed:.2f}s  "
          f"({ok} OK, {err} errors)\n")

    for i, prompt in enumerate(BATCH_PROMPTS):
        tag = "✓" if i in results else "✗"
        print(f"  [{i+1}] {tag}  {prompt}")
        if i in results:
            snippet = results[i].strip().replace("\n", " ")[:200]
            print(f"        → {snippet}")
        else:
            print(f"        ERROR: {errors[i][:120]}")
        print()

    return err == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok1 = test_single()
    if ok1:
        test_batch()
    else:
        print(
            "\n✗  Single request failed — check:\n"
            "   1. BEDROCK_API_KEY in .env is the latest key from the Bedrock console\n"
            f"   2. AWS_REGION is correct (currently '{AWS_REGION}')\n"
            "   3. The model is enabled in your account\n"
            f"      Model: {MODEL_ID}\n"
            "   4. Key Status is Active (not expired)\n"
        )

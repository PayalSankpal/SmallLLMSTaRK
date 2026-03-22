import os
import requests
from dotenv import load_dotenv

load_dotenv()

BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY")
MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"
AWS_REGION = "us-east-1"
BASE_URL = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com"

def test_prompt(prompt):
    url = f"{BASE_URL}/model/{MODEL_ID}/invoke"
    
    # Llama 3 prompt format
    body = {
        "prompt": (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "max_gen_len": 128,
        "temperature": 0.0,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
    }
    
    print(f"Testing model: {MODEL_ID}...")
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=60)
        if not resp.ok:
            print(f"Error ({resp.status_code}): {resp.text}")
            return
        
        data = resp.json()
        print("Response received!")
        print("-" * 50)
        print(data.get("generation", str(data)).strip())
        print("-" * 50)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    if not BEDROCK_API_KEY:
        print("Missing BEDROCK_API_KEY in environment!")
    else:
        test_prompt("Output a JSON array containing strictly the numbers 1, 2, and 3.")
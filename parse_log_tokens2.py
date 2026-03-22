import json
import glob
import os

token_sum = 0
input_tokens = 0
output_tokens = 0

response_files = glob.glob('/raid/adityasd314/BTechProject/experiments/batch_reranking_sort/*_res.jsonl')

for res_file in response_files:
    try:
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                usage = data.get("response", {}).get("body", {}).get("usage", {})
                tokens = usage.get("total_tokens", 0)
                input_tokens += usage.get("prompt_tokens", 0)
                output_tokens += usage.get("completion_tokens", 0)
                token_sum += tokens
    except Exception as e:
        print(f"Error reading {res_file}: {e}")

print(f"Input Tokens: {input_tokens:,}")
print(f"Output Tokens: {output_tokens:,}")
print(f"Total API Tokens computed from responses: {token_sum:,} tokens")

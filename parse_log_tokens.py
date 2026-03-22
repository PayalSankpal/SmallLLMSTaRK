import json
import glob
import os

token_sum = 0
request_files = glob.glob('/raid/adityasd314/BTechProject/experiments/batch_reranking_sort/*_req.jsonl')
response_files = glob.glob('/raid/adityasd314/BTechProject/experiments/batch_reranking_sort/*_res.jsonl')

total_comparisons = 0

for res_file in response_files:
    try:
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                usage = data.get("response", {}).get("body", {}).get("usage", {})
                tokens = usage.get("total_tokens", 0)
                token_sum += tokens
                total_comparisons += 1
    except Exception as e:
        print(f"Error reading {res_file}: {e}")

print(f"Total Comparions Made: {total_comparisons}")
print(f"Total API Tokens computed from responses: {token_sum:,} tokens")

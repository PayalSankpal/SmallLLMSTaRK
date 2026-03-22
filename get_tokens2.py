import json
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

with open("experiments/batch_reranking_sort/round_1_req.jsonl", "r") as f:
    line = f.readline()
    data = json.loads(line)
    
prompt = data["body"]["messages"][0]["content"]

parts = prompt.split("\n")
doc1 = parts[2]
doc2 = parts[3]

doc1_tokens = len(enc.encode(doc1))
doc2_tokens = len(enc.encode(doc2))
full_tokens = len(enc.encode(prompt))

print("--- Token Sizes ---")
print(f"Sample Document 1 Tokens: {doc1_tokens:,}")
print(f"Sample Document 2 Tokens: {doc2_tokens:,}")
print(f"Total Tokens for ONE Pairwise Comparison Query: {full_tokens:,}")

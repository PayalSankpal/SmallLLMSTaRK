import os
import ast
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import math
import sys
import threading
import concurrent.futures
import queue
import argparse

sys.path.append('..')
from stark_qa import load_skb

parser = argparse.ArgumentParser(description="RankGPT Listwise Sort via NVIDIA NIM")
parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to dictate output folder")
parser.add_argument("--complete-run", action="store_true", help="Run on all queries (bypasses num-good and num-bad)")
parser.add_argument("--num-good", type=int, default=20, help="Number of good queries to run")
parser.add_argument("--num-bad", type=int, default=20, help="Number of bad queries to run")
args = parser.parse_args()

# Load Environment
load_dotenv()
nv_keys = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
print(f"Loaded {len(nv_keys)} NVIDIA API keys.")

if not nv_keys:
    raise ValueError("No NVIDIA API Keys found!")

# Initialize a queue of clients, one per key
client_queue = queue.Queue()
for key in nv_keys:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key
    )
    client_queue.put(client)

MODEL = "qwen/qwen2.5-coder-7b-instruct"

# Load Data
print("Loading Knowledge Base...")
kb = load_skb('prime', download_processed=True)

print("Building Few-Shot Prompt from teacher_cot_evaluations.csv...")
few_shot_examples = ""
try:
    import pandas as pd
    import ast
    df_train = pd.read_csv('../teacher_cot_evaluations.csv').head(2)
    for idx, row in df_train.iterrows():
        try:
            resp = row['teacher_response']
            if not isinstance(resp, str): continue
            
            reasoning = resp.split('<reasoning>')[1].split('</reasoning>')[0].strip()
            ranking_str = resp.split('<ranking>')[1].split('</ranking>')[0].strip()
            
            candidates = ast.literal_eval(row['candidates'])
            
            docs_info = []
            for d in candidates:
                d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
                n_type = str(kb.get_node_type_by_id(d))
                docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")
            fs_docs_str = '\n'.join(docs_info)
            
            example = f"""
Example {idx+1}:
Query: {row['query']}

Candidate Documents:
{fs_docs_str}

Response:
<reasoning>
{reasoning}
</reasoning>
<ranking>
{ranking_str}
</ranking>
"""
            few_shot_examples += example + "\n"
        except Exception as e:
            continue
    print("Few-Shot Prompt built successfully.")
except Exception as e:
    print("Could not build few-shot prompt:", e)
DATA_DUMP_PATH = "../experiments/prime/LLM_SAVED_RESPONSES/full_data_dump.csv"
df = pd.read_csv(DATA_DUMP_PATH)

def extract_ground_truths(gt_val):
    if pd.isna(gt_val): return []
    if isinstance(gt_val, str):
        try: return ast.literal_eval(gt_val)
        except: return []
    return gt_val

def extract_answer_list(results_str):
    if pd.isna(results_str): return []
    if isinstance(results_str, str):
        try:
            res_dict = ast.literal_eval(results_str)
            return res_dict.get('answer_list', [])[:20]
        except: return []
    return []

df['ground_truths_list'] = df['ground_truths'].apply(extract_ground_truths)
df['top_20_nodes'] = df['results'].apply(extract_answer_list)

def get_target_queries(df, complete_run=False, num_bad=20, num_good=20):
    def rank_of_first_gt(row):
        gt_set = set(row['ground_truths_list'])
        for i, pred in enumerate(row['top_20_nodes']):
            if pred in gt_set: return i + 1
        return 999
    
    df = df.copy()
    df['first_gt_rank'] = df.apply(rank_of_first_gt, axis=1)
    
    if complete_run:
        return df.to_dict('records')
    
    bad_df = df[(df['first_gt_rank'] >= 5) & (df['first_gt_rank'] <= 20)].sort_values(by='first_gt_rank', ascending=False)
    good_df = df[df['first_gt_rank'] == 1]
    
    bad_q = bad_df.head(num_bad).to_dict('records')
    good_q = good_df.head(num_good).to_dict('records')
    
    return bad_q + good_q

queries = get_target_queries(df, complete_run=args.complete_run, num_bad=args.num_bad, num_good=args.num_good)
print(f"Selected {len(queries)} total queries for listwise reranking.")

def rerank_sublist(query_text, docs_list, kb, client, retries=3):
    docs_info = []
    for d in docs_list:
        d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
        n_type = str(kb.get_node_type_by_id(d))
        docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")
    docs_str = '\n'.join(docs_info)
        
    prompt = f"""You are an advanced search relevance ranker. Below are some examples followed by a new search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.
For your response to the final query, output your response in the same format: first provide a <reasoning> block with your reasoning, then provide a <ranking> block containing ONLY a JSON array of the document IDs in the sorted order.

{few_shot_examples}

=== NEW QUERY TO RANK ===
Query: {query_text}

Candidate Documents:
{docs_str}

Response:
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            
            ranking_str = raw
            try:
                if '<ranking>' in raw and '</ranking>' in raw:
                    ranking_str = raw.split('<ranking>')[1].split('</ranking>')[0].strip()
            except Exception:
                pass
                
            # Clean possible markdown
            if ranking_str.startswith("```json"): ranking_str = ranking_str[7:]
            if ranking_str.startswith("```"): ranking_str = ranking_str[3:]
            if ranking_str.endswith("```"): ranking_str = ranking_str[:-3]
            ranking_str = ranking_str.strip()
            
            parsed = json.loads(ranking_str)
            if isinstance(parsed, list):
                # Only keep IDs that were actually in the chunk
                valid = [d for d in parsed if d in docs_list]
                missing = [d for d in docs_list if d not in valid]
                return valid + missing
        
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "limit" in msg or "429" in msg:
                print(f"    [!] Rate limit hit, sleeping before retry...")
                time.sleep(2)
            else:
                print(f"    [Error] Parsing/API error: {e}")
                time.sleep(2)
                
    # Fallback to original order if all retries fail
    print("    [!] Failed to rerank sublist after retries, keeping original order.")
    return docs_list

def rankgpt_sliding_window(query_text, docs, kb, client, window_size=3, step=2):
    w_docs = docs.copy()
    n = len(w_docs)
    
    if n <= window_size:
        return rerank_sublist(query_text, w_docs, kb, client)
        
    # Standard RankGPT moves from bottom to top to bubble up the best
    starts = list(range(n - window_size, -1, -step))
    if len(starts) == 0 or starts[-1] != 0:
        starts.append(0)
        
    for i in starts:
        current_window = w_docs[i : i+window_size]
        new_window = rerank_sublist(query_text, current_window, kb, client)
        w_docs[i : i+window_size] = new_window
        
    return w_docs

def mrr(lst, gts):
    for i, v in enumerate(lst):
        if v in gts: return 1.0/(i+1)
    return 0.0

# RUN EXPERIMENT
print("\n" + "="*50)
window = 3
stride = 2
print(f"Starting Sliding Window Listwise Sort (Window={window}, Stride={stride}) - PARALLEL MODE")
print("="*50)

def process_query(idx, q):
    print(f"Processing Query {idx+1}/{len(queries)} (ID: {q['id']})...")
    client = client_queue.get()
    try:
        orig_docs = q['top_20_nodes']
        reranked = rankgpt_sliding_window(q['query'], orig_docs, kb, client, window_size=3, step=2)
        
        gts = set(q['ground_truths_list'])  
        return {
            'id': q['id'],
            'orig_mrr': mrr(orig_docs, gts),
            'new_mrr': mrr(reranked, gts),
            'orig_h1': 1.0 if orig_docs and orig_docs[0] in gts else 0.0,
            'new_h1': 1.0 if reranked and reranked[0] in gts else 0.0,
            'orig_h5': 1.0 if any(d in gts for d in orig_docs[:5]) else 0.0,
            'new_h5': 1.0 if any(d in gts for d in reranked[:5]) else 0.0
        }
    finally:
        client_queue.put(client)

results = []
# Using ThreadPoolExecutor limited to the number of available keys
num_workers = len(nv_keys)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_idx = {executor.submit(process_query, i, q): i for i, q in enumerate(queries)}
    for future in concurrent.futures.as_completed(future_to_idx):
        try:
            res = future.result()
            results.append(res)
        except Exception as exc:
            print(f"[!] Query generated an exception: {exc}")

df_res = pd.DataFrame(results)

mean_orig_mrr = df_res['orig_mrr'].mean()
mean_new_mrr = df_res['new_mrr'].mean()
mean_orig_h1 = df_res['orig_h1'].mean()
mean_new_h1 = df_res['new_h1'].mean()
mean_orig_h5 = df_res['orig_h5'].mean()
mean_new_h5 = df_res['new_h5'].mean()

print("\n" + "="*50)
print("=== METRICS SNAPSHOT ===")
print(f"MRR:      {mean_orig_mrr:.3f} -> {mean_new_mrr:.3f} (Delta: {mean_new_mrr - mean_orig_mrr:+.3f})")
print(f"Hit@1:    {mean_orig_h1:.3f} -> {mean_new_h1:.3f} (Delta: {mean_new_h1 - mean_orig_h1:+.3f})")
print(f"Hit@5:    {mean_orig_h5:.3f} -> {mean_new_h5:.3f} (Delta: {mean_new_h5 - mean_orig_h5:+.3f})")
print("="*50)

# Save
exp_dir = f"../experiments/{args.experiment_name}"
os.makedirs(exp_dir, exist_ok=True)
df_res.to_csv(f"{exp_dir}/sliding_window_metrics.csv", index=False)

agg_res = pd.DataFrame([{
    'Metric': 'MRR', 'New': mean_new_mrr, 'Old': mean_orig_mrr, 'Delta': mean_new_mrr - mean_orig_mrr
}, {
    'Metric': 'Hit@1', 'New': mean_new_h1, 'Old': mean_orig_h1, 'Delta': mean_new_h1 - mean_orig_h1
}, {
    'Metric': 'Hit@5', 'New': mean_new_h5, 'Old': mean_orig_h5, 'Delta': mean_new_h5 - mean_orig_h5
}])
agg_res.to_csv(f"{exp_dir}/aggregate_result.csv", index=False)

print(f"Saved metrics to {exp_dir}/sliding_window_metrics.csv")
print(f"Saved aggregate to {exp_dir}/aggregate_result.csv")


import os
import ast
import json
import time
import pandas as pd
import requests
from dotenv import load_dotenv
import sys
import concurrent.futures
import argparse

sys.path.append('..')
from stark_qa import load_skb

parser = argparse.ArgumentParser(description="RankGPT Listwise Sort via AWS Bedrock NIM")
parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to dictate output folder")
parser.add_argument("--complete-run", action="store_true", help="Run on all queries (bypasses num-good and num-bad)")
parser.add_argument("--num-good", type=int, default=20, help="Number of good queries to run")
parser.add_argument("--num-bad", type=int, default=20, help="Number of bad queries to run")
args = parser.parse_args()

# Load Environment
load_dotenv()
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY")

if not BEDROCK_API_KEY:
    raise ValueError("No BEDROCK_API_KEY found in environment!")

MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"
AWS_REGION = "us-east-1"
BASE_URL = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/model/{MODEL_ID}/invoke"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {BEDROCK_API_KEY}",
}

# Create a shared requests session since it handles thread-pooling well
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Load Data
print("Loading Knowledge Base...")
kb = load_skb('prime', download_processed=True)
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
print(f"Selected {len(queries)} total queries for listwise reranking on Bedrock.")

def parse_bedrock_json(raw_text, docs_list):
    # Strip markdown code blocks
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[-1].split("```")[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[-1].split("```")[0]
        
    start_idx = raw_text.find('[')
    end_idx = raw_text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_text = raw_text[start_idx:end_idx+1]
        
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            valid = [d for d in parsed if d in docs_list]
            missing = [d for d in docs_list if d not in valid]
            return valid + missing
    except json.JSONDecodeError:
        pass
        
    return None

def rerank_sublist(query_text, docs_list, kb, retries=3):
    docs_info = []
    for d in docs_list:
        d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
        n_type = str(kb.get_node_type_by_id(d))
        docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")
    docs_str = '\n'.join(docs_info)
        
    system_instruction = "You are an advanced search relevance ranker."
    prompt = f"""Below is a search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.

Query: {query_text}

Candidate Documents:
{docs_str}

Output your response strictly as a JSON array containing ONLY the document IDs in the sorted order. Do not include any explanations, markdown syntax, or other text.
Example format:
[1234, 5678, 91011]
"""

    payload = {
        "prompt": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_instruction}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        ),
        "max_gen_len": 200,
        "temperature": 0.0,
    }

    for attempt in range(retries):
        try:
            resp = SESSION.post(BASE_URL, json=payload, timeout=120)
            
            if resp.status_code == 429:
                print("    [!] Rate limit hit (429), sleeping 3s before retry...")
                time.sleep(3)
                continue
            
            if not resp.ok:
                print(f"    [Error] Bedrock API returned {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
                continue
                
            data = resp.json()
            raw = data.get("generation", "").strip()
            
            parsed = parse_bedrock_json(raw, docs_list)
            if parsed is not None:
                return parsed
                
        except requests.exceptions.RequestException as e:
            print(f"    [Error] Request Exception: {e}")
            time.sleep(2)
            
    print("    [!] Failed to rerank sublist after retries, keeping original order.")
    return docs_list

def mrr(lst, gts):
    for i, v in enumerate(lst):
        if v in gts: return 1.0/(i+1)
    return 0.0

# RUN EXPERIMENT
print("\n" + "="*50)
print(f"Starting Sliding Window Listwise Sort (Window=3, Stride=2) - BEDROCK ROUND-BY-ROUND BATCHING")
print("="*50)

window_size = 3
step = 2

# Initialize state for all queries
queries_state = []
for q in queries:
    docs_copy = q['top_20_nodes'].copy()
    queries_state.append({
        'id': q['id'],
        'query': q['query'],
        'docs': docs_copy,
        'orig_docs': docs_copy.copy(),
        'gts': set(q['ground_truths_list'])
    })

n = len(queries_state[0]['docs']) if queries_state else 0
if n <= window_size:
    starts = [0]
else:
    starts = list(range(n - window_size, -1, -step))
    if len(starts) == 0 or starts[-1] != 0:
        starts.append(0)

# Process Round-by-Round across ALL queries efficiently
for round_idx, start_pos in enumerate(starts, 1):
    print(f"\n>> Executing Round {round_idx}/{len(starts)} | Window Indices: [ {start_pos} : {start_pos+window_size} ]")
    
    def process_query_window(state):
        current_window = state['docs'][start_pos : start_pos+window_size]
        # Skip API call if the sliding window index falls entirely out-of-bounds for smaller arrays
        if len(current_window) <= 1:
            return state['id'], current_window
            
        new_window = rerank_sublist(state['query'], current_window, kb)
        return state['id'], new_window

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_state = {executor.submit(process_query_window, s): s for s in queries_state}
        for future in concurrent.futures.as_completed(future_to_state):
            try:
                q_id, new_window = future.result()
                # Apply the sorted window back into the state
                state = next(s for s in queries_state if s['id'] == q_id)
                state['docs'][start_pos : start_pos+window_size] = new_window
            except Exception as exc:
                q_id = future_to_state[future]['id']
                print(f"[!] Query {q_id} generated an exception during round {round_idx}: {exc}")

results = []
for state in queries_state:
    orig_docs = state['orig_docs']
    reranked = state['docs']
    gts = state['gts']
    results.append({
        'id': state['id'],
        'orig_mrr': mrr(orig_docs, gts),
        'new_mrr': mrr(reranked, gts),
        'orig_h1': 1.0 if orig_docs and orig_docs[0] in gts else 0.0,
        'new_h1': 1.0 if reranked and reranked[0] in gts else 0.0,
        'orig_h5': 1.0 if any(d in gts for d in orig_docs[:5]) else 0.0,
        'new_h5': 1.0 if any(d in gts for d in reranked[:5]) else 0.0
    })

df_res = pd.DataFrame(results)

mean_orig_mrr = df_res['orig_mrr'].mean() if not df_res.empty else 0
mean_new_mrr = df_res['new_mrr'].mean() if not df_res.empty else 0
mean_orig_h1 = df_res['orig_h1'].mean() if not df_res.empty else 0
mean_new_h1 = df_res['new_h1'].mean() if not df_res.empty else 0
mean_orig_h5 = df_res['orig_h5'].mean() if not df_res.empty else 0
mean_new_h5 = df_res['new_h5'].mean() if not df_res.empty else 0

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
import os
import ast
import time
import argparse
import pandas as pd
import cohere
import threading
import concurrent.futures
from collections import deque
from tqdm import tqdm
from stark_qa import load_skb

# API Keys for Cohere
API_KEYS = [
    "zUIi4u2suP7wMEOGZOeQaVP4LcXHvs7U1I4Avswd",
    "ki5sgOxJy797ufk7fhQiK7LE5l3DJiK9CTyGefQ2",
    "PS45TDnnGiQXwPafsH0SoN2w0IekBgjGLfJ9aiBJ",
    "NK3nihzYwIq73MmVEJavCGdUUvBsfoakrihCw0ef"
]

class RateLimitedKeyManager:
    def __init__(self, api_keys, rate_limit_per_min=10):
        self.api_keys = api_keys
        self.rate_limit = rate_limit_per_min
        self.usage = {key: deque() for key in api_keys}
        self.lock = threading.Lock()
    
    def get_key(self):
        while True:
            with self.lock:
                current_time = time.time()
                min_wait_time = float('inf')
                
                for key in self.api_keys:
                    timestamps = self.usage[key]
                    while timestamps and timestamps[0] < current_time - 62:
                        timestamps.popleft()
                    
                    if len(timestamps) < self.rate_limit:
                        timestamps.append(current_time)
                        return key
                    
                    if timestamps:
                        wait_time = timestamps[0] + 62 - current_time
                        if wait_time < min_wait_time:
                            min_wait_time = wait_time
                
                if min_wait_time == float('inf'):
                    min_wait_time = 1.0
            
            time.sleep(min_wait_time)

key_manager = RateLimitedKeyManager(API_KEYS)

def rerank_with_cohere_direct(query, documents, top_n=None):
    try:
        api_key = key_manager.get_key()
        co = cohere.ClientV2(api_key)
        
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=top_n,
            max_tokens_per_doc=48000
        )
        
        results = []
        for result in response.results:
            results.append({
                'index': result.index,
                'relevance_score': result.relevance_score
            })
        return results
    except Exception as e:
        print(f"Error invoking Cohere Direct API: {e}")
        return []

def rerank(top_k_node_ids, query, kb, max_k=20, compact_docs=False, add_rel=False):
    # Simplified rerank for Method 4 only
    if not top_k_node_ids:
        return []
        
    top_k_node_ids = top_k_node_ids[:max_k]
    
    documents = []
    for node_id in top_k_node_ids:
        doc_text = kb.get_doc_info(node_id, add_rel=add_rel, compact=compact_docs)
        documents.append(doc_text)
        
    rerank_results = rerank_with_cohere_direct(query, documents, top_n=len(documents))
    
    if not rerank_results:
            print("Cohere Direct Rerank failed or returned no results. Returning original order.")
            sorted_node_ids = top_k_node_ids
    else:
        sorted_node_ids = []
        for result in rerank_results:
            original_index = result['index']
            sorted_node_ids.append(top_k_node_ids[original_index])

    return sorted_node_ids

def process_single_query_with_retry(row, kb, max_k, compact_docs, add_rel, max_retries=3):
    query_text = row["query"]
    answer_list = row["answer_list"]
    
    for attempt in range(max_retries):
        try:
            return rerank(
                answer_list,
                query_text,
                kb=kb,
                max_k=max_k,
                compact_docs=compact_docs,
                add_rel=add_rel
            )
        except Exception as e:
            print(f"Error processing query '{query_text[:30]}...' (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return answer_list

def rerank_queries_parallel(df_to_process, kb, max_k, compact_docs, add_rel, max_workers=40):
    print(f"Starting parallel reranking with {max_workers} workers...")
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df_to_process.iterrows():
            futures.append(executor.submit(
                process_single_query_with_retry, 
                row, kb, max_k, compact_docs, add_rel
            ))
            
        results = []
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Reranking"):
            pass
            
    # We need to preserve order, so we can't just append in as_completed
    # Re-iterate futures in order
    results = [f.result() for f in futures]
    return results

def calculate_metrics(predicted_nodes, ground_truth_nodes):
    if len(predicted_nodes) < 1:
        return {'hit_at_1': 0.0, 'hit_at_5': 0.0, 'mrr': 0.0, 'recall_at_20': 0.0}
    
    gt_set = set(ground_truth_nodes)
    hit_at_1 = 1.0 if predicted_nodes[0] in gt_set else 0.0
    hit_at_5 = 1.0 if any(node in gt_set for node in predicted_nodes[:5]) else 0.0
    
    mrr = 0.0
    for rank, node in enumerate(predicted_nodes, 1):
        if node in gt_set:
            mrr = 1.0 / rank
            break
    
    predicted_set = set(predicted_nodes)
    relevant_found = len(gt_set.intersection(predicted_set))
    recall_at_20 = relevant_found / len(gt_set) if len(gt_set) > 0 else 0.0
    
    return {'hit_at_1': hit_at_1, 'hit_at_5': hit_at_5, 'mrr': mrr, 'recall_at_20': recall_at_20}

def evaluate_dataframe_metrics(df):
    metrics_list = []
    
    for i in range(len(df)):
        reranked = df["reranked_answers"].iloc[i]
        ground_truth = df["ground_truths"].iloc[i]
        
        if isinstance(reranked, str): reranked = ast.literal_eval(reranked)
        if isinstance(ground_truth, str): ground_truth = ast.literal_eval(ground_truth)

        metrics = calculate_metrics(reranked, ground_truth)
        metrics['id'] = df['id'].iloc[i]
        metrics['query'] = df['query'].iloc[i]
        metrics_list.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_list)
    
    print("="*20, "Reranked Metrics", "="*20)
    print(f"Hit@1: {metrics_df['hit_at_1'].mean():.4f}")
    print(f"Hit@5: {metrics_df['hit_at_5'].mean():.4f}")
    print(f"MRR: {metrics_df['mrr'].mean():.4f}")
    print(f"Recall@20: {metrics_df['recall_at_20'].mean():.4f}")
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description="Rerank queries using Cohere Direct API")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--dataset", type=str, default="prime", help="Dataset name (prime, amazon, mag)")
    parser.add_argument("--max_workers", type=int, default=40, help="Max parallel workers")
    
    args = parser.parse_args()
    
    print(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Preprocessing
    if "results" in df.columns:
        df["results"] = df["results"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Assuming metrics_on=1 logic from notebook where answer_list comes from vss_merged_candidates
        if "vss_merged_candidates" in df.columns:
             df["vss_merged_candidates"] = df["vss_merged_candidates"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
             df["answer_list"] = df["vss_merged_candidates"]
        else:
             # Fallback to extracting from results if vss_merged_candidates not present
             answers = []
             for x in df["results"]:
                 ans = x.get('answer_list', [])
                 answers.append(ans)
             df["answer_list"] = answers
    
    df["answer_list"] = df["answer_list"].map(lambda x: x[:20])
    if "ground_truths" in df.columns:
        df["ground_truths"] = df["ground_truths"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    print(f"Loading SKB for dataset: {args.dataset}")
    kb = load_skb(args.dataset, download_processed=True)
    
    print("Starting reranking...")
    reranked_answers = rerank_queries_parallel(
        df, 
        kb=kb, 
        max_k=20, 
        compact_docs=False, 
        add_rel=True,
        max_workers=args.max_workers
    )
    
    df_new = pd.DataFrame()
    df_new["id"] = df["id"]
    df_new["ground_truths"] = df["ground_truths"]
    df_new["query"] = df["query"]
    df_new["original_answers"] = df["answer_list"]
    df_new["reranked_answers"] = reranked_answers
    
    print(f"Saving results to: {args.output_csv}")
    df_new.to_csv(args.output_csv, index=False)
    
    print("Calculating metrics...")
    metrics_df = evaluate_dataframe_metrics(df_new)
    
    # Save summary and details
    base_dir = os.path.dirname(args.output_csv)
    base_name = os.path.splitext(os.path.basename(args.output_csv))[0]
    
    # Detailed metrics per query
    details_path = os.path.join(base_dir, f"{base_name}_details.csv")
    metrics_df.to_csv(details_path, index=False)
    print(f"Saved detailed metrics to: {details_path}")
    
    # Aggregate stats
    summary_stats = pd.DataFrame([{
        'Hit@1': metrics_df['hit_at_1'].mean(),
        'Hit@5': metrics_df['hit_at_5'].mean(),
        'MRR': metrics_df['mrr'].mean(),
        'Recall@20': metrics_df['recall_at_20'].mean()
    }])
    summary_stats_path = os.path.join(base_dir, f"{base_name}_summary.csv")
    summary_stats.to_csv(summary_stats_path, index=False)
    print(f"Saved aggregate summary to: {summary_stats_path}")

if __name__ == "__main__":
    main()
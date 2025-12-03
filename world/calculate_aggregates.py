import pandas as pd
import ast
import json
import sys
import os

def parse_results_column(result_str):
    """
    Safely parses the 'results' column which might be a stringified Python dict 
    or a JSON string.
    """
    if pd.isna(result_str) or result_str == "":
        return None
    
    try:
        # Try parsing as a Python literal (handles single quotes)
        return ast.literal_eval(result_str)
    except (ValueError, SyntaxError):
        try:
            # Try parsing as standard JSON (double quotes)
            return json.loads(result_str)
        except json.JSONDecodeError:
            return None

def main(input_file, output_file="aggregate_results_recalculated.csv"):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'results' not in df.columns:
        print("Error: Column 'results' not found in the CSV.")
        sys.exit(1)

    # Containers for metrics
    grounding_metrics = []
    vss_metrics = []

    print(f"Processing {len(df)} rows...")

    for index, row in df.iterrows():
        # Skip failed queries if necessary, or count them as 0s depending on your logic.
        # Here we only aggregate queries that actually produced a result dict.
        res_data = parse_results_column(row['results'])
        
        if not res_data:
            continue

        # Extract Step 4 Grounding Metrics
        g_met = res_data.get('metrics', {})
        if g_met:
            grounding_metrics.append(g_met)

        # Extract Step 5 VSS Merged Metrics
        v_met = res_data.get('vss_merged_metrics', {})
        if v_met:
            vss_metrics.append(v_met)

    if not grounding_metrics:
        print("No valid metrics found in the data dump.")
        sys.exit(0)

    # --- Calculate Averages ---
    num_queries = len(grounding_metrics)
    
    # Helper to safely get value or 0
    def get_val(metric_dict, key):
        return float(metric_dict.get(key, 0.0))

    avg_results = {
        'total_queries_processed': num_queries,
        'avg_total_answers':   sum(get_val(m, 'total_answers') for m in grounding_metrics) / num_queries,
        'avg_retrieved_count': sum(get_val(m, 'retrieved_count') for m in grounding_metrics) / num_queries,
        'avg_missed_count':    sum(get_val(m, 'missed_count') for m in grounding_metrics) / num_queries,
        
        # Grounding (Step 4)
        'avg_recall@20': sum(get_val(m, 'recall@20') for m in grounding_metrics) / num_queries,
        'avg_recall@50': sum(get_val(m, 'recall@50') for m in grounding_metrics) / num_queries,
        'avg_hit@1':     sum(get_val(m, 'hit_at_1') for m in grounding_metrics) / num_queries,
        'avg_hit@5':     sum(get_val(m, 'hit_at_5') for m in grounding_metrics) / num_queries,
        'avg_mrr':       sum(get_val(m, 'mrr') for m in grounding_metrics) / num_queries,
        
        # VSS Merged (Step 5)
        'recall@20_vss_merged': sum(get_val(m, 'recall@20') for m in vss_metrics) / num_queries if vss_metrics else 0.0
    }

    # --- Save to CSV ---
    output_df = pd.DataFrame([avg_results])
    output_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("AGGREGATE RESULTS")
    print("="*40)
    for k, v in avg_results.items():
        print(f"{k:<25}: {v:.4f}")
    print("="*40)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_aggregates.py <path_to_full_data_dump.csv>")
    else:
        main(sys.argv[1])
import csv
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate aggregate results from a pipeline results CSV.")
    parser.add_argument("--pipeline_results", required=True, help="Path to the pipeline_results.csv file")
    parser.add_argument("--output", default="aggregate_results.csv", help="Path to save the output aggregate CSV")
    return parser.parse_args()

def calculate_aggregate(pipeline_results_path, output_path):
    all_metrics = []
    
    try:
        with open(pipeline_results_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Extract standard metrics
                    metrics = {
                        "total_answers": float(row.get("total_answers", 0) or 0),
                        "retrieved_count": float(row.get("retrieved_count", 0) or 0),
                        "missed_count": float(row.get("missed_count", 0) or 0),
                        "recall@20": float(row.get("recall@20", 0) or 0),
                        "recall@50": float(row.get("recall@50", 0) or 0),
                        "hit_at_1": float(row.get("hit@1", 0) or 0),
                        "hit_at_5": float(row.get("hit@5", 0) or 0),
                        "mrr": float(row.get("mrr", 0) or 0),
                    }
                    
                    # Extract vss metrics
                    for a in range(12, 21):
                        prop = f"recall@20_alpha_{a}"
                        metrics[prop] = float(row.get(prop, 0) or 0)
                        
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error parsing row {row.get('query_id')}: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find pipeline results file at {pipeline_results_path}")
        return

    if not all_metrics:
        print("No valid metrics found in the pipeline results to calculate aggregate.")
        return

    num_queries = len(all_metrics)
    avg_metrics = {
        "total_queries":       num_queries,
        "avg_total_answers":   sum(m.get("total_answers",   0)   for m in all_metrics) / num_queries,
        "avg_retrieved_count": sum(m.get("retrieved_count", 0)   for m in all_metrics) / num_queries,
        "avg_missed_count":    sum(m.get("missed_count",    0)   for m in all_metrics) / num_queries,
        "avg_recall@20":       sum(m.get("recall@20",       0.0) for m in all_metrics) / num_queries,
        "avg_recall@50":       sum(m.get("recall@50",       0.0) for m in all_metrics) / num_queries,
        "avg_hit@1":           sum(m.get("hit_at_1",        0.0) for m in all_metrics) / num_queries,
        "avg_hit@5":           sum(m.get("hit_at_5",        0.0) for m in all_metrics) / num_queries,
        "avg_mrr":             sum(m.get("mrr",             0.0) for m in all_metrics) / num_queries,
    }
    
    for a in range(12, 21):
        prop = f"recall@20_alpha_{a}"
        total = sum(m.get(prop, 0.0) for m in all_metrics)
        avg_metrics[f"avg_{prop}"] = total / num_queries

    print(f"\n[DONE] Aggregate Calculation from {pipeline_results_path}")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_metrics.keys()))
        writer.writeheader()
        writer.writerow(avg_metrics)
        
    print(f"✓ Aggregate results saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    calculate_aggregate(args.pipeline_results, args.output)
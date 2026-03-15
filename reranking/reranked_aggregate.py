import pandas as pd
import argparse
import ast


def calculate_metrics(predicted_nodes, ground_truth_nodes):
    """
    Calculate Hit@1, Hit@5, MRR, and Recall@20 metrics.
    
    Args:
        predicted_nodes: List of 20 predicted nodes ranked in decreasing order
        ground_truth_nodes: List of ground truth nodes
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    if len(predicted_nodes) < 1:
        return {
            'hit_at_1': 0.0,
            'hit_at_5': 0.0,
            'mrr': 0.0,
            'recall_at_20': 0.0
            }
    
    # Convert ground truth to set for O(1) lookup
    gt_set = set(ground_truth_nodes)
    
    # Hit@1: Check if the first predicted node is in ground truth
    hit_at_1 = 1.0 if predicted_nodes[0] in gt_set else 0.0
    
    # Hit@5: Check if any of the first 5 predicted nodes are in ground truth
    hit_at_5 = 1.0 if any(node in gt_set for node in predicted_nodes[:5]) else 0.0
    
    # MRR (Mean Reciprocal Rank): Find the rank of the first relevant item
    mrr = 0.0
    for rank, node in enumerate(predicted_nodes, 1):
        if node in gt_set:
            mrr = 1.0 / rank
            break
    
    # Recall@20: Fraction of ground truth nodes found in top 20 predictions
    predicted_set = set(predicted_nodes)
    relevant_found = len(gt_set.intersection(predicted_set))
    recall_at_20 = relevant_found / len(gt_set) if len(gt_set) > 0 else 0.0
    
    return {
        'hit_at_1': hit_at_1,
        'hit_at_5': hit_at_5,
        'mrr': mrr,
        'recall_at_20': recall_at_20
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Rerank SKB answers with LLM.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="prime",
        help="Dataset name for load_skb (e.g., 'prime', 'amazon', 'mag')."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV (full_data_dump.csv)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV (reranked_full_data_dump.csv)."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.input_csv)

    # Parse results column
    df["reranked_answers"] = df["reranked_answers"].map(lambda x: ast.literal_eval(x))
    df["original_answers"] = df["original_answers" ].map(lambda x: ast.literal_eval(x))
    df["ground_truths"] = df["ground_truths"].map(lambda x: ast.literal_eval(x))

    hit_at_1 = []
    hit_at_5 = []
    mrr = []
    recall_at_20 = []

    hit_at_1_org = []
    hit_at_5_org = []
    mrr_org = []
    recall_at_20_org = []

    for i in range(len(df)):
        metrics =calculate_metrics(df["reranked_answers"][i], df["ground_truths"][i])
        metrics_org =calculate_metrics(df["original_answers"][i], df["ground_truths"][i])

        hit_at_1.append(metrics['hit_at_1'])
        hit_at_5.append(metrics['hit_at_5'])
        mrr.append(metrics['mrr'])
        recall_at_20.append(metrics['recall_at_20'])

        hit_at_1_org.append(metrics_org['hit_at_1'])
        hit_at_5_org.append(metrics_org['hit_at_5'])
        mrr_org.append(metrics_org['mrr'])
        recall_at_20_org.append(metrics_org['recall_at_20'])
    

    df["hit_at_1"] = hit_at_1
    df["hit_at_5"] = hit_at_5
    df["mrr"] = mrr     
    df["recall_at_20"] = recall_at_20

    df.to_csv(args.output_csv, index=False)

    print("="*20, "Original metrics", "="*20)
    print(f"Hit@1: {sum(hit_at_1_org)/len(hit_at_1_org):.4f}")
    print(f"Hit@5: {sum(hit_at_5_org)/len(hit_at_5_org):.4f}")
    print(f"MRR: {sum(mrr_org)/len(mrr_org):.4f}")
    print(f"Recall@20: {sum(recall_at_20_org)/len(recall_at_20_org):.4f}")

    print("="*20, "New metrics", "="*20)
    print(f"Hit@1: {df['hit_at_1'].mean():.4f}")
    print(f"Hit@5: {df['hit_at_5'].mean():.4f}")
    print(f"MRR: {df['mrr'].mean():.4f}")
    print(f"Recall@20: {df['recall_at_20'].mean():.4f}")
    
    

if __name__ == "__main__":
    main()

    
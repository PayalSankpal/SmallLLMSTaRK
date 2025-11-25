import pandas as pd
import ast
import numpy as np


def parse_list_string(s):
    """Parse string representation of list to actual list."""
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


def compute_metrics_for_combined(ground_truths, combined_list):
    """
    Compute evaluation metrics for combined answer list.
    
    Args:
        ground_truths: List of ground truth node IDs
        combined_list: Combined answer list
    
    Returns:
        Dictionary containing metrics
    """
    gt_set = set(ground_truths)
    retrieved_set = set(combined_list)
    
    # Count retrieved ground truths
    retrieved_gts = gt_set.intersection(retrieved_set)
    missed_gts = gt_set - retrieved_set
    
    total_answers = len(ground_truths)
    retrieved_count = len(retrieved_gts)
    missed_count = len(missed_gts)
    
    # Recall
    recall = retrieved_count / total_answers if total_answers > 0 else 0.0
    
    # Hit@k
    hit_1 = 1.0 if len(combined_list) > 0 and combined_list[0] in gt_set else 0.0
    hit_5 = 1.0 if any(rid in gt_set for rid in combined_list[:5]) else 0.0
    hit_10 = 1.0 if any(rid in gt_set for rid in combined_list[:10]) else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, rid in enumerate(combined_list):
        if rid in gt_set:
            mrr = 1.0 / (idx + 1)
            break
    
    return {
        'total_answers': total_answers,
        'retrieved_count': retrieved_count,
        'missed_count': missed_count,
        'recall': recall,
        'hit@1': hit_1,
        'hit@5': hit_5,
        'hit@10': hit_10,
        'mrr': mrr
    }


def merge_and_evaluate(df1, df2, output_csv='combined_results.csv'):
    """
    Merge df1 and df2, create combined answer lists, and calculate metrics.
    
    Args:
        df1: DataFrame with columns ['id', 'query', 'results', ...]
        df2: DataFrame with columns ['q_id', 'q_string', 'top_20_vss_array', 'ground_truths_array', ...]
        output_csv: Path to save the output CSV
    
    Returns:
        DataFrame with combined results and metrics
    """
    # Merge dataframes on id/q_id
    merged = pd.merge(df1, df2, left_on='id', right_on='q_id', how='inner')
    
    print(f"Merged {len(merged)} rows")
    
    results_list = []
    
    for idx, row in merged.iterrows():
        # Parse df1 results
        results_dict = parse_list_string(row['results'])
        df1_answer_list = results_dict.get('answer_list', [])
        
        # Parse df2 top_20_vss_array
        top_20_vss = parse_list_string(row['top_20_vss_array'])
        
        # Parse ground truths
        ground_truths = parse_list_string(row['ground_truths_array'])
        
        # Create combined list:
        combined_answer_list = df1_answer_list.copy()[0:12]
        df1_set = set(combined_answer_list)
        
        # Add up to 8 unique elements from top_20_vss
        added_count = 0
        for item in top_20_vss:
            if item not in df1_set and added_count < 8:
                combined_answer_list.append(item)
                added_count += 1
        
        # Calculate metrics for combined list
        metrics = compute_metrics_for_combined(ground_truths, combined_answer_list)
        
        # Store results
        results_list.append({
            'q_id': row['q_id'],
            'q_string': row['q_string'],
            'df1_answer_list': df1_answer_list,
            'top_20_vss_array': top_20_vss,
            'combined_answer_list': combined_answer_list,
            'ground_truths': ground_truths,
            'total_answers': metrics['total_answers'],
            'retrieved_count': metrics['retrieved_count'],
            'missed_count': metrics['missed_count'],
            'recall': metrics['recall'],
            'hit@1': metrics['hit@1'],
            'hit@5': metrics['hit@5'],
            'hit@10': metrics['hit@10'],
            'mrr': metrics['mrr']
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(merged)} rows")
    
    # Create result dataframe
    result_df = pd.DataFrame(results_list)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    
    # Print aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE METRICS FOR COMBINED ANSWER LISTS")
    print("="*80)
    print(f"Total Queries:           {len(result_df)}")
    print(f"Avg Total Answers:       {result_df['total_answers'].mean():.2f}")
    print(f"Avg Retrieved Count:     {result_df['retrieved_count'].mean():.2f}")
    print(f"Avg Missed Count:        {result_df['missed_count'].mean():.2f}")
    print(f"Avg Recall:              {result_df['recall'].mean():.4f}")
    print(f"Avg Hit@1:               {result_df['hit@1'].mean():.4f}")
    print(f"Avg Hit@5:               {result_df['hit@5'].mean():.4f}")
    print(f"Avg Hit@10:              {result_df['hit@10'].mean():.4f}")
    print(f"Avg MRR:                 {result_df['mrr'].mean():.4f}")
    print("="*80)
    
    # Save aggregate results
    aggregate_csv = output_csv.replace('.csv', '_aggregate.csv')
    agg_results = pd.DataFrame([{
        'metric': 'total_queries',
        'value': len(result_df)
    }, {
        'metric': 'avg_total_answers',
        'value': f"{result_df['total_answers'].mean():.2f}"
    }, {
        'metric': 'avg_retrieved_count',
        'value': f"{result_df['retrieved_count'].mean():.2f}"
    }, {
        'metric': 'avg_missed_count',
        'value': f"{result_df['missed_count'].mean():.2f}"
    }, {
        'metric': 'avg_recall',
        'value': f"{result_df['recall'].mean():.4f}"
    }, {
        'metric': 'avg_hit@1',
        'value': f"{result_df['hit@1'].mean():.4f}"
    }, {
        'metric': 'avg_hit@5',
        'value': f"{result_df['hit@5'].mean():.4f}"
    }, {
        'metric': 'avg_hit@10',
        'value': f"{result_df['hit@10'].mean():.4f}"
    }, {
        'metric': 'avg_mrr',
        'value': f"{result_df['mrr'].mean():.4f}"
    }])
    
    agg_results.to_csv(aggregate_csv, index=False)
    print(f"Saved aggregate results to {aggregate_csv}\n")
    
    return result_df


# Usage example:
if __name__ == "__main__":
    # Load your dataframes
    df1 = pd.read_csv("output/AMAZON/full_data_dump.csv")
    df2 = pd.read_csv("vss_results/results.csv")
    
    # Merge and evaluate
    result_df = merge_and_evaluate(df1, df2, 'combined_results.csv')
    
    # Display first few rows
    print("\nFirst 5 rows of combined results:")
    print(result_df[['q_id', 'q_string', 'total_answers', 'retrieved_count', 
                     'recall', 'hit@1', 'mrr']].head())
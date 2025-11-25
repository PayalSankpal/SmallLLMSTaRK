import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from stark_qa import load_skb, load_qa
from vss import VSS

def compute_metrics(ground_truths: List[int], retrieved_ids: List[int], retrieved_scores: List[float]) -> dict:
    """
    Compute evaluation metrics for a single query.
    
    Args:
        ground_truths: List of ground truth node IDs
        retrieved_ids: List of retrieved node IDs (top-k)
        retrieved_scores: List of similarity scores for retrieved nodes
    
    Returns:
        Dictionary containing metrics
    """
    gt_set = set(ground_truths)
    retrieved_set = set(retrieved_ids)
    
    # Count retrieved ground truths
    retrieved_gts = gt_set.intersection(retrieved_set)
    missed_gts = gt_set - retrieved_set
    
    total_answers = len(ground_truths)
    retrieved_count = len(retrieved_gts)
    missed_count = len(missed_gts)
    
    # Recall@k
    recall_20 = retrieved_count / total_answers if total_answers > 0 else 0.0
    recall_50 = recall_20  # Same as recall@20 since we only retrieve top 20
    
    # Hit@k
    hit_1 = 1.0 if len(retrieved_ids) > 0 and retrieved_ids[0] in gt_set else 0.0
    hit_5 = 1.0 if any(rid in gt_set for rid in retrieved_ids[:5]) else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            mrr = 1.0 / (idx + 1)
            break
    
    return {
        'retrieved_gts': list(retrieved_gts),
        'missed_gts': list(missed_gts),
        'total_answers': total_answers,
        'retrieved_count': retrieved_count,
        'missed_count': missed_count,
        'recall@20': recall_20,
        'recall@50': recall_50,
        'hit@1': hit_1,
        'hit@5': hit_5,
        'mrr': mrr
    }


def calculate_optimal_batch_size(num_products: int, embedding_dim: int, gpu_memory_gb: float = 8.0):
    """
    Calculate optimal batch size based on GPU memory.
    
    Memory calculation:
    - Product embeddings: num_products × embedding_dim × 4 bytes (float32)
    - Query batch embeddings: batch_size × embedding_dim × 4 bytes
    - Similarity matrix: batch_size × num_products × 4 bytes
    
    Args:
        num_products: Number of product embeddings
        embedding_dim: Dimension of embeddings
        gpu_memory_gb: Available GPU memory in GB
    
    Returns:
        Optimal batch size
    """
    bytes_per_float = 4
    available_bytes = gpu_memory_gb * 1024**3
    
    # Reserve 2GB for other operations and overhead
    usable_bytes = available_bytes - (2 * 1024**3)
    
    # Product embeddings (loaded once)
    product_emb_bytes = num_products * embedding_dim * bytes_per_float
    
    # Remaining memory for batch operations
    remaining_bytes = usable_bytes - product_emb_bytes
    
    # For each query in batch: query_emb + row in similarity matrix
    bytes_per_query = (embedding_dim * bytes_per_float) + (num_products * bytes_per_float)
    
    # Calculate batch size
    batch_size = max(1, int(remaining_bytes / bytes_per_query))
    
    # Memory report
    print("\n" + "="*80)
    print("GPU MEMORY CALCULATION")
    print("="*80)
    print(f"Available GPU memory:        {gpu_memory_gb:.2f} GB")
    print(f"Reserved for overhead:       2.00 GB")
    print(f"Usable memory:               {usable_bytes / 1024**3:.2f} GB")
    print(f"Product embeddings size:     {product_emb_bytes / 1024**3:.2f} GB ({num_products} × {embedding_dim})")
    print(f"Remaining for batches:       {remaining_bytes / 1024**3:.2f} GB")
    print(f"Memory per query:            {bytes_per_query / 1024**2:.2f} MB")
    print(f"Optimal batch size:          {batch_size} queries")
    print(f"Similarity matrix per batch: {batch_size * num_products * bytes_per_float / 1024**3:.2f} GB")
    print("="*80 + "\n")
    
    return batch_size


def evaluate_vss_parallel(vss, queries: List[Tuple], output_csv: str, k: int = 20, 
                         batch_size: int = None, gpu_memory_gb: float = 8.0,
                         product_chunk_size: int = None):
    """
    Evaluate VSS retrieval using parallelized batch processing with chunked product embeddings.
    
    Args:
        vss: VSS object with embeddings
        queries: List of tuples (q_string, q_id, ground_truth_array, None)
        output_csv: Path to output CSV file
        k: Number of top nodes to retrieve (default: 20)
        batch_size: Number of queries to process in parallel (auto-calculated if None)
        gpu_memory_gb: Available GPU memory in GB (default: 8.0)
        product_chunk_size: Number of products to load at once (auto-calculated if None)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get product node type embeddings and IDs
    node_type = 'product'
    if node_type not in vss.node_emb_dict:
        raise ValueError(f"Node type '{node_type}' not found in embeddings")
    
    # Keep product embeddings on CPU initially
    product_embs_cpu = vss.node_emb_dict[node_type]  # Shape: [num_products, emb_dim]
    product_ids = vss.node_ids_by_type[node_type]
    
    num_products = len(product_ids)
    emb_dim = product_embs_cpu.shape[1]
    
    print(f"Total {num_products} product embeddings of dimension {emb_dim}")
    
    # Auto-calculate product chunk size if not provided
    if product_chunk_size is None:
        # Use half the products per chunk to be safe with memory
        product_chunk_size = num_products // 2
        print(f"Auto-calculated product chunk size: {product_chunk_size}")
    
    num_product_chunks = (num_products + product_chunk_size - 1) // product_chunk_size
    print(f"Will process products in {num_product_chunks} chunks")
    
    # Calculate optimal batch size based on chunk size
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(product_chunk_size, emb_dim, gpu_memory_gb)
    else:
        print(f"Using provided batch size: {batch_size}")
    
    # Collect all query embeddings
    query_embs_list = []
    valid_queries = []
    
    print("Loading query embeddings...")
    for q_string, q_id, ground_truths, _ in queries:
        try:
            q_emb = vss.get_query_emb(q_string, q_id)
            # Ensure embedding is 1D
            if q_emb.dim() > 1:
                q_emb = q_emb.reshape(-1)
            query_embs_list.append(q_emb)
            valid_queries.append((q_string, q_id, ground_truths))
        except Exception as e:
            print(f"Error loading embedding for query {q_id}: {e}")
            continue
    
    num_queries = len(valid_queries)
    print(f"Successfully loaded {num_queries} query embeddings")
    
    # Verify all embeddings have the same dimension
    if num_queries > 0:
        emb_dims = [emb.shape[0] for emb in query_embs_list]
        if len(set(emb_dims)) > 1:
            print(f"WARNING: Query embeddings have different dimensions: {set(emb_dims)}")
        print(f"Query embedding dimension: {query_embs_list[0].shape[0]}")
    
    # Process queries in batches
    results = []
    all_metrics = []
    
    num_batches = (num_queries + batch_size - 1) // batch_size
    print(f"\nProcessing {num_queries} queries in {num_batches} batches of up to {batch_size} queries each")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_queries)
        current_batch_size = batch_end - batch_start
        
        print(f"\n{'='*80}")
        print(f"Query Batch {batch_idx + 1}/{num_batches}: Processing queries {batch_start + 1}-{batch_end}")
        print(f"{'='*80}")
        
        # Get batch of query embeddings and move to GPU
        batch_query_embs = torch.stack(query_embs_list[batch_start:batch_end])
        
        # Debug: print shape before moving to GPU
        print(f"Query batch shape (CPU): {batch_query_embs.shape}")
        
        batch_query_embs = batch_query_embs.to(device)
        
        # Normalize - ensure we're normalizing along the embedding dimension (dim=1)
        if batch_query_embs.dim() == 2:
            batch_query_embs = torch.nn.functional.normalize(batch_query_embs, p=2, dim=1)
        else:
            print(f"WARNING: Unexpected query batch dimensions: {batch_query_embs.shape}")
            batch_query_embs = batch_query_embs.reshape(current_batch_size, -1)
            batch_query_embs = torch.nn.functional.normalize(batch_query_embs, p=2, dim=1)
        
        print(f"Query batch shape: {batch_query_embs.shape}")
        
        # Initialize storage for top-k results across all product chunks
        # We'll keep top-k from each chunk and then merge
        all_top_k_scores = []
        all_top_k_indices = []
        
        # Process products in chunks
        for prod_chunk_idx in range(num_product_chunks):
            prod_start = prod_chunk_idx * product_chunk_size
            prod_end = min(prod_start + product_chunk_size, num_products)
            
            print(f"\n  Product Chunk {prod_chunk_idx + 1}/{num_product_chunks}: products {prod_start}-{prod_end}")
            
            # Load this chunk of products to GPU
            product_chunk = product_embs_cpu[prod_start:prod_end]
            
            # Debug: print shape before moving to GPU
            print(f"  Product chunk shape (CPU): {product_chunk.shape}")
            
            product_chunk = product_chunk.to(device)
            
            # Normalize - ensure correct dimensions
            if product_chunk.dim() == 2:
                product_chunk = torch.nn.functional.normalize(product_chunk, p=2, dim=1)
            else:
                print(f"  WARNING: Unexpected product chunk dimensions: {product_chunk.shape}")
                product_chunk = product_chunk.reshape(-1, emb_dim)
                product_chunk = torch.nn.functional.normalize(product_chunk, p=2, dim=1)
            
            print(f"  Product chunk shape: {product_chunk.shape}")
            
            # *** MATRIX MULTIPLICATION FOR THIS CHUNK ***
            with torch.no_grad():
                similarities = torch.matmul(batch_query_embs, product_chunk.T)
            
            print(f"  Similarity matrix: {similarities.shape}")
            
            # Get top-k for this chunk
            chunk_top_k_scores, chunk_top_k_indices = torch.topk(
                similarities, k=min(k, product_chunk.shape[0]), 
                dim=1, largest=True, sorted=True
            )
            
            # Adjust indices to global product indices
            chunk_top_k_indices = chunk_top_k_indices + prod_start
            
            # Store results from this chunk
            all_top_k_scores.append(chunk_top_k_scores.cpu())
            all_top_k_indices.append(chunk_top_k_indices.cpu())
            
            # Clear GPU memory for this product chunk
            del product_chunk, similarities
            torch.cuda.empty_cache()
            
            print(f"  Completed chunk {prod_chunk_idx + 1}/{num_product_chunks}")
        
        # Clear query embeddings from GPU
        del batch_query_embs
        torch.cuda.empty_cache()
        
        print(f"\n  Merging results from {num_product_chunks} product chunks...")
        
        # Merge top-k results from all chunks
        # Concatenate all top-k scores and indices
        merged_scores = torch.cat(all_top_k_scores, dim=1)  # [batch_size, k * num_chunks]
        merged_indices = torch.cat(all_top_k_indices, dim=1)  # [batch_size, k * num_chunks]
        
        # Get final top-k across all chunks
        final_top_k_scores, topk_positions = torch.topk(merged_scores, k=min(k, merged_scores.shape[1]), 
                                                         dim=1, largest=True, sorted=True)
        
        # Get corresponding indices
        final_top_k_indices = torch.gather(merged_indices, 1, topk_positions)
        
        print(f"  Final top-{k} results shape: {final_top_k_indices.shape}")
        
        # Process each query in the batch
        for i in range(current_batch_size):
            query_idx = batch_start + i
            q_string, q_id, ground_truths = valid_queries[query_idx]
            
            # Get top-k node IDs for this query
            top_k_node_ids = [product_ids[idx] for idx in final_top_k_indices[i].tolist()]
            scores = final_top_k_scores[i].tolist()
            
            # Compute metrics
            metrics = compute_metrics(ground_truths, top_k_node_ids, scores)
            all_metrics.append(metrics)
            
            # Store results
            results.append({
                'q_id': q_id,
                'q_string': q_string,
                'top_20_vss_array': top_k_node_ids,
                'ground_truths_array': ground_truths,
                'ground_truths_missed_array': metrics['missed_gts'],
                'total_answers': metrics['total_answers'],
                'retrieved_count': metrics['retrieved_count'],
                'missed_count': metrics['missed_count'],
                'recall@20': metrics['recall@20'],
                'recall@50': metrics['recall@50'],
                'hit@1': metrics['hit@1'],
                'hit@5': metrics['hit@5'],
                'mrr': metrics['mrr']
            })
        
        print(f"\nCompleted query batch {batch_idx + 1}/{num_batches}")
        
        # Final cleanup
        torch.cuda.empty_cache()
    
    # Write results to CSV
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['q_id', 'q_string', 'top_20_vss_array', 'ground_truths_array', 
                     'ground_truths_missed_array', 'total_answers', 'retrieved_count', 
                     'missed_count', 'recall@20', 'recall@50', 'hit@1', 'hit@5', 'mrr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Compute aggregate metrics
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    
    total_queries = len(all_metrics)
    avg_total_answers = np.mean([m['total_answers'] for m in all_metrics])
    avg_retrieved_count = np.mean([m['retrieved_count'] for m in all_metrics])
    avg_missed_count = np.mean([m['missed_count'] for m in all_metrics])
    avg_recall_20 = np.mean([m['recall@20'] for m in all_metrics])
    avg_recall_50 = np.mean([m['recall@50'] for m in all_metrics])
    avg_hit_1 = np.mean([m['hit@1'] for m in all_metrics])
    avg_hit_5 = np.mean([m['hit@5'] for m in all_metrics])
    avg_mrr = np.mean([m['mrr'] for m in all_metrics])
    
    print(f"Total Queries:           {total_queries}")
    print(f"Avg Total Answers:       {avg_total_answers:.2f}")
    print(f"Avg Retrieved Count:     {avg_retrieved_count:.2f}")
    print(f"Avg Missed Count:        {avg_missed_count:.2f}")
    print(f"Avg Recall@20:           {avg_recall_20:.4f}")
    print(f"Avg Recall@50:           {avg_recall_50:.4f}")
    print(f"Avg Hit@1:               {avg_hit_1:.4f}")
    print(f"Avg Hit@5:               {avg_hit_5:.4f}")
    print(f"Avg MRR:                 {avg_mrr:.4f}")
    print("="*80)
    
    # Save aggregate results
    aggregate_csv = output_csv.replace('.csv', '_aggregate.csv')
    with open(aggregate_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_queries', total_queries])
        writer.writerow(['avg_total_answers', f'{avg_total_answers:.2f}'])
        writer.writerow(['avg_retrieved_count', f'{avg_retrieved_count:.2f}'])
        writer.writerow(['avg_missed_count', f'{avg_missed_count:.2f}'])
        writer.writerow(['avg_recall@20', f'{avg_recall_20:.4f}'])
        writer.writerow(['avg_recall@50', f'{avg_recall_50:.4f}'])
        writer.writerow(['avg_hit@1', f'{avg_hit_1:.4f}'])
        writer.writerow(['avg_hit@5', f'{avg_hit_5:.4f}'])
        writer.writerow(['avg_mrr', f'{avg_mrr:.4f}'])
    
    print(f"\nAggregate results saved to {aggregate_csv}")
    print(f"Detailed results saved to {output_csv}")
    
    return results, all_metrics

# Example usage:
if __name__ == "__main__":
    # Example: Load your VSS object and queries
    # from your_module import VSS, load_queries
    
    # vss = VSS(...)  # Your initialized VSS object
    kb = load_skb("amazon")
    qa_dataset = load_qa('amazon')
    qa = qa_dataset.split_indices["test"].reshape(-1).tolist()

    qa = qa[:int(len(qa) * 0.1)]
    queries = [qa_dataset[i] for i in qa]
    emb_model = 'text-embedding-ada-002'

    dataset = "amazon"
    node_ids_by_type = {}
    for n_type in kb.node_type_lst():
        node_ids_by_type[n_type] = kb.get_node_ids_by_type(n_type)
    vss = VSS(
                kb,
                Path(f"emb/{dataset}"),
                f"{dataset}",
                queries,
                emb_model,
                node_ids_by_type,
                False
            )
    
    
    results, metrics = results, metrics = evaluate_vss_parallel(
    vss, queries, 'results.csv', 
    k=20,
    product_chunk_size=300000  # ~300K products per chunk
)

    
    print("Script ready. Call evaluate_vss_gpu(vss, queries, output_csv) to run evaluation.")
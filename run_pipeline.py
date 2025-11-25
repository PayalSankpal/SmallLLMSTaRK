import pandas as pd
from stark_qa import load_skb
import regex as re
import ast
import torch
from custom_pipeline import llm_bridge
import vss
from stark_qa import load_qa
import sys
from urllib import response
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Any
from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.llm_bridge import LlmBridge
from custom_pipeline.query import Query
from custom_pipeline.prompt_generator import get_entity_extraction_prompt, get_relation_extraction_prompt
from custom_pipeline.vss_retriever_gpu import VSSRetriever
from custom_pipeline.candidate_context import CandidateContext
from custom_pipeline.grounders.grounder3 import PriorityQueueGrounding
import os
import contextlib
import io
import traceback
from tqdm import tqdm
import csv
from pathlib import Path
from typing import Optional
import heapq
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from custom_pipeline.candidate_context import CandidateContext
import json


def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    return response[0][0]


def step1_identify_entities(query: Query, llm_bridge: LlmBridge, dataset_name: str,use_saved_llm_responses: bool = False, llm_response: Optional[Dict] = None):
    
    if use_saved_llm_responses and llm_response is not None:
        response_string = llm_response.get('entities', "")
    else:
        prompt = get_entity_extraction_prompt(query.query, dataset_name)
        response_string = get_llm_response(prompt, llm_bridge)

    query.entity_id_response = response_string
    
    if response_string == '':
        query.status = "FAILED"
        return
    else:
        try:
            identified_entities = parse_entity_response(response_string)
            query.entities = identified_entities
            print(f"\n✅ STEP 1 SUCCESS - Entities: {list(query.entities.keys())}")  # ADD THIS
            print(query)  # ADD THIS
        except ValueError as e:
            print(f"\n❌ STEP 1 FAILED - Error: {e}")  # ADD THIS
            query.status = "FAILED"
            return
        
def step2_identify_relations(query: Query, llm_bridge: LlmBridge, dataset_name: str,use_saved_llm_responses: bool = False, llm_response: Optional[Dict] = None):
    if use_saved_llm_responses and llm_response is not None:
        response_string = llm_response.get("relations", "")
    else:
        prompt = get_relation_extraction_prompt(dataset_name , query.query, query.entity_id_response)  # Pass query.entities directly, not str(query.entities)
        response_string = get_llm_response(prompt, llm_bridge)
    query.relations_id_response = response_string

    if response_string == '' or response_string == '{}':  # Also check for empty dict
        query.status = "FAILED"
        query.relations = {}  # Set empty dict as default
        return
    else:
        try:
            identified_relations = parse_relation_string(response_string)
            print("\n\n\n\nIDENTIFIED RELATIONS\n\n\n", identified_relations,"\n\n")
            query.relations = identified_relations
            print(query)
        except ValueError as e:
            query.status = "FAILED"
            query.relations = {}  # Set empty dict as default
            return

def get_initial_candidates_for_entity(entity_info, entity_key, kb, retriever):
    candidates = []
    entity_types = entity_info.get("type", [])
    name_constraint = entity_info.get("lexical", {}).get("name", None)
    semantic_constraints = entity_info.get("semantic", []).copy()
    
    for key in entity_info.get("lexical", {}):
        semantic_constraints.append(f" {entity_info['lexical'][key]}")
   
    nodes_by_name = []

    if name_constraint:
        for etype in entity_types:
            nodes_by_name = kb.get_node_ids_by_value(node_type=etype, key="name", value=name_constraint)
            candidates.extend([CandidateContext(node_id=x, entity=entity_key, score=1) for x in nodes_by_name])

    vss_candidates_count = 25 - len(candidates)

    for etype in entity_types:
        sem = "".join(semantic_constraints)
        nodes_by_desc_ids, vss_scores = retriever.get_top_k_nodes(
            search_str=sem, k=vss_candidates_count, node_type=etype, cutoff=0.65
        )
        candidates.extend([
            CandidateContext(node_id=nodes_by_desc_ids[x], entity=entity_key, score=vss_scores[x])
            for x in range(len(nodes_by_desc_ids)) if nodes_by_desc_ids[x] not in nodes_by_name
        ])
    
    return candidates


def step3_get_initial_candidates(current_query, kb, retriever):
    initial_candidates = {}
    
    for entity in current_query.entities:
        if entity == "ANSWER":
            continue
        initial_candidates[entity] = get_initial_candidates_for_entity(
            current_query.entities[entity], entity_key=entity, kb=kb, retriever=retriever
        )
    current_query.initial_symbol_candidates = initial_candidates

    print(current_query)


def run_priority_queue_grounding(
        
    query_obj,
    kb,
    vss_retriever,
    max_candidates_per_symbol: int = 1000,
    max_answer_candidates: int = 50,
    top_k_neighbors: int = 10,
    score_decay: float = 0.9,
    support_boost: float = 0.15,
    verbose: bool = False
) -> Dict[str, List[CandidateContext]]:
    grounder = PriorityQueueGrounding(
        query_obj=query_obj,
        kb=kb,
        vss_retriever=vss_retriever,
        max_candidates_per_symbol=max_candidates_per_symbol,
        max_answer_candidates=max_answer_candidates,
        top_k_neighbors=top_k_neighbors,
        score_decay=score_decay,
        support_boost=support_boost,
        verbose=verbose
    )
    
    return grounder.ground()
    

def step4_grounding(query: Query, kb, retriever):
    final_candidates = run_priority_queue_grounding(
        query_obj=query,
        kb=kb,
        vss_retriever=retriever,
        max_candidates_per_symbol=100,
        max_answer_candidates=20,
        top_k_neighbors=15,
        support_boost=0,
        score_decay=0.9,
        verbose=False
    )
    answers = [cc.node_id for cc in sorted(
        final_candidates["ANSWER"], key=lambda x: (x.score, -x.support), reverse=True
    )]
    query.grounding_candidates = answers
        # print(query)

    return final_candidates

def step5_merge_vss_candidates(query: Query, retriever: VSSRetriever, kb, querywise_vss_candidates, use_saved_vss_candidates: bool = False,alpha:int = 12):
    vss_candidates = []
    if not use_saved_vss_candidates:
        
        possible_node_types = query.entities["ANSWER"]["type"].copy()
        for node_type in possible_node_types:
            print("\n\n\nSEARCHING VSS FOR NODE TYPE:", node_type,"\n\n")
            print(query.query)
            top_canididates = retriever.get_top_k_nodes(
                search_str=query.query, k=20, node_type=node_type, cutoff=0.6
            )
            print("\n\n\nTOP CANDIDATES FROM VSS\n\n\n", top_canididates,"\n\n")
            vss_candidates.extend(list(zip(top_canididates[0], top_canididates[1])))
    else:
        vss_candidates = querywise_vss_candidates.get(query.id, [])

    print("\n\n\nVSS CANDIDATES\n\n\n", vss_candidates,"\n\n")
    vss_candidates.sort(key=lambda x: x[1], reverse=True)

    existing_candidate_ids = query.grounding_candidates[:alpha]

    new_vss_candidates = []
    for node, score in vss_candidates:
        if node not in existing_candidate_ids:
            new_vss_candidates.append(node)
        if len(new_vss_candidates) == 20 - alpha:
            break
    
    merged_candidates = existing_candidate_ids + new_vss_candidates
    query.vss_merged_candidates = merged_candidates
    evaluate_results_after_merging_vss_candidates(query)
    return merged_candidates


def evaluate_results_after_merging_vss_candidates(query: Query):
    results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = results['metrics']
    print("\n\nRESULTS AFTER MERGING VSS CANDIDATES\n\n", results,"\n\n")
    return results

def evaluate_results(predicted_nodes, ground_truth_nodes):
    ground_truth_set = set(ground_truth_nodes)
    
    if not predicted_nodes:
        return {
            "answer_set": set(),
            "ground_truth_set": ground_truth_set,
            "retrieved_ground_truths": set(),
            "missed_ground_truths": ground_truth_set,
            "metrics": {
                "total_answers": len(ground_truth_set),
                "retrieved_count": 0,
                "missed_count": len(ground_truth_set),
                "recall@50": 0.0,
                "hit_at_1": 0.0,
                "hit_at_5": 0.0,
                "mrr": 0.0,
            }
        }
    
    hit_at_1 = 1.0 if predicted_nodes[0] in ground_truth_set else 0.0
    hit_at_5 = 1.0 if any(node in ground_truth_set for node in predicted_nodes[:5]) else 0.0
    
    mrr = 0.0
    for rank, node in enumerate(predicted_nodes, 1):
        if node in ground_truth_set:
            mrr = 1.0 / rank
            break
    
    answer_set = set(predicted_nodes)
    retieved_ground_truths = answer_set.intersection(ground_truth_set)
    missed_ground_truths = ground_truth_set.difference(retieved_ground_truths)
    retieved_ground_truths_count = len(retieved_ground_truths)
    missed_ground_truths_count = len(missed_ground_truths)
    recall = retieved_ground_truths_count / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
    
    top_20_predictions = predicted_nodes[:20]
    retrieved_20 = set(top_20_predictions).intersection(ground_truth_set)
    recall_20 = len(retrieved_20) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0

    results = {
        "answer_list": predicted_nodes,
        "ground_truth_set": ground_truth_set,
        "retrieved_ground_truths": retieved_ground_truths,
        "missed_ground_truths": missed_ground_truths,
        "metrics": {
            "total_answers": len(ground_truth_set),
            "retrieved_count": retieved_ground_truths_count,
            "missed_count": missed_ground_truths_count,
            "recall@50": recall,        
            "recall@20": recall_20, 
            'hit_at_1': hit_at_1,
            'hit_at_5': hit_at_5,
            'mrr': mrr,
        }
    }
    return results


def save_results(query: Query, csv_path: str = "pipeline_results.csv", append: bool = True):
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    write_headers = not file_exists or not append
    mode = 'a' if append else 'w'
    
    # 1. Ensure metrics are calculated without wiping existing data
    grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
    vss_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)

    # 2. Update the query object safely
    if query.results is None:
        query.results = {}
    
    query.results.update(grounding_results) # Updates 'metrics', 'answer_list', etc.
    query.results['vss_merged_metrics'] = vss_results['metrics'] # Explicitly save VSS metrics
    
    metrics = query.results.get('metrics', {})
    vss_metrics = query.results.get('vss_merged_metrics', {})
    
    row_data = {
        'query_id': query.id,
        'query_text': query.query,
        'total_answers': metrics.get('total_answers', 0),
        'retrieved_count': metrics.get('retrieved_count', 0),
        'missed_count': metrics.get('missed_count', 0),
        'recall@20': metrics.get('recall@20', 0.0),
        'recall@50': metrics.get('recall@50', 0.0),
        'hit@1': metrics.get('hit_at_1', 0.0),
        'hit@5': metrics.get('hit_at_5', 0.0),
        'mrr': metrics.get('mrr', 0.0),
        # Retrieve 'recall@20' from the VSS dictionary
        'recall@20_vss_merged': vss_metrics.get('recall@20', 0.0) 
    }
    
    fieldnames = [
        'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
        'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr', 'recall@20_vss_merged'
    ]
    
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_headers:
            writer.writeheader()
        writer.writerow(row_data)


def save_aggregate_results(queries: list, csv_path: str = "aggregate_results.csv"):
    if not queries:
        return
    
    # SEPARATE lists for grounding and VSS metrics
    grounding_metrics_list = []
    vss_metrics_list = []

    for query in queries:
        # 1. Collect Grounding Metrics
        if hasattr(query, 'results') and query.results is not None:
            grounding_metrics_list.append(query.results.get('metrics', {}))
        
        # 2. Collect VSS Metrics
        if hasattr(query, 'results') and 'vss_merged_metrics' in query.results:
            vss_metrics_list.append(query.results.get('vss_merged_metrics', {}))

    if not grounding_metrics_list:
        return
    
    num_queries = len(grounding_metrics_list)
    
    # Calculate averages using the distinct lists
    avg_metrics = {
        'total_queries': num_queries,
        
        # --- Grounding Metrics (Standard) ---
        'avg_total_answers': sum(m.get('total_answers', 0) for m in grounding_metrics_list) / num_queries,
        'avg_retrieved_count': sum(m.get('retrieved_count', 0) for m in grounding_metrics_list) / num_queries,
        'avg_missed_count': sum(m.get('missed_count', 0) for m in grounding_metrics_list) / num_queries,
        'avg_recall@20': sum(m.get('recall@20', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_recall@50': sum(m.get('recall@50', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_hit@1': sum(m.get('hit_at_1', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_hit@5': sum(m.get('hit_at_5', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_mrr': sum(m.get('mrr', 0.0) for m in grounding_metrics_list) / num_queries,
        
        # --- VSS Merged Metrics ---
        # Note: We look for 'recall@20' INSIDE the vss list, not 'recall@20_vss_merged'
        'recall@20_vss_merged': sum(m.get('recall@20', 0.0) for m in vss_metrics_list) / num_queries if vss_metrics_list else 0.0,        
    }
    
    fieldnames = list(avg_metrics.keys())
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(avg_metrics)
    
    print(f"\n[SAVED] Aggregate results saved to {csv_path}")
    print(f"\nAggregate Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

def create_experiment_dir(exp_name: str = "test_run", base_dir: str = './output/'):
    try:
        os.mkdir(f"{base_dir}/{exp_name}")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"Error creating directory: {e}")


def pipeline(query, llm_bridge, kb, retriever, dataset_name: str, exp_name: str, use_saved_llm_responses: bool = False, llm_response: Optional[dict] = None,alpha:int = 12,use_saved_vss_candidates : bool = False):

    step1_identify_entities(query, llm_bridge, dataset_name,use_saved_llm_responses, llm_response)
    step2_identify_relations(query, llm_bridge, dataset_name,use_saved_llm_responses, llm_response)
    step3_get_initial_candidates(query, kb, retriever)
    step4_grounding(query, kb, retriever)
    step5_merge_vss_candidates(query, retriever, kb, {}, use_saved_vss_candidates= use_saved_vss_candidates,alpha=alpha)
    save_results(query, csv_path=f"./output/{exp_name}/pipeline_results.csv")


def save_data_dump(results, csv_path="aggregate_results.csv"):
    if not results:
        return
    
    def serialize_value(value):
        if value is None:
            return ""
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [serialize_value(item) for item in value]
        else:
            return str(value)
    
    fieldnames = [
        'id', 'query', 'ground_truths', 'status', 
        'entities',
        'relations', 'results','grounding_candidates','vss_merged_candidates'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for query_obj in results:
            row = {}
            for field in fieldnames:
                value = getattr(query_obj, field, "")
                
                if value is None:
                    row[field] = ""
                elif isinstance(value, dict):
                    try:
                        serialized = serialize_value(value)
                        row[field] = json.dumps(serialized) if serialized else ""
                    except Exception:
                        row[field] = str(value)
                elif isinstance(value, (list, tuple)):
                    try:
                        serialized = serialize_value(value)
                        row[field] = json.dumps(serialized) if serialized else ""
                    except Exception:
                        row[field] = str(value)
                else:
                    row[field] = str(value)
            
            writer.writerow(row)
    
    print(f"✓ Successfully saved {len(results)} results to {csv_path}")


def main(args):
    data_split = args.split
    dataset_name = args.dataset
    embedding_dir = args.embedding_dir
    exp_name = args.exp_name
    use_saved_llm_responses = args.use_saved_llm_responses
    llm_responses_file = args.llm_responses_file

    emb_model = 'text-embedding-ada-002'
    configs_path = 'configs.json'

    qa_dataset = load_qa(dataset_name)
    qa = qa_dataset.split_indices[data_split].reshape(-1).tolist()
    qa = qa[:int(len(qa) * 0.1)]

    test_queries = [qa_dataset[i] for i in qa]
    if args.test_run:
        test_queries = test_queries[:15]

    kb = load_skb(dataset_name, download_processed=True)

    model_name = 'meta/llama-3.3-70b-instruct'
    llm_bridge = LlmBridge(model_name=model_name, configs_path="configs.json")

    retriever = VSSRetriever(
        kb=kb,
        emb_base_path=f"./emb/{dataset_name}/",
        emb_model="text-embedding-ada-002",
        qa_dataset=qa_dataset,
        dataset_name=data_split,
        use_vss=True
    )

    create_experiment_dir(exp_name=exp_name, base_dir='./output/')

    failed_queries = []
    results = []
    if use_saved_llm_responses:
        llm_respose_df = pd.read_csv(llm_responses_file)

    with open(f"./output/{exp_name}/temp.txt", 'w', encoding='utf-8') as temp_file:
            for query in tqdm(test_queries, desc="Processing queries", unit="query"):
                try:
                    with contextlib.redirect_stdout(temp_file), contextlib.redirect_stderr(temp_file):
                        current_query = Query(id=query[1], query=query[0], ground_truths=query[2])
                        if use_saved_llm_responses:
                            llm_response = llm_respose_df[llm_respose_df['query_id'] == current_query.id].to_dict(orient='records')
                            llm_response = llm_response[0]
                        else:
                            llm_response = None
                        pipeline(current_query, llm_bridge, kb, retriever, dataset_name, exp_name, use_saved_llm_responses=args.use_saved_llm_responses, llm_response=llm_response)
                    results.append(current_query)
                    
                except Exception as e:
                    tb = traceback.format_exc()
                    failed_queries.append({
                        'query_id': query[1],
                        'query_text': query[0][:100],
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'traceback': tb
                    })
                    tqdm.write(f"✗ Query {query[1]} failed: {type(e).__name__}: {str(e)}")
                    continue
    save_aggregate_results(results, csv_path=f"./output/{exp_name}/aggregate_results.csv")

    if failed_queries:
        with open(f"./output/{exp_name}/failed_queries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_id', 'query_text', 'error_type', 'error', 'traceback'])
            writer.writeheader()
            writer.writerows(failed_queries)
        
        print(f"\n⚠ {len(failed_queries)} queries failed. See ./output/{exp_name}/failed_queries.csv for details.")
        
        from collections import Counter
        error_counts = Counter(fq['error_type'] for fq in failed_queries)
        print(f"\nError breakdown:")
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
    else:
        print(f"\n✓ All {len(results)} queries processed successfully!")

    print(f"\n📊 Successfully processed: {len(results)} queries")
    save_data_dump(results, csv_path=f"./output/{exp_name}/full_data_dump.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA pipeline on knowledge base")
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        required=True,
        help="Directory containing embeddings"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split (e.g., train, test, val)"
    )
    
    parser.add_argument(
        "--test-run",
        action='store_true',
        help="Run on subset of 10 queries for testing"
    )
    
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the experiment for saving outputs"
    )

    parser.add_argument(
        "--use-saved-llm-responses",
        action='store_true',
        help="use pre saved LLM responses from file instead of querying the LLM"
    )

    parser.add_argument(
        "--llm-responses-file",
        type=str,
        default="llm_responses.csv",
        help="Path to the CSV file containing saved LLM responses"
    )

    parser.add_argument(
        "--use-saved-vss-candidates",
        action='store_true',
        help="use pre saved VSS candidates instead of querying the VSS retriever"
    )
    
    args = parser.parse_args()
    
    if args.exp_name is None:
        args.exp_name = f"{args.dataset}_{args.split}_results"
    
    main(args)
    
    print(f"\nScript completed successfully!")
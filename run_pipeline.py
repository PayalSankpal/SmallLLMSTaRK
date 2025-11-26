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
from custom_pipeline.prompt_generator import get_entity_extraction_prompt, get_relation_extraction_prompt,get_query_expansion_prompt
from custom_pipeline.vss_retreiver import VSSRetriever
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

# --- OPTIMIZATION CONFIG ---
MAX_WORKERS = 4 
QUERY_TIMEOUT_SECONDS = 300  # 5 minutes timeout per query
# ---------------------------

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Query execution timed out")


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
            print(f"Entities found: {list(query.entities.keys())}") 
        except ValueError as e:
            print(f"Error parsing entities: {e}")
            query.status = "FAILED"
            return
        
def step2_identify_relations(query: Query, llm_bridge: LlmBridge, dataset_name: str,use_saved_llm_responses: bool = False, llm_response: Optional[Dict] = None):
    if use_saved_llm_responses and llm_response is not None:
        response_string = llm_response.get("relations", "")
    else:
        prompt = get_relation_extraction_prompt(dataset_name , query.query, query.entity_id_response)  # Pass query.entities directly, not str(query.entities)
        response_string = get_llm_response(prompt, llm_bridge)
    query.relations_id_response = response_string

    if response_string == '' or response_string == '{}':
        query.status = "FAILED"
        query.relations = {}
        return
    else:
        try:
            identified_relations = parse_relation_string(response_string)
            query.relations = identified_relations
            edge_type_dict = {
                'affiliated_with': 'author___affiliated_with___institution',
                'cites': 'paper___cites___paper', 
                'has_topic': 'paper___has_topic___field_of_study',
                'writes': 'author___writes___paper'
            } 
            for pair in query.relations:
                rels = query.relations[pair]
                for i in range(len(rels)):
                    if rels[i] in edge_type_dict:
                        rels[i] = edge_type_dict[rels[i]]
            print(f"Relations found: {query.relations}")
        except ValueError as e:
            query.status = "FAILED"
            query.relations = {}
            return

def get_initial_candidates_for_entity(entity_info, entity_key, kb, retriever):
    candidates = []
    entity_types = entity_info.get("type", [])
    name_constraint = entity_info.get("lexical", {}).get("name", None)
    
    semantic_parts = entity_info.get("semantic", []).copy()
    lexical_info = entity_info.get("lexical", {})
    if lexical_info:
        semantic_parts.extend([f" {val}" for val in lexical_info.values()])
    
    search_sem = "".join(semantic_parts)
    nodes_by_name = set() 

    if name_constraint:
        for etype in entity_types:
            exact_matches = kb.get_node_ids_by_value(node_type=etype, key="name", value=name_constraint)
            if exact_matches:
                nodes_by_name.update(exact_matches)
                candidates.extend([CandidateContext(node_id=x, entity=entity_key, score=1.0) for x in exact_matches])

    existing_ids = nodes_by_name

    current_count = len(candidates)
    if current_count < 25:
        vss_needed = 25 - current_count
        for etype in entity_types:
            nodes_by_desc_ids, vss_scores = retriever.get_top_k_nodes(
                search_str=search_sem, k=vss_needed, node_type=etype, cutoff=0.65
            )
            
            for i, node_id in enumerate(nodes_by_desc_ids):
                if node_id not in existing_ids:
                    candidates.append(CandidateContext(node_id=node_id, entity=entity_key, score=vss_scores[i]))

    return entity_key, candidates


def step3_get_initial_candidates(current_query, kb, retriever):
    initial_candidates = {}
    
    entities_to_process = {
        k: v for k, v in current_query.entities.items() if k != "ANSWER"
    }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entity = {
            executor.submit(
                get_initial_candidates_for_entity, 
                entity_info, 
                entity_key, 
                kb, 
                retriever
            ): entity_key 
            for entity_key, entity_info in entities_to_process.items()
        }

        for future in as_completed(future_to_entity):
            try:
                e_key, candidates = future.result()
                initial_candidates[e_key] = candidates
            except Exception as exc:
                print(f"Entity generation generated an exception for {future_to_entity[future]}: {exc}")

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
    



def evaluate_results_after_merging_vss_candidates(query: Query):
    results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = results['metrics']
    print("\n\nRESULTS AFTER MERGING VSS CANDIDATES\n\n", results,"\n\n")
    return results

def evaluate_results(predicted_nodes, ground_truth_nodes):
    """Evaluate prediction results against ground truth."""
    ground_truth_set = set(ground_truth_nodes)
    
    if not predicted_nodes:
        return {
            "answer_list": [],
            "answer_set": set(),
            "ground_truth_set": ground_truth_set,
            "retrieved_ground_truths": set(),
            "missed_ground_truths": ground_truth_set,
            "metrics": {
                "total_answers": len(ground_truth_set),
                "retrieved_count": 0,
                "missed_count": len(ground_truth_set),
                "recall@50": 0.0,
                "recall@20": 0.0,
                "hit_at_1": 0.0,
                "hit_at_5": 0.0,
                "mrr": 0.0,
            }
        }
    
    # Calculate metrics
    hit_at_1 = 1.0 if predicted_nodes[0] in ground_truth_set else 0.0
    hit_at_5 = 1.0 if any(node in ground_truth_set for node in predicted_nodes[:5]) else 0.0
    
    mrr = 0.0
    for rank, node in enumerate(predicted_nodes, 1):
        if node in ground_truth_set:
            mrr = 1.0 / rank
            break
    
    answer_set = set(predicted_nodes)
    retrieved_ground_truths = answer_set.intersection(ground_truth_set)
    missed_ground_truths = ground_truth_set.difference(retrieved_ground_truths)
    
    recall_50 = len(retrieved_ground_truths) / len(ground_truth_set) if ground_truth_set else 0.0
    
    top_20 = set(predicted_nodes[:20]).intersection(ground_truth_set)
    recall_20 = len(top_20) / len(ground_truth_set) if ground_truth_set else 0.0

    return {
        "answer_list": predicted_nodes,
        "answer_set": answer_set,
        "ground_truth_set": ground_truth_set,
        "retrieved_ground_truths": retrieved_ground_truths,
        "missed_ground_truths": missed_ground_truths,
        "metrics": {
            "total_answers": len(ground_truth_set),
            "retrieved_count": len(retrieved_ground_truths),
            "missed_count": len(missed_ground_truths),
            "recall@50": recall_50,
            "recall@20": recall_20,
            'hit_at_1': hit_at_1,
            'hit_at_5': hit_at_5,
            'mrr': mrr,
        }
    }


def step4_grounding(query: Query, kb, retriever):
    # --- PERFORMANCE CONFIG ---
    # max_candidates_per_symbol REDUCED TO 100 as requested
    final_candidates = run_priority_queue_grounding(
        query_obj=query,
        kb=kb,
        vss_retriever=retriever,
        max_candidates_per_symbol=3000, # <--- CHANGED FROM 1000 to 100
        max_answer_candidates=20,      
        top_k_neighbors=15,            
        support_boost=0,
        score_decay=0.9,
        verbose=False 
    )
    
    if "ANSWER" in final_candidates:
        answers = [cc.node_id for cc in sorted(
            final_candidates["ANSWER"], key=lambda x: (x.score, -x.support), reverse=True
        )]
    else:
        answers = []
    query.grounding_candidates = answers  # ✅ ADD THIS LINE
    query.final_candidates = final_candidates
    return final_candidates


def get_expanded_query(query,  dataset_name: str,kb, llm_bridge) -> str:
    docs_list = []
    print(query.grounding_candidates)
    if not hasattr(query, 'grounding_candidates') :
        return query.query
    
    for candidate in query.grounding_candidates[:3]:
        print(f"Candidate: {candidate} ")
        print(kb.get_doc_info(candidate))
        docs_list.append(kb.get_doc_info(candidate))
    expanded_prompt = get_query_expansion_prompt(query, dataset_name, docs_list)
    expanded_query = get_llm_response(expanded_prompt, llm_bridge)
    print("[EXPANED QUERY]: ", expanded_query)
    return expanded_query

def step5_merge_vss_candidates(query: Query, retriever: VSSRetriever, kb,
                                use_saved_vss_candidates: bool = False,
                                alpha: int = 12,
                                vss_candidates: dict = {},llm_bridge: LlmBridge=None,dataset_name: str = "") -> List[int]:
    """
    Step 5: Merge grounding candidates with VSS candidates.
    """
    vss_candidates_list = []
    print(f"[Step 5] Merging VSS candidates with alpha={alpha} USE_SAVED={use_saved_vss_candidates} \n\n VSS CANDIDATES: {vss_candidates}\n\n")
    if not use_saved_vss_candidates:
        all_candidates = []
        
        expanded_query = get_expanded_query(query, dataset_name=dataset_name,kb=kb,llm_bridge=llm_bridge)
        print(query)
        print("[Step 5] Expanded Query for VSS:", expanded_query)
        possible_node_types = query.entities["ANSWER"]["type"].copy()
        for node_type in possible_node_types:
            print(f"[Step 5] Searching VSS for node type: {node_type}")
            top_candidates = retriever.get_top_k_nodes(
                search_str=expanded_query, k=20, node_type=node_type, cutoff=0.6
            )
            all_candidates.extend(list(zip(top_candidates[0], top_candidates[1])))
        vss_candidates_list = list(map (lambda x: x[0], sorted(all_candidates, key=lambda x: x[1], reverse=True)))
    else:
        vss_candidates_list = vss_candidates.get(str(query.id), [])
        print("READ VSS CANDIDATES FROM FILE:", vss_candidates_list)

    
    # Take top alpha candidates from grounding
    existing_candidate_ids = query.grounding_candidates[:alpha]
    
    # Add new VSS candidates not in grounding top-alpha
    new_vss_candidates = []
    for node in vss_candidates_list:
        if node not in existing_candidate_ids:
            new_vss_candidates.append(node)
        if len(new_vss_candidates) == 20 - alpha:
            break
    
    # Merge: top alpha from grounding + remaining from VSS
    merged_candidates = existing_candidate_ids + new_vss_candidates
    query.vss_merged_candidates = merged_candidates
    
    # ✅ Initialize results dict if needed
    if query.results is None:
        query.results = {}
    
    # Evaluate grounding-only metrics (if not already done)
    if 'metrics' not in query.results:
        grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
        query.results.update(grounding_results)
        print(f"[Step 5] Grounding Recall@20: {query.results['metrics']['recall@20']:.3f}")
    
    # Evaluate VSS-merged metrics
    vss_merged_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = vss_merged_results['metrics']
    
    print(f"[Step 5] VSS Merged Recall@20: {query.results['vss_merged_metrics']['recall@20']:.3f}")
    
    return merged_candidates


def save_results(query: Query, csv_path: str = "pipeline_results.csv", append: bool = True):
    """
    Save query results to CSV.
    Assumes metrics are already calculated in step5.
    """
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    mode = 'a' if append else 'w'
    
    # Safety check: Initialize if missing
    if query.results is None:
        print(f"[WARNING] Query {query.id} has no results, calculating now...")
        query.results = {}
        
        # Calculate grounding metrics
        if hasattr(query, 'grounding_candidates') and query.grounding_candidates:
            grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
            query.results.update(grounding_results)
        
        # Calculate VSS merged metrics
        if hasattr(query, 'vss_merged_candidates') and query.vss_merged_candidates:
            vss_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
            query.results['vss_merged_metrics'] = vss_results['metrics']
    
    # Extract metrics for CSV
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
        'recall@20_vss_merged': vss_metrics.get('recall@20', 0.0)
    }
    
    fieldnames = [
        'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
        'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr', 'recall@20_vss_merged'
    ]
    
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
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

def create_experiment_dir(exp_name: str = "test_run", base_dir: str = './output/'):
    try:
        os.makedirs(f"{base_dir}/{exp_name}", exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")


def pipeline(query, llm_bridge, kb, retriever, dataset_name: str, exp_name: str, 
             use_saved_llm_responses: bool = False, llm_response: Optional[dict] = None,
             alpha: int = 12, use_saved_vss_candidates: bool = False, vss_candidates: dict = {}):
    """
    Execute full pipeline with timing information.
    """
    pipeline_start = time.time()
    
    t0 = time.time()
    step1_identify_entities(query, llm_bridge, dataset_name, use_saved_llm_responses, llm_response)
    print(f"[TIMING] Step 1 (Entities): {time.time() - t0:.4f}s")
    
    if query.status == "FAILED":
        print(f"[ERROR] Query {query.id} failed at Step 1")
        return
    
    t0 = time.time()
    step2_identify_relations(query, llm_bridge, dataset_name, use_saved_llm_responses, llm_response)
    print(f"[TIMING] Step 2 (Relations): {time.time() - t0:.4f}s")
    
    if query.status == "FAILED":
        print(f"[ERROR] Query {query.id} failed at Step 2")
        return
    
    t0 = time.time()
    step3_get_initial_candidates(query, kb, retriever)
    print(f"[TIMING] Step 3 (Candidates): {time.time() - t0:.4f}s")
    
    t0 = time.time()
    step4_grounding(query, kb, retriever)
    print(f"[TIMING] Step 4 (Grounding): {time.time() - t0:.4f}s")
    
    t0 = time.time()
    step5_merge_vss_candidates(query, retriever, kb,  
                               use_saved_vss_candidates=use_saved_vss_candidates, vss_candidates=vss_candidates,
                               alpha=alpha,llm_bridge=llm_bridge,dataset_name=dataset_name)
    print(f"[TIMING] Step 5 (VSS Merge): {time.time() - t0:.4f}s")
    
    t0 = time.time()
    save_results(query, csv_path=f"./output/{exp_name}/pipeline_results.csv")
    print(f"[TIMING] Save Results: {time.time() - t0:.4f}s")
    
    print(f"[TIMING] TOTAL: {time.time() - pipeline_start:.4f}s")
def save_data_dump(results, csv_path="aggregate_results.csv"):
    if not results:
        return
    
    def serialize_value(value):
        if value is None: return ""
        if isinstance(value, (str, int, float, bool)): return value
        try: return json.dumps(value)
        except: return str(value)
    
    fieldnames = [
        'id', 'query', 'ground_truths', 'status', 
        'entities',
        'relations', 'results','grounding_candidates','vss_merged_candidates'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_obj in results:
            row = {field: serialize_value(getattr(query_obj, field, "")) for field in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Successfully saved {len(results)} results to {csv_path}")


def main(args):
    data_split = args.split
    dataset_name = args.dataset
    exp_name = args.exp_name
    use_saved_llm_responses = args.use_saved_llm_responses
    llm_responses_file = args.llm_responses_file
    use_saved_vss_candidates = args.use_saved_vss_candidates
    vss_candidates_json_path = args.vss_candidates_json_path

    emb_model = 'text-embedding-ada-002'
    configs_path = 'configs.json'

    qa_dataset = load_qa(dataset_name)
    qa = qa_dataset.split_indices[data_split].reshape(-1).tolist()
    qa = qa[:int(len(qa) * 0.1)] 
    
    vss_candidates = {}
    if use_saved_vss_candidates:
        with open(vss_candidates_json_path, 'r', encoding='utf-8') as f:
            vss_candidates = json.load(f)


    test_queries = [qa_dataset[i] for i in qa]
    if args.test_run:
        test_queries = test_queries[:10]

    print("Loading Knowledge Base...")
    kb = load_skb(dataset_name, download_processed=True)
    
    print("Loading LLM Bridge...")
    llm_bridge = LlmBridge(model_name='meta/llama-3.3-70b-instruct', configs_path="configs.json")
    
    print("Loading Retriever...")
    retriever = VSSRetriever(
        kb=kb,
        emb_base_path=f"./emb/{dataset_name}/",
        emb_model="text-embedding-ada-002",
        qa_dataset=qa_dataset,
        dataset_name=data_split,
        use_vss=True,
        use_gpu=True
    )

    create_experiment_dir(exp_name=exp_name, base_dir='./output/')
    
    failed_queries = []
    results = []
    if use_saved_llm_responses:
        llm_respose_df = pd.read_csv(llm_responses_file)

    log_file_path = f"./output/{exp_name}/temp.txt"
    print(f"Processing queries (Logs: {log_file_path})...")
    
    with open(log_file_path, 'w', encoding='utf-8') as temp_file:
        for query in tqdm(test_queries, desc="Pipeline", unit="query"):
            try:
                with contextlib.redirect_stdout(temp_file):
                    current_query = Query(id=query[1], query=query[0], ground_truths=query[2])
                    
                    # Set timeout alarm
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(QUERY_TIMEOUT_SECONDS)
                    
                    try:
                        if use_saved_llm_responses:
                            llm_response = llm_respose_df[llm_respose_df['id'] == current_query.id].to_dict(orient='records')
                            print("LOADED SAVED LLM RESPONSE:", llm_response)
                            llm_response = llm_response[0]
                        else:
                            llm_response = None
                        pipeline(current_query, llm_bridge, kb, retriever, dataset_name, exp_name, use_saved_llm_responses=args.use_saved_llm_responses, llm_response=llm_response, use_saved_vss_candidates=args.use_saved_vss_candidates, vss_candidates = vss_candidates)
                        signal.alarm(0)  # Cancel the alarm
                    except TimeoutException:
                        signal.alarm(0)  # Cancel the alarm
                        current_query.status = "TIMEOUT"
                        raise TimeoutException(f"Query exceeded {QUERY_TIMEOUT_SECONDS}s timeout")
                    
                results.append(current_query)
                
            except TimeoutException as e:
                failed_queries.append({
                    'query_id': query[1], 
                    'error': f'TIMEOUT: {str(e)}', 
                    'traceback': f'Query exceeded {QUERY_TIMEOUT_SECONDS} second timeout'
                })
                tqdm.write(f"⏱ Query {query[1]} timed out after {QUERY_TIMEOUT_SECONDS}s")
                temp_file.write(f"\nTIMEOUT Query {query[1]}: Exceeded {QUERY_TIMEOUT_SECONDS}s\n")
                continue
                
            except Exception as e:
                signal.alarm(0)  # Cancel alarm in case of other exceptions
                tb = traceback.format_exc()
                failed_queries.append({
                    'query_id': query[1], 
                    'error': str(e), 
                    'traceback': tb
                })
                tqdm.write(f"✗ Query {query[1]} failed: {e}")
                temp_file.write(f"\nFATAL ERROR Query {query[1]}:\n{tb}\n")
                continue

    save_aggregate_results(results, csv_path=f"./output/{exp_name}/aggregate_results.csv")
    save_data_dump(results, csv_path=f"./output/{exp_name}/full_data_dump.csv")

    if failed_queries:
        with open(f"./output/{exp_name}/failed_queries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_id', 'error', 'traceback'])
            writer.writeheader()
            writer.writerows(failed_queries)
        print(f"\n⚠ {len(failed_queries)} queries failed.")
    else:
        print(f"\n✓ All {len(results)} queries processed successfully!")


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
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each query"
    )
    parser.add_argument(
        "--vss-candidates-json-path",
        type=str,
        default="vss_candidates.json",
        help="Path to the JSON file containing saved VSS candidates"
    )
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = f"{args.dataset}_{args.split}_results"
    
    QUERY_TIMEOUT_SECONDS = args.timeout
    
    main(args)

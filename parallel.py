import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import csv
import functools
from pathlib import Path
import pandas as pd
from stark_qa import load_skb, load_qa
import regex as re
import ast
import vss
import sys
import os
import time
import traceback
import logging
from tqdm import tqdm
from urllib import response
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Any, Optional
import json
import signal
import warnings
import concurrent.futures

# Custom imports (assumed available in environment)
from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.llm_bridge import LlmBridge
from custom_pipeline.query import Query
from custom_pipeline.prompt_generator import get_entity_extraction_prompt, get_relation_extraction_prompt, get_query_expansion_prompt
from custom_pipeline.vss_retreiver import VSSRetriever
from custom_pipeline.candidate_context import CandidateContext
from custom_pipeline.grounders.grounder3 import PriorityQueueGrounding
from custom_pipeline.thread_safe_writers import ThreadSafeCSVWriter

# --- CONFIGURATION CONSTANTS ---
QUERY_TIMEOUT = 300  # Seconds allowed per query before killing/retrying
MAX_RETRIES = 3      # Number of times to retry a failed query
DUMP_INTERVAL = 5    # Dump results to file every N queries

class TimeoutException(Exception):
    pass

def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    return response[0][0]

def step1_identify_entities(query: Query, llm_bridge: LlmBridge, dataset_name: str, use_saved: bool = False, saved_response: Optional[Dict] = None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get('entities', "")
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
        
def step2_identify_relations(query: Query, llm_bridge: LlmBridge, dataset_name: str, use_saved: bool = False, saved_response: Optional[Dict] = None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get("relations", "")
    else:
        prompt = get_relation_extraction_prompt(dataset_name , query.query, query.entity_id_response)
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

def get_initial_candidates_for_entity(entity_info, entity_key, kb, retriever, limit=25, cutoff=0.65):
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
    if current_count < limit:
        vss_needed = limit - current_count
        for etype in entity_types:
            nodes_by_desc_ids, vss_scores = retriever.get_top_k_nodes(
                search_str=search_sem, k=vss_needed, node_type=etype, cutoff=cutoff
            )
            
            for i, node_id in enumerate(nodes_by_desc_ids):
                if node_id not in existing_ids:
                    candidates.append(CandidateContext(node_id=node_id, entity=entity_key, score=vss_scores[i]))

    return entity_key, candidates


def step3_get_initial_candidates(current_query, kb, retriever, config):
    initial_candidates = {}
    entities_to_process = {
        k: v for k, v in current_query.entities.items() if k != "ANSWER"
    }

    step3_config = config['retrieval_params']['step3_candidate_generation']

    for entity_key, entity_info in entities_to_process.items():
        try:
            e_key, candidates = get_initial_candidates_for_entity(
                entity_info, 
                entity_key, 
                kb, 
                retriever,
                step3_config['initial_limit'],
                step3_config['vss_cutoff']
            )
            initial_candidates[e_key] = candidates
        except Exception as exc:
            print(f"Entity generation generated an exception for {entity_key}: {exc}")

    current_query.initial_symbol_candidates = initial_candidates
    print(current_query)


def run_priority_queue_grounding(query_obj, kb, vss_retriever, params, verbose: bool = False) -> Dict[str, List[CandidateContext]]:
    grounder = PriorityQueueGrounding(
        query_obj=query_obj,
        kb=kb,
        vss_retriever=vss_retriever,
        max_candidates_per_symbol=params['max_candidates_per_symbol'],
        max_answer_candidates=params['max_answer_candidates'],
        top_k_neighbors=params['top_k_neighbors'],
        score_decay=params['score_decay'],
        support_boost=params['support_boost'],
        verbose=verbose
    )
    print(f"[Grounding] Running grounding for Query ID: {query_obj.id}")
    print(f"[Grounding] Relations: { query_obj.relations } ")
    print(f"[Grounding] Initial Entities: { query_obj.entities } ")
    return grounder.ground()
    
def evaluate_results(predicted_nodes, ground_truth_nodes):
    ground_truth_set = set(ground_truth_nodes)
    if not predicted_nodes:
        return {
            "answer_list": [], "answer_set": set(), "ground_truth_set": ground_truth_set,
            "retrieved_ground_truths": set(), "missed_ground_truths": ground_truth_set,
            "metrics": {"total_answers": len(ground_truth_set), "retrieved_count": 0, "missed_count": len(ground_truth_set), "recall@50": 0.0, "recall@20": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0}
        }
    
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
        "answer_list": predicted_nodes, "answer_set": answer_set, "ground_truth_set": ground_truth_set,
        "retrieved_ground_truths": retrieved_ground_truths, "missed_ground_truths": missed_ground_truths,
        "metrics": {"total_answers": len(ground_truth_set), "retrieved_count": len(retrieved_ground_truths), "missed_count": len(missed_ground_truths), "recall@50": recall_50, "recall@20": recall_20, 'hit_at_1': hit_at_1, 'hit_at_5': hit_at_5, 'mrr': mrr}
    }


def step4_grounding(query: Query, kb, retriever, config):
    g_params = config['grounding_params']
    
    final_candidates = run_priority_queue_grounding(
        query_obj=query,
        kb=kb,
        vss_retriever=retriever,
        params=g_params,
        verbose=False 
    )
    
    if "ANSWER" in final_candidates:
        answers = [cc.node_id for cc in sorted(
            final_candidates["ANSWER"], key=lambda x: (x.score, -x.support), reverse=True
        )]
    else:
        answers = []
    query.grounding_candidates = answers
    query.final_candidates = final_candidates
    return final_candidates

def get_expanded_query(query, dataset_name: str, kb, llm_bridge) -> str:
    docs_list = []
    print(query.grounding_candidates)
    if not hasattr(query, 'grounding_candidates') :
        return query.query
    
    for candidate in query.grounding_candidates[:4]:
        print(f"Candidate: {candidate} ")
        print(kb.get_doc_info(candidate))
        docs_list.append(kb.get_doc_info(candidate))
    expanded_prompt = get_query_expansion_prompt(query, dataset_name, docs_list)
    expanded_query = get_llm_response(expanded_prompt, llm_bridge)
    print("[EXPANED QUERY]: ", expanded_query)
    return expanded_query

def step5_merge_vss_candidates(query: Query, retriever: VSSRetriever, kb, config, use_saved: bool = False, vss_candidates: dict = {}, llm_bridge: LlmBridge=None, dataset_name: str = "") -> List[int]:
    
    step5_params = config['retrieval_params']['step5_vss_merge']
    alpha = min(step5_params['alpha'], len(query.grounding_candidates)) 
    k_val = step5_params['top_k']
    cutoff_val = step5_params['cutoff']
    
    vss_candidates_list = []
    if not use_saved:
        all_candidates = []
        expanded_query = get_expanded_query(query, dataset_name=dataset_name, kb=kb, llm_bridge=llm_bridge)
        print(query)
        print("[Step 5] Expanded Query for VSS:", expanded_query)
        possible_node_types = query.entities["ANSWER"]["type"].copy()
        for node_type in possible_node_types:
            print(f"[Step 5] Searching VSS for node type: {node_type}")
            top_candidates = retriever.get_top_k_nodes(
                search_str=expanded_query, k=k_val, node_type=node_type, cutoff=cutoff_val
            )
            all_candidates.extend(list(zip(top_candidates[0], top_candidates[1])))
        vss_candidates_list = list(map (lambda x: x[0], sorted(all_candidates, key=lambda x: x[1], reverse=True)))
    else:
        vss_candidates_list = vss_candidates.get(str(query.id), [])
        print("READ VSS CANDIDATES FROM FILE:", vss_candidates_list)

    existing_candidate_ids = query.grounding_candidates[:alpha]
    new_vss_candidates = []
    for node in vss_candidates_list:
        if node not in existing_candidate_ids[:20-alpha] :
            new_vss_candidates.append(node)
        if len(new_vss_candidates) == 20 - alpha:
            break
    print(f"[Step 5] Merging VSS candidates with alpha={alpha} USE_SAVED={use_saved} \n\n VSS CANDIDATES: {new_vss_candidates}\n\n")

    merged_candidates = existing_candidate_ids + new_vss_candidates
    query.vss_merged_candidates = merged_candidates
    
    if query.results is None:
        query.results = {}
    
    if 'metrics' not in query.results:
        grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
        query.results.update(grounding_results)
        print(f"[Step 5] Grounding Recall@20: {query.results['metrics']['recall@20']:.3f}")
    
    vss_merged_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = vss_merged_results['metrics']
    print(f"[Step 5] VSS Merged Recall@20: {query.results['vss_merged_metrics']['recall@20']:.3f}")
    
    return merged_candidates


def save_results_threadsafe(query: Query, csv_writer: ThreadSafeCSVWriter):
    if query.results is None:
        print(f"[WARNING] Query {query.id} has no results, calculating now...")
        query.results = {}
        if hasattr(query, 'grounding_candidates') and query.grounding_candidates:
            grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
            query.results.update(grounding_results)
        if hasattr(query, 'vss_merged_candidates') and query.vss_merged_candidates:
            vss_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
            query.results['vss_merged_metrics'] = vss_results['metrics']
    
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
    csv_writer.write_row(row_data)


def save_aggregate_results(queries: list, csv_path: str = "aggregate_results.csv"):
    # This function now mostly acts as a final merger or check, 
    # since we are saving incrementally in the process
    if not queries: return
    # Logic remains similar for calculating final averages if needed
    # ... (Implementation kept as utility)

def serialize_value(value):
    if value is None: return ""
    if isinstance(value, (str, int, float, bool)): return value
    try:
        return json.dumps(value, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
    except Exception as e:
        return str(value)

def save_partial_dump(results, csv_path, fieldnames):
    """Appends results to the full dump file."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for query_obj in results:
            row = {field: serialize_value(getattr(query_obj, field, "")) for field in fieldnames}
            # Add failure reason if it exists
            row['failure_reason'] = getattr(query_obj, 'failure_reason', "")
            writer.writerow(row)

def pipeline_worker(query, llm_bridge, kb, retriever, dataset_name, 
                    csv_writer, config,
                    use_saved: bool = False, saved_response: Optional[dict] = None,
                    use_saved_vss: bool = False, vss_candidates: dict = {}):
    
    print(f"--- Processing Query {query.id} ---")
    
    # Step 1
    start_time_s1 = time.time()
    step1_identify_entities(query, llm_bridge, dataset_name, use_saved, saved_response)
    print(f"[TIMING] Query {query.id} - Step 1 took {time.time() - start_time_s1:.4f} seconds")
    if query.status == "FAILED": return query
    
    # Step 2
    start_time_s2 = time.time()
    step2_identify_relations(query, llm_bridge, dataset_name, use_saved, saved_response)
    print(f"[TIMING] Query {query.id} - Step 2 took {time.time() - start_time_s2:.4f} seconds")
    if query.status == "FAILED": return query
    
    # Step 3
    start_time_s3 = time.time()
    step3_get_initial_candidates(query, kb, retriever, config)
    print(f"[TIMING] Query {query.id} - Step 3 took {time.time() - start_time_s3:.4f} seconds")

    # Step 4
    start_time_s4 = time.time()
    step4_grounding(query, kb, retriever, config)
    print(f"[TIMING] Query {query.id} - Step 4 took {time.time() - start_time_s4:.4f} seconds")

    # Step 5
    start_time_s5 = time.time()
    step5_merge_vss_candidates(query, retriever, kb, config,
                               use_saved=use_saved_vss, 
                               vss_candidates=vss_candidates,
                               llm_bridge=llm_bridge, 
                               dataset_name=dataset_name)
    print(f"[TIMING] Query {query.id} - Step 5 took {time.time() - start_time_s5:.4f} seconds")
    
    # Save Metrics to the metric-specific CSV
    save_results_threadsafe(query, csv_writer)
    print(f"Finished Query {query.id}")
    
    return query

def create_experiment_dir(exp_name: str, base_dir: str):
    try: os.makedirs(f"{base_dir}/{exp_name}", exist_ok=True)
    except Exception as e: pass

def run_with_timeout(func, args, timeout):
    """Run a function with a timeout using threads."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutException(f"Execution timed out after {timeout} seconds")

def process_batch(batch_queries, config, dataset_name, model_config, data_config, 
                  result_queue, failed_queue, progress_queue):
    
    process_id = mp.current_process().pid
    exp_name = config['experiment'].get('exp_name', f"{dataset_name}_results")
    output_base = config['experiment'].get('output_base_dir', './output/')
    
    # --- REDIRECT STDOUT TO FILE, KEEP STDERR ---
    # We open a specific log file for this process
    log_file_path = f"{output_base}/{exp_name}/process_{process_id}.log"
    sys.stdout = open(log_file_path, "a", encoding="utf-8")
    # sys.stderr is inherited, so it prints to the console (progress bar)
    
    dump_fieldnames = ['id', 'query', 'ground_truths', 'status', 'entities', 'relations',
                  'initial_symbol_candidates', 'final_candidates', 'results',
                  'grounding_candidates', 'vss_merged_candidates', 'failure_reason']
    
    dump_file_path = f"{output_base}/{exp_name}/full_dump_process_{process_id}.csv"

    # Buffer for data dump
    batch_results_buffer = []

    try:
        # Initialize resources
        kb = load_skb(dataset_name, download_processed=True)
        llm_bridge = LlmBridge(model_name=model_config['llm_name'], 
                              configs_path=model_config['llm_config_path'])
        
        qa_dataset = load_qa(dataset_name)
        retriever = VSSRetriever(
            kb=kb, 
            emb_base_path=f"{model_config['embedding_base_path']}/{dataset_name}/", 
            emb_model=model_config['embedding_model'], 
            qa_dataset=qa_dataset, 
            dataset_name=config['experiment']['split'], 
            use_vss=True, 
            use_gpu=True
        )
        
        vss_candidates = {}
        if data_config['use_saved_vss_candidates']:
            with open(data_config['vss_candidates_json_path'], 'r', encoding='utf-8') as f:
                vss_candidates = json.load(f)
        
        llm_respose_df = None
        if data_config['use_saved_llm_responses']:
            llm_respose_df = pd.read_csv(data_config['llm_responses_file'])

        csv_fieldnames = [
            'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
            'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr', 'recall@20_vss_merged'
        ]
        
        csv_writer = ThreadSafeCSVWriter(
            csv_path=f"{output_base}/{exp_name}/pipeline_results_process_{process_id}.csv",
            fieldnames=csv_fieldnames
        )
        
        # Process each query
        for i, query_data in enumerate(batch_queries):
            current_query = Query(id=query_data[1], query=query_data[0], ground_truths=query_data[2])
            
            llm_response = None
            if data_config['use_saved_llm_responses'] and llm_respose_df is not None:
                llm_response_rows = llm_respose_df[llm_respose_df['id'] == current_query.id].to_dict(orient='records')
                if llm_response_rows: 
                    llm_response = llm_response_rows[0]
            
            # --- RETRY & TIMEOUT LOGIC ---
            success = False
            last_error = None
            
            for attempt in range(MAX_RETRIES):
                try:
                    # Run the worker with a timeout wrapper
                    args = (current_query, llm_bridge, kb, retriever, dataset_name,
                            csv_writer, config, data_config['use_saved_llm_responses'],
                            llm_response, data_config['use_saved_vss_candidates'], vss_candidates)
                    
                    result = run_with_timeout(pipeline_worker, args, QUERY_TIMEOUT)
                    
                    if result:
                        batch_results_buffer.append(result)
                        success = True
                        break # Success, exit retry loop
                        
                except Exception as e:
                    last_error = str(e)
                    print(f"[WARNING] Query {current_query.id} failed attempt {attempt+1}/{MAX_RETRIES}: {e}")
                    traceback.print_exc()
            
            if not success:
                # Store failed query info in buffer with reason
                current_query.status = "FAILED"
                current_query.failure_reason = f"Max Retries ({MAX_RETRIES}) reached. Last error: {last_error}"
                batch_results_buffer.append(current_query)
                failed_queue.put({
                    'query_id': current_query.id, 
                    'error': last_error, 
                    'traceback': traceback.format_exc()
                })

            progress_queue.put(1)

            # --- INCREMENTAL DUMP ---
            if len(batch_results_buffer) >= DUMP_INTERVAL:
                save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
                # Send to main process queue for final aggregation if needed, but file is safer
                for res in batch_results_buffer:
                    result_queue.put(res) 
                batch_results_buffer = [] # Clear buffer

    except Exception as e:
        sys.stderr.write(f"[CRITICAL] Batch process {process_id} crashed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        
    finally:
        # Dump any remaining results in buffer (handle Ctrl+C or crash)
        if batch_results_buffer:
            print(f"Dumping remaining {len(batch_results_buffer)} queries before exit...")
            save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
            for res in batch_results_buffer:
                result_queue.put(res)
        
        sys.stdout.close()


# ============================================
# MAIN FUNCTION
# ============================================
def main(config_path):
    # 0. LOAD CONFIG
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    exp_config = config['experiment']
    pipe_config = config['pipeline']
    data_config = config['data_paths']
    model_config = config['models']

    dataset_name = exp_config['dataset']
    split_name = exp_config['split']
    
    exp_name = exp_config.get('exp_name')
    if not exp_name:
        exp_name = f"{dataset_name}_{split_name}_results"
        
    output_base = exp_config.get('output_base_dir', './output/')
    create_experiment_dir(exp_name=exp_name, base_dir=output_base)
    
    # Main log file for setup info (stdout of main process)
    # Note: Worker processes will write to their own files
    sys.stdout = open(f"{output_base}/{exp_name}/main_process.log", 'a')
    # stderr remains console
    
    try:
        qa_dataset = load_qa(dataset_name)
        qa = qa_dataset.split_indices[split_name].reshape(-1).tolist()
        qa = qa[:int(len(qa) * 0.1)] 
        all_test_queries = [qa_dataset[i] for i in qa]
        
        if exp_config.get('test_run', False):
            limit = pipe_config.get('max_queries_test_run', 100)
            all_test_queries = all_test_queries[:limit]
        
        # --- RESUME CAPABILITY ---
        # Check if full_data_dump (merged) or partial dumps exist to skip queries
        completed_ids = set()
        
        # 1. Check main merged dump
        main_dump_path = f"{output_base}/{exp_name}/full_data_dump.csv"
        if os.path.exists(main_dump_path):
            try:
                df = pd.read_csv(main_dump_path)
                completed_ids.update(df['id'].astype(str).tolist())
            except Exception: pass
            
        # 2. Check process-specific partial dumps (in case merge didn't happen)
        for fname in os.listdir(f"{output_base}/{exp_name}"):
            if fname.startswith("full_dump_process_") and fname.endswith(".csv"):
                try:
                    df = pd.read_csv(f"{output_base}/{exp_name}/{fname}")
                    completed_ids.update(df['id'].astype(str).tolist())
                except Exception: pass
                
        sys.stderr.write(f"Found {len(completed_ids)} already completed queries. Filtering...\n")
        
        # Filter queries
        test_queries = [q for q in all_test_queries if str(q[1]) not in completed_ids]
        
        sys.stderr.write(f"Starting {len(test_queries)} remaining queries (Original: {len(all_test_queries)}).\n")
        
        if not test_queries:
            sys.stderr.write("All queries completed. Exiting.\n")
            return

        # Split queries into batches
        num_workers = pipe_config['max_workers']
        batch_size = len(test_queries) // num_workers
        if len(test_queries) % num_workers != 0:
            batch_size += 1
            
        # Handle case where test_queries < num_workers
        if batch_size == 0: batch_size = 1
        
        query_batches = [
            test_queries[i:i + batch_size] 
            for i in range(0, len(test_queries), batch_size)
        ]
        
        sys.stderr.write(f"Split into {len(query_batches)} batches for {num_workers} workers\n")
        
        manager = Manager()
        result_queue = manager.Queue()
        failed_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        processes = []
        for batch in query_batches:
            if not batch: continue
            p = Process(
                target=process_batch,
                args=(batch, config, dataset_name, model_config, data_config,
                      result_queue, failed_queue, progress_queue)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress via TQDM on stderr
        completed = 0
        total = len(test_queries)
        
        # Explicitly file=sys.stderr for tqdm
        with tqdm(total=total, desc="Pipeline", unit="query", file=sys.stderr) as pbar:
            while completed < total:
                # Check if all processes are dead (unexpected exit)
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
                    
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                except:
                    pass
        
        for p in processes:
            p.join()
        
        # Collect results
        results = []
        failed_queries = []
        
        while not result_queue.empty():
            results.append(result_queue.get())
        while not failed_queue.empty():
            failed_queries.append(failed_queue.get())
            
        sys.stderr.write(f"\nProcessed {len(results)} queries successfully in this run\n")
        
        # --- MERGE PROCESS OUTPUTS ---
        # 1. Merge Metrics
        merged_csv_path = f"{output_base}/{exp_name}/pipeline_results.csv"
        csv_fieldnames = [
            'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
            'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr', 'recall@20_vss_merged'
        ]
        
        # We append to existing merged file if it exists (since we resumed)
        file_exists = os.path.exists(merged_csv_path)
        
        with open(merged_csv_path, 'a', newline='', encoding='utf-8') as merged_file:
            writer = csv.DictWriter(merged_file, fieldnames=csv_fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for p in processes:
                # Note: We rely on the process ID that ran. 
                # Ideally, glob the directory for pipeline_results_process_*.csv
                pass 
                
        # Better merging strategy: Glob all partial files in directory
        import glob
        all_metric_files = glob.glob(f"{output_base}/{exp_name}/pipeline_results_process_*.csv")
        with open(merged_csv_path, 'a', newline='', encoding='utf-8') as merged_file:
            writer = csv.DictWriter(merged_file, fieldnames=csv_fieldnames)
            if not file_exists: writer.writeheader()
            for fpath in all_metric_files:
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            writer.writerow(row)
                    # os.remove(fpath) # Optional: cleanup
                except Exception: pass

        # 2. Merge Full Data Dumps
        full_dump_path = f"{output_base}/{exp_name}/full_data_dump.csv"
        dump_fieldnames = ['id', 'query', 'ground_truths', 'status', 'entities', 'relations',
                  'initial_symbol_candidates', 'final_candidates', 'results',
                  'grounding_candidates', 'vss_merged_candidates', 'failure_reason']
        
        dump_exists = os.path.exists(full_dump_path)
        all_dump_files = glob.glob(f"{output_base}/{exp_name}/full_dump_process_*.csv")
        
        with open(full_dump_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dump_fieldnames)
            if not dump_exists: writer.writeheader()
            for fpath in all_dump_files:
                try:
                    with open(fpath, 'r', encoding='utf-8') as pf:
                        reader = csv.DictReader(pf)
                        for row in reader:
                            writer.writerow(row)
                    # os.remove(fpath) # Optional cleanup
                except Exception: pass

        if failed_queries:
            with open(f"{output_base}/{exp_name}/failed_queries.csv", 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['query_id', 'error', 'traceback'])
                if f.tell() == 0: writer.writeheader()
                writer.writerows(failed_queries)

    except Exception as e:
        sys.stderr.write(f"CRITICAL MAIN ERROR: {e}\n")
        traceback.print_exc()

    finally:
        sys.stderr.write("Run complete.\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_config.json>")
        sys.exit(1)
    main(sys.argv[1])

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
    alpha = min(step5_params['alpha'], len(query.grounding_candidates)) # if grounding returns less than aplha candidates
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
    if not queries: return
    grounding_metrics_list = []
    vss_metrics_list = []

    for query in queries:
        if hasattr(query, 'results') and query.results is not None:
            grounding_metrics_list.append(query.results.get('metrics', {}))
        if hasattr(query, 'results') and 'vss_merged_metrics' in query.results:
            vss_metrics_list.append(query.results.get('vss_merged_metrics', {}))

    if not grounding_metrics_list: return
    num_queries = len(grounding_metrics_list)
    
    avg_metrics = {
        'total_queries': num_queries,
        'avg_total_answers': sum(m.get('total_answers', 0) for m in grounding_metrics_list) / num_queries,
        'avg_retrieved_count': sum(m.get('retrieved_count', 0) for m in grounding_metrics_list) / num_queries,
        'avg_missed_count': sum(m.get('missed_count', 0) for m in grounding_metrics_list) / num_queries,
        'avg_recall@20': sum(m.get('recall@20', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_recall@50': sum(m.get('recall@50', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_hit@1': sum(m.get('hit_at_1', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_hit@5': sum(m.get('hit_at_5', 0.0) for m in grounding_metrics_list) / num_queries,
        'avg_mrr': sum(m.get('mrr', 0.0) for m in grounding_metrics_list) / num_queries,
        'recall@20_vss_merged': sum(m.get('recall@20', 0.0) for m in vss_metrics_list) / num_queries if vss_metrics_list else 0.0,        
    }
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_metrics.keys()))
        writer.writeheader()
        writer.writerow(avg_metrics)
    print(f"\n[SAVED] Aggregate results saved to {csv_path}")

def save_data_dump(results, csv_path="aggregate_results.csv"):
    if not results: return

    def serialize_value(value):
        if value is None: return ""
        if isinstance(value, (str, int, float, bool)): return value
        try:
            return json.dumps(value, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        except Exception as e:
            return str(value)
    
    fieldnames = ['id', 'query', 'ground_truths', 'status', 'entities', 'relations',
                  'initial_symbol_candidates', 'final_candidates', 'results',
                  'grounding_candidates', 'vss_merged_candidates']
                  
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_obj in results:
            row = {field: serialize_value(getattr(query_obj, field, "")) for field in fieldnames}
            writer.writerow(row)
            
    print(f"✓ Successfully saved {len(results)} results to {csv_path}")


def pipeline_worker(query, llm_bridge, kb, retriever, dataset_name, 
                    csv_writer, config,
                    use_saved: bool = False, saved_response: Optional[dict] = None,
                    use_saved_vss: bool = False, vss_candidates: dict = {}):
    
    print(f"--- Processing Query {query.id} ---")
    
    start_time_s1 = time.time()
    step1_identify_entities(query, llm_bridge, dataset_name, use_saved, saved_response)
    end_time_s1 = time.time()
    print(f"[TIMING] Query {query.id} - Step 1 took {end_time_s1 - start_time_s1:.4f} seconds")

    if query.status == "FAILED":
        print(f"FAILED at Step 1")
        return query
    
    start_time_s2 = time.time()
    step2_identify_relations(query, llm_bridge, dataset_name, use_saved, saved_response)
    end_time_s2 = time.time()
    print(f"[TIMING] Query {query.id} - Step 2 took {end_time_s2 - start_time_s2:.4f} seconds")

    if query.status == "FAILED":
        print(f"FAILED at Step 2")
        return query
    
    start_time_s3 = time.time()
    step3_get_initial_candidates(query, kb, retriever, config)
    end_time_s3 = time.time()
    print(f"[TIMING] Query {query.id} - Step 3 took {end_time_s3 - start_time_s3:.4f} seconds")

    start_time_s4 = time.time()
    step4_grounding(query, kb, retriever, config)
    end_time_s4 = time.time()
    print(f"[TIMING] Query {query.id} - Step 4 took {end_time_s4 - start_time_s4:.4f} seconds")

    start_time_s5 = time.time()
    step5_merge_vss_candidates(query, retriever, kb, config,
                               use_saved=use_saved_vss, 
                               vss_candidates=vss_candidates,
                               llm_bridge=llm_bridge, 
                               dataset_name=dataset_name)
    end_time_s5 = time.time()
    print(f"[TIMING] Query {query.id} - Step 5 took {end_time_s5 - start_time_s5:.4f} seconds")
    
    save_results_threadsafe(query, csv_writer)
    print(f"Finished Query {query.id}")
    
    return query

def create_experiment_dir(exp_name: str, base_dir: str):
    try: os.makedirs(f"{base_dir}/{exp_name}", exist_ok=True)
    except Exception as e: pass


def process_batch(batch_queries, config, dataset_name, model_config, data_config, 
                  result_queue, failed_queue, progress_queue):
    """
    Process a batch of queries in a separate process.
    """
    try:
        # Initialize resources for this process
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
        
        # Each process writes to its own CSV file
        process_id = mp.current_process().pid
        exp_name = config['experiment'].get('exp_name', f"{dataset_name}_results")
        output_base = config['experiment'].get('output_base_dir', './output/')
        csv_writer = ThreadSafeCSVWriter(
            csv_path=f"{output_base}/{exp_name}/pipeline_results_process_{process_id}.csv",
            fieldnames=csv_fieldnames
        )
        
        # Process each query in the batch
        for query_data in batch_queries:
            try:
                current_query = Query(id=query_data[1], query=query_data[0], 
                                     ground_truths=query_data[2])
                
                llm_response = None
                if data_config['use_saved_llm_responses'] and llm_respose_df is not None:
                    llm_response_rows = llm_respose_df[llm_respose_df['id'] == current_query.id].to_dict(orient='records')
                    if llm_response_rows: 
                        llm_response = llm_response_rows[0]
                
                result = pipeline_worker(
                    current_query,
                    llm_bridge,
                    kb,
                    retriever,
                    dataset_name,
                    csv_writer,
                    config,
                    data_config['use_saved_llm_responses'],
                    llm_response,
                    data_config['use_saved_vss_candidates'],
                    vss_candidates
                )
                
                if result:
                    result_queue.put(result)
                progress_queue.put(1)
                
            except Exception as e:
                print(f"[ERROR] Query {query_data[1]}: {e}\n{traceback.format_exc()}")
                failed_queue.put({
                    'query_id': query_data[1], 
                    'error': str(e), 
                    'traceback': traceback.format_exc()
                })
                progress_queue.put(1)
                
    except Exception as e:
        print(f"[CRITICAL] Batch process error: {e}\n{traceback.format_exc()}")


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

    # 1. SETUP LOGGING
    dataset_name = exp_config['dataset']
    split_name = exp_config['split']
    
    exp_name = exp_config.get('exp_name')
    if not exp_name:
        exp_name = f"{dataset_name}_{split_name}_results"
        
    output_base = exp_config.get('output_base_dir', './output/')
    create_experiment_dir(exp_name=exp_name, base_dir=output_base)
    
    log_path = f"{output_base}/{exp_name}/full_debug_log.txt"
    
    console_out = sys.stdout
    console_err = sys.stderr
    log_file = open(log_path, 'w', encoding='utf-8')
    
    sys.stdout = log_file
    sys.stderr = log_file
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    logging.basicConfig(stream=log_file, level=getattr(logging, exp_config.get('logging_level', 'INFO')), force=True)

    try:
        qa_dataset = load_qa(dataset_name)
        qa = qa_dataset.split_indices[split_name].reshape(-1).tolist()
        qa = qa[:int(len(qa) * 0.1)] 
        test_queries = [qa_dataset[i] for i in qa]
        
        if exp_config.get('test_run', False):
            limit = pipe_config.get('max_queries_test_run', 100)
            test_queries = test_queries[:limit]
        
        console_out.write(f"Starting {len(test_queries)} queries. Logs: {log_path}\n")
        
        # Split queries into batches for multiprocessing
        num_workers = pipe_config['max_workers']
        batch_size = len(test_queries) // num_workers
        if len(test_queries) % num_workers != 0:
            batch_size += 1
        
        query_batches = [
            test_queries[i:i + batch_size] 
            for i in range(0, len(test_queries), batch_size)
        ]
        
        console_out.write(f"Split into {len(query_batches)} batches for {num_workers} workers\n")
        
        # Create queues for communication between processes
        manager = Manager()
        result_queue = manager.Queue()
        failed_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        # Create and start processes
        processes = []
        for batch in query_batches:
            p = Process(
                target=process_batch,
                args=(batch, config, dataset_name, model_config, data_config,
                      result_queue, failed_queue, progress_queue)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress
        results = []
        failed_queries = []
        completed = 0
        total = len(test_queries)
        
        with tqdm(total=total, desc="Pipeline", unit="query", file=console_out) as pbar:
            while completed < total:
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                except:
                    pass
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results from queues
        while not result_queue.empty():
            results.append(result_queue.get())
        
        while not failed_queue.empty():
            failed_queries.append(failed_queue.get())
        
        console_out.write(f"\nProcessed {len(results)} queries successfully\n")
        console_out.write(f"Failed queries: {len(failed_queries)}\n")
        
        # Merge results from all process-specific CSV files
        csv_fieldnames = [
            'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
            'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr', 'recall@20_vss_merged'
        ]
        
        merged_csv_path = f"{output_base}/{exp_name}/pipeline_results.csv"
        with open(merged_csv_path, 'w', newline='', encoding='utf-8') as merged_file:
            writer = csv.DictWriter(merged_file, fieldnames=csv_fieldnames)
            writer.writeheader()
            
            # Read and merge all process-specific CSV files
            for p in processes:
                process_csv = f"{output_base}/{exp_name}/pipeline_results_process_{p.pid}.csv"
                if os.path.exists(process_csv):
                    with open(process_csv, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            writer.writerow(row)
                    # Optionally remove process-specific CSV
                    # os.remove(process_csv)
        
        save_aggregate_results(results, csv_path=f"{output_base}/{exp_name}/aggregate_results.csv")
        save_data_dump(results, csv_path=f"{output_base}/{exp_name}/full_data_dump.csv")
        
        if failed_queries:
            with open(f"{output_base}/{exp_name}/failed_queries.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['query_id', 'error', 'traceback'])
                writer.writeheader()
                writer.writerows(failed_queries)

    except Exception as e:
        sys.stdout = console_out
        sys.stderr = console_err
        print(f"CRITICAL MAIN ERROR: {e}")
        traceback.print_exc()

    finally:
        sys.stdout = console_out
        sys.stderr = console_err
        try:
            log_file.close()
        except:
            pass
        
        console_out.write("Run complete.\n")

if __name__ == "__main__":
    # Required for multiprocessing on Windows and to avoid recursion
    mp.set_start_method('spawn', force=True)
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_config.json>")
        sys.exit(1)
        
    main(sys.argv[1])
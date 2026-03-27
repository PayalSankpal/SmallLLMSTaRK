import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import csv
import functools
from pathlib import Path
import pandas as pd
from stark_qa import load_skb, load_qa
import regex as re
import ast
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
import glob
import shutil  # Added for copying files

# Custom imports (assumed available in environment)
import custom_pipeline.vss
from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.llm_bridge_old import LlmBridge
from custom_pipeline.query import Query
from custom_pipeline.prompt_generator import get_entity_extraction_prompt, get_relation_extraction_prompt, get_query_expansion_prompt
from custom_pipeline.vss_retreiver import VSSRetriever
from custom_pipeline.candidate_context import CandidateContext
from custom_pipeline.grounders.priority_queue_grounder import PriorityQueueGrounding
from custom_pipeline.thread_safe_writers import ThreadSafeCSVWriter

# --- CONFIGURATION CONSTANTS ---
QUERY_TIMEOUT = 300  # Seconds allowed per query before killing
DUMP_INTERVAL = 5    # Dump results to file every N queries
MAX_RETRIES = 3      # Number of retries before falling back

class TimeoutException(Exception):
    pass

class PipelineStepFailed(Exception):
    """Custom exception for when LLM fails Step 1 or Step 2 with a hard error."""
    pass

class PipelineSkipped(Exception):
    """Raised when a query is intentionally skipped (LLM parse failed after MAX_LLM_RETRIES)."""
    pass

MAX_LLM_RETRIES = 3   # Max attempts per step before marking query as SKIPPED

def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    print(response)
    return response[0][0]

def step1_identify_entities(query: Query, llm_bridge: LlmBridge, dataset_name: str, use_saved: bool = False, saved_response: Optional[Dict] = None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get('entities', "")
        query.entity_id_response = response_string
        if response_string == '':
            query.status = "FAILED"
            return
        try:
            query.entities = parse_entity_response(response_string)
        except ValueError:
            query.status = "FAILED"
        return

    prompt = get_entity_extraction_prompt(query.query, dataset_name)
    for attempt in range(MAX_LLM_RETRIES):
        response_string = get_llm_response(prompt, llm_bridge)
        if not response_string:
            print(f"[Step1 RETRY] Attempt {attempt+1}/{MAX_LLM_RETRIES}: empty response, retrying...")
            continue
        try:
            identified_entities = parse_entity_response(response_string)
            query.entity_id_response = response_string
            query.entities = identified_entities
            return
        except ValueError as e:
            print(f"[Step1 RETRY] Attempt {attempt+1}/{MAX_LLM_RETRIES}: parse failed ({e}), retrying...")

    print(f"[Step1 SKIPPED] Query {query.id}: all {MAX_LLM_RETRIES} attempts failed — marking as SKIPPED.")
    query.entity_id_response = response_string
    query.status = "SKIPPED"
        
def step2_identify_relations(query: Query, llm_bridge: LlmBridge, dataset_name: str, use_saved: bool = False, saved_response: Optional[Dict] = None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get("relations", "")
        query.relations_id_response = response_string
        if response_string == '' or response_string == '{}':
            query.status = "FAILED"
            query.relations = {}
            return
        try:
            identified_relations = parse_relation_string(response_string)
            query.relations = identified_relations
        except ValueError:
            query.status = "FAILED"
            query.relations = {}
        return

    prompt = get_relation_extraction_prompt(dataset_name, query.query, query.entity_id_response)
    for attempt in range(MAX_LLM_RETRIES):
        response_string = get_llm_response(prompt, llm_bridge)
        if not response_string or response_string == '{}':
            print(f"[Step2 RETRY] Attempt {attempt+1}/{MAX_LLM_RETRIES}: empty response, retrying...")
            continue
        try:
            identified_relations = parse_relation_string(response_string)
            # Validate: all values must be lists (not nested dicts)
            for k, v in identified_relations.items():
                if not isinstance(v, list):
                    raise ValueError(f"Relation value for key {k!r} is not a list (got {type(v).__name__}). LLM may have returned nested-dict format.")
            query.relations_id_response = response_string
            query.relations = identified_relations
            edge_type_dict = {
                'affiliated_with': 'author___affiliated_with___institution',
                'cites': 'paper___cites___paper', 
                'has_topic': 'paper___has_topic___field_of_study',
                'writes': 'author___writes___paper'
            } 
            for pair in query.relations:
                rels = query.relations[pair]
                if not isinstance(rels, list):
                    continue  # safety guard; should not reach here after validation above
                for i in range(len(rels)):
                    if rels[i] in edge_type_dict:
                        rels[i] = edge_type_dict[rels[i]]
            return
        except ValueError as e:
            print(f"[Step2 RETRY] Attempt {attempt+1}/{MAX_LLM_RETRIES}: parse failed ({e}), retrying...")

    print(f"[Step2 SKIPPED] Query {query.id}: all {MAX_LLM_RETRIES} attempts failed — marking as SKIPPED.")
    query.relations_id_response = response_string
    query.relations = {}
    query.status = "SKIPPED"

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
    # Sanitize Node Types
    try:
        valid_node_types = set(kb.node_type_lst())
        for entity_key, entity_data in current_query.entities.items():
            predicted_types = entity_data.get("type", [])
            sanitized_types = []
            for t in predicted_types:
                if t in valid_node_types:
                    sanitized_types.append(t)
            entity_data["type"] = sanitized_types
    except Exception as e:
        print(f"[Step 3 Warning] Node type validation failed: {e}")

    initial_candidates = {}
    entities_to_process = {k: v for k, v in current_query.entities.items() if k != "ANSWER"}
    step3_config = config['retrieval_params']['step3_candidate_generation']
    for entity_key, entity_info in entities_to_process.items():
        try:
            e_key, candidates = get_initial_candidates_for_entity(
                entity_info, entity_key, kb, retriever,
                step3_config['initial_limit'], step3_config['vss_cutoff']
            )
            initial_candidates[e_key] = candidates
        except Exception as exc:
            pass
    current_query.initial_symbol_candidates = initial_candidates

def run_priority_queue_grounding(query_obj, kb, vss_retriever, params, verbose: bool = False):
    grounder = PriorityQueueGrounding(
        query_obj=query_obj, kb=kb, vss_retriever=vss_retriever,
        max_candidates_per_symbol=params['max_candidates_per_symbol'],
        max_answer_candidates=params['max_answer_candidates'],
        top_k_neighbors=params['top_k_neighbors'],
        score_decay=params['score_decay'],
        support_boost=params['support_boost'], verbose=verbose
    )
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
    
    # Sanitize Relations
    try:
        valid_edge_types = set(kb.rel_type_lst()) 
        clean_relations = {}
        if hasattr(query, 'relations') and query.relations:
            for entity_pair, rel_list in query.relations.items():
                valid_rels_for_pair = [r for r in rel_list if r in valid_edge_types]
                if valid_rels_for_pair:
                    clean_relations[entity_pair] = valid_rels_for_pair
        query.relations = clean_relations

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
    except Exception as e:
        raise e

def get_expanded_query(query, dataset_name: str, kb, llm_bridge) -> str:
    docs_list = []
    if not hasattr(query, 'grounding_candidates'): return query.query
    for candidate in query.grounding_candidates[:4]:
        docs_list.append(kb.get_doc_info(candidate))
    expanded_prompt = get_query_expansion_prompt(query, dataset_name, docs_list)
    expanded_query = get_llm_response(expanded_prompt, llm_bridge)
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
        possible_node_types = query.entities["ANSWER"]["type"].copy()
        for node_type in possible_node_types:
            top_candidates = retriever.get_top_k_nodes(
                search_str=expanded_query, k=k_val, node_type=node_type, cutoff=cutoff_val
            )
            all_candidates.extend(list(zip(top_candidates[0], top_candidates[1])))
        vss_candidates_list = list(map (lambda x: x[0], sorted(all_candidates, key=lambda x: x[1], reverse=True)))
    else:
        vss_candidates_list = vss_candidates.get(str(query.id), [])

    existing_candidate_ids = query.grounding_candidates[:alpha]
    new_vss_candidates = []
    for node in vss_candidates_list:
        if node not in existing_candidate_ids[:alpha] :
            new_vss_candidates.append(node)
        if len(new_vss_candidates) == 20 - alpha:
            break

    merged_candidates = existing_candidate_ids + new_vss_candidates
    query.vss_merged_candidates = merged_candidates
    
    if query.results is None: query.results = {}
    
    if 'metrics' not in query.results:
        grounding_results = evaluate_results(query.grounding_candidates, query.ground_truths)
        query.results.update(grounding_results)
    
    vss_merged_results = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = vss_merged_results['metrics']
    
    return merged_candidates

def save_results_threadsafe(query: Query, csv_writer: ThreadSafeCSVWriter):
    if query.results is None: query.results = {}
    grounding_cands = getattr(query, 'grounding_candidates', [])
    vss_merged_cands = getattr(query, 'vss_merged_candidates', [])

    if 'metrics' not in query.results and grounding_cands:
        query.results.update(evaluate_results(grounding_cands, query.ground_truths))
    
    if 'vss_merged_metrics' not in query.results and vss_merged_cands:
        res = evaluate_results(vss_merged_cands, query.ground_truths)
        query.results['vss_merged_metrics'] = res['metrics']

    metrics = query.results.get('metrics', {})
    vss_metrics = query.results.get('vss_merged_metrics', {})
    
    row_data = {
        'query_id': query.id,
        'query_text': query.query,
        'total_answers': metrics.get('total_answers', len(query.ground_truths)),
        'retrieved_count': metrics.get('retrieved_count', 0),
        'missed_count': metrics.get('missed_count', len(query.ground_truths)),
        'recall@20': metrics.get('recall@20', 0.0),
        'recall@50': metrics.get('recall@50', 0.0),
        'hit@1': metrics.get('hit_at_1', 0.0),
        'hit@5': metrics.get('hit_at_5', 0.0),
        'mrr': metrics.get('mrr', 0.0),
        'recall@20_vss_merged': vss_metrics.get('recall@20', 0.0)
    }
    csv_writer.write_row(row_data)

def serialize_value(value):
    if value is None: return ""
    if isinstance(value, (str, int, float, bool)): return value
    try:
        return json.dumps(value, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
    except Exception as e:
        return str(value)

def save_partial_dump(results, csv_path, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for query_obj in results:
            row = {field: serialize_value(getattr(query_obj, field, "")) for field in fieldnames}
            row['failure_reason'] = getattr(query_obj, 'failure_reason', "")
            writer.writerow(row)

# --- FALLBACK STRATEGY ---
def run_fallback_strategy(query: Query, retriever: VSSRetriever, kb, csv_writer, failure_reason: str):
    """
    If the pipeline failed, run a pure VSS search on common types as fallback.
    """
    print(f"--- [FALLBACK] Query {query.id}: Falling back to VSS. Reason: {failure_reason}")
    query.status = "FALLBACK"
    query.failure_reason = failure_reason
    
    # Check for detected type
    node_types = ['disease'] # Default
    if hasattr(query, 'entities') and query.entities and "ANSWER" in query.entities:
        detected = query.entities["ANSWER"].get("type", [])
        if detected:
            node_types = detected
            print(f"--- [FALLBACK] Using detected ANSWER types: {node_types}")
        else:
            print(f"--- [FALLBACK] No ANSWER type detected. Defaulting to: {node_types}")
    else:
        print(f"--- [FALLBACK] Step 1 failed/missing. Defaulting to: {node_types}")

    all_candidates = []
    # Search for all candidate types (usually just 1, but list allows flexibility)
    for nt in node_types:
        try:
            top_ids, scores = retriever.get_top_k_nodes(
                search_str=query.query, k=20, node_type=nt, cutoff=0.0
            )
            all_candidates.extend(list(zip(top_ids, scores)))
        except Exception as e:
            print(f"--- [FALLBACK] VSS failed for type {nt}: {e}")
      
    sorted_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:20]
    final_ids = [x[0] for x in sorted_candidates]
    
    query.vss_merged_candidates = final_ids
    query.grounding_candidates = [] 
    
    vss_results = evaluate_results(final_ids, query.ground_truths)
    query.results = {
        'metrics': {'recall@20': 0.0},
        'vss_merged_metrics': vss_results['metrics']
    }
    
    save_results_threadsafe(query, csv_writer)
    return query

def pipeline_worker(query, llm_bridge, kb, retriever, dataset_name, 
                    csv_writer, config,
                    use_saved: bool = False, saved_response: Optional[dict] = None,
                    use_saved_vss: bool = False, vss_candidates: dict = {}):
    
    print(f"--- Processing Query {query.id} ---")
    
    step1_identify_entities(query, llm_bridge, dataset_name, use_saved, saved_response)
    if query.status == "SKIPPED": raise PipelineSkipped("Step 1 Skipped (Entity Extraction parse failed)")
    if query.status == "FAILED":  raise PipelineStepFailed("Step 1 Failed (Entity Extraction)")
    
    step2_identify_relations(query, llm_bridge, dataset_name, use_saved, saved_response)
    if query.status == "SKIPPED": raise PipelineSkipped("Step 2 Skipped (Relation Extraction parse failed)")
    if query.status == "FAILED":  raise PipelineStepFailed("Step 2 Failed (Relation Extraction)")
    
    step3_get_initial_candidates(query, kb, retriever, config)
    step4_grounding(query, kb, retriever, config)
    step5_merge_vss_candidates(query, retriever, kb, config,
                               use_saved=use_saved_vss, 
                               vss_candidates=vss_candidates,
                               llm_bridge=llm_bridge, 
                               dataset_name=dataset_name)
    
    save_results_threadsafe(query, csv_writer)
    print(f"Finished Query {query.id}")
    return query

def run_with_timeout(func, args, timeout):
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
    
    log_file_path = f"{output_base}/{exp_name}/process_{process_id}.log"
    sys.stdout = open(log_file_path, "a", encoding="utf-8")
    
    dump_fieldnames = ['id', 'query', 'ground_truths', 'status', 'entities', 'relations',
                  'initial_symbol_candidates', 'final_candidates', 'results',
                  'grounding_candidates', 'vss_merged_candidates', 'failure_reason']
    
    dump_file_path = f"{output_base}/{exp_name}/full_dump_process_{process_id}.csv"
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
            use_gpu=False
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
        print("hello")        
        for i, query_data in enumerate(batch_queries):
            current_query = Query(id=query_data[1], query=query_data[0], ground_truths=query_data[2])
            
            llm_response = None
            if data_config['use_saved_llm_responses'] and llm_respose_df is not None:
                llm_response_rows = llm_respose_df[llm_respose_df['id'] == current_query.id].to_dict(orient='records')
                if llm_response_rows: llm_response = llm_response_rows[0]
            
            success = False
            skipped = False
            last_error = ""
            
            print("hhhhhhhhhelli")

            # --- RETRY LOOP ---
            for attempt in range(MAX_RETRIES):
                try:
                    args = (current_query, llm_bridge, kb, retriever, dataset_name,
                            csv_writer, config, data_config['use_saved_llm_responses'],
                            llm_response, data_config['use_saved_vss_candidates'], vss_candidates)
                    
                    result = run_with_timeout(pipeline_worker, args, QUERY_TIMEOUT)
                    batch_results_buffer.append(result)
                    success = True
                    break # Success! Exit retry loop.

                except PipelineSkipped as e:
                    # LLM parse failed after MAX_LLM_RETRIES — record as SKIPPED, no fallback
                    print(f"[SKIPPED] Query {current_query.id}: {e}")
                    current_query.status = "SKIPPED"
                    current_query.failure_reason = str(e)
                    batch_results_buffer.append(current_query)  # saved to dump, NOT to metrics CSV
                    skipped = True
                    break

                except (PipelineStepFailed, TimeoutException) as e:
                    last_error = str(e)
                    print(f"[WARNING] Query {current_query.id} failed attempt {attempt+1}/{MAX_RETRIES}: {e}")
                    # Continue to next retry
                except Exception as e:
                    last_error = str(e)
                    print(f"[ERROR] Query {current_query.id} crashed attempt {attempt+1}/{MAX_RETRIES}: {e}")
                    traceback.print_exc()
                    # Continue to next retry

            # If all retries failed (not skipped), trigger Fallback
            if not success and not skipped:
                print(f"[CRITICAL] Query {current_query.id} failed {MAX_RETRIES} attempts. Triggering Fallback.")
                try:
                    fallback_result = run_fallback_strategy(
                        current_query, retriever, kb, csv_writer, failure_reason=last_error
                    )
                    batch_results_buffer.append(fallback_result)
                except Exception as fb_err:
                    print(f"[FATAL] Fallback also failed for {current_query.id}: {fb_err}")
                    failed_queue.put({'query_id': current_query.id, 'error': str(fb_err)})

            progress_queue.put(1)

            if len(batch_results_buffer) >= DUMP_INTERVAL:
                save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
                for res in batch_results_buffer: result_queue.put(res) 
                batch_results_buffer = []

    except Exception as e:
        sys.stderr.write(f"[CRITICAL] Batch process {process_id} crashed: {e}\n")
    finally:
        if batch_results_buffer:
            save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
            for res in batch_results_buffer: result_queue.put(res)
        sys.stdout.close()

def create_experiment_dir(exp_name: str, base_dir: str):
    try: os.makedirs(f"{base_dir}/{exp_name}", exist_ok=True)
    except Exception: pass

def print_aggregate_report(pipeline_results_path):
    if not os.path.exists(pipeline_results_path): return

    try:
        df = pd.read_csv(pipeline_results_path)
        if df.empty: return

        num_queries = len(df)
        avg_metrics = {
            'total_queries': num_queries,
            'avg_total_answers': df['total_answers'].mean(),
            'avg_retrieved_count': df['retrieved_count'].mean(),
            'avg_missed_count': df['missed_count'].mean(),
            'avg_recall@20': df['recall@20'].mean(),
            'avg_recall@50': df['recall@50'].mean(),
            'avg_hit@1': df['hit@1'].mean(),
            'avg_hit@5': df['hit@5'].mean(),
            'avg_mrr': df['mrr'].mean(),
            'recall@20_vss_merged': df['recall@20_vss_merged'].mean()
        }

        print("\n" + "="*50, file=sys.__stdout__)
        print(f" FINAL AGGREGATE REPORT (N={num_queries})", file=sys.__stdout__)
        print("="*50, file=sys.__stdout__)
        for k, v in avg_metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"{k:<30} | {val_str:<10}", file=sys.__stdout__)
        print("="*50 + "\n", file=sys.__stdout__)

        output_dir = os.path.dirname(pipeline_results_path)
        agg_path = os.path.join(output_dir, "aggregate_results.csv")
        with open(agg_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(avg_metrics.keys()))
            writer.writeheader()
            writer.writerow(avg_metrics)

    except Exception as e:
        print(f"[Report Error] Failed to calculate aggregates: {e}", file=sys.__stdout__)
def main():
    # 1. SETUP ARGUMENTS
    parser = argparse.ArgumentParser(description="Run StarkQA Pipeline")
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file")
    
    # CLI Overrides
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--alpha", type=int, default=None, help="Override Step 5 alpha")
    parser.add_argument("--score_decay", type=float, default=None, help="Override grounding score decay")
    parser.add_argument("--num_workers", type=int, default=None, help="Override max workers")

    parser.add_argument("--split", type=str, default=None, help="Override data split (validation/test)")
    parser.add_argument("--retry", action="store_true", help="Re-run only queries that have FAILED or FALLBACK status in the existing dump")
    parser.add_argument("--run_skipped", action="store_true", help="Re-run only queries that have SKIPPED status in the existing dump files")
    parser.add_argument("--query_ids", type=str, default=None, help="Comma-separated list of query IDs to run (e.g. 5883,10405,1109). Overrides skip/retry logic.")

    args = parser.parse_args()

    # 2. LOAD CONFIG
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # 3. APPLY OVERRIDES
    if args.exp_name: config['experiment']['exp_name'] = args.exp_name
    if args.dataset: config['experiment']['dataset'] = args.dataset
    if args.alpha is not None: config['retrieval_params']['step5_vss_merge']['alpha'] = args.alpha
    if args.score_decay is not None: config['grounding_params']['score_decay'] = args.score_decay
    if args.num_workers is not None: config['pipeline']['max_workers'] = args.num_workers
    if args.split: config['experiment']['split'] = args.split

    # --- NEW: PRINT FINALIZED PARAMS (COMPACT) ---
    print("\n" + "="*80)
    print("FINALIZED PARAMETERS:")
    flat_params = []
    for section, content in config.items():
        if isinstance(content, dict):
            for k, v in content.items():
                flat_params.append(f"{section}.{k}: {v}")
        else:
            flat_params.append(f"{section}: {content}")
            
    # Print 3 parameters per line for compactness
    for i in range(0, len(flat_params), 3):
        print(" | ".join(f"{p:<35}" for p in flat_params[i:i+3]))
    print("="*80 + "\n")
    # ---------------------------------------------

    # 4. SETUP DIRECTORIES
    exp_config = config['experiment']
    pipe_config = config['pipeline']
    data_config = config['data_paths']
    model_config = config['models']

    dataset_name = exp_config['dataset']
    split_name = exp_config['split']
    exp_name = exp_config.get('exp_name', f"{dataset_name}_{split_name}_results")
    output_base = exp_config.get('output_base_dir', './output/')
    
    exp_dir = f"{output_base}/{exp_name}"
    create_experiment_dir(exp_name, output_base)
    
    # --- ARCHIVE CONFIG AND CODE ---
    try:
        # Copy Config
        shutil.copy(args.config_path, f"{exp_dir}/config_snapshot.json")
        # Copy Script
        shutil.copy(__file__, f"{exp_dir}/script_snapshot.py")
        # Copy prompt templates (so we always know which prompts produced these results)
        for dataset_p_name in [dataset_name, 'prime', 'mag', 'amazon']:
            for prompt_kind in ['entity', 'relation']:
                src = f"custom_pipeline/prompt_examples/{prompt_kind}_{dataset_p_name}.txt"
                if os.path.exists(src):
                    shutil.copy(src, f"{exp_dir}/{prompt_kind}_{dataset_p_name}_prompt.txt")
        # Also copy prompt_generator.py
        pg_src = "custom_pipeline/prompt_generator.py"
        if os.path.exists(pg_src):
            shutil.copy(pg_src, f"{exp_dir}/prompt_generator_snapshot.py")
        print(f"[Archived] Config, Code, and Prompts saved to {exp_dir}", file=sys.__stdout__)
    except Exception as e:
        print(f"[Warning] Failed to archive files: {e}", file=sys.__stdout__)

    sys.stdout = open(f"{exp_dir}/main_process.log", 'a')

    try:
        qa_dataset = load_qa(dataset_name)
        qa = qa_dataset.split_indices[split_name].reshape(-1).tolist()
        qa = qa[:int(len(qa) * 0.1)] 
        all_test_queries = [qa_dataset[i] for i in qa]
        
        if exp_config.get('test_run', False):
            all_test_queries = all_test_queries[:pipe_config.get('max_queries_test_run', 100)]
        
        completed_ids = set()
        main_dump = f"{exp_dir}/full_data_dump.csv"

        if args.query_ids:
            # Run ONLY the explicitly specified query IDs
            target_ids = set(str(x.strip()) for x in args.query_ids.split(',') if x.strip())
            test_queries = [q for q in all_test_queries if str(q[1]) in target_ids]
            print(f"[query_ids] Running {len(test_queries)} specified queries: {sorted(target_ids)}", file=sys.__stdout__)
        elif args.retry:
            # Collect IDs that have FAILED or FALLBACK status — these need to be re-run
            retry_ids = set()
            for dump_path in [main_dump] + glob.glob(f"{exp_dir}/full_dump_process_*.csv"):
                if os.path.exists(dump_path):
                    try:
                        dump_df = pd.read_csv(dump_path)
                        if 'status' in dump_df.columns:
                            bad = dump_df[dump_df['status'].isin(['FAILED', 'FALLBACK'])]['id'].astype(str)
                            retry_ids.update(bad.tolist())
                            # Mark the rest (good ones) as already completed
                            good = dump_df[~dump_df['status'].isin(['FAILED', 'FALLBACK'])]['id'].astype(str)
                            completed_ids.update(good.tolist())
                    except Exception as e:
                        print(f"[Retry] Could not read {dump_path}: {e}", file=sys.__stdout__)
            print(f"[Retry mode] Re-running {len(retry_ids)} FAILED/FALLBACK queries.", file=sys.__stdout__)
            print(f"[Retry mode] Query IDs: {sorted(retry_ids)}", file=sys.__stdout__)
            test_queries = [q for q in all_test_queries if str(q[1]) in retry_ids]
        elif args.run_skipped:
            # Collect IDs with SKIPPED status — re-run them, treat everything else as done
            skipped_ids = set()
            for dump_path in [main_dump] + glob.glob(f"{exp_dir}/full_dump_process_*.csv"):
                if os.path.exists(dump_path):
                    try:
                        dump_df = pd.read_csv(dump_path)
                        if 'status' in dump_df.columns:
                            skipped = dump_df[dump_df['status'] == 'SKIPPED']['id'].astype(str)
                            skipped_ids.update(skipped.tolist())
                            done = dump_df[dump_df['status'] != 'SKIPPED']['id'].astype(str)
                            completed_ids.update(done.tolist())
                    except Exception as e:
                        print(f"[run_skipped] Could not read {dump_path}: {e}", file=sys.__stdout__)
            print(f"[run_skipped] Re-running {len(skipped_ids)} SKIPPED queries.", file=sys.__stdout__)
            print(f"[run_skipped] Query IDs: {sorted(skipped_ids)}", file=sys.__stdout__)
            test_queries = [q for q in all_test_queries if str(q[1]) in skipped_ids]
        else:
            if os.path.exists(main_dump):
                try: completed_ids.update(pd.read_csv(main_dump)['id'].astype(str).tolist())
                except: pass
            for f in glob.glob(f"{exp_dir}/full_dump_process_*.csv"):
                try: completed_ids.update(pd.read_csv(f)['id'].astype(str).tolist())
                except: pass
            test_queries = [q for q in all_test_queries if str(q[1]) not in completed_ids]
        sys.stderr.write(f"Starting {len(test_queries)} queries (Skipped {len(completed_ids)}).\n")

        if test_queries:
            num_workers = pipe_config['max_workers']
            batch_size = max(1, len(test_queries) // num_workers + (1 if len(test_queries) % num_workers != 0 else 0))
            query_batches = [test_queries[i:i + batch_size] for i in range(0, len(test_queries), batch_size)]
            
            manager = Manager()
            result_queue, failed_queue, progress_queue = manager.Queue(), manager.Queue(), manager.Queue()
            
            processes = []
            for batch in query_batches:
                if not batch: continue
                p = Process(target=process_batch, args=(batch, config, dataset_name, model_config, data_config, result_queue, failed_queue, progress_queue))
                p.start()
                processes.append(p)
            
            completed = 0
            with tqdm(total=len(test_queries), desc="Pipeline", file=sys.stderr) as pbar:
                while completed < len(test_queries):
                    if not any(p.is_alive() for p in processes) and progress_queue.empty(): break
                    try:
                        progress_queue.get(timeout=1)
                        completed += 1
                        pbar.update(1)
                    except: pass
            
            for p in processes: p.join()

        sys.stderr.write("\nMerging results...\n")
        
        metric_files = glob.glob(f"{exp_dir}/pipeline_results_process_*.csv")
        final_metrics_path = f"{exp_dir}/pipeline_results.csv"
        
        dfs = []
        if os.path.exists(final_metrics_path):
            try: dfs.append(pd.read_csv(final_metrics_path))
            except: pass
        for f in metric_files:
            try: dfs.append(pd.read_csv(f))
            except: pass
            
        if dfs:
            full_metrics = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['query_id'], keep='last')
            full_metrics.to_csv(final_metrics_path, index=False)
            print_aggregate_report(final_metrics_path)

        dump_files = glob.glob(f"{exp_dir}/full_dump_process_*.csv")
        final_dump_path = f"{exp_dir}/full_data_dump.csv"
        
        dump_dfs = []
        if os.path.exists(final_dump_path):
            try: dump_dfs.append(pd.read_csv(final_dump_path))
            except: pass
        for f in dump_files:
            try: dump_dfs.append(pd.read_csv(f))
            except: pass

        if dump_dfs:
            # keep='last' ensures re-run results (appended last) override old SKIPPED/FAILED records
            full_dump = pd.concat(dump_dfs, ignore_index=True).drop_duplicates(subset=['id'], keep='last')
            full_dump.to_csv(final_dump_path, index=False)

    except Exception as e:
        sys.stderr.write(f"CRITICAL MAIN ERROR: {e}\n")
        traceback.print_exc()
    finally:
        sys.stdout.close()
        
        try:
            log_dir = os.path.join(exp_dir, "log")
            config_dir = os.path.join(exp_dir, "config")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            
            for pat in ["*_process_*.csv", "*.log"]:
                for f in glob.glob(os.path.join(exp_dir, pat)):
                    try: shutil.move(f, os.path.join(log_dir, os.path.basename(f)))
                    except: pass
                    
            for pat in ["*.json", "*.txt", "*.py", "*.pyc"]:
                for f in glob.glob(os.path.join(exp_dir, pat)):
                    try: shutil.move(f, os.path.join(config_dir, os.path.basename(f)))
                    except: pass
        except Exception as oe:
            sys.stderr.write(f"Error organizing files: {oe}\n")

        sys.stderr.write("Run complete.\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

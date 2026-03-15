"""
parallel_v2.py — Redesigned StarkQA Pipeline with 3 new stages:

  Step 2.5  Entity Anchor Verification
            After VSS retrieval, for each constant anchor whose lexical exact-match
            FAILS (score < 1.0), show top-3 candidates to the LLM and ask it to
            confirm the best match.  Fixes wrong anchor grounding before the graph
            traversal even begins.

  Step 4.5  ANSWER-Direct VSS Fallback
            If priority-queue grounding returns fewer than GROUNDING_MIN_THRESHOLD
            ANSWER candidates, run a targeted VSS query using the ANSWER entity's
            type + the full query text + ANSWER semantic.  Merged back into
            grounding_candidates so step-5 can still do its alpha-blend.

  Step 6    LLM Reranking
            After step-5 produces the top-50 candidate pool, ask the LLM to rank
            them from most to least likely to be correct.  Returns top-20 reranked.
            This is the most powerful new stage: semantic understanding at the last
            mile, where the existing pipeline is purely score-based.

Config keys added (all inside params_v2.json):
  pipeline_v2.use_anchor_verification   (bool, default true)
  pipeline_v2.use_answer_direct_vss     (bool, default true)
  pipeline_v2.answer_direct_vss_k       (int,  default 100)
  pipeline_v2.grounding_min_threshold   (int,  default 5)
  pipeline_v2.use_llm_reranking         (bool, default true)
  pipeline_v2.rerank_pool_size          (int,  default 50)
  pipeline_v2.rerank_top_k              (int,  default 20)
"""

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
from typing import List, Dict, Any, Optional, Tuple
import json
import signal
import warnings
import concurrent.futures
import glob
import shutil

from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.llm_bridge import LlmBridge
from custom_pipeline.query import Query
from custom_pipeline.prompt_generator import (
    get_entity_extraction_prompt,
    get_relation_extraction_prompt,
    get_query_expansion_prompt,
    get_entity_anchor_verification_prompt,
    get_llm_reranking_prompt,
)
from custom_pipeline.vss_retreiver import VSSRetriever
from custom_pipeline.candidate_context import CandidateContext
from custom_pipeline.grounders.priority_queue_grounder import PriorityQueueGrounding
from custom_pipeline.thread_safe_writers import ThreadSafeCSVWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUERY_TIMEOUT   = 300
DUMP_INTERVAL   = 5
MAX_RETRIES     = 3
MAX_LLM_RETRIES = 3

# How few ANSWER grounding candidates trigger the direct-VSS fallback
GROUNDING_MIN_THRESHOLD = 5


class TimeoutException(Exception):
    pass

class PipelineStepFailed(Exception):
    pass

class PipelineSkipped(Exception):
    pass


# ---------------------------------------------------------------------------
# Shared helpers (unchanged from parallel_final.py)
# ---------------------------------------------------------------------------

def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    print(response)
    return response[0][0]


def step1_identify_entities(query: Query, llm_bridge, dataset_name: str,
                             use_saved=False, saved_response=None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get('entities', "")
        query.entity_id_response = response_string
        if not response_string:
            query.status = "FAILED"; return
        try:
            query.entities = parse_entity_response(response_string)
        except ValueError:
            query.status = "FAILED"
        return

    prompt = get_entity_extraction_prompt(query.query, dataset_name)
    for attempt in range(MAX_LLM_RETRIES):
        response_string = get_llm_response(prompt, llm_bridge)
        if not response_string:
            print(f"[Step1 RETRY] {attempt+1}/{MAX_LLM_RETRIES}: empty response"); continue
        try:
            query.entity_id_response = response_string
            query.entities = parse_entity_response(response_string)
            return
        except ValueError as e:
            print(f"[Step1 RETRY] {attempt+1}/{MAX_LLM_RETRIES}: parse failed ({e})")

    print(f"[Step1 SKIPPED] Query {query.id}: all attempts failed.")
    query.entity_id_response = response_string
    query.status = "SKIPPED"


def step2_identify_relations(query: Query, llm_bridge, dataset_name: str,
                              use_saved=False, saved_response=None):
    if use_saved and saved_response is not None:
        response_string = saved_response.get("relations", "")
        query.relations_id_response = response_string
        if not response_string or response_string == '{}':
            query.status = "FAILED"; query.relations = {}; return
        try:
            query.relations = parse_relation_string(response_string)
        except ValueError:
            query.status = "FAILED"; query.relations = {}
        return

    prompt = get_relation_extraction_prompt(dataset_name, query.query, query.entity_id_response)
    for attempt in range(MAX_LLM_RETRIES):
        response_string = get_llm_response(prompt, llm_bridge)
        if not response_string or response_string == '{}':
            print(f"[Step2 RETRY] {attempt+1}/{MAX_LLM_RETRIES}: empty response"); continue
        try:
            identified_relations = parse_relation_string(response_string)
            for k, v in identified_relations.items():
                if not isinstance(v, list):
                    raise ValueError(f"Relation for {k!r} not a list")
            query.relations_id_response = response_string
            query.relations = identified_relations
            return
        except ValueError as e:
            print(f"[Step2 RETRY] {attempt+1}/{MAX_LLM_RETRIES}: parse failed ({e})")

    print(f"[Step2 SKIPPED] Query {query.id}: all attempts failed.")
    query.relations_id_response = response_string
    query.relations = {}
    query.status = "SKIPPED"


def get_initial_candidates_for_entity(entity_info, entity_key, kb, retriever,
                                       limit=25, cutoff=0.65):
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
        # Parallel VSS across entity types
        def _vss_one_type(etype):
            return retriever.get_top_k_nodes(
                search_str=search_sem, k=vss_needed, node_type=etype, cutoff=cutoff
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(entity_types) or 1) as pool:
            futures = {pool.submit(_vss_one_type, et): et for et in entity_types}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    node_ids, vss_scores = fut.result()
                    for i, node_id in enumerate(node_ids):
                        if node_id not in existing_ids:
                            candidates.append(CandidateContext(node_id=node_id, entity=entity_key, score=vss_scores[i]))
                except Exception:
                    pass
    return entity_key, candidates


def step3_get_initial_candidates(current_query: Query, kb, retriever, config):
    try:
        valid_node_types = set(kb.node_type_lst())
        for entity_key, entity_data in current_query.entities.items():
            entity_data["type"] = [t for t in entity_data.get("type", []) if t in valid_node_types]
    except Exception as e:
        print(f"[Step3 Warning] Node type validation failed: {e}")

    initial_candidates = {}
    step3_config = config['retrieval_params']['step3_candidate_generation']
    anchor_entities = [(k, v) for k, v in current_query.entities.items() if k != "ANSWER"]

    # Run VSS for all anchor entities in parallel
    def _fetch_entity(entity_key, entity_info):
        return get_initial_candidates_for_entity(
            entity_info, entity_key, kb, retriever,
            step3_config['initial_limit'], step3_config['vss_cutoff']
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(anchor_entities))) as pool:
        futs = {pool.submit(_fetch_entity, k, v): k for k, v in anchor_entities}
        for fut in concurrent.futures.as_completed(futs):
            try:
                e_key, candidates = fut.result()
                initial_candidates[e_key] = candidates
            except Exception:
                pass

    current_query.initial_symbol_candidates = initial_candidates


# ---------------------------------------------------------------------------
# NEW Step 2.5 — Entity Anchor Verification
# ---------------------------------------------------------------------------

def step2_5_verify_anchors(query: Query, kb, retriever, llm_bridge, v2_config: dict):
    """
    For each constant anchor entity whose initial retrieval did NOT find a
    score=1.0 exact match, show top-3 VSS candidates to the LLM and ask it
    to confirm the best node.  If LLM picks a different option, we update
    the initial_symbol_candidates accordingly.
    """
    if not v2_config.get('use_anchor_verification', True):
        return

    print("[Step2.5] Running entity anchor verification ...")

    for entity_key, entity_info in query.entities.items():
        if entity_key == "ANSWER":
            continue
        if not entity_info.get('constant', False):
            continue  # Only verify constant anchors

        name_constraint = entity_info.get('lexical', {}).get('name', '')
        if not name_constraint:
            continue  # Nothing to verify without a lexical name

        existing = query.initial_symbol_candidates.get(entity_key, [])
        # Check if any exact match already exists (score = 1.0)
        exact = [c for c in existing if c.score >= 1.0]
        if exact:
            print(f"  [{entity_key}] exact match found ({exact[0].node_id}) — skipping verification")
            continue

        # Build top-3 candidates for LLM
        candidates_for_llm = []
        seen = set()
        for c in existing[:10]:
            if c.node_id not in seen:
                doc_str = kb.get_doc_info(c.node_id, compact=True)
                candidates_for_llm.append((c.node_id, c.score, doc_str))
                seen.add(c.node_id)
            if len(candidates_for_llm) >= 3:
                break

        if not candidates_for_llm:
            print(f"  [{entity_key}] no candidates to verify")
            continue

        prompt = get_entity_anchor_verification_prompt(
            query=query.query,
            entity_key=entity_key,
            entity_info=entity_info,
            candidates=candidates_for_llm,
            kb_doc_fn=kb.get_doc_info,
        )

        try:
            response_str = get_llm_response(prompt, llm_bridge).strip()
            choice = int(response_str)
        except (ValueError, TypeError):
            print(f"  [{entity_key}] LLM gave unparseable response: {response_str!r}")
            continue

        if choice == 0:
            print(f"  [{entity_key}] LLM says none match — keeping VSS order")
            continue

        if 1 <= choice <= len(candidates_for_llm):
            verified_node_id, verified_score, _ = candidates_for_llm[choice - 1]
            # Promote this node to score=1.0 and move to front
            new_cands = [CandidateContext(node_id=verified_node_id, entity=entity_key, score=1.0)]
            for c in existing:
                if c.node_id != verified_node_id:
                    new_cands.append(c)
            query.initial_symbol_candidates[entity_key] = new_cands
            print(f"  [{entity_key}] LLM verified node {verified_node_id} as best anchor → promoted to score=1.0")
        else:
            print(f"  [{entity_key}] LLM choice {choice} out of range — ignoring")


# ---------------------------------------------------------------------------
# Priority queue grounding (unchanged wrapper)
# ---------------------------------------------------------------------------

def run_priority_queue_grounding(query_obj, kb, vss_retriever, params, verbose=False):
    grounder = PriorityQueueGrounding(
        query_obj=query_obj, kb=kb, vss_retriever=vss_retriever,
        max_candidates_per_symbol=params['max_candidates_per_symbol'],
        max_answer_candidates=params['max_answer_candidates'],
        top_k_neighbors=params['top_k_neighbors'],
        score_decay=params['score_decay'],
        support_boost=params['support_boost'],
        verbose=verbose,
    )
    return grounder.ground()


def step4_grounding(query: Query, kb, retriever, config):
    g_params = config['grounding_params']
    try:
        valid_edge_types = set(kb.rel_type_lst())
        clean_relations = {}
        if hasattr(query, 'relations') and query.relations:
            for entity_pair, rel_list in query.relations.items():
                valid_rels = [r for r in rel_list if r in valid_edge_types]
                if valid_rels:
                    clean_relations[entity_pair] = valid_rels
        query.relations = clean_relations

        final_candidates = run_priority_queue_grounding(
            query_obj=query, kb=kb, vss_retriever=retriever,
            params=g_params, verbose=False,
        )
        if "ANSWER" in final_candidates:
            answers = [cc.node_id for cc in sorted(
                final_candidates["ANSWER"],
                key=lambda x: (x.score, -x.support), reverse=True,
            )]
        else:
            answers = []
        query.grounding_candidates = answers
        query.final_candidates = final_candidates
        return final_candidates
    except Exception as e:
        raise e


# ---------------------------------------------------------------------------
# NEW Step 4.5 — ANSWER-Direct VSS Fallback
# ---------------------------------------------------------------------------

def step4_5_answer_direct_vss(query: Query, retriever, kb, config, v2_config: dict):
    """
    If the priority-queue grounder found very few ANSWER candidates, supplement
    them with a direct VSS search on the ANSWER entity type using the full query
    text + ANSWER semantic description.

    The new candidates are deduplicated and APPENDED to grounding_candidates so
    that step-5 alpha-blend still works (they are ranked after proper grounding
    results but before pure VSS).
    """
    if not v2_config.get('use_answer_direct_vss', True):
        return

    current_count = len(query.grounding_candidates)
    threshold = v2_config.get('grounding_min_threshold', GROUNDING_MIN_THRESHOLD)
    if current_count >= threshold:
        print(f"[Step4.5] Grounding has {current_count} candidates — skipping direct VSS")
        return

    print(f"[Step4.5] Only {current_count} grounding candidates — running ANSWER-direct VSS ...")

    answer_entity = query.entities.get("ANSWER", {})
    answer_types = answer_entity.get("type", [])
    answer_semantic = answer_entity.get("semantic", [])

    # Build a rich query string: original query + ANSWER semantic hints
    combined_query = query.query
    if answer_semantic:
        combined_query += " " + " ".join(answer_semantic)

    k_direct = v2_config.get('answer_direct_vss_k', 100)
    cutoff = config['retrieval_params']['step3_candidate_generation'].get('vss_cutoff', 0.6)

    all_direct = []
    def _direct_vss_type(node_type):
        return retriever.get_top_k_nodes(
            search_str=combined_query, k=k_direct, node_type=node_type, cutoff=cutoff
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(answer_types))) as pool:
        futs = {pool.submit(_direct_vss_type, nt): nt for nt in answer_types}
        for fut in concurrent.futures.as_completed(futs):
            try:
                node_ids, scores = fut.result()
                all_direct.extend(list(zip(node_ids, scores)))
            except Exception as e:
                print(f"[Step4.5] VSS thread failed: {e}")

    # Sort by score, deduplicate, append to grounding_candidates
    all_direct.sort(key=lambda x: x[1], reverse=True)
    existing_set = set(query.grounding_candidates)
    new_ids = []
    for node_id, score in all_direct:
        if node_id not in existing_set:
            new_ids.append(node_id)
            existing_set.add(node_id)

    query.grounding_candidates = query.grounding_candidates + new_ids
    print(f"[Step4.5] Added {len(new_ids)} direct VSS candidates. Total grounding: {len(query.grounding_candidates)}")


# ---------------------------------------------------------------------------
# Step 5 — VSS merge (adapted from parallel_final.py)
# ---------------------------------------------------------------------------

def get_expanded_query(query, dataset_name, kb, llm_bridge):
    docs_list = []
    if not hasattr(query, 'grounding_candidates'):
        return query.query
    for candidate in query.grounding_candidates[:4]:
        docs_list.append(kb.get_doc_info(candidate))
    expanded_prompt = get_query_expansion_prompt(query, dataset_name, docs_list)
    return get_llm_response(expanded_prompt, llm_bridge)


def step5_merge_vss_candidates(query: Query, retriever, kb, config,
                                use_saved=False, vss_candidates=None,
                                llm_bridge=None, dataset_name=""):
    vss_candidates = vss_candidates or {}
    step5_params = config['retrieval_params']['step5_vss_merge']
    alpha   = min(step5_params['alpha'], len(query.grounding_candidates))
    k_val   = step5_params['top_k']
    cutoff  = step5_params['cutoff']

    vss_list = []
    if not use_saved:
        expanded_query = get_expanded_query(query, dataset_name=dataset_name, kb=kb, llm_bridge=llm_bridge)
        all_candidates = []
        for node_type in query.entities["ANSWER"]["type"]:
            top_ids, top_scores = retriever.get_top_k_nodes(
                search_str=expanded_query, k=k_val, node_type=node_type, cutoff=cutoff
            )
            all_candidates.extend(list(zip(top_ids, top_scores)))
        vss_list = [x[0] for x in sorted(all_candidates, key=lambda x: x[1], reverse=True)]
    else:
        vss_list = vss_candidates.get(str(query.id), [])

    existing_ids = query.grounding_candidates[:alpha]
    existing_set = set(existing_ids[:alpha])
    new_vss = []
    for node in vss_list:
        if node not in existing_set:
            new_vss.append(node)
        if len(new_vss) == 20 - alpha:
            break

    merged = existing_ids + new_vss
    query.vss_merged_candidates = merged

    if query.results is None:
        query.results = {}
    if 'metrics' not in query.results:
        query.results.update(evaluate_results(query.grounding_candidates, query.ground_truths))
    vss_metrics = evaluate_results(query.vss_merged_candidates, query.ground_truths)
    query.results['vss_merged_metrics'] = vss_metrics['metrics']

    return merged


# ---------------------------------------------------------------------------
# NEW Step 6 — LLM Reranking
# ---------------------------------------------------------------------------

def _parse_reranking_response(response_str: str, n_candidates: int) -> List[int]:
    """
    Parse LLM reranking response (comma-separated 1-based indices) into
    a list of 0-based positions, filtering invalid values.
    """
    indices = []
    seen = set()
    for token in re.split(r'[\s,]+', response_str.strip()):
        token = token.strip().rstrip('.')
        try:
            idx = int(token)
            if 1 <= idx <= n_candidates and idx not in seen:
                indices.append(idx - 1)   # convert to 0-based
                seen.add(idx)
        except ValueError:
            pass
    # Append any missing indices at the end (so we always return a full ranking)
    for i in range(n_candidates):
        if i not in seen:
            indices.append(i)
    return indices


def step6_llm_rerank(query: Query, kb, llm_bridge, v2_config: dict) -> List[int]:
    """
    Expands the candidate pool to rerank_pool_size, asks LLM to rank them,
    and returns top rerank_top_k node IDs.
    The result replaces query.vss_merged_candidates.
    """
    if not v2_config.get('use_llm_reranking', True):
        return query.vss_merged_candidates

    pool_size = v2_config.get('rerank_pool_size', 50)
    top_k     = v2_config.get('rerank_top_k', 20)

    # Use vss_merged_candidates as the pool (already contains best scored nodes)
    pool = query.vss_merged_candidates[:pool_size]
    if not pool:
        return query.vss_merged_candidates

    print(f"[Step6] LLM reranking {len(pool)} candidates ...")

    # Build compact descriptions for each candidate
    candidate_pairs = []
    for node_id in pool:
        try:
            doc_str = kb.get_doc_info(node_id, compact=True)
        except Exception:
            doc_str = f"Node {node_id}"
        candidate_pairs.append((node_id, doc_str))

    prompt = get_llm_reranking_prompt(query=query.query, candidates=candidate_pairs)

    try:
        response_str = get_llm_response(prompt, llm_bridge)
        ranked_positions = _parse_reranking_response(response_str, len(pool))
    except Exception as e:
        print(f"[Step6] LLM reranking failed: {e} — falling back to original order")
        ranked_positions = list(range(len(pool)))

    reranked = [pool[i] for i in ranked_positions[:top_k]]

    # Record in query
    query.reranked_candidates = reranked

    # Update vss_merged_metrics with reranked results
    rerank_eval = evaluate_results(reranked, query.ground_truths)
    if query.results is None:
        query.results = {}
    query.results['reranked_metrics'] = rerank_eval['metrics']

    print(f"[Step6] Reranked top-{top_k}: recall@20={rerank_eval['metrics']['recall@20']:.3f}")
    return reranked


# ---------------------------------------------------------------------------
# Shared evaluation + save helpers
# ---------------------------------------------------------------------------

def evaluate_results(predicted_nodes, ground_truth_nodes):
    ground_truth_set = set(ground_truth_nodes)
    if not predicted_nodes:
        return {
            "answer_list": [], "answer_set": set(), "ground_truth_set": ground_truth_set,
            "retrieved_ground_truths": set(), "missed_ground_truths": ground_truth_set,
            "metrics": {"total_answers": len(ground_truth_set), "retrieved_count": 0,
                        "missed_count": len(ground_truth_set), "recall@50": 0.0,
                        "recall@20": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0},
        }
    hit_at_1 = 1.0 if predicted_nodes[0] in ground_truth_set else 0.0
    hit_at_5 = 1.0 if any(n in ground_truth_set for n in predicted_nodes[:5]) else 0.0
    mrr = 0.0
    for rank, node in enumerate(predicted_nodes, 1):
        if node in ground_truth_set:
            mrr = 1.0 / rank; break
    retrieved   = set(predicted_nodes).intersection(ground_truth_set)
    missed      = ground_truth_set.difference(retrieved)
    recall_50   = len(retrieved) / len(ground_truth_set) if ground_truth_set else 0.0
    top_20_hit  = set(predicted_nodes[:20]).intersection(ground_truth_set)
    recall_20   = len(top_20_hit) / len(ground_truth_set) if ground_truth_set else 0.0
    return {
        "answer_list": predicted_nodes, "answer_set": set(predicted_nodes),
        "ground_truth_set": ground_truth_set, "retrieved_ground_truths": retrieved,
        "missed_ground_truths": missed,
        "metrics": {"total_answers": len(ground_truth_set), "retrieved_count": len(retrieved),
                    "missed_count": len(missed), "recall@50": recall_50,
                    "recall@20": recall_20, "hit_at_1": hit_at_1, "hit_at_5": hit_at_5,
                    "mrr": mrr},
    }


def save_results_threadsafe(query: Query, csv_writer: ThreadSafeCSVWriter):
    if query.results is None:
        query.results = {}
    grounding_cands = getattr(query, 'grounding_candidates', [])
    vss_merged_cands = getattr(query, 'vss_merged_candidates', [])
    reranked_cands = getattr(query, 'reranked_candidates', [])

    if 'metrics' not in query.results and grounding_cands:
        query.results.update(evaluate_results(grounding_cands, query.ground_truths))
    if 'vss_merged_metrics' not in query.results and vss_merged_cands:
        query.results['vss_merged_metrics'] = evaluate_results(vss_merged_cands, query.ground_truths)['metrics']
    if 'reranked_metrics' not in query.results and reranked_cands:
        query.results['reranked_metrics'] = evaluate_results(reranked_cands, query.ground_truths)['metrics']

    metrics         = query.results.get('metrics', {})
    vss_metrics     = query.results.get('vss_merged_metrics', {})
    rerank_metrics  = query.results.get('reranked_metrics', {})

    row_data = {
        'query_id':   query.id,
        'query_text': query.query,
        'total_answers':   metrics.get('total_answers', len(query.ground_truths)),
        'retrieved_count': metrics.get('retrieved_count', 0),
        'missed_count':    metrics.get('missed_count', len(query.ground_truths)),
        'recall@20':       metrics.get('recall@20', 0.0),
        'recall@50':       metrics.get('recall@50', 0.0),
        'hit@1':           metrics.get('hit_at_1', 0.0),
        'hit@5':           metrics.get('hit_at_5', 0.0),
        'mrr':             metrics.get('mrr', 0.0),
        'recall@20_vss_merged': vss_metrics.get('recall@20', 0.0),
        'recall@20_reranked':   rerank_metrics.get('recall@20', 0.0),
        'mrr_reranked':         rerank_metrics.get('mrr', 0.0),
        'hit@1_reranked':       rerank_metrics.get('hit_at_1', 0.0),
    }
    csv_writer.write_row(row_data)


def serialize_value(value):
    if value is None: return ""
    if isinstance(value, (str, int, float, bool)): return value
    try:
        return json.dumps(value, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
    except Exception:
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


def run_fallback_strategy(query: Query, retriever, kb, csv_writer, failure_reason: str):
    print(f"--- [FALLBACK] Query {query.id}: {failure_reason}")
    query.status = "FALLBACK"
    query.failure_reason = failure_reason

    node_types = ['disease']
    if hasattr(query, 'entities') and query.entities and "ANSWER" in query.entities:
        detected = query.entities["ANSWER"].get("type", [])
        if detected:
            node_types = detected

    all_candidates = []
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
    query.reranked_candidates   = final_ids
    query.grounding_candidates  = []
    query.results = {
        'metrics': {'recall@20': 0.0},
        'vss_merged_metrics': evaluate_results(final_ids, query.ground_truths)['metrics'],
        'reranked_metrics':   evaluate_results(final_ids, query.ground_truths)['metrics'],
    }
    save_results_threadsafe(query, csv_writer)
    return query


# ---------------------------------------------------------------------------
# NEW pipeline_worker (v2)
# ---------------------------------------------------------------------------

def pipeline_worker_v2(query: Query, llm_bridge, kb, retriever,
                        dataset_name: str, csv_writer, config, v2_config: dict,
                        use_saved=False, saved_response=None,
                        use_saved_vss=False, vss_candidates=None):
    """
    Full v2 pipeline:
      1.  Entity extraction      (LLM)
      2.  Relation extraction    (LLM)
      2.5 Anchor verification    (LLM, optional)
      3.  Initial VSS candidates
      4.  Priority-queue grounding
      4.5 ANSWER-direct VSS      (auto when grounding < threshold)
      5.  VSS merge              (LLM for query expansion)
      6.  LLM reranking          (LLM, optional)
    """
    print(f"--- [V2] Processing Query {query.id} ---")
    vss_candidates = vss_candidates or {}

    # Steps 1 & 2
    step1_identify_entities(query, llm_bridge, dataset_name, use_saved, saved_response)
    if query.status == "SKIPPED": raise PipelineSkipped("Step 1 Skipped")
    if query.status == "FAILED":  raise PipelineStepFailed("Step 1 Failed")

    step2_identify_relations(query, llm_bridge, dataset_name, use_saved, saved_response)
    if query.status == "SKIPPED": raise PipelineSkipped("Step 2 Skipped")
    if query.status == "FAILED":  raise PipelineStepFailed("Step 2 Failed")

    # Step 3 (needs to happen before 2.5 so we have initial_symbol_candidates)
    step3_get_initial_candidates(query, kb, retriever, config)

    # Step 2.5 — verify ambiguous anchors with LLM
    step2_5_verify_anchors(query, kb, retriever, llm_bridge, v2_config)

    # Step 4
    step4_grounding(query, kb, retriever, config)

    # Step 4.5 — direct VSS fallback when grounding is sparse
    step4_5_answer_direct_vss(query, retriever, kb, config, v2_config)

    # Step 5
    step5_merge_vss_candidates(
        query, retriever, kb, config,
        use_saved=use_saved_vss,
        vss_candidates=vss_candidates,
        llm_bridge=llm_bridge,
        dataset_name=dataset_name,
    )

    # Step 6 — LLM reranking
    step6_llm_rerank(query, kb, llm_bridge, v2_config)

    save_results_threadsafe(query, csv_writer)
    print(f"[V2] Finished Query {query.id}")
    return query


# ---------------------------------------------------------------------------
# run_with_timeout (unchanged)
# ---------------------------------------------------------------------------

def run_with_timeout(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutException(f"Timed out after {timeout}s")


# ---------------------------------------------------------------------------
# process_batch (v2)
# ---------------------------------------------------------------------------

def process_batch(batch_queries, config, dataset_name, model_config, data_config,
                  v2_config, result_queue, failed_queue, progress_queue, worker_index=0):

    process_id = mp.current_process().pid
    exp_name    = config['experiment'].get('exp_name', f"{dataset_name}_results")
    output_base = config['experiment'].get('output_base_dir', './output/')

    log_file_path = f"{output_base}/{exp_name}/process_{process_id}.log"
    sys.stdout = open(log_file_path, "a", encoding="utf-8")

    dump_fieldnames = [
        'id', 'query', 'ground_truths', 'status', 'entities', 'relations',
        'initial_symbol_candidates', 'final_candidates', 'results',
        'grounding_candidates', 'vss_merged_candidates', 'reranked_candidates',
        'failure_reason',
    ]
    dump_file_path = f"{output_base}/{exp_name}/full_dump_process_{process_id}.csv"
    batch_results_buffer = []

    try:
        kb = load_skb(dataset_name, download_processed=True)
        llm_bridge = LlmBridge(
            model_name=model_config['llm_name'],
            configs_path=model_config['llm_config_path'],
            dataset=config['experiment']['dataset'],
            worker_index=worker_index,
        )
        qa_dataset = load_qa(dataset_name)
        retriever = VSSRetriever(
            kb=kb,
            emb_base_path=f"{model_config['embedding_base_path']}/{dataset_name}/",
            emb_model=model_config['embedding_model'],
            qa_dataset=qa_dataset,
            dataset_name=config['experiment']['split'],
            use_vss=True,
            use_gpu=False,
        )

        vss_cands = {}
        if data_config['use_saved_vss_candidates']:
            with open(data_config['vss_candidates_json_path'], 'r', encoding='utf-8') as f:
                vss_cands = json.load(f)

        llm_response_df = None
        if data_config['use_saved_llm_responses']:
            llm_response_df = pd.read_csv(data_config['llm_responses_file'])

        csv_fieldnames = [
            'query_id', 'query_text', 'total_answers', 'retrieved_count', 'missed_count',
            'recall@50', 'recall@20', 'hit@1', 'hit@5', 'mrr',
            'recall@20_vss_merged', 'recall@20_reranked', 'mrr_reranked', 'hit@1_reranked',
        ]
        csv_writer = ThreadSafeCSVWriter(
            csv_path=f"{output_base}/{exp_name}/pipeline_results_process_{process_id}.csv",
            fieldnames=csv_fieldnames,
        )

        for i, query_data in enumerate(batch_queries):
            current_query = Query(id=query_data[1], query=query_data[0], ground_truths=query_data[2])

            saved_llm_response = None
            if data_config['use_saved_llm_responses'] and llm_response_df is not None:
                rows = llm_response_df[llm_response_df['id'] == current_query.id].to_dict(orient='records')
                if rows:
                    saved_llm_response = rows[0]

            success = False
            skipped = False
            last_error = ""

            for attempt in range(MAX_RETRIES):
                try:
                    args = (
                        current_query, llm_bridge, kb, retriever,
                        dataset_name, csv_writer, config, v2_config,
                        data_config['use_saved_llm_responses'], saved_llm_response,
                        data_config['use_saved_vss_candidates'], vss_cands,
                    )
                    result = run_with_timeout(pipeline_worker_v2, args, QUERY_TIMEOUT)
                    batch_results_buffer.append(result)
                    success = True
                    break

                except PipelineSkipped as e:
                    print(f"[SKIPPED] Query {current_query.id}: {e}")
                    current_query.status = "SKIPPED"
                    current_query.failure_reason = str(e)
                    batch_results_buffer.append(current_query)
                    skipped = True
                    break

                except (PipelineStepFailed, TimeoutException) as e:
                    last_error = str(e)
                    print(f"[WARNING] Query {current_query.id} attempt {attempt+1}: {e}")
                except Exception as e:
                    last_error = str(e)
                    print(f"[ERROR]   Query {current_query.id} attempt {attempt+1}: {e}")
                    traceback.print_exc()

            if not success and not skipped:
                print(f"[CRITICAL] Query {current_query.id} failed all retries — fallback.")
                try:
                    fallback_result = run_fallback_strategy(
                        current_query, retriever, kb, csv_writer, last_error
                    )
                    batch_results_buffer.append(fallback_result)
                except Exception as fb_err:
                    print(f"[FATAL] Fallback failed for {current_query.id}: {fb_err}")
                    failed_queue.put({'query_id': current_query.id, 'error': str(fb_err)})

            progress_queue.put(1)

            if len(batch_results_buffer) >= DUMP_INTERVAL:
                save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
                for res in batch_results_buffer:
                    result_queue.put(res)
                batch_results_buffer = []

    except Exception as e:
        sys.stderr.write(f"[CRITICAL] Batch process {process_id} crashed: {e}\n")
    finally:
        if batch_results_buffer:
            save_partial_dump(batch_results_buffer, dump_file_path, dump_fieldnames)
            for res in batch_results_buffer:
                result_queue.put(res)
        sys.stdout.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def create_experiment_dir(exp_name, base_dir):
    try:
        os.makedirs(f"{base_dir}/{exp_name}", exist_ok=True)
    except Exception:
        pass


def print_aggregate_report(pipeline_results_path):
    if not os.path.exists(pipeline_results_path):
        return
    try:
        df = pd.read_csv(pipeline_results_path)
        if df.empty:
            return
        n = len(df)

        def m(col): return df[col].mean() if col in df.columns else 0.0

        avg = {
            'total_queries':                   n,
            # Grounding stage
            'avg_mrr':                         m('mrr'),
            'avg_hit@1':                       m('hit@1'),
            'avg_recall@20_grounding':         m('recall@20'),
            'avg_recall@50':                   m('recall@50'),
            # VSS-merged (before reranking)
            'avg_recall@20_before_reranking':  m('recall@20_vss_merged'),
            # After LLM reranking
            'avg_recall@20_reranked':          m('recall@20_reranked'),
            'avg_mrr_reranked':                m('mrr_reranked'),
            'avg_hit@1_reranked':              m('hit@1_reranked'),
        }

        sep  = "=" * 62
        thin = "-" * 62
        print("\n" + sep, file=sys.__stdout__)
        print(f"  FINAL AGGREGATE REPORT  (N={n})", file=sys.__stdout__)
        print(sep, file=sys.__stdout__)
        print(f"  {'Stage':<36} {'R@20':>8} {'MRR':>8} {'H@1':>8}", file=sys.__stdout__)
        print(thin, file=sys.__stdout__)
        print(f"  {'Grounding (priority queue)':<36} "
              f"{avg['avg_recall@20_grounding']:>8.4f} "
              f"{avg['avg_mrr']:>8.4f} "
              f"{avg['avg_hit@1']:>8.4f}", file=sys.__stdout__)
        print(f"  {'VSS merge  (before reranking)':<36} "
              f"{avg['avg_recall@20_before_reranking']:>8.4f} "
              f"{'—':>8} {'—':>8}", file=sys.__stdout__)
        print(f"  {'LLM reranking':<36} "
              f"{avg['avg_recall@20_reranked']:>8.4f} "
              f"{avg['avg_mrr_reranked']:>8.4f} "
              f"{avg['avg_hit@1_reranked']:>8.4f}", file=sys.__stdout__)
        print(thin, file=sys.__stdout__)
        print(f"  R@50 (grounding): {avg['avg_recall@50']:.4f}", file=sys.__stdout__)
        print(sep + "\n", file=sys.__stdout__)

        agg_path = os.path.join(os.path.dirname(pipeline_results_path), "aggregate_results.csv")
        with open(agg_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(avg.keys()))
            writer.writeheader()
            writer.writerow(avg)
    except Exception as e:
        print(f"[Report Error] {e}", file=sys.__stdout__)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run StarkQA Pipeline V2")
    parser.add_argument("config_path", type=str)
    parser.add_argument("--exp_name",    type=str,   default=None)
    parser.add_argument("--dataset",     type=str,   default=None)
    parser.add_argument("--alpha",       type=int,   default=None)
    parser.add_argument("--score_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int,   default=None)
    parser.add_argument("--split",       type=str,   default=None)
    parser.add_argument("--retry",       action="store_true")
    parser.add_argument("--run_skipped", action="store_true")
    parser.add_argument("--query_ids",   type=str,   default=None)
    # V2-specific flags
    parser.add_argument("--no_verify",   action="store_true", help="Disable Step 2.5 anchor verification")
    parser.add_argument("--no_direct_vss", action="store_true", help="Disable Step 4.5 direct VSS")
    parser.add_argument("--no_rerank",   action="store_true", help="Disable Step 6 LLM reranking")
    parser.add_argument("--rerank_pool", type=int, default=None, help="Reranking pool size (default 50)")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Apply overrides
    if args.exp_name:       config['experiment']['exp_name']                   = args.exp_name
    if args.dataset:        config['experiment']['dataset']                    = args.dataset
    if args.alpha is not None:
        config['retrieval_params']['step5_vss_merge']['alpha']                 = args.alpha
    if args.score_decay is not None:
        config['grounding_params']['score_decay']                              = args.score_decay
    if args.num_workers is not None:
        config['pipeline']['max_workers']                                      = args.num_workers
    if args.split:          config['experiment']['split']                      = args.split

    # Build v2 config — read from file if present, otherwise use defaults
    raw_v2 = config.get('pipeline_v2', {})
    v2_config = {
        'use_anchor_verification': not args.no_verify and raw_v2.get('use_anchor_verification', True),
        'use_answer_direct_vss':   not args.no_direct_vss and raw_v2.get('use_answer_direct_vss', True),
        'answer_direct_vss_k':     raw_v2.get('answer_direct_vss_k', 100),
        'grounding_min_threshold': raw_v2.get('grounding_min_threshold', GROUNDING_MIN_THRESHOLD),
        'use_llm_reranking':       not args.no_rerank and raw_v2.get('use_llm_reranking', True),
        'rerank_pool_size':        args.rerank_pool or raw_v2.get('rerank_pool_size', 50),
        'rerank_top_k':            raw_v2.get('rerank_top_k', 20),
    }

    # Print config
    print("\n" + "=" * 80)
    print("FINALIZED V2 PARAMETERS:")
    for k, v in v2_config.items():
        print(f"  {k}: {v}")
    print("=" * 80 + "\n")

    exp_config   = config['experiment']
    pipe_config  = config['pipeline']
    data_config  = config['data_paths']
    model_config = config['models']

    dataset_name = exp_config['dataset']
    split_name   = exp_config['split']
    exp_name     = exp_config.get('exp_name', f"{dataset_name}_{split_name}_v2")
    output_base  = exp_config.get('output_base_dir', './output/')
    exp_dir      = f"{output_base}/{exp_name}"

    create_experiment_dir(exp_name, output_base)

    # Archive config + scripts + prompts
    try:
        shutil.copy(args.config_path, f"{exp_dir}/config_snapshot.json")
        shutil.copy(__file__,          f"{exp_dir}/script_snapshot.py")
        for dataset_p in [dataset_name, 'prime', 'mag', 'amazon']:
            for kind in ['entity', 'relation']:
                src = f"custom_pipeline/prompt_examples/{kind}_{dataset_p}.txt"
                if os.path.exists(src):
                    shutil.copy(src, f"{exp_dir}/{kind}_{dataset_p}_prompt.txt")
        pg_src = "custom_pipeline/prompt_generator.py"
        if os.path.exists(pg_src):
            shutil.copy(pg_src, f"{exp_dir}/prompt_generator_snapshot.py")
        print(f"[Archived] Config, code, and prompts → {exp_dir}", file=sys.__stdout__)
    except Exception as e:
        print(f"[Warning] Archive failed: {e}", file=sys.__stdout__)

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
            target_ids = set(str(x.strip()) for x in args.query_ids.split(',') if x.strip())
            test_queries = [q for q in all_test_queries if str(q[1]) in target_ids]
            print(f"[query_ids] Running {len(test_queries)} specified queries", file=sys.__stdout__)

        elif args.retry:
            retry_ids = set()
            for dump_path in [main_dump] + glob.glob(f"{exp_dir}/full_dump_process_*.csv"):
                if os.path.exists(dump_path):
                    try:
                        df = pd.read_csv(dump_path)
                        if 'status' in df.columns:
                            retry_ids.update(df[df['status'].isin(['FAILED','FALLBACK'])]['id'].astype(str).tolist())
                            completed_ids.update(df[~df['status'].isin(['FAILED','FALLBACK'])]['id'].astype(str).tolist())
                    except Exception:
                        pass
            # Exclude any query that later succeeded in a newer dump file
            retry_ids -= completed_ids
            test_queries = [q for q in all_test_queries if str(q[1]) in retry_ids]
            print(f"[Retry] Re-running {len(retry_ids)} FAILED/FALLBACK queries", file=sys.__stdout__)

        elif args.run_skipped:
            skipped_ids = set()
            for dump_path in [main_dump] + glob.glob(f"{exp_dir}/full_dump_process_*.csv"):
                if os.path.exists(dump_path):
                    try:
                        df = pd.read_csv(dump_path)
                        if 'status' in df.columns:
                            skipped_ids.update(df[df['status']=='SKIPPED']['id'].astype(str).tolist())
                            completed_ids.update(df[df['status']!='SKIPPED']['id'].astype(str).tolist())
                    except Exception:
                        pass
            # A query may appear as SKIPPED in an old dump but succeeded in a newer
            # per-process dump — only re-run queries that are STILL skipped.
            skipped_ids -= completed_ids
            test_queries = [q for q in all_test_queries if str(q[1]) in skipped_ids]
            print(f"[run_skipped] Re-running {len(skipped_ids)} SKIPPED queries", file=sys.__stdout__)

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
            batch_size  = max(1, len(test_queries) // num_workers +
                               (1 if len(test_queries) % num_workers != 0 else 0))
            query_batches = [test_queries[i:i + batch_size] for i in range(0, len(test_queries), batch_size)]

            manager = Manager()
            result_queue   = manager.Queue()
            failed_queue   = manager.Queue()
            progress_queue = manager.Queue()

            processes = []
            for batch_idx, batch in enumerate(query_batches):
                if not batch:
                    continue
                p = Process(
                    target=process_batch,
                    args=(batch, config, dataset_name, model_config, data_config,
                          v2_config, result_queue, failed_queue, progress_queue, batch_idx),
                )
                p.start()
                processes.append(p)

            completed = 0
            with tqdm(total=len(test_queries), desc="Pipeline V2", file=sys.stderr) as pbar:
                while completed < len(test_queries):
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
            full_dump = pd.concat(dump_dfs, ignore_index=True).drop_duplicates(subset=['id'], keep='last')
            full_dump.to_csv(final_dump_path, index=False)

    except Exception as e:
        sys.stderr.write(f"CRITICAL MAIN ERROR: {e}\n")
        traceback.print_exc()
    finally:
        sys.stdout.close()
        sys.stderr.write("Run complete.\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

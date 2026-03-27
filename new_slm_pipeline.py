import multiprocessing
"""
Pipeline script converted from pipeline_new_feb.ipynb

Usage:
    python pipeline.py --dataset prime --model meta/llama-3.3-70b-instruct --top_k 15 --score_decay 0.9 --exp_name my_experiment

    # Test run (10 queries only):
    python pipeline.py --dataset prime --model meta/llama-3.3-70b-instruct --top_k 15 --score_decay 0.9 --exp_name test_run --test_run
"""

import argparse
import contextlib
import csv
import difflib
import io
import json
import os
import re
import ast
import signal
import time
import traceback
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from dotenv import load_dotenv
from stark_qa import load_skb, load_qa
from tqdm.auto import tqdm

from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.llm_bridge_old import LlmBridge
from custom_pipeline.vss_retreiver import VSSRetriever
from custom_pipeline.candidate_context import CandidateContext
from custom_pipeline.grounders.priority_queue_grounder import PriorityQueueGrounding
from custom_pipeline.prompt_generator import get_query_expansion_prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the STARK QA pipeline.")

    # Required
    parser.add_argument("--dataset",    required=True,  help="Dataset name, e.g. 'prime'")
    parser.add_argument("--model",      required=True,  help="LLM model name, e.g. 'meta/llama-3.3-70b-instruct'")
    parser.add_argument("--exp_name",   required=True,  help="Experiment name used for output directory")

    # Core hyperparams
    parser.add_argument("--top_k",      type=int,   default=15,   help="top_k_neighbors for grounding (default: 15)")
    parser.add_argument("--score_decay",type=float, default=0.9,  help="Score decay for grounding (default: 0.9)")

    # Optional grounding params
    parser.add_argument("--max_candidates_per_symbol", type=int,   default=1000, help="Max candidates per entity symbol (default: 1000)")
    parser.add_argument("--max_answer_candidates",     type=int,   default=20,   help="Max candidates for ANSWER entity (default: 20)")
    parser.add_argument("--support_boost",             type=float, default=0.25, help="Support boost for grounding (default: 0.25)")

    # Flags
    parser.add_argument("--test_run", action="store_true", help="If set, only run on first 10 queries")
    parser.add_argument("--data_split", default="test", help="Dataset split to use (default: 'test')")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query class
# ---------------------------------------------------------------------------

class Query:
    def __init__(self, qa_query):
        self.query = qa_query[0]
        self.id = qa_query[1]
        self.ground_truths = qa_query[2]
        self.status = "IN_PROGRESS"

        self.deconstructed_facts = ""
        self.step2_entities_response = ""
        self.step3_relation_response = ""
        self.step3_relation_validation_response = ""

        self.entities = {}
        self.relations = []
        self.initial_symbol_candidates = {}
        self.results = {}

    def __repr__(self) -> str:
        return (f"\nQuery(id={self.id!r} query={self.query!r} "
                f"ground_truths={self.ground_truths!r})")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    return response[0][0]


# ---------------------------------------------------------------------------
# Step 1 — Deconstruct query into facts
# ---------------------------------------------------------------------------

def step1_deconstruct_query(query, llm_bridge, dataset_name):
    with open(f"SLLM_Prompts/step1_deconstruct_{dataset_name}.txt", "r", encoding="utf-8") as f:
        base_prompt = f.read()

    step1_prompt = base_prompt.replace("<query string here>", query.query)
    response = get_llm_response(step1_prompt, llm_bridge)
    query.deconstructed_facts = response
    print(response)
    return response


# ---------------------------------------------------------------------------
# Step 2 — Extract entities
# ---------------------------------------------------------------------------

def generate_step2_prompt(template_filepath: str, facts_str: str, query_str: str) -> str:
    with open(template_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("<Insert Deconstructed Facts Here>", facts_str.strip())
    prompt = prompt.replace("<query string here>", query_str.strip())
    return prompt


def parse_step2_output(llm_output_string: str) -> dict:
    cleaned_string = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", llm_output_string.strip())
    cleaned_string = cleaned_string.replace("```json", "").replace("```", "").strip()

    try:
        parsed_dict = json.loads(cleaned_string)
        validated_dict = {}
        for key, value in parsed_dict.items():
            if isinstance(value, dict) and "type" in value:
                validated_dict[key] = value
            else:
                print(f"Skipping malformed entity key '{key}': {value}")
        return validated_dict
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}\nRaw String: {llm_output_string}")
        return {}


def step2_extract_entities(query, llm_bridge, dataset_name):
    step2_prompt = generate_step2_prompt(
        f"SLLM_Prompts/step2_entities_{dataset_name}.txt",
        query.deconstructed_facts,
        query.query
    )
    response = get_llm_response(step2_prompt, llm_bridge)
    query.entities = parse_step2_output(response)
    query.step2_entities_response = response
    return response


# ---------------------------------------------------------------------------
# Entity cleaning
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {
    "disease", "gene/protein", "molecular_function", "drug", "pathway",
    "anatomy", "effect/phenotype", "biological_process", "cellular_component", "exposure"
}

TYPE_CORRECTIONS = {
    "gene": "gene/protein",
    "protein": "gene/protein",
    "effect": "effect/phenotype",
    "phenotype": "effect/phenotype"
}


def clean_extracted_entities(entities_dict: dict) -> dict:
    cleaned = {}
    ans = entities_dict.get("ANSWER", {})
    ans_name = ans.get("name", "").strip().lower()
    ans_info = ans.get("information", "").strip().lower()

    for key, entity in entities_dict.items():
        ent_type = entity.get("type", "")

        if ent_type in TYPE_CORRECTIONS:
            ent_type = TYPE_CORRECTIONS[ent_type]
            entity["type"] = ent_type

        ent_name = entity.get("name", "").strip().lower()
        ent_info = entity.get("information", "").strip().lower()

        if key == "ANSWER":
            cleaned[key] = entity
            continue

        if ent_type not in ALLOWED_TYPES:
            print("\nremoving ", entity, "\n due to INVALID TYPE")
            continue

        if ans_name and ent_name == ans_name:
            print("\nremoving ", entity, "\n due to answer NAME duplication")
            continue

        if ans_info and ent_info:
            similarity = difflib.SequenceMatcher(None, ent_info, ans_info).ratio()
            if similarity >= 0.85:
                print("\nremoving ", entity, "\n due to answer duplication")
                continue

        cleaned[key] = entity

    print("Filtered out", len(entities_dict) - len(cleaned), "entities.")
    return cleaned


def clean_entities(query):
    pass
    # query.entities = clean_extracted_entities(query.entities)


# ---------------------------------------------------------------------------
# Step 3 — Relation identification & validation
# ---------------------------------------------------------------------------

def generate_step3_prompt(template_filepath: str, entity_str: str, facts_str: str, query_str: str) -> str:
    with open(template_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace(
        "<insert the python-cleaned dictionary of ANSWER and all other entities here>", entity_str)
    prompt = prompt.replace("<Insert Deconstructed Facts Here>", facts_str.strip())
    prompt = prompt.replace("<query string here>", query_str.strip())
    return prompt


def parse_step3_relations_tuples(llm_output_string: str) -> list:
    cleaned_string = re.sub(r"```(?:json|python)?\s*([\s\S]*?)\s*```", r"\1", llm_output_string.strip())
    for tag in ["```json", "```python", "```"]:
        cleaned_string = cleaned_string.replace(tag, "")
    cleaned_string = cleaned_string.strip()

    if not cleaned_string:
        return []

    try:
        parsed_data = ast.literal_eval(cleaned_string)
        if not isinstance(parsed_data, list):
            print(f"Validation Error: Expected a list, got {type(parsed_data)}")
            return []
        validated_relations = []
        for item in parsed_data:
            if isinstance(item, tuple) and len(item) == 3:
                validated_relations.append((str(item[0]), str(item[1]), str(item[2])))
            else:
                print(f"Skipping malformed relation item: {item}")
        return validated_relations
    except (SyntaxError, ValueError) as e:
        print(f"AST Parsing Error in Relation Parsing: {e}\nRaw String: {llm_output_string}")
        return []


def step3_relation_identification(query, llm_bridge, dataset_name):
    entity_json_string = json.dumps(query.entities, indent=2)
    facts_string = getattr(query, "deconstructed_facts", "")

    step3_prompt = generate_step3_prompt(
        f"SLLM_Prompts/step3_relations_{dataset_name}.txt",
        entity_json_string,
        facts_string,
        query.query
    )

    response = get_llm_response(step3_prompt, llm_bridge)
    query.relations = set(parse_step3_relations_tuples(response))
    query.step3_relation_response = response
    return response


VALID_RELATIONS = {
    "ppi", "carrier", "enzyme", "target", "transporter", "contraindication",
    "indication", "off-label use", "synergistic interaction", "associated with",
    "parent-child", "phenotype absent", "phenotype present", "side effect",
    "interacts with", "linked to", "expression present", "expression absent"
}


def generate_validation_prompt(template_filepath: str, entity_str: str, relations_str: str) -> str:
    with open(template_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace(
        "<insert the python-cleaned dictionary of ANSWER and all other entities here>", entity_str)
    prompt = prompt.replace("<insert the current relations here>", relations_str)
    return prompt


def clean_extracted_relations(query, llm_bridge, dataset_name):
    entity_json_string = json.dumps(query.entities, indent=2)
    relations_list_str = str(list(query.relations))

    validation_prompt = generate_validation_prompt(
        f"SLLM_Prompts/step3_validate_relations_{dataset_name}.txt",
        entity_json_string,
        relations_list_str
    )

    response = get_llm_response(validation_prompt, llm_bridge)
    query.step3_relation_validation_response = response
    validated_relations = parse_step3_relations_tuples(response)

    final_relations = set()
    for (src, tgt, rel) in validated_relations:
        rel_lower = rel.lower().strip()
        if rel_lower in VALID_RELATIONS:
            final_relations.add((src, tgt, rel_lower))
        else:
            print(f"Post-LLM validation dropped invalid relation: {rel}")

    query.relations = final_relations


# ---------------------------------------------------------------------------
# Format mapping
# ---------------------------------------------------------------------------

def map_entities_to_old_format(query: Query):
    new_entities_dict = {}
    for key, value in query.entities.items():
        key_dict = {}
        key_dict["type"] = [value.get("type")]
        key_dict["lexical"] = {"name": value.get("name")}
        key_dict["semantic"] = [value.get("information")]
        key_dict["constant"] = value.get("constant")
        new_entities_dict[key] = key_dict
    query.entities = new_entities_dict


def map_relations_to_old_format(query: Query):
    new_relations_map = {}
    pairs_set = set()

    for relation in query.relations:
        id_string_1 = relation[0] + relation[1]
        id_string_2 = relation[1] + relation[0]
        if id_string_1 in pairs_set or id_string_2 in pairs_set:
            continue
        pairs_set.add(id_string_1)
        pairs_set.add(id_string_2)
        new_relations_map[(relation[0], relation[1])] = [relation[2]]

    query.relations = new_relations_map


# ---------------------------------------------------------------------------
# VSS candidate retrieval
# ---------------------------------------------------------------------------

def get_initial_candidates_for_entity(entity_info, entity_key, kb, retriever, num_candidates=20):
    print(entity_info)
    candidates = []
    entity_types = entity_info.get("type", [])
    name_constraint = entity_info.get("lexical", {}).get("name", None)
    semantic_constraints = entity_info.get("semantic", []).copy()

    for key in entity_info.get("lexical", {}):
        semantic_constraints.append(f" {entity_info['lexical'][key]}")

    nodes_by_name = []
    if name_constraint:
        for etype in entity_types:
            print("Searching for entity type:", etype)
            nodes_by_name = kb.get_node_ids_by_value(node_type=etype, key="name", value=name_constraint)
            print("Found nodes by name:", nodes_by_name)
            candidates.extend([CandidateContext(node_id=x, entity=entity_key, score=1) for x in nodes_by_name])

    vss_candidates_count = num_candidates - len(candidates)
    for etype in entity_types:
        sem = "".join(semantic_constraints)
        print("Searching for semantic constraint:", sem)
        nodes_by_desc_ids, vss_scores = retriever.get_top_k_nodes(
            search_str=sem, k=vss_candidates_count, node_type=etype, cutoff=0.65)
        candidates.extend([
            CandidateContext(node_id=nodes_by_desc_ids[x], entity=entity_key, score=vss_scores[x])
            for x in range(len(nodes_by_desc_ids)) if nodes_by_desc_ids[x] not in nodes_by_name
        ])
        print("Found nodes by description:", len(nodes_by_desc_ids))

    print("Candidates so far:", candidates)
    return candidates


def step3_get_initial_candidates(current_query, kb, retriever):
    initial_candidates = {}
    print(initial_candidates)
    for entity in current_query.entities:
        print("Entity:", entity)
        if entity == "ANSWER":
            print(entity)
            continue
        initial_candidates[entity] = get_initial_candidates_for_entity(
            current_query.entities[entity], entity_key=entity, kb=kb, retriever=retriever)
    current_query.initial_symbol_candidates = initial_candidates
    return initial_candidates


# ---------------------------------------------------------------------------
# Step 2.5 — Entity anchor verification
# ---------------------------------------------------------------------------

def get_entity_anchor_verification_prompt_small(query, entity_key, entity_info, candidates) -> str:
    entity_name = entity_info.get("lexical", {}).get("name", "")
    entity_type = (entity_info.get("type", [""])[0]
                   if isinstance(entity_info.get("type"), list) else entity_info.get("type", ""))
    entity_desc = entity_info.get("semantic", [""])[0] if entity_info.get("semantic") else ""

    candidates_block = ""
    for i, (node_id, score, doc_str) in enumerate(candidates, 1):
        truncated = doc_str[:150].split(".")[0].strip() if doc_str else ""
        candidates_block += f"{i}. {truncated}\n"

    prompt = f"""### TASK
You are matching an entity to its correct entry in a knowledge graph.

You will be given:
- An entity NAME and TYPE
- A short DESCRIPTION of how this entity appears in a query
- A numbered list of CANDIDATES from the knowledge graph

Pick the candidate that refers to the exact same real-world entity as the given name and description.

### RULES
1. Read the NAME carefully. Look for an exact or near-exact name match first.
2. If no name match exists, use the DESCRIPTION and TYPE to find the best fit.
3. Pick 0 if none of the candidates match. Do not guess.
4. Write one short sentence in "observation" identifying the key reason for your choice.

### INPUT
Original Query: {query}
Entity Name: {entity_name}
Entity Type: {entity_type}
Description: {entity_desc}

Candidates:
{candidates_block.strip()}

### OUTPUT FORMAT
Your response must be ONLY a valid JSON object. No explanation, no markdown, no text outside the JSON.
{{
    "observation": "one sentence explaining the key reason for your choice",
    "choice": 1
}}"""

    return prompt


def parse_verification_response_small(response_str: str, num_candidates: int) -> int:
    if not response_str:
        return 0

    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_str.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        choice = int(parsed.get("choice", 0))
        if 0 <= choice <= num_candidates:
            return choice
        return 0
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        json_match = re.search(r"\{[\s\S]*?\}", cleaned)
        if json_match:
            parsed = json.loads(json_match.group())
            choice = int(parsed.get("choice", 0))
            if 0 <= choice <= num_candidates:
                return choice
        return 0
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    if len(cleaned) <= 20:
        digit_match = re.search(r"\b([0-5])\b", cleaned)
        if digit_match:
            choice = int(digit_match.group(1))
            if 0 <= choice <= num_candidates:
                return choice

    return 0


def step2_5_verify_anchors_small(query: Query, kb, retriever, llm_bridge):
    vss_confidence_threshold = 0.80
    top_k_verify = 3
    promotion_score = 0.99

    print("[Step2.5 SMALL] Running entity anchor verification ...")

    for entity_key, entity_info in query.entities.items():
        if entity_key == "ANSWER":
            continue
        if not entity_info.get("constant", False):
            continue

        name_constraint = entity_info.get("lexical", {}).get("name", "")
        if not name_constraint:
            continue

        existing = query.initial_symbol_candidates.get(entity_key, [])

        exact = [c for c in existing if c.score >= 1.0]
        if exact:
            print(f"  [{entity_key}] exact match found ({exact[0].node_id}) — skipping")
            continue

        top_score = max((c.score for c in existing), default=0)
        if top_score >= vss_confidence_threshold:
            print(f"  [{entity_key}] top VSS score {top_score:.3f} >= threshold — skipping")
            continue

        candidates_for_llm = []
        seen = set()
        for c in existing[:15]:
            if c.node_id not in seen:
                doc_str = kb.get_doc_info(c.node_id, compact=True)
                candidates_for_llm.append((c.node_id, c.score, doc_str))
                seen.add(c.node_id)
            if len(candidates_for_llm) >= top_k_verify:
                break

        if not candidates_for_llm:
            print(f"  [{entity_key}] no candidates to verify — skipping")
            continue

        prompt = get_entity_anchor_verification_prompt_small(
            query=query.query,
            entity_key=entity_key,
            entity_info=entity_info,
            candidates=candidates_for_llm,
        )

        try:
            response_str = get_llm_response(prompt, llm_bridge).strip()
        except Exception as e:
            print(f"  [{entity_key}] LLM call failed: {e} — skipping")
            continue

        choice = parse_verification_response_small(response_str, len(candidates_for_llm))
        print(f"  [{entity_key}] raw response: {response_str[:120]!r}")

        if choice == 0:
            print(f"  [{entity_key}] LLM says none match (or unparseable) — keeping VSS order")
            continue

        verified_node_id, verified_score, _ = candidates_for_llm[choice - 1]
        new_cands = [CandidateContext(node_id=verified_node_id, entity=entity_key, score=promotion_score)]
        for c in existing:
            if c.node_id != verified_node_id:
                new_cands.append(c)

        query.initial_symbol_candidates[entity_key] = new_cands
        print(f"  [{entity_key}] LLM verified node {verified_node_id} "
              f"(was rank ~{choice}, score {verified_score:.3f}) "
              f"→ promoted to {promotion_score}")


# ---------------------------------------------------------------------------
# Step 4 — Grounding
# ---------------------------------------------------------------------------

def run_priority_queue_grounding(
    query_obj,
    kb,
    vss_retriever,
    max_candidates_per_symbol: int = 1000,
    max_answer_candidates: int = 20,
    top_k_neighbors: int = 15,
    score_decay: float = 0.9,
    support_boost: float = 0.25,
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


def evaluate_result(predicted_nodes, ground_truth_nodes):
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
                "recall@20": 0.0,
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
    retrieved_ground_truths = answer_set.intersection(ground_truth_set)
    missed_ground_truths = ground_truth_set.difference(retrieved_ground_truths)

    top_20_predictions = predicted_nodes[:20]
    retrieved_20 = set(top_20_predictions).intersection(ground_truth_set)
    recall_20 = len(retrieved_20) / len(ground_truth_set) if ground_truth_set else 0.0
    recall = len(retrieved_ground_truths) / len(ground_truth_set) if ground_truth_set else 0.0

    return {
        "answer_list": predicted_nodes,
        "ground_truth_set": ground_truth_set,
        "retrieved_ground_truths": retrieved_ground_truths,
        "missed_ground_truths": missed_ground_truths,
        "metrics": {
            "total_answers": len(ground_truth_set),
            "retrieved_count": len(retrieved_ground_truths),
            "missed_count": len(missed_ground_truths),
            "recall@50": recall,
            "recall@20": recall_20,
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
        }
    }


def step4_grounding(query: Query, kb, retriever, top_k, score_decay,
                    max_candidates_per_symbol, max_answer_candidates, support_boost):
    final_candidates = run_priority_queue_grounding(
        query_obj=query,
        kb=kb,
        vss_retriever=retriever,
        max_candidates_per_symbol=max_candidates_per_symbol,
        max_answer_candidates=max_answer_candidates,
        top_k_neighbors=top_k,
        support_boost=support_boost,
        score_decay=score_decay,
        verbose=True
    )

    if "ANSWER" in final_candidates:
        ans_candidates = final_candidates["ANSWER"]
        answers = [cc.node_id for cc in sorted(
            ans_candidates,
            key=lambda x: (x.support, x.score),
            reverse=True
        )]
    else:
        answers = []

    results = evaluate_result(answers, query.ground_truths)
    query.results = results
    query.grounding_candidates = answers
    query.final_candidates = final_candidates
    return final_candidates


# ---------------------------------------------------------------------------
# Step 5 — Merge VSS candidates
# ---------------------------------------------------------------------------

def step5_merge_vss_candidates(query: Query, retriever, llm_bridge) -> list:
    search_str = query.query
    all_candidates = []

    possible_node_types = query.entities.get("ANSWER", {}).get("type", [])
    if not possible_node_types:
        possible_node_types = [""]

    for node_type in possible_node_types:
        try:
            top_candidates = retriever.get_top_k_nodes(
                search_str=search_str, k=40, node_type=node_type, cutoff=0.65)
            if top_candidates and len(top_candidates) == 2:
                all_candidates.extend(list(zip(top_candidates[0], top_candidates[1])))
        except Exception as e:
            print(f"VSS step failed for {node_type}: {e}")

    vss_candidates_list = list(map(
        lambda x: x[0], sorted(all_candidates, key=lambda x: x[1], reverse=True)))

    if "vss_metrics" not in query.results:
        query.results["vss_metrics"] = {}

    alphas = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    final_merged_candidates = []

    for alpha in alphas:
        a_val = min(alpha, len(query.grounding_candidates))
        existing_candidate_ids = query.grounding_candidates[:a_val]

        new_vss_candidates = []
        for node in vss_candidates_list:
            if node not in existing_candidate_ids:
                new_vss_candidates.append(node)
            if len(new_vss_candidates) == 20 - a_val:
                break

        merged = existing_candidate_ids + new_vss_candidates
        vss_merged_results = evaluate_result(merged, query.ground_truths)
        query.results["vss_metrics"][f"recall@20_alpha_{alpha}"] = vss_merged_results["metrics"]["recall@20"]

        if alpha == 20:
            final_merged_candidates = merged

    query.vss_merged_candidates = final_merged_candidates
    return final_merged_candidates


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(query: Query, csv_path: str = "pipeline_results.csv", append: bool = True):
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    write_headers = not file_exists or not append
    mode = "a" if append else "w"

    if not hasattr(query, "results") or query.results is None:
        print(f"[WARNING] Query {query.id} has no results. Skipping.")
        return

    metrics = query.results.get("metrics", {})
    vss_metrics = query.results.get("vss_metrics", {})

    row_data = {
        "query_id": query.id,
        "query_text": query.query,
        "total_answers": metrics.get("total_answers", 0),
        "retrieved_count": metrics.get("retrieved_count", 0),
        "missed_count": metrics.get("missed_count", 0),
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "hit@1": metrics.get("hit_at_1", 0.0),
        "hit@5": metrics.get("hit_at_5", 0.0),
        "mrr": metrics.get("mrr", 0.0),
    }
    for a in range(12, 21):
        row_data[f"recall@20_alpha_{a}"] = vss_metrics.get(f"recall@20_alpha_{a}", 0.0)

    fieldnames = [
        "query_id", "query_text", "total_answers", "retrieved_count", "missed_count",
        "recall@50", "recall@20", "hit@1", "hit@5", "mrr"
    ] + [f"recall@20_alpha_{a}" for a in range(12, 21)]

    with open(csv_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_headers:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"[SAVED] Results for query {query.id} saved to {csv_path}")


def save_aggregate_results(queries: list, csv_path: str = "aggregate_results.csv"):
    if not queries:
        print("[WARNING] No queries provided for aggregation.")
        return

    all_metrics = []
    all_vss_metrics = []
    for query in queries:
        if hasattr(query, "results") and query.results is not None:
            all_metrics.append(query.results.get("metrics", {}))
            all_vss_metrics.append(query.results.get("vss_metrics", {}))

    if not all_metrics:
        print("[WARNING] No valid results found in queries.")
        return

    num_queries = len(all_metrics)
    avg_metrics = {
        "total_queries": num_queries,
        "avg_total_answers":   sum(m.get("total_answers", 0)   for m in all_metrics) / num_queries,
        "avg_retrieved_count": sum(m.get("retrieved_count", 0) for m in all_metrics) / num_queries,
        "avg_missed_count":    sum(m.get("missed_count", 0)    for m in all_metrics) / num_queries,
        "avg_recall@20":       sum(m.get("recall@20", 0.0)     for m in all_metrics) / num_queries,
        "avg_recall@50":       sum(m.get("recall@50", 0.0)     for m in all_metrics) / num_queries,
        "avg_hit@1":           sum(m.get("hit_at_1", 0.0)      for m in all_metrics) / num_queries,
        "avg_hit@5":           sum(m.get("hit_at_5", 0.0)      for m in all_metrics) / num_queries,
        "avg_mrr":             sum(m.get("mrr", 0.0)           for m in all_metrics) / num_queries,
    }
    for a in range(12, 21):
        total_recall = sum(vm.get(f"recall@20_alpha_{a}", 0.0) for vm in all_vss_metrics)
        avg_metrics[f"avg_recall@20_alpha_{a}"] = total_recall / num_queries

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_metrics.keys()))
        writer.writeheader()
        writer.writerow(avg_metrics)

    print(f"[SAVED] Aggregate results saved to {csv_path}")
    print("\nAggregate Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def save_data_dump(results, csv_path="aggregate_results.csv"):
    if not results:
        print("No results to save.")
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
        "id", "query", "ground_truths", "status", "deconstructed_facts",
        "step2_entities_response", "step3_relation_response",
        "step3_relation_validation_response", "entities",
        "initial_symbol_candidates", "relations", "results"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Full pipeline per query
# ---------------------------------------------------------------------------

def pipeline(query, llm_bridge, kb, retriever, dataset_name,
             top_k, score_decay, max_candidates_per_symbol,
             max_answer_candidates, support_boost, exp_name):

    # Phase 1: Deconstruct query into facts
    step1_deconstruct_query(query, llm_bridge, dataset_name)
    time.sleep(0.5)

    # Phase 2: Extract entities from facts
    step2_extract_entities(query, llm_bridge, dataset_name)
    time.sleep(0.5)

    # Phase 3: Clean and normalise entities
    # clean_entities(query)

    # Phase 4: Identify relations
    step3_relation_identification(query, llm_bridge, dataset_name)
    time.sleep(0.5)

    # Phase 4.5: Validate relations via LLM
    clean_extracted_relations(query, llm_bridge, dataset_name)

    # Bridge to legacy format
    map_relations_to_old_format(query)
    map_entities_to_old_format(query)

    # Phase 5: Retrieve candidates & grounding
    step3_get_initial_candidates(query, kb, retriever)
    time.sleep(0.5)
    step2_5_verify_anchors_small(query, kb, retriever, llm_bridge)

    step4_grounding(query, kb, retriever, top_k, score_decay,
                    max_candidates_per_symbol, max_answer_candidates, support_boost)

    # Phase 6: Merge VSS candidates
    step5_merge_vss_candidates(query, retriever, llm_bridge=llm_bridge)

    if not getattr(query, "results", {}).get("answer_list"):
        print(f"\\n[!] Empty answer list for {query.id}. Triggering VSS Fallback.")
        apply_vss_fallback(query, retriever, exp_name)
        return

    # Save per-query results
    save_results(query, csv_path=f"./output/{exp_name}/pipeline_results.csv")


# ---------------------------------------------------------------------------
# VSS fallback
# ---------------------------------------------------------------------------

def apply_vss_fallback(query, retriever, exp_name):
    try:
        possible_node_types = (
            query.entities.get("ANSWER", {}).get("type", [])
            if hasattr(query, "entities") and query.entities else []
        )
        if not possible_node_types:
            possible_node_types = [""]

        all_candidates = []
        for node_type in possible_node_types:
            try:
                top_candidates = retriever.get_top_k_nodes(
                    search_str=query.query, k=20, node_type=node_type, cutoff=0.0)
                if top_candidates and len(top_candidates) == 2:
                    all_candidates.extend(list(zip(top_candidates[0], top_candidates[1])))
            except Exception as e:
                print(f"Fallback VSS step failed for {node_type}: {e}")

        fallback_nodes = list(dict.fromkeys(
            map(lambda x: x[0], sorted(all_candidates, key=lambda x: x[1], reverse=True))
        ))[:20]

        query.results = evaluate_result(fallback_nodes, query.ground_truths)
        query.vss_merged_candidates = fallback_nodes
        query.status = "VSS_FALLBACK"

        if "vss_metrics" not in query.results:
            query.results["vss_metrics"] = {}
        for alpha in range(12, 21):
            query.results["vss_metrics"][f"recall@20_alpha_{alpha}"] = \
                query.results["metrics"].get("recall@20", 0.0)

        save_results(query, csv_path=f"./output/{exp_name}/pipeline_results.csv")

    except Exception as e:
        print(f"Fallback also failed: {e}")
        query.status = "FAILED"
        if not hasattr(query, "results") or not query.results:
            query.results = evaluate_result([], query.ground_truths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_previous_results(csv_path):
    import pandas as pd
    import ast
    import json
    prev_results = {}
    if not os.path.exists(csv_path):
        return prev_results
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    for _, row in df.iterrows():
        qa_query = (row['query'], row['id'], 
                    ast.literal_eval(row['ground_truths']) if isinstance(row['ground_truths'], str) else row['ground_truths'])
        q = Query(qa_query)
        q.status = row['status']
        q.deconstructed_facts = row['deconstructed_facts']
        q.step2_entities_response = row['step2_entities_response']
        q.step3_relation_response = row['step3_relation_response']
        q.step3_relation_validation_response = row['step3_relation_validation_response']
        try:
            q.entities = json.loads(row['entities']) if row['entities'] else {}
        except: pass
        try:
            q.relations = ast.literal_eval(row['relations']) if row['relations'] else []
        except: pass
        res = row['results']
        if isinstance(res, str) and res:
            try:
                q.results = ast.literal_eval(res)
            except:
                q.results = {}
        else:
            q.results = res if isinstance(res, dict) else {}
            
        prev_results[q.id] = q
    return prev_results

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Query execution timed out after 120 seconds")


def main():
    load_dotenv(override=True)
    args = parse_args()

    dataset_name = args.dataset
    model_name   = args.model
    exp_name     = args.exp_name
    top_k        = args.top_k
    score_decay  = args.score_decay

    print(f"Dataset:     {dataset_name}")
    print(f"Model:       {model_name}")
    print(f"Experiment:  {exp_name}")
    print(f"top_k:       {top_k}")
    print(f"score_decay: {score_decay}")
    print(f"Test run:    {args.test_run}")

    # --- Single LLM bridge used everywhere ---
    llm_bridge = LlmBridge(model_name=model_name, configs_path="configs.json", verbose=False)

    # --- Load dataset ---
    qa_dataset = load_qa(dataset_name)
    qa = qa_dataset.split_indices[args.data_split].reshape(-1).tolist()
    qa = qa[:int(len(qa) * 0.1)]
    test_queries = [qa_dataset[i] for i in qa]

    if args.test_run:
        test_queries = test_queries[:10]
        print(f"\n[TEST RUN] Running on {len(test_queries)} queries only.\n")

    kb = load_skb(dataset_name, download_processed=True)

    # --- VSS retriever ---
    retriever = VSSRetriever(
        kb=kb,
        emb_base_path=f"../BtechProject/stark/emb/{dataset_name}",
        emb_model="text-embedding-ada-002",
        qa_dataset=qa_dataset,
        dataset_name="test",
        use_vss=True
    )

    # --- Output directory ---
    output_dir = f"./output/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Resume logic ---
    existing_dump = os.path.join(output_dir, "full_data_dump.csv")
    prev_queries = load_previous_results(existing_dump)
    # Clear pipeline_results.csv since we will append newly or resave
    if os.path.exists(f"{output_dir}/pipeline_results.csv"):
        # We don't necessarily delete it, we can just let it overwrite or append.
        # But to be clean we could write over it at the end anyway.
        pass

    # --- Run pipeline ---
    failed_queries = []
    results = []
    fallback_count = 0

    for query_raw in tqdm(test_queries, desc="Processing queries", unit="query"):
        q_id = query_raw[1]
        
        # Check if we should resume or fallback
        if q_id in prev_queries:
            old_q = prev_queries[q_id]
            ans_list = old_q.results.get('answer_list', []) if isinstance(old_q.results, dict) else []
            if len(ans_list) > 0:
                results.append(old_q)
                if old_q.status == "VSS_FALLBACK":
                    fallback_count += 1
                continue
            else:
                # Answer list is empty, quickly apply fallback!
                tqdm.write(f"✓ Found old empty query {q_id}, applying quick VSS Fallback.")
                apply_vss_fallback(old_q, retriever, exp_name)
                results.append(old_q)
                fallback_count += 1
                continue
                
        current_query = Query(query_raw)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pipeline(
                    current_query, llm_bridge, kb, retriever, dataset_name,
                    top_k, score_decay,
                    args.max_candidates_per_symbol,
                    args.max_answer_candidates,
                    args.support_boost,
                    exp_name
                )
                current_query.status = "SUCCESS"

            signal.alarm(0)
            results.append(current_query)

        except TimeoutException as e:
            signal.alarm(0)
            tqdm.write(f"✗ Query {query_raw[1]} timed out. Using VSS Fallback: {str(e)}")
            apply_vss_fallback(current_query, retriever, exp_name)
            results.append(current_query)
            fallback_count += 1
            failed_queries.append({
                "query_id": query_raw[1],
                "query_text": query_raw[0][:100],
                "error": str(e),
                "error_type": "TimeoutException (VSS Fallback Applied)",
                "traceback": ""
            })

        except Exception as e:
            signal.alarm(0)
            tb = traceback.format_exc()
            tqdm.write(f"✗ Query {query_raw[1]} failed. Using VSS Fallback: {type(e).__name__}: {str(e)}")
            apply_vss_fallback(current_query, retriever, exp_name)
            results.append(current_query)
            fallback_count += 1
            failed_queries.append({
                "query_id": query_raw[1],
                "query_text": query_raw[0][:100],
                "error": str(e),
                "error_type": f"{type(e).__name__} (VSS Fallback Applied)",
                "traceback": tb
            })

    # --- Aggregate & dump ---
    save_aggregate_results(results, csv_path=f"{output_dir}/aggregate_results.csv")
    save_data_dump(results, csv_path=f"{output_dir}/full_data_dump.csv")

    if failed_queries:
        with open(f"{output_dir}/failed_queries.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["query_id", "query_text", "error_type", "error", "traceback"])
            writer.writeheader()
            writer.writerows(failed_queries)
        print(f"\n⚠ {len(failed_queries)} queries used VSS Fallback. "
              f"See {output_dir}/failed_queries.csv for details.")
        error_counts = Counter(fq["error_type"] for fq in failed_queries)
        print("\nError breakdown:")
        for error_type, count in error_counts.items():
            print(f"  {error_type}: {count}")
    else:
        print(f"\n✓ All {len(results)} queries processed successfully without fallback!")

    print(f"\n📊 Successfully processed: {len(results)} queries")
    print(f"🔄 Total queries using VSS Fallback: {fallback_count}")
    print("\nDONE !!!")


if __name__ == "__main__":
    main()
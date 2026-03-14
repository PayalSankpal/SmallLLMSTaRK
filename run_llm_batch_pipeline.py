"""
run_llm_batch_pipeline.py — Two-stage OpenAI Batch API pipeline for Entity and Relation extraction.

Stage 1: Generate entity extraction prompts -> send to OpenAI Batch -> save stage 1 results.
Stage 2: Use entities from Stage 1 -> generate relation extraction prompts -> send to OpenAI Batch -> save final CSV.

Outputs in the same format as llm_response_dataset_prime.csv:
id,query,entities,relations

Usage:
    # 1. Run Stage 1 (Entity Extraction)
    python run_llm_batch_pipeline.py --stage 1 --dataset prime --split test --model gpt-4.1-mini-2025-04-14

    # 2. Run Stage 2 (Relation Extraction)
    python run_llm_batch_pipeline.py --stage 2 --dataset prime --split test --model gpt-4.1-mini-2025-04-14

    # Optional: Resume polling if interrupted
    python run_llm_batch_pipeline.py --stage 1 --batch_id batch_XYZ...
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from stark_qa import load_qa

# Custom imports
from custom_pipeline.prompt_generator import (
    get_entity_extraction_prompt, 
    get_relation_extraction_prompt
)

load_dotenv()


# ── Configuration ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run LLM Pipeline via OpenAI Batch API")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2],
                   help="1 for Entity Extraction, 2 for Relation Extraction")
    p.add_argument("--dataset", default="prime", help="Dataset name (e.g. prime)")
    p.add_argument("--split", default="test", help="Data split to run on (e.g. test, val)")
    p.add_argument("--model", default="gpt-4.1-mini-2025-04-14", help="OpenAI Model ID")
    p.add_argument("--poll_interval", type=int, default=15, help="Seconds between API polling")
    p.add_argument("--batch_id", default=None, help="Resume polling an existing Batch ID")
    p.add_argument("--limit", type=int, default=None, help="Limit number of queries for testing")
    p.add_argument("--fraction", type=float, default=0.1, help="Fraction of the split to use (default 0.1 like parallel_final)")
    return p.parse_args()


def get_paths(model: str, dataset: str, split: str):
    # Ensure safe filename representation
    model_safe = model.replace(".", "-").replace(":", "-")
    base = Path(f"llm_responses_batch/{dataset}/{split}/{model_safe}")
    base.mkdir(parents=True, exist_ok=True)
    
    return {
        "s1_jsonl":    base / "stage1_requests.jsonl",
        "s1_res":      base / "stage1_results.jsonl",
        "s1_batch":    base / "stage1_batch_id.txt",
        "s1_csv":      base / f"stage1_entities_{model_safe}.csv",
        
        "s2_jsonl":    base / "stage2_requests.jsonl",
        "s2_res":      base / "stage2_results.jsonl",
        "s2_batch":    base / "stage2_batch_id.txt",
        "final_csv":   base / f"llm_response_dataset_{dataset}_{model_safe}.csv"
    }


# ── Core Data Loading ─────────────────────────────────────────────────────────

def load_target_queries(dataset: str, split: str, fraction: float, limit: int = None):
    print(f"\nLoading queries for {dataset} (split: {split}) ...")
    qa_dataset = load_qa(dataset)
    indices = qa_dataset.split_indices[split].reshape(-1).tolist()
    
    # Matching the parallel_final logic: qa = qa[:int(len(qa) * 0.1)]
    if fraction < 1.0:
        indices = indices[:int(len(indices) * fraction)]
        
    queries = []
    for idx in indices:
        item = qa_dataset[idx]
        query_text = item[0]
        query_id = item[1]
        queries.append({"id": int(query_id), "query": str(query_text)})
        
    if limit:
        queries = queries[:limit]
        
    print(f"Loaded {len(queries)} queries to process.")
    return queries


# ── Batch API Helpers ─────────────────────────────────────────────────────────

def submit_batch(client: OpenAI, jsonl_path: Path, batch_id_txt: Path) -> str:
    print(f"\nUploading {jsonl_path.name} ...")
    with jsonl_path.open("rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    
    print(f"Submitting batch job ...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id_txt.write_text(batch.id)
    print(f"Batch created: {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, interval: int) -> str:
    print(f"\nPolling batch {batch_id} every {interval}s ...")
    terminal_states = {"completed", "failed", "expired", "cancelled"}
    
    while True:
        b = client.batches.retrieve(batch_id)
        c = b.request_counts
        
        # Avoid crash if request_counts is not yet populated
        if c is not None:
            completed, total, failed = c.completed, c.total, c.failed
        else:
            completed, total, failed = 0, 0, 0
            
        print(f"[{time.strftime('%H:%M:%S')}] status={b.status:<12} "
              f"completed={completed}/{total} failed={failed}", flush=True)
              
        if b.status in terminal_states:
            break
        time.sleep(interval)

    if b.status != "completed":
        sys.exit(f"\nBatch ended prematurely with status '{b.status}'. Check OpenAI dashboard.")
        
    print(f"Batch completed! Output file: {b.output_file_id}")
    return b.output_file_id


def download_results(client: OpenAI, file_id: str, out_path: Path):
    print(f"\nDownloading batch results to {out_path} ...")
    data = client.files.content(file_id)
    out_path.write_bytes(data.content)
    print(f"Downloaded OK.")


# ── STAGE 1: Entity Extraction ────────────────────────────────────────────────

def run_stage1(args, client, paths):
    if args.batch_id:
        batch_id = args.batch_id
    else:
        queries = load_target_queries(args.dataset, args.split, args.fraction, args.limit)
        
        # 1. Build JSONL
        print(f"\nBuilding Stage 1 Request JSONL...")
        with paths["s1_jsonl"].open("w", encoding="utf-8") as f:
            for q in queries:
                prompt = get_entity_extraction_prompt(q["query"], args.dataset)
                req = {
                    "custom_id": f"qid-{q['id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        # Response format guarantees strictly JSON output if the model supports it
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0
                    }
                }
                f.write(json.dumps(req) + "\n")
        
        # 2. Submit
        batch_id = submit_batch(client, paths["s1_jsonl"], paths["s1_batch"])
        
    # 3. Poll
    out_file_id = poll_batch(client, batch_id, args.poll_interval)
    
    # 4. Download
    download_results(client, out_file_id, paths["s1_res"])
    
    # 5. Parse and save intermediate CSV
    print("\nMerging queries and Stage 1 responses...")
    queries = load_target_queries(args.dataset, args.split, args.fraction, args.limit)
    q_dict = {q["id"]: q["query"] for q in queries}
    
    results = []
    with paths["s1_res"].open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["custom_id"].removeprefix("qid-"))
            err = data.get("error")
            resp = data.get("response", {})
            
            if err or resp.get("status_code") != 200:
                entities = ""
                print(f"WARNING: qid {qid} failed in batch.")
            else:
                entities = resp["body"]["choices"][0]["message"]["content"]
                
            results.append({
                "id": qid,
                "query": q_dict.get(qid, ""),
                "entities": entities
            })
            
    # Write intermediate CSV
    with paths["s1_csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "query", "entities"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nStage 1 completed. Output written to:\n{paths['s1_csv']}\n")


# ── STAGE 2: Relation Extraction ──────────────────────────────────────────────

def run_stage2(args, client, paths):
    if not paths["s1_csv"].exists():
        sys.exit(f"ERROR: Stage 1 CSV not found: {paths['s1_csv']}\nPlease run --stage 1 first.")
        
    if args.batch_id:
        batch_id = args.batch_id
    else:
        # Load Entities from Stage 1
        print(f"Loading outputs from Stage 1...")
        s1_data = []
        with paths["s1_csv"].open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s1_data.append(row)
                
        # 1. Build JSONL
        print(f"\nBuilding Stage 2 Request JSONL...")
        with paths["s2_jsonl"].open("w", encoding="utf-8") as f:
            for item in s1_data:
                # If entity extraction failed or was empty, skip
                if not item["entities"] or item["entities"].strip() == "":
                    continue
                    
                prompt = get_relation_extraction_prompt(args.dataset, item["query"], item["entities"])
                req = {
                    "custom_id": f"qid-{item['id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        # Can optionally enforce JSON on the relations format too if prompt supports it
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0
                    }
                }
                f.write(json.dumps(req) + "\n")
                
        # 2. Submit
        batch_id = submit_batch(client, paths["s2_jsonl"], paths["s2_batch"])
        
    # 3. Poll
    out_file_id = poll_batch(client, batch_id, args.poll_interval)
    
    # 4. Download
    download_results(client, out_file_id, paths["s2_res"])
    
    # 5. Parse and merge to final CSV
    print("\nDrafting final CSV...")
    s1_dict = {}
    with paths["s1_csv"].open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s1_dict[int(row["id"])] = row

    with paths["s2_res"].open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qid = int(data["custom_id"].removeprefix("qid-"))
            err = data.get("error")
            resp = data.get("response", {})
            
            if err or resp.get("status_code") != 200:
                relations = "{}"
            else:
                relations = resp["body"]["choices"][0]["message"]["content"]
                
            if qid in s1_dict:
                s1_dict[qid]["relations"] = relations
                
    # Write final CSV
    final_fields = ["id", "query", "entities", "relations"]
    with paths["final_csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        # Some rows might not have 'relations' if skipped above, fill with empty JSON
        for row in s1_dict.values():
            if "relations" not in row:
                row["relations"] = "{}"
            writer.writerow({k: row[k] for k in final_fields})
            
    print(f"\n✅ Pipeline Complete. Final Output written to:\n{paths['final_csv']}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not found in environment or .env file")
        
    client = OpenAI(api_key=api_key)
    paths = get_paths(args.model, args.dataset, args.split)
    
    if args.stage == 1:
        run_stage1(args, client, paths)
    elif args.stage == 2:
        run_stage2(args, client, paths)


if __name__ == "__main__":
    main()

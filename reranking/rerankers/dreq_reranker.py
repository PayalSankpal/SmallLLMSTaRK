"""
dreq_reranker.py
================
Zero-shot LLM adaptation of DREQ (Document Re-Ranking Using Entity-based
Query Understanding, Chatterjee et al. ECIR 2024).

Core DREQ idea adapted for our KB-graph setting:
  1) Entity extraction: LLM identifies key entities/concepts in the query
     with importance weights (high / medium / low → 1.0 / 0.6 / 0.3)
  2) Entity-centric scoring: for each candidate document, the LLM judges
     relevance w.r.t. EACH query entity independently → weighted sum
     maps to DREQ's V^Q_ed (query-specific entity-centric doc embedding)
  3) Text-centric scoring: pointwise LLM relevance score for the document
     maps to DREQ's V^Q_td (text-centric doc embedding)
  4) Hybrid score: α·entity_score + (1-α)·text_score

All 20 candidates are batched into a SINGLE LLM call per query,
keeping total LLM calls to 2 per query (entity extraction + batch scoring).

Usage:
    python dreq_reranker.py [params_json] [options]

Options:
    --dump          Path to full_data_dump.csv
    --dataset       Dataset name [prime|amazon|mag]
    --N             Candidate pool size           (default: 20)
    --alpha         Entity/text mixing weight     (default: 0.6)
    --max_queries   Limit queries
    --worst_n       N worst-MRR queries (overrides max_queries)
    --fixable_only  Only queries where GT is in top-N
    --workers       Parallel workers              (default: 16)
    --out           Output CSV path
    --debug         Verbose LLM output for first query
"""

import argparse
import ast
import json
import math
import os
import queue
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stark_qa import load_skb
from custom_pipeline.llm_bridge import LlmBridge

# ─── shared print lock ────────────────────────────────────────────────────────
_print_lock = Lock()

def cprint(*a, **kw):
    with _print_lock:
        print(*a, **kw)

# ─── helpers ──────────────────────────────────────────────────────────────────

def safe_parse_list(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val).strip())
    except Exception:
        try:
            return json.loads(str(val).strip())
        except Exception:
            return []


def compute_metrics(ranked_list, ground_truths, k=20):
    gt_set = set(ground_truths)
    if not gt_set:
        return {}
    top_k = ranked_list[:k]
    recall = len(set(top_k) & gt_set) / len(gt_set)
    hit1   = 1.0 if (top_k and top_k[0] in gt_set) else 0.0
    hit5   = 1.0 if any(n in gt_set for n in top_k[:5]) else 0.0
    mrr    = 0.0
    for r, n in enumerate(top_k, 1):
        if n in gt_set:
            mrr = 1.0 / r
            break
    dcg  = sum(1.0 / math.log2(r + 1) for r, n in enumerate(top_k, 1) if n in gt_set)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gt_set), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"recall@20": recall, "hit@1": hit1, "hit@5": hit5, "mrr": mrr, "ndcg@20": ndcg}


# ─── DREQ prompt templates ────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """\
You are an expert biomedical information retrieval assistant.
Identify the key entities and concepts in the following query.
For each entity, provide:
  - "entity": the entity name (concise, canonical)
  - "type": one of [disease, drug, gene, protein, pathway, anatomy, phenotype, other]
  - "weight": 1.0 (essential to the query), 0.6 (important but secondary), or 0.3 (background context)

Query: {query}

Respond with ONLY a JSON array. Example:
[{{"entity": "Warfarin", "type": "drug", "weight": 1.0}}, ...]
If no clear entities exist, return [].
JSON array:"""


HYBRID_SCORING_PROMPT = """\
You are a relevance scoring assistant for biomedical search.

Query: {query}

Key query entities (name → importance weight 0–1):
{entity_str}

For each candidate document below, provide TWO scores (both 0–10):
  - entity_score: how well the document directly mentions or relates to the query entities above
  - text_score:   overall semantic relevance of the document to the query regardless of specific entities

Candidates:
{doc_str}

Respond with ONLY a JSON array of objects, one per candidate, in the same order:
[{{"id": 1, "entity_score": X, "text_score": Y}}, ...]
JSON array:"""


# ─── DREQ core ────────────────────────────────────────────────────────────────

class DREQReranker:
    def __init__(self, alpha: float = 0.6, debug: bool = False):
        self.alpha = alpha   # weight for entity-centric score
        self.debug = debug

    def _extract_query_entities(self, query: str, llm_bridge) -> list[dict]:
        """Step 1: Extract weighted entity list from query. Returns list of {entity, type, weight}."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        try:
            answers, _ = llm_bridge.ask_llm_batch([prompt])
            raw = answers[0].strip()
            # strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            # find JSON array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                entities = json.loads(m.group())
                # normalise weights
                for e in entities:
                    e["weight"] = float(e.get("weight", 0.6))
                if self.debug:
                    cprint(f"  [DREQ] Extracted {len(entities)} query entities: "
                           f"{[(e['entity'], e['weight']) for e in entities[:5]]}")
                return entities
        except Exception as ex:
            if self.debug:
                cprint(f"  [DREQ] Entity extraction failed: {ex}")
        return []

    def _score_candidates(self, query: str, entities: list[dict],
                          candidates: list[int], doc_cache: dict,
                          llm_bridge) -> dict[int, tuple[float, float]]:
        """
        Step 2: Batch-score all candidates in one LLM call.
        Returns {node_id: (entity_score_0-10, text_score_0-10)}.
        """
        if not entities:
            # fall back to text-only scoring
            entity_str = "(no specific entities identified)"
        else:
            entity_str = "\n".join(
                f"  - {e['entity']} ({e['type']}, weight={e['weight']:.1f})"
                for e in entities
            )

        doc_lines = []
        for disp_i, nid in enumerate(candidates, 1):
            doc_text = doc_cache.get(nid, f"Node {nid}")[:350].strip()
            doc_lines.append(f"[{disp_i}] {doc_text}")
        doc_str = "\n".join(doc_lines)

        prompt = HYBRID_SCORING_PROMPT.format(
            query=query,
            entity_str=entity_str,
            doc_str=doc_str,
        )

        try:
            answers, _ = llm_bridge.ask_llm_batch([prompt])
            raw = answers[0].strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                scores_list = json.loads(m.group())
                results = {}
                for item in scores_list:
                    disp_id = int(item.get("id", 0))
                    if 1 <= disp_id <= len(candidates):
                        nid = candidates[disp_id - 1]
                        es = float(item.get("entity_score", 5.0))
                        ts = float(item.get("text_score",   5.0))
                        results[nid] = (es, ts)
                if self.debug:
                    cprint(f"  [DREQ] Scored {len(results)}/{len(candidates)} candidates")
                return results
        except Exception as ex:
            if self.debug:
                cprint(f"  [DREQ] Batch scoring failed: {ex}")
        return {}

    def rerank(self, query: str, candidates: list[int], kb, llm_bridge,
               query_id=None) -> tuple[list[int], dict]:
        """
        Full DREQ-inspired pipeline for one query.
        Returns (reranked_list, stats).
        """
        N = len(candidates)
        if N == 0:
            return [], {}

        # pre-fetch doc descriptions
        doc_cache = {}
        for nid in candidates:
            try:
                doc_cache[nid] = kb.get_doc_info(nid, compact=True)
            except Exception:
                doc_cache[nid] = f"Node {nid}"

        # Step 1: entity extraction
        entities = self._extract_query_entities(query, llm_bridge)
        llm_calls = 1

        # Step 2: hybrid scoring
        scores = self._score_candidates(query, entities, candidates, doc_cache, llm_bridge)
        llm_calls += 1

        # Step 3: hybrid score = α * entity_score + (1-α) * text_score
        final_scores = {}
        for nid in candidates:
            if nid in scores:
                es, ts = scores[nid]
                final_scores[nid] = self.alpha * es + (1 - self.alpha) * ts
            else:
                # fallback: mid-range
                final_scores[nid] = 5.0

        reranked = sorted(candidates, key=lambda n: final_scores[n], reverse=True)

        stats = {
            "llm_calls":  llm_calls,
            "n_entities": len(entities),
            "entity_names": [e["entity"] for e in entities[:6]],
            "final_scores": final_scores,
        }
        return reranked, stats


# ─── per-query worker ─────────────────────────────────────────────────────────

def process_query(row, kb, llm_bridge, dreq, N, debug=False):
    qid   = row["id"]
    query = row["query"]
    gt    = safe_parse_list(row["ground_truths"])
    cands = safe_parse_list(row.get("vss_merged_candidates", ""))[:N]
    if not cands:
        cands = safe_parse_list(row.get("grounding_candidates", ""))[:N]
    if not cands or not gt:
        return None

    cur_m    = compute_metrics(cands, gt)
    reranked, stats = dreq.rerank(query, cands, kb, llm_bridge, query_id=qid)
    new_m    = compute_metrics(reranked, gt)

    return {
        "qid":           qid,
        "query":         query[:120],
        "llm_calls":     stats.get("llm_calls", 0),
        "n_entities":    stats.get("n_entities", 0),
        "entity_names":  str(stats.get("entity_names", [])),
        "cur_recall@20": cur_m.get("recall@20", 0),
        "cur_hit@1":     cur_m.get("hit@1", 0),
        "cur_hit@5":     cur_m.get("hit@5", 0),
        "cur_mrr":       cur_m.get("mrr", 0),
        "cur_ndcg@20":   cur_m.get("ndcg@20", 0),
        "ts_recall@20":  new_m.get("recall@20", 0),
        "ts_hit@1":      new_m.get("hit@1", 0),
        "ts_hit@5":      new_m.get("hit@5", 0),
        "ts_mrr":        new_m.get("mrr", 0),
        "ts_ndcg@20":    new_m.get("ndcg@20", 0),
        "reranked_list": str(reranked),
    }


# ─── summary ──────────────────────────────────────────────────────────────────

def print_summary(df, alpha, N):
    n = len(df)
    metrics = ["recall@20", "hit@1", "hit@5", "mrr", "ndcg@20"]
    print(f"\n{'═'*72}")
    print(f"  DREQ-inspired Reranker  (α={alpha}, N={N})")
    print(f"  Queries evaluated: {n}")
    print(f"{'═'*72}")
    print(f"  {'Metric':<14} {'Before':>10} {'DREQ':>12} {'Gain':>10} {'Gain%':>8}")
    print(f"  {'─'*14} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    for m in metrics:
        cur = df[f"cur_{m}"].mean()
        new = df[f"ts_{m}"].mean()
        g   = new - cur
        pct = g / cur * 100 if cur > 0 else float("inf")
        print(f"  {m:<14} {cur:>10.4f} {new:>12.4f} {g:>+10.4f} {pct:>7.1f}%")
    print(f"{'─'*72}")
    imp  = (df["ts_mrr"] > df["cur_mrr"]).sum()
    same = (df["ts_mrr"] == df["cur_mrr"]).sum()
    hurt = (df["ts_mrr"] < df["cur_mrr"]).sum()
    calls = df["llm_calls"].sum()
    print(f"\n  MRR improved : {imp}/{n}   same: {same}/{n}   hurt: {hurt}/{n}")
    print(f"  Avg entities extracted per query: {df['n_entities'].mean():.1f}")
    print(f"  Total LLM calls: {int(calls)} ({calls/n:.1f}/query avg)")
    print(f"{'═'*72}\n")

    r20_drift = (df["ts_recall@20"] - df["cur_recall@20"]).abs().max()
    if r20_drift < 1e-9:
        print("  ✓ recall@20 unchanged (reranking only reorders)\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DREQ-inspired LLM reranker")
    p.add_argument("params",        nargs="?", default="params_new.json")
    p.add_argument("--dump",        default=None)
    p.add_argument("--dataset",     default=None)
    p.add_argument("--N",           type=int,   default=20)
    p.add_argument("--alpha",       type=float, default=0.6,  help="Entity score weight (0=text-only, 1=entity-only)")
    p.add_argument("--max_queries", type=int,   default=None)
    p.add_argument("--worst_n",     type=int,   default=None, help="N worst-MRR queries")
    p.add_argument("--fixable_only",action="store_true",       help="Only queries where GT is in top-N")
    p.add_argument("--workers",     type=int,   default=16)
    p.add_argument("--out",         default=None)
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.params) as f:
        config = json.load(f)

    dataset_name = args.dataset or config["experiment"]["dataset"]
    exp_name     = config["experiment"]["exp_name"]
    output_base  = config["experiment"].get("output_base_dir", "./world/")
    model_cfg    = config["models"]

    dump_path = args.dump or f"{output_base}{exp_name}/full_data_dump.csv"
    out_path  = args.out  or f"{output_base}{exp_name}/dreq_results.csv"

    print(f"\n{'─'*60}")
    print(f"  DREQ-inspired Reranker")
    print(f"  Dataset : {dataset_name}   alpha={args.alpha}   N={args.N}")
    print(f"  Dump    : {dump_path}")
    print(f"{'─'*60}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(dump_path)
    df = df[~df["status"].isin(["SKIPPED", "FAILED", "ERROR"])].copy()

    def _row_metrics(row):
        c = safe_parse_list(row.get("vss_merged_candidates", ""))[:args.N]
        g = safe_parse_list(row.get("ground_truths", ""))
        m = compute_metrics(c, g)
        return pd.Series({"_cur_mrr": m.get("mrr", 0.0),
                          "_cur_r20": m.get("recall@20", 0.0)})

    _m = df.apply(_row_metrics, axis=1)
    df["_cur_mrr"] = _m["_cur_mrr"]
    df["_cur_r20"] = _m["_cur_r20"]

    if args.fixable_only:
        before = len(df)
        df = df[df["_cur_r20"] > 0].reset_index(drop=True)
        print(f"fixable_only: kept {len(df)}/{before} queries (GT in top-{args.N})")

    if args.worst_n:
        df = df.nsmallest(args.worst_n, "_cur_mrr").reset_index(drop=True)
        print(f"Selected {len(df)} worst-MRR queries")
    elif args.max_queries:
        df = df.head(args.max_queries)

    df = df.drop(columns=["_cur_mrr", "_cur_r20"])
    print(f"Loaded {len(df)} queries")

    # ── KB ────────────────────────────────────────────────────────────────────
    print(f"Loading KB ({dataset_name})...")
    kb = load_skb(dataset_name, download_processed=True)
    print("KB loaded.")

    # ── bridge pool ───────────────────────────────────────────────────────────
    n_workers = min(args.workers, len(df))
    print(f"Building {n_workers} LLM bridge(s)...")
    bridge_pool: queue.Queue = queue.Queue()
    for wi in range(n_workers):
        bridge_pool.put(LlmBridge(
            model_name=model_cfg["llm_name"],
            configs_path=model_cfg["llm_config_path"],
            dataset=dataset_name,
            worker_index=wi,
        ))
    print(f"{n_workers} bridge(s) ready.\n")

    dreq = DREQReranker(alpha=args.alpha, debug=args.debug)

    # ── run ───────────────────────────────────────────────────────────────────
    rows       = df.to_dict(orient="records")
    total      = len(rows)
    results    = []
    done_count = [0]

    def _worker(i, row):
        bridge = bridge_pool.get()
        try:
            local_dreq = DREQReranker(alpha=args.alpha, debug=(args.debug and i == 0))
            return process_query(row, kb, bridge, local_dreq, N=args.N,
                                 debug=(args.debug and i == 0))
        finally:
            bridge_pool.put(bridge)

    print(f"  {'#':>4}  {'qid':>8}  {'MRR before':>10}  {'MRR after':>10}  "
          f"{'Δ MRR':>8}  {'H@1 bef':>7}  {'H@1 aft':>7}  {'entities':>8}  {'calls':>5}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  "
          f"{'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, i, row): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
                    done_count[0] += 1
                    delta = res["ts_mrr"] - res["cur_mrr"]
                    cprint(
                        f"  {done_count[0]:>4}  {res['qid']:>8}  "
                        f"{res['cur_mrr']:>10.4f}  {res['ts_mrr']:>10.4f}  "
                        f"{delta:>+8.4f}  {int(res['cur_hit@1']):>7}  "
                        f"{int(res['ts_hit@1']):>7}  {res['n_entities']:>8}  "
                        f"{res['llm_calls']:>5}"
                    )
                else:
                    cprint(f"  [qid {rows[i]['id']}] skipped")
            except Exception as e:
                cprint(f"  [qid {rows[i]['id']}] ERROR: {e}")

    if not results:
        print("No results produced.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print_summary(results_df, alpha=args.alpha, N=args.N)


if __name__ == "__main__":
    main()

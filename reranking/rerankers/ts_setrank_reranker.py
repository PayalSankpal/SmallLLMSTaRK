"""
ts_setrank_reranker.py
======================
Thompson Sampling for Setwise Reranking (TS-SetRank)

Based on: "Contextual Relevance and Adaptive Sampling for LLM-Based Document
Reranking" (Huang et al., 2025)

Operates on an existing full_data_dump.csv produced by the pipeline.
For each query it:
  1. Takes the top-N vss_merged_candidates as the candidate pool
  2. Runs TS-SetRank over T rounds with batch size b
  3. Produces a reranked list sorted by posterior mean relevance
  4. Computes and prints metrics vs the original ranking

Usage:
    python ts_setrank_reranker.py [params_json] [options]

Options:
    --dump          Path to full_data_dump.csv  (default: from params exp dir)
    --dataset       Dataset name [prime|amazon|mag] (default: from params)
    --T             Total inference rounds      (default: 30)
    --Tf            Exploration rounds          (default: 15)
    --b             Batch size                  (default: 5)
    --N             Candidate pool size         (default: 20)
    --max_queries   Limit number of queries     (default: all)
    --worst_n       Run on N worst-MRR queries  (default: off, overrides max_queries)
    --workers       Parallel workers            (default: 16)
    --out           Output CSV path             (default: <exp_dir>/ts_setrank_results.csv)
    --debug         Print per-round details for first query
"""

import argparse
import ast
import json
import math
import os
import queue
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

# ─── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from stark_qa import load_skb
from custom_pipeline.llm_bridge import LlmBridge

# ─── helpers ──────────────────────────────────────────────────────────────────

def safe_parse_list(val):
    if pd.isna(val) or val is None or str(val).strip() == "":
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []


def compute_metrics(ranked_list, ground_truths, k=20):
    gt_set = set(ground_truths)
    if not gt_set:
        return {}
    top_k = ranked_list[:k]
    recall_20 = len(set(top_k) & gt_set) / len(gt_set)
    hit_1  = 1.0 if (top_k and top_k[0] in gt_set) else 0.0
    hit_5  = 1.0 if any(n in gt_set for n in top_k[:5]) else 0.0
    mrr    = 0.0
    for rank, node in enumerate(top_k, 1):
        if node in gt_set:
            mrr = 1.0 / rank
            break
    dcg  = sum(1.0 / math.log2(r + 1) for r, n in enumerate(top_k, 1) if n in gt_set)
    ideal = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"recall@20": recall_20, "hit@1": hit_1, "hit@5": hit_5,
            "mrr": mrr, "ndcg@20": ndcg}


# ─── setwise LLM prompt ───────────────────────────────────────────────────────

SETWISE_SYSTEM = (
    "You are a relevance judge. Given a query and a set of candidate documents, "
    "identify which documents are relevant to answering the query. "
    "Be strict: only mark a document relevant if it directly helps answer the query."
)

def make_setwise_prompt(query: str, batch_docs: list[tuple]) -> str:
    """
    batch_docs: list of (display_idx, node_id, doc_str)
    Returns a prompt asking LLM to output comma-separated relevant indices.
    """
    doc_lines = "\n".join(
        f"  [{i}] {doc_str[:300].strip()}"
        for i, nid, doc_str in batch_docs
    )
    return f"""RELEVANCE JUDGMENT TASK

Query: {query}

Candidate documents:
{doc_lines}

Which of the numbered documents above are DIRECTLY RELEVANT to answering the query?
List ALL relevant document numbers as a comma-separated list (e.g. "1, 3").
If none are relevant, respond with "None".

Respond with ONLY the comma-separated numbers or "None". No explanation."""


def parse_setwise_response(response: str, max_idx: int) -> set[int]:
    """Parse LLM response to a set of 1-based display indices."""
    response = response.strip()
    if response.lower() == "none" or not response:
        return set()
    nums = set()
    for tok in re.split(r"[,\s]+", response):
        tok = tok.strip(" .")
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= max_idx:
                nums.add(idx)
    return nums


# ─── TS-SetRank core ──────────────────────────────────────────────────────────

class TSSetRank:
    def __init__(self, T: int = 30, Tf: int = 15, b: int = 5, debug: bool = False):
        self.T  = T    # total rounds
        self.Tf = Tf   # uniform exploration rounds
        self.b  = b    # batch size
        self.debug = debug

    def rerank(self, query: str, candidates: list[int], kb, llm_bridge,
               query_id=None) -> tuple[list[int], dict]:
        """
        Run TS-SetRank over candidates.
        Phase I  (t <= Tf): ALL exploration prompts sent as one batched LLM call
                            so the bridge can issue them concurrently via its
                            internal thread pool.
        Phase II (t > Tf) : Sequential Thompson-sampling rounds (each depends
                            on the posterior updated by the previous round).
        Returns (reranked_list, stats_dict).
        """
        N = len(candidates)
        if N == 0:
            return [], {}

        b = min(self.b, N)

        # --- Beta-Bernoulli posteriors: α_i = 1, β_i = 1 (uniform prior) ---
        alpha = np.ones(N, dtype=float)
        beta_ = np.ones(N, dtype=float)

        # Fetch doc descriptions once
        doc_cache = {}
        for nid in candidates:
            try:
                doc_cache[nid] = kb.get_doc_info(nid, compact=True)
            except Exception:
                doc_cache[nid] = f"Node {nid}"

        llm_calls = 0

        # ── Phase I: batch ALL exploration prompts in one ask_llm_batch call ─
        # Pre-generate all Tf random batches (no dependency between them)
        phase1_batches = []
        phase1_prompts = []
        for _ in range(self.Tf):
            idx = random.sample(range(N), b)
            node_ids = [candidates[i] for i in idx]
            docs = [(di + 1, nid, doc_cache[nid]) for di, nid in enumerate(node_ids)]
            phase1_batches.append(idx)
            phase1_prompts.append(make_setwise_prompt(query, docs))

        if phase1_prompts:
            try:
                phase1_answers, _ = llm_bridge.ask_llm_batch(phase1_prompts)
                llm_calls += len(phase1_prompts)
            except Exception as e:
                if self.debug:
                    print(f"  [Phase I] LLM batch error: {e}")
                phase1_answers = ["None"] * len(phase1_prompts)
                time.sleep(2)

            for t_idx, (batch_indices, raw) in enumerate(
                    zip(phase1_batches, phase1_answers)):
                relevant_disp = parse_setwise_response(raw, b)
                for disp_i, orig_idx in enumerate(batch_indices):
                    if (disp_i + 1) in relevant_disp:
                        alpha[orig_idx] += 1
                    else:
                        beta_[orig_idx] += 1
                if self.debug:
                    means = alpha / (alpha + beta_)
                    best  = candidates[int(np.argmax(means))]
                    node_ids = [candidates[i] for i in batch_indices]
                    print(f"  [round {t_idx+1:02d}|EXPLORE] batch={node_ids} "
                          f"relevant={relevant_disp} best={best}")

        # ── Phase II: Thompson sampling rounds (sequential) ───────────────────
        for t in range(self.Tf + 1, self.T + 1):
            samples = np.random.beta(alpha, beta_)
            batch_indices = list(np.argsort(samples)[::-1][:b])
            batch_node_ids = [candidates[i] for i in batch_indices]

            batch_docs = [
                (di + 1, nid, doc_cache[nid])
                for di, nid in enumerate(batch_node_ids)
            ]
            prompt = make_setwise_prompt(query, batch_docs)

            try:
                answers, _ = llm_bridge.ask_llm_batch([prompt])
                raw = answers[0]
                llm_calls += 1
            except Exception as e:
                if self.debug:
                    print(f"  [round {t}|EXPLOIT] LLM error: {e}")
                time.sleep(2)
                continue

            relevant_disp = parse_setwise_response(raw, b)
            for disp_i, orig_idx in enumerate(batch_indices):
                if (disp_i + 1) in relevant_disp:
                    alpha[orig_idx] += 1
                else:
                    beta_[orig_idx] += 1

            if self.debug:
                means = alpha / (alpha + beta_)
                best  = candidates[int(np.argmax(means))]
                print(f"  [round {t:02d}|EXPLOIT] batch={batch_node_ids} "
                      f"relevant={relevant_disp} best={best}")

        # ── final ranking by posterior mean ──────────────────────────────────
        means = alpha / (alpha + beta_)
        order = np.argsort(means)[::-1]
        reranked = [candidates[i] for i in order]

        stats = {
            "llm_calls": llm_calls,
            "final_means": {candidates[i]: float(means[i]) for i in range(N)},
        }
        return reranked, stats


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TS-SetRank reranker")
    p.add_argument("params", nargs="?", default="params_new.json",
                   help="Params JSON file (same one used for the pipeline run)")
    p.add_argument("--dump",        default=None,   help="Override full_data_dump.csv path")
    p.add_argument("--dataset",     default=None,   help="Override dataset name")
    p.add_argument("--T",           type=int, default=30,  help="Total TS rounds per query")
    p.add_argument("--Tf",          type=int, default=15,  help="Uniform exploration rounds")
    p.add_argument("--b",           type=int, default=5,   help="Batch size")
    p.add_argument("--N",           type=int, default=20,  help="Candidate pool size")
    p.add_argument("--max_queries", type=int, default=None, help="Limit queries (for testing)")
    p.add_argument("--worst_n",     type=int, default=None, help="Run on N queries with worst current MRR (overrides max_queries)")
    p.add_argument("--fixable_only", action="store_true",   help="Only include queries where GT is in top-N (reranking can actually help)")
    p.add_argument("--workers",     type=int, default=16,  help="Parallel query workers (one LLM bridge per worker)")
    p.add_argument("--out",         default=None,   help="Output CSV path")
    p.add_argument("--debug",       action="store_true",   help="Verbose per-round output for first query")
    return p.parse_args()


# shared print lock so concurrent workers don't interleave lines
_print_lock = Lock()

def cprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def process_query(row, kb, llm_bridge, ts, N, debug=False):
    """Process a single query row. Returns metrics dict."""
    qid   = row["id"]
    query = row["query"]
    gt    = safe_parse_list(row["ground_truths"])
    cands = safe_parse_list(row.get("vss_merged_candidates", ""))[:N]
    if not cands:
        cands = safe_parse_list(row.get("grounding_candidates", ""))[:N]

    if not cands or not gt:
        return None

    current_metrics  = compute_metrics(cands, gt)
    reranked, stats  = ts.rerank(query, cands, kb, llm_bridge, query_id=qid)
    reranked_metrics = compute_metrics(reranked, gt)

    return {
        "qid":              qid,
        "query":            query[:120],
        "llm_calls":        stats.get("llm_calls", 0),
        # current (pre-rerank)
        "cur_recall@20":    current_metrics.get("recall@20", 0),
        "cur_hit@1":        current_metrics.get("hit@1", 0),
        "cur_hit@5":        current_metrics.get("hit@5", 0),
        "cur_mrr":          current_metrics.get("mrr", 0),
        "cur_ndcg@20":      current_metrics.get("ndcg@20", 0),
        # TS-SetRank reranked
        "ts_recall@20":     reranked_metrics.get("recall@20", 0),
        "ts_hit@1":         reranked_metrics.get("hit@1", 0),
        "ts_hit@5":         reranked_metrics.get("hit@5", 0),
        "ts_mrr":           reranked_metrics.get("mrr", 0),
        "ts_ndcg@20":       reranked_metrics.get("ndcg@20", 0),
        "reranked_list":    str(reranked),
    }


def print_summary(results_df, T, Tf, b, N):
    metrics = ["recall@20", "hit@1", "hit@5", "mrr", "ndcg@20"]
    n = len(results_df)
    print("\n" + "═" * 72)
    print(f"  TS-SetRank Reranking Results  (T={T}, Tf={Tf}, b={b}, N={N})")
    print(f"  Queries evaluated: {n}")
    print("═" * 72)
    hdr = f"  {'Metric':<14} {'Before':>10} {'TS-SetRank':>12} {'Gain':>10} {'Gain%':>8}"
    print(hdr)
    print("─" * 72)
    for m in metrics:
        cur  = results_df[f"cur_{m}"].mean()
        ts   = results_df[f"ts_{m}"].mean()
        gain = ts - cur
        pct  = (gain / cur * 100) if cur > 0 else 0.0
        print(f"  {m:<14} {cur:>10.4f} {ts:>12.4f} {gain:>+10.4f} {pct:>7.1f}%")
    print("─" * 72)

    improved = (results_df["ts_mrr"] > results_df["cur_mrr"]).sum()
    same     = (results_df["ts_mrr"] == results_df["cur_mrr"]).sum()
    hurt     = (results_df["ts_mrr"] < results_df["cur_mrr"]).sum()
    total_calls = results_df["llm_calls"].sum()
    print(f"\n  MRR improved : {improved}/{n}")
    print(f"  MRR unchanged: {same}/{n}")
    print(f"  MRR hurt     : {hurt}/{n}")
    print(f"  Total LLM calls: {total_calls} ({total_calls/n:.1f}/query avg)")
    print("═" * 72 + "\n")

    # recall@20 unchanged confirmation
    r20_diff = (results_df["ts_recall@20"] - results_df["cur_recall@20"]).abs().max()
    if r20_diff < 1e-9:
        print("  ✓ recall@20 is identical before/after (reranking only reorders)\n")
    else:
        print(f"  ! recall@20 changed by up to {r20_diff:.4f} "
              "(unexpected — check candidate list handling)\n")


def main():
    args = parse_args()

    # ── load params ───────────────────────────────────────────────────────────
    with open(args.params, "r") as f:
        config = json.load(f)

    dataset_name = args.dataset or config["experiment"]["dataset"]
    exp_name     = config["experiment"]["exp_name"]
    output_base  = config["experiment"].get("output_base_dir", "./world/")
    model_config = config["models"]

    dump_path = args.dump or f"{output_base}{exp_name}/full_data_dump.csv"
    out_path  = args.out  or f"{output_base}{exp_name}/ts_setrank_results.csv"

    print(f"\n{'─'*60}")
    print(f"  TS-SetRank Reranker")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Dump     : {dump_path}")
    print(f"  T={args.T}, Tf={args.Tf}, b={args.b}, N={args.N}")
    print(f"{'─'*60}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(dump_path)
    df = df[~df["status"].isin(["SKIPPED", "FAILED", "ERROR"])].copy()

    # ── pre-compute per-row metrics for filtering ────────────────────────────
    def _row_metrics(row):
        cands = safe_parse_list(row.get("vss_merged_candidates", ""))[:args.N]
        gt    = safe_parse_list(row.get("ground_truths", ""))
        m     = compute_metrics(cands, gt)
        return pd.Series({"_cur_mrr": m.get("mrr", 0.0),
                          "_cur_r20": m.get("recall@20", 0.0)})

    _metrics = df.apply(_row_metrics, axis=1)
    df["_cur_mrr"] = _metrics["_cur_mrr"]
    df["_cur_r20"] = _metrics["_cur_r20"]

    # ── fixable_only: GT must be in top-N candidate pool ─────────────────────
    if args.fixable_only:
        before = len(df)
        df = df[df["_cur_r20"] > 0].reset_index(drop=True)
        print(f"fixable_only: kept {len(df)}/{before} queries (GT in top-{args.N})")

    # ── worst_n: sort by MRR ascending, take N ───────────────────────────────
    if args.worst_n:
        df = df.nsmallest(args.worst_n, "_cur_mrr").reset_index(drop=True)
        print(f"Selected {len(df)} worst-MRR queries")
    elif args.max_queries:
        df = df.head(args.max_queries)

    df = df.drop(columns=["_cur_mrr", "_cur_r20"])
    print(f"Loaded {len(df)} queries from {dump_path}")

    # ── load KB ───────────────────────────────────────────────────────────────
    print(f"Loading KB ({dataset_name})...")
    kb = load_skb(dataset_name, download_processed=True)
    print("KB loaded.")

    # ── build per-worker LLM bridge pool ─────────────────────────────────────
    # One bridge per worker avoids lock contention on shared state.
    # worker_index distributes across API keys in configs.json.
    n_workers = min(args.workers, len(df))
    print(f"Building {n_workers} LLM bridge(s) for {model_config['llm_name']}...")
    bridge_pool: queue.Queue = queue.Queue()
    for wi in range(n_workers):
        bridge_pool.put(LlmBridge(
            model_name=model_config["llm_name"],
            configs_path=model_config["llm_config_path"],
            dataset=dataset_name,
            worker_index=wi,
        ))
    print(f"{n_workers} bridge(s) ready.")

    # ── process queries ───────────────────────────────────────────────────────
    all_results = []
    rows = df.to_dict(orient="records")
    total = len(rows)
    done_count = [0]  # mutable counter for live display

    def _worker(i, row):
        bridge = bridge_pool.get()  # blocks until a bridge is free
        try:
            debug_this = args.debug and i == 0
            ts_local = TSSetRank(T=args.T, Tf=args.Tf, b=args.b, debug=debug_this)
            return process_query(row, kb, bridge, ts_local, N=args.N, debug=debug_this)
        finally:
            bridge_pool.put(bridge)  # always return to pool

    print(f"\nProcessing {total} queries with {n_workers} parallel worker(s)...\n")
    print(f"  {'#':>4}  {'qid':>8}  {'MRR before':>10}  {'MRR after':>10}  "
          f"{'Δ MRR':>8}  {'H@1 bef':>8}  {'H@1 aft':>8}  {'LLM calls':>9}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  "
          f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, i, row): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
                if res:
                    all_results.append(res)
                    done_count[0] += 1
                    cprint(
                        f"  {done_count[0]:>4}  {res['qid']:>8}  "
                        f"{res['cur_mrr']:>10.4f}  {res['ts_mrr']:>10.4f}  "
                        f"{res['ts_mrr']-res['cur_mrr']:>+8.4f}  "
                        f"{res['cur_hit@1']:>8.0f}  {res['ts_hit@1']:>8.0f}  "
                        f"{res['llm_calls']:>9}"
                    )
                else:
                    cprint(f"  [qid {rows[i]['id']}] skipped (no candidates or GT)")
            except Exception as e:
                cprint(f"  [qid {rows[i]['id']}] ERROR: {e}")

    if not all_results:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    print_summary(results_df, T=args.T, Tf=args.Tf, b=args.b, N=args.N)


if __name__ == "__main__":
    main()

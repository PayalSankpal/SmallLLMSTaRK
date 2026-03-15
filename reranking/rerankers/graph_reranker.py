"""
Graph Neighborhood Overlap Reranker (GNOR) for PRIME
=====================================================

PRIME-adapted reranking: 0 LLM calls, pure graph structure.

Motivation
----------
PRIME answers are reached via multi-hop KB traversal. The correct answer node
IS connected (within K hops) to the anchor entity nodes extracted from the
query. Incorrect VSS candidates that float up via semantic similarity may NOT
be well-connected to those anchors.

Algorithm
---------
1. For each query, extract anchor nodes from `initial_symbol_candidates`:
   - All non-ANSWER entity nodes with match_score >= score_thresh
   - Weighted by their match score (high-confidence anchors count more)

2. For each anchor, get its K-hop neighbor set using kb.k_hop_neighbor().

3. For each candidate in vss_merged_candidates[:N]:
   - graph_score = Σ(anchor_score × I[candidate ∈ K-hop(anchor)]) / Σ(anchor_score)
   - This is 0.0 if no anchor can reach the candidate within K hops.

4. Final score = alpha * graph_score + (1-alpha) * rank_score
   where rank_score = (N - original_rank) / N  (preserves original order for ties).

5. Rerank descending by final_score.

Usage
-----
  python graph_reranker.py params_new.json \\
    [--dump PATH] [--N 20] [--K 2] [--alpha 0.7] \\
    [--score_thresh 0.5] [--top_anchors 5] \\
    [--workers 16] [--worst_n N] [--fixable_only] \\
    [--out PATH]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from stark_qa import load_skb

# ─── utility ──────────────────────────────────────────────────────────────────

def cprint(*args, **kwargs):
    """Thread-safe print."""
    print(*args, **kwargs, flush=True)


def safe_parse_list(val):
    if val is None:
        return []
    try:
        if pd.isna(val) or str(val).strip() == "":
            return []
    except Exception:
        pass
    if isinstance(val, list):
        return [int(x) for x in val]
    try:
        r = ast.literal_eval(str(val).strip())
        return [int(x) for x in r] if r else []
    except Exception:
        try:
            r = json.loads(str(val).strip())
            return [int(x) for x in r] if r else []
        except Exception:
            return []


def safe_parse_dict(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return {}
    try:
        return ast.literal_eval(str(val).strip())
    except Exception:
        try:
            return json.loads(str(val).strip())
        except Exception:
            return {}


def compute_metrics(ranked_list, ground_truths, k=20):
    gt_set = set(ground_truths)
    if not gt_set:
        return {}
    top_k  = ranked_list[:k]
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


# ─── anchor extraction ────────────────────────────────────────────────────────

def extract_anchors(initial_symbol_candidates_raw: str,
                    score_thresh: float = 0.5,
                    top_k: int = 5,
                    exclude_entity: str = "ANSWER") -> list[tuple[int, float]]:
    """
    Parse initial_symbol_candidates and return (node_id, score) pairs for
    non-ANSWER entities, filtered by score_thresh and limited to top_k per entity.

    Returns a flat list of (node_id, score) with duplicates removed (max score kept).
    """
    isc = safe_parse_dict(initial_symbol_candidates_raw)
    anchor_map: dict[int, float] = {}
    for entity_key, cands in isc.items():
        if entity_key == exclude_entity:
            continue
        if not isinstance(cands, list):
            continue
        # sort by score descending, take top_k
        cands_sorted = sorted(cands, key=lambda c: c.get("score", 0.0), reverse=True)
        for c in cands_sorted[:top_k]:
            nid   = int(c.get("node_id", -1))
            score = float(c.get("score", 0.0))
            if nid < 0 or score < score_thresh:
                continue
            # keep max score across entities if node appears for multiple
            anchor_map[nid] = max(anchor_map.get(nid, 0.0), score)
    return list(anchor_map.items())   # [(node_id, score), ...]


# ─── graph neighborhood cache ─────────────────────────────────────────────────

# Per-process LRU-style cache for k_hop results (keyed by (node_id, K))
_khop_cache: dict[tuple[int, int], frozenset[int]] = {}
_khop_lock  = threading.Lock()


def get_khop_set(kb, node_id: int, K: int) -> frozenset[int]:
    """Return the set of node IDs within K hops of node_id (bidirectional)."""
    key = (node_id, K)
    with _khop_lock:
        if key in _khop_cache:
            return _khop_cache[key]
    try:
        subset, _, _, _ = kb.k_hop_neighbor(node_id, K)
        result = frozenset(int(x) for x in subset.tolist())
    except Exception:
        try:
            result = frozenset(int(x) for x in kb.get_neighbor_nodes(node_id, edge_type="*"))
        except Exception:
            result = frozenset()
    with _khop_lock:
        _khop_cache[key] = result
    return result


# ─── reranker ─────────────────────────────────────────────────────────────────

class GNORReranker:
    """Graph Neighborhood Overlap Reranker."""

    def __init__(self,
                 K: int = 2,
                 alpha: float = 0.7,
                 score_thresh: float = 0.5,
                 top_anchors: int = 5,
                 debug: bool = False):
        self.K            = K
        self.alpha        = alpha      # weight for graph_score (1-alpha → rank_score)
        self.score_thresh = score_thresh
        self.top_anchors  = top_anchors
        self.debug        = debug

    def rerank(self, candidates: list[int],
               initial_symbol_candidates_raw: str,
               kb,
               query_id=None) -> tuple[list[int], dict]:
        N = len(candidates)

        # ── 1. extract anchors ─────────────────────────────────────────────
        anchors = extract_anchors(
            initial_symbol_candidates_raw,
            score_thresh=self.score_thresh,
            top_k=self.top_anchors,
        )
        if not anchors:
            # no anchors → return original order
            stats = {"anchors": 0, "reranked": False, "reason": "no_anchors"}
            return list(candidates), stats

        total_anchor_weight = sum(s for _, s in anchors)

        # ── 2. build K-hop sets for each anchor ────────────────────────────
        anchor_nbhds: list[tuple[frozenset[int], float]] = []
        for nid, score in anchors:
            nbhd = get_khop_set(kb, nid, self.K)
            anchor_nbhds.append((nbhd, score))

        if self.debug:
            avg_nbhd = sum(len(n) for n, _ in anchor_nbhds) / max(len(anchor_nbhds), 1)
            cprint(f"  [GNOR] {len(anchors)} anchors, avg K={self.K}-hop size: {avg_nbhd:.0f}")

        # ── 3. score each candidate ────────────────────────────────────────
        graph_scores: dict[int, float] = {}
        for rank_i, cnd in enumerate(candidates):
            gs = sum(w for nbhd, w in anchor_nbhds if cnd in nbhd)
            graph_scores[cnd] = gs / total_anchor_weight if total_anchor_weight > 0 else 0.0

        # rank score: rank-1 → 1.0, rank-N → ~0
        rank_scores = {cnd: (N - rank_i) / N for rank_i, cnd in enumerate(candidates)}

        # ── 4. final score & rerank ────────────────────────────────────────
        final_scores = {
            cnd: self.alpha * graph_scores[cnd] + (1 - self.alpha) * rank_scores[cnd]
            for cnd in candidates
        }
        reranked = sorted(candidates, key=lambda c: final_scores[c], reverse=True)

        if self.debug:
            top3_gs = [(c, graph_scores[c]) for c in reranked[:3]]
            cprint(f"  [GNOR] top-3 graph scores: {top3_gs}")

        # how many candidates had graph_score > 0?
        nonzero = sum(1 for c in candidates if graph_scores[c] > 0)
        stats = {
            "anchors":  len(anchors),
            "nonzero":  nonzero,
            "reranked": True,
            "top1_gs":  graph_scores[reranked[0]],
            "avg_gs":   sum(graph_scores.values()) / N,
        }
        return reranked, stats


# ─── per-query worker ─────────────────────────────────────────────────────────

def process_query(row: dict, kb, reranker: GNORReranker, N: int, debug: bool = False) -> dict | None:
    qid   = row["id"]
    query = row["query"]
    gt    = safe_parse_list(row["ground_truths"])
    cands = safe_parse_list(row.get("vss_merged_candidates", ""))[:N]
    if not cands:
        cands = safe_parse_list(row.get("grounding_candidates", ""))[:N]
    if not cands or not gt:
        return None

    isc_raw = row.get("initial_symbol_candidates", "{}")
    cur_m   = compute_metrics(cands, gt)

    reranked, stats = reranker.rerank(cands, isc_raw, kb, query_id=qid)
    new_m = compute_metrics(reranked, gt)

    if debug:
        cprint(f"\n[Q {qid}] {query[:80]}...")
        cprint(f"  anchors={stats['anchors']}  nonzero_cands={stats.get('nonzero',0)}/{len(cands)}")
        cprint(f"  MRR: {cur_m.get('mrr',0):.4f} → {new_m.get('mrr',0):.4f}")

    return {
        "qid":          qid,
        "query":        query[:120],
        "n_anchors":    stats.get("anchors", 0),
        "nonzero_cands": stats.get("nonzero", 0),
        "top1_gs":      stats.get("top1_gs", 0.0),
        "avg_gs":       stats.get("avg_gs", 0.0),
        "cur_recall@20": cur_m.get("recall@20", 0),
        "cur_hit@1":    cur_m.get("hit@1", 0),
        "cur_hit@5":    cur_m.get("hit@5", 0),
        "cur_mrr":      cur_m.get("mrr", 0),
        "cur_ndcg@20":  cur_m.get("ndcg@20", 0),
        "ts_recall@20": new_m.get("recall@20", 0),
        "ts_hit@1":     new_m.get("hit@1", 0),
        "ts_hit@5":     new_m.get("hit@5", 0),
        "ts_mrr":       new_m.get("mrr", 0),
        "ts_ndcg@20":   new_m.get("ndcg@20", 0),
        "reranked_list": str(reranked),
    }


# ─── summary ──────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, K: int, alpha: float, N: int):
    n = len(df)
    metrics = ["recall@20", "hit@1", "hit@5", "mrr", "ndcg@20"]
    print(f"\n{'═'*72}")
    print(f"  Graph Neighborhood Overlap Reranker  (K={K}, α={alpha}, N={N})")
    print(f"  Queries evaluated : {n}")
    print(f"  Avg anchors/query : {df['n_anchors'].mean():.1f}")
    print(f"  Avg candidates with graph_score>0: {df['nonzero_cands'].mean():.1f}/{N}")
    print(f"  Avg top-1 graph score: {df['top1_gs'].mean():.3f}")
    print(f"{'═'*72}")
    print(f"  {'Metric':<14} {'Before':>10} {'GNOR':>12} {'Gain':>10} {'Gain%':>8}")
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
    print(f"\n  MRR improved: {imp}/{n}   same: {same}/{n}   hurt: {hurt}/{n}")
    print(f"{'═'*72}\n")

    r20_drift = (df["ts_recall@20"] - df["cur_recall@20"]).abs().max()
    if r20_drift < 1e-9:
        print("  ✓ recall@20 unchanged (reranking only reorders)\n")

    # Bucket analysis
    print("  MRR bucket analysis:")
    print(f"  {'Bucket':<18} {'N':>5}  {'Before':>8}  {'After':>8}  {'Gain':>8}  hurt  imp")
    buckets = [
        ("MRR = 1.0",  (df["cur_mrr"] == 1.0)),
        ("MRR 0.5–1.0", (df["cur_mrr"] >= 0.5) & (df["cur_mrr"] < 1.0)),
        ("MRR 0.2–0.5", (df["cur_mrr"] >= 0.2) & (df["cur_mrr"] < 0.5)),
        ("MRR < 0.2",   (df["cur_mrr"] < 0.2)),
    ]
    for label, mask in buckets:
        sub = df[mask]
        if len(sub) == 0:
            continue
        b_mrr = sub["cur_mrr"].mean()
        a_mrr = sub["ts_mrr"].mean()
        hurt  = (sub["ts_mrr"] < sub["cur_mrr"]).sum()
        imp   = (sub["ts_mrr"] > sub["cur_mrr"]).sum()
        print(f"  {label:<18} {len(sub):>5}  {b_mrr:>8.4f}  {a_mrr:>8.4f}  {a_mrr-b_mrr:>+8.4f}  {hurt:>4}  {imp:>3}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Graph Neighborhood Overlap Reranker (GNOR)")
    p.add_argument("params",         nargs="?", default="params_new.json")
    p.add_argument("--dump",         default=None)
    p.add_argument("--dataset",      default=None)
    p.add_argument("--N",            type=int,   default=20)
    p.add_argument("--K",            type=int,   default=2,
                   help="Number of hops for neighborhood expansion (default: 2)")
    p.add_argument("--alpha",        type=float, default=0.7,
                   help="Weight for graph_score vs rank_score (default: 0.7)")
    p.add_argument("--score_thresh", type=float, default=0.5,
                   help="Min anchor match score to include (default: 0.5)")
    p.add_argument("--top_anchors",  type=int,   default=5,
                   help="Max anchor nodes per entity (default: 5)")
    p.add_argument("--max_queries",  type=int,   default=None)
    p.add_argument("--worst_n",      type=int,   default=None, help="N worst-MRR queries")
    p.add_argument("--fixable_only", action="store_true",
                   help="Only queries where GT is in top-N")
    p.add_argument("--workers",      type=int,   default=16)
    p.add_argument("--out",          default=None)
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.params) as f:
        config = json.load(f)

    dataset_name = args.dataset or config["experiment"]["dataset"]
    exp_name     = config["experiment"]["exp_name"]
    output_base  = config["experiment"].get("output_base_dir", "./world/")

    dump_path = args.dump or f"{output_base}{exp_name}/full_data_dump.csv"
    out_path  = args.out  or f"{output_base}{exp_name}/gnor_results.csv"

    print(f"\n{'─'*60}")
    print(f"  Graph Neighborhood Overlap Reranker (GNOR)")
    print(f"  Dataset : {dataset_name}   K={args.K}  alpha={args.alpha}  N={args.N}")
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
    print("KB loaded.\n")

    # ── reranker ─────────────────────────────────────────────────────────────
    reranker = GNORReranker(
        K=args.K,
        alpha=args.alpha,
        score_thresh=args.score_thresh,
        top_anchors=args.top_anchors,
        debug=args.debug,
    )
    print(f"GNOR: K={args.K}, alpha={args.alpha}, score_thresh={args.score_thresh}, "
          f"top_anchors={args.top_anchors}\n")

    # ── run ───────────────────────────────────────────────────────────────────
    rows       = df.to_dict(orient="records")
    total      = len(rows)
    results    = []
    done_count = [0]
    lock       = threading.Lock()

    print(f"{'idx':>6}  {'qid':>6}  {'cur_mrr':>8}  {'gnor_mrr':>9}  {'Δmrr':>8}  {'h@1':>4}  {'h@5':>4}  anchors")
    print("─"*72)

    def _worker(i, row):
        local_reranker = GNORReranker(
            K=args.K, alpha=args.alpha,
            score_thresh=args.score_thresh,
            top_anchors=args.top_anchors,
            debug=(args.debug and i == 0),
        )
        return process_query(row, kb, local_reranker, N=args.N,
                             debug=(args.debug and i == 0))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, i, row): (i, row) for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            with lock:
                done_count[0] += 1
                results.append(res)
                d_mrr = res["ts_mrr"] - res["cur_mrr"]
                cprint(f"{done_count[0]:>6}  {res['qid']:>6}  "
                       f"{res['cur_mrr']:>8.4f}  {res['ts_mrr']:>9.4f}  "
                       f"{d_mrr:>+8.4f}  "
                       f"{int(res['ts_hit@1']):>4}  {int(res['ts_hit@5']):>4}  "
                       f"{res['n_anchors']:>7}")

    if not results:
        print("No results — check dump path.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print_summary(results_df, K=args.K, alpha=args.alpha, N=args.N)


if __name__ == "__main__":
    main()

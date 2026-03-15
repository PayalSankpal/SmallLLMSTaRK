"""
Relation-Conditioned Edge Checker (RCEC) Reranker for PRIME.

Zero LLM calls. Uses the exact KB edge types from the `relations` field of
each query to check which ANSWER candidates actually satisfy the required
graph-structural constraints. Completely orthogonal signal from the grounding
harmonic-mean VSS score.

Algorithm:
  1. Parse `relations` dict: e.g. {('A','ANSWER'): ['ppi','interacts with'],
                                    ('B','ANSWER'): ['associated with'],
                                    ('ANSWER','C'): ['phenotype present']}
  2. Extract anchor nodes for each non-ANSWER entity from
     `initial_symbol_candidates` (top-K by VSS score).
  3. For each ANSWER constraint (src→ANSWER or ANSWER→tgt):
       - Forward  (src →ANSWER):  candidate ∈ get_neighbor_nodes(anchor, rel)?
       - Backward (ANSWER→ tgt): anchor    ∈ get_neighbor_nodes(candidate, rel)?
     (Bio KGs are mostly symmetric, so forward check on both sides works well
      as a fast approximation.)
  4. RCEC score = satisfied_constraints / total_constraints   (0.0 – 1.0)
  5. Final rerank score = alpha * rcec_score + (1-alpha) * orig_rank_score
     where orig_rank_score = 1 - (rank-1)/(N-1)

Usage:
  python rcec_reranker.py \
      --dump   world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/full_data_dump.csv \
      --out    world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/rcec_scores.csv \
      --alpha  0.6  --top_anchors 3 \
      --fixable_only  --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import pandas as pd
from stark_qa import load_skb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_parse(s: Any) -> Any:
    try:
        return ast.literal_eval(str(s))
    except Exception:
        try:
            return json.loads(str(s))
        except Exception:
            return {}


def get_mrr(row: pd.Series) -> float:
    r = safe_parse(row["results"])
    if not isinstance(r, dict):
        return 0.0
    m = r.get("vss_merged_metrics", r.get("metrics", {}))
    return float(m.get("mrr", m.get("MRR", 0.0)))


def get_vmc(row: pd.Series, top_n: int = 20) -> list[int]:
    return [int(x) for x in safe_parse(row["vss_merged_candidates"])][:top_n]


def get_gt(row: pd.Series) -> set[int]:
    return set(int(x) for x in safe_parse(row["ground_truths"]))


def compute_mrr_for_list(ranked: list[int], gt: set[int]) -> float:
    for i, n in enumerate(ranked):
        if n in gt:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Neighbour cache (thread-safe)
# ---------------------------------------------------------------------------

_neigh_cache: dict[tuple[int, str], frozenset[int]] = {}
_neigh_lock = threading.Lock()


def get_neighbors_cached(kb, node_id: int, rel_type: str) -> frozenset[int]:
    key = (node_id, rel_type)
    if key not in _neigh_cache:
        with _neigh_lock:
            if key not in _neigh_cache:          # double-check
                try:
                    ns = kb.get_neighbor_nodes(node_id, edge_type=rel_type)
                    _neigh_cache[key] = frozenset(int(x) for x in ns)
                except Exception:
                    _neigh_cache[key] = frozenset()
    return _neigh_cache[key]


# ---------------------------------------------------------------------------
# Per-query RCEC scoring
# ---------------------------------------------------------------------------

def score_query(kb, row: pd.Series, top_anchors: int, alpha: float,
                valid_rel_types: set[str], min_gap: float = 0.3,
                max_rcec_cands: int = 5, verbose: bool = False) -> dict:
    """
    Returns a dict with keys: qid, orig_mrr, rcec_mrr, improved, hurt, same,
    n_constraints, avg_rcec_rank1, avg_rcec_gt.
    """
    qid      = row["id"]
    orig_vmc = get_vmc(row, 20)
    gt       = get_gt(row)
    orig_mrr = compute_mrr_for_list(orig_vmc, gt)

    # ── 1. Parse relations ──────────────────────────────────────────────────
    rels = safe_parse(row["relations"])
    if not isinstance(rels, dict) or not rels:
        return {
            "qid": qid, "orig_mrr": orig_mrr, "rcec_mrr": orig_mrr,
            "improved": 0, "hurt": 0, "same": 1,
            "n_constraints": 0, "orig_rank1_rcec": 0.0,
            "max_rcec": 0.0, "avg_rcec_gt": 0.0, "gated": 1,
        }

    # Constraints that point TO or FROM ANSWER
    constraints: list[tuple[str, str, str]] = []  # (src_role, tgt_role, rel)
    for (src_role, tgt_role), rel_list in rels.items():
        if "ANSWER" not in (src_role, tgt_role):
            continue
        for rel in rel_list:
            if rel not in valid_rel_types:
                continue
            constraints.append((src_role, tgt_role, rel))

    if not constraints:
        return {
            "qid": qid, "orig_mrr": orig_mrr, "rcec_mrr": orig_mrr,
            "improved": 0, "hurt": 0, "same": 1,
            "n_constraints": 0, "orig_rank1_rcec": 0.0,
            "max_rcec": 0.0, "avg_rcec_gt": 0.0, "gated": 1,
        }

    # ── 2. Collect anchor nodes for each non-ANSWER entity ─────────────────
    isc = safe_parse(row["initial_symbol_candidates"])
    anchor_nodes: dict[str, list[int]] = {}  # role → top anchor node IDs
    if isinstance(isc, dict):
        for role, cands in isc.items():
            if role == "ANSWER" or not cands:
                continue
            sorted_cands = sorted(cands, key=lambda c: -c.get("score", 0.0))
            anchor_nodes[role] = [c["node_id"] for c in sorted_cands[:top_anchors]]

    # ── 3. Build per-candidate RCEC score ───────────────────────────────────
    # Direction rules:
    #   (src, ANSWER, rel):  candidate ∈ get_neighbor_nodes(src_anchor, rel)
    #                         [anchor has outgoing edge to candidate]
    #   (ANSWER, tgt, rel):  tgt_anchor ∈ get_neighbor_nodes(candidate, rel)
    #                         [candidate has outgoing edge to tgt_anchor]
    N = len(orig_vmc)
    scores: dict[int, float] = {c: 0.0 for c in orig_vmc}
    total_constraints = 0

    # Pre-compute forward anchor neighbour sets for (src→ANSWER) constraints
    for src_role, tgt_role, rel in constraints:
        if src_role != "ANSWER":
            # Forward: src_anchor → candidate
            partner_role = src_role
            if partner_role not in anchor_nodes:
                continue
            total_constraints += 1
            for anchor_id in anchor_nodes[partner_role]:
                nbrs = get_neighbors_cached(kb, anchor_id, rel)
                for cand in orig_vmc:
                    if cand in nbrs:
                        scores[cand] += 1.0 / len(anchor_nodes[partner_role])
        else:
            # Backward: candidate → tgt_anchor
            partner_role = tgt_role
            if partner_role not in anchor_nodes:
                continue
            total_constraints += 1
            tgt_anchor_set = set(anchor_nodes[partner_role])
            for cand in orig_vmc:
                cand_nbrs = get_neighbors_cached(kb, cand, rel)
                matched = tgt_anchor_set.intersection(cand_nbrs)
                if matched:
                    scores[cand] += len(matched) / len(anchor_nodes[partner_role])

    if total_constraints == 0:
        return {
            "qid": qid, "orig_mrr": orig_mrr, "rcec_mrr": orig_mrr,
            "improved": 0, "hurt": 0, "same": 1,
            "n_constraints": 0, "orig_rank1_rcec": 0.0,
            "max_rcec": 0.0, "avg_rcec_gt": 0.0, "gated": 1,
        }

    # Normalise by total constraints
    for cand in orig_vmc:
        scores[cand] /= total_constraints

    # ── 4. Gate: only rerank if evidence is strong and specific ─────────────
    orig_rank1_rcec  = scores[orig_vmc[0]]
    max_rcec         = max(scores.values())
    rcec_gap         = max_rcec - orig_rank1_rcec
    n_positive       = sum(1 for s in scores.values() if s > 0.0)

    gt_nodes_in_vmc = [c for c in orig_vmc if c in gt]
    avg_rcec_gt     = (sum(scores[n] for n in gt_nodes_in_vmc)
                       / max(len(gt_nodes_in_vmc), 1))

    # Skip reranking if the gap is too small (ambiguous) or too many positive
    gated = (rcec_gap < min_gap or n_positive > max_rcec_cands)
    if gated:
        if verbose:
            direction = "="
            print(f"  qid={qid}  orig={orig_mrr:.4f}  rcec=GATED  ="
                  f"  constraints={total_constraints}"
                  f"  gap={rcec_gap:.3f}  n_pos={n_positive}"
                  f"  rank1_rcec={orig_rank1_rcec:.3f}  GT_rcec={avg_rcec_gt:.3f}")
        return {
            "qid": qid, "orig_mrr": orig_mrr, "rcec_mrr": orig_mrr,
            "improved": 0, "hurt": 0, "same": 1,
            "n_constraints": total_constraints,
            "orig_rank1_rcec": orig_rank1_rcec,
            "max_rcec": max_rcec, "avg_rcec_gt": avg_rcec_gt, "gated": 1,
        }

    # ── 5. Fuse with original rank score and rerank ──────────────────────────
    rank_scores  = {cand: 1.0 - i / max(N - 1, 1) for i, cand in enumerate(orig_vmc)}
    final_scores = {
        cand: alpha * scores[cand] + (1.0 - alpha) * rank_scores[cand]
        for cand in orig_vmc
    }
    reranked = sorted(orig_vmc, key=lambda c: -final_scores[c])
    rcec_mrr = compute_mrr_for_list(reranked, gt)

    if verbose:
        direction = "↑" if rcec_mrr > orig_mrr else ("↓" if rcec_mrr < orig_mrr else "=")
        print(f"  qid={qid}  orig={orig_mrr:.4f}  rcec={rcec_mrr:.4f}  {direction}  "
              f"constraints={total_constraints}  gap={rcec_gap:.3f}  n_pos={n_positive}"
              f"  rank1_rcec={orig_rank1_rcec:.3f}  GT_rcec={avg_rcec_gt:.3f}")
        if rcec_mrr < orig_mrr:
            gt_node = gt_nodes_in_vmc[0] if gt_nodes_in_vmc else None
            badnode = reranked[0]
            print(f"    HURT: rank1_orig={orig_vmc[0]} (rcec={scores[orig_vmc[0]]:.3f})"
                  f" → rank1_new={badnode} (rcec={scores[badnode]:.3f})"
                  f"  GT={gt_node} (rcec={scores.get(gt_node, 0):.3f})"
                  f"  orig_GT_rank={orig_vmc.index(gt_node)+1 if gt_node in orig_vmc else 'N/A'}")

    improved = 1 if rcec_mrr > orig_mrr else 0
    hurt     = 1 if rcec_mrr < orig_mrr else 0
    same     = 1 if rcec_mrr == orig_mrr else 0
    return {
        "qid": qid, "orig_mrr": orig_mrr, "rcec_mrr": rcec_mrr,
        "improved": improved, "hurt": hurt, "same": same,
        "n_constraints": total_constraints,
        "orig_rank1_rcec": orig_rank1_rcec,
        "max_rcec": max_rcec, "avg_rcec_gt": avg_rcec_gt, "gated": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Relation-Conditioned Edge Checker reranker")
    p.add_argument("--dump",         default="world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/full_data_dump.csv")
    p.add_argument("--out",          default="world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/rcec_scores.csv")
    p.add_argument("--alpha",        type=float, default=0.6,
                   help="Weight for RCEC score vs original rank score")
    p.add_argument("--top_anchors",  type=int,   default=3,
                   help="Number of top anchor nodes per entity (by VSS score)")
    p.add_argument("--workers",      type=int,   default=8)
    p.add_argument("--min_gap",      type=float, default=0.3,
                   help="Min RCEC gap between best candidate and rank-1 to trigger reranking")
    p.add_argument("--max_rcec_cands", type=int, default=5,
                   help="Skip reranking if more than this many candidates have RCEC>0 (ambiguous)")
    p.add_argument("--fixable_only", action="store_true",
                   help="Only score queries where GT is in top-20")
    p.add_argument("--worst_n",      type=int,   default=0,
                   help="Only test the N queries with lowest current MRR (0=all)")
    p.add_argument("--verbose",      action="store_true")
    args = p.parse_args()

    # Load KB
    print("Loading PRIME KB …")
    kb = load_skb("prime", download_processed=True,
                  root="/raid/adityasd314/BTechProject/data")
    valid_rel_types = set(kb.rel_type_lst())
    print(f"KB loaded. Relation types: {valid_rel_types}\n")

    # Load dump
    dump = pd.read_csv(args.dump)
    print(f"Loaded {len(dump)} rows from {args.dump}")

    # Filter rows
    rows = []
    for _, row in dump.iterrows():
        orig_mrr = get_mrr(row)
        vmc = get_vmc(row, 20)
        gt  = get_gt(row)
        gt_in_vmc = bool(gt.intersection(set(vmc)))
        if args.fixable_only and not gt_in_vmc:
            continue
        rows.append((row, orig_mrr))

    if args.worst_n > 0:
        rows.sort(key=lambda x: x[1])
        rows = rows[:args.worst_n]

    print(f"Evaluating {len(rows)} queries (fixable_only={args.fixable_only}, worst_n={args.worst_n})\n")

    # Evaluate — sequential to avoid race conditions on sparse_adj_by_type
    # (Torch sparse tensors are not thread-safe for indexing)
    results = []
    for row, _ in rows:
        rec = score_query(kb, row, args.top_anchors, args.alpha, valid_rel_types,
                          min_gap=args.min_gap,
                          max_rcec_cands=args.max_rcec_cands,
                          verbose=args.verbose)
        results.append(rec)

    # Stats
    improved = sum(r["improved"] for r in results)
    hurt     = sum(r["hurt"]     for r in results)
    same_    = sum(r["same"]     for r in results)
    gated_n  = sum(r.get("gated", 0) for r in results)
    orig_avg = sum(r["orig_mrr"] for r in results) / len(results)
    rcec_avg = sum(r["rcec_mrr"] for r in results) / len(results)

    print(f"\n{'─'*60}")
    print(f"Queries evaluated : {len(results)}")
    print(f"Gated (no rerank) : {gated_n}")
    print(f"Reranked          : {len(results) - gated_n}")
    print(f"  Improved        : {improved}")
    print(f"  Same            : {same_ - gated_n}")
    print(f"  Hurt            : {hurt}")
    print(f"Orig avg MRR      : {orig_avg:.4f}")
    print(f"RCEC avg MRR      : {rcec_avg:.4f}")
    print(f"MRR delta         : {rcec_avg - orig_avg:+.4f}")
    print(f"{'─'*60}\n")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

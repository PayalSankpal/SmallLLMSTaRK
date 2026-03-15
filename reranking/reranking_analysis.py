"""
reranking_analysis.py
=====================
Compares current retrieval metrics vs the ideal upper-bound achievable
only by reranking (i.e. without adding new candidates, just reordering
the existing top-20 list).

Usage:
    python reranking_analysis.py <path_to_full_data_dump.csv>

If no path is provided it defaults to the PRIME experiment folder.
"""

import sys
import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────── helpers ──────────────────────────────────────

def safe_parse_list(val):
    """Parse a string-encoded Python/JSON list to a Python list. Returns []."""
    if pd.isna(val) or val == "" or val is None:
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
    """
    Compute ranking metrics given an ordered candidate list and GT set.
    ranked_list : ordered list of node IDs (position 0 = rank 1)
    ground_truths: set / list of correct node IDs
    """
    gt_set = set(ground_truths)
    if not gt_set:
        return None

    top_k = ranked_list[:k]

    # Recall@20
    recall_20 = len(set(top_k) & gt_set) / len(gt_set)

    # Hit@1
    hit_1 = 1.0 if (top_k and top_k[0] in gt_set) else 0.0

    # Hit@5
    hit_5 = 1.0 if any(n in gt_set for n in top_k[:5]) else 0.0

    # MRR
    mrr = 0.0
    for rank, node in enumerate(top_k, 1):
        if node in gt_set:
            mrr = 1.0 / rank
            break

    # NDCG@20 (binary relevance)
    dcg = sum(1.0 / np.log2(r + 1) for r, n in enumerate(top_k, 1) if n in gt_set)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    ndcg_20 = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall@20": recall_20,
        "hit@1":     hit_1,
        "hit@5":     hit_5,
        "mrr":       mrr,
        "ndcg@20":   ndcg_20,
    }


def ideal_ranking(ranked_list, ground_truths):
    """
    Return the ideal reranked list: GT nodes first (in their original order),
    then the remaining non-GT nodes (in their original order).
    This is the best reranking can do without adding new candidates.
    """
    gt_set = set(ground_truths)
    gt_nodes    = [n for n in ranked_list if n in gt_set]
    other_nodes = [n for n in ranked_list if n not in gt_set]
    return gt_nodes + other_nodes


# ──────────────────────────── main ─────────────────────────────────────────

def main():
    dump_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN/full_data_dump.csv")

    print(f"\nLoading: {dump_path}")
    df = pd.read_csv(dump_path)

    # Keep only rows that finished processing
    df = df[~df["status"].isin(["SKIPPED", "FAILED", "ERROR"])].copy()
    print(f"Rows after filtering SKIPPED/FAILED: {len(df)}")

    rows_current, rows_ideal, rows_gain = [], [], []

    for _, row in df.iterrows():
        gt = safe_parse_list(row["ground_truths"])
        # Use vss_merged_candidates as the "current" ranked list
        candidates = safe_parse_list(row.get("vss_merged_candidates", ""))
        if not candidates:
            candidates = safe_parse_list(row.get("grounding_candidates", ""))
        if not gt or not candidates:
            continue

        cur  = compute_metrics(candidates, gt)
        best = compute_metrics(ideal_ranking(candidates, gt), gt)
        if cur is None or best is None:
            continue

        rows_current.append(cur)
        rows_ideal.append(best)
        gain = {k: best[k] - cur[k] for k in cur}
        rows_gain.append(gain)

    if not rows_current:
        print("No valid rows found. Check column names / data format.")
        return

    cur_df  = pd.DataFrame(rows_current)
    best_df = pd.DataFrame(rows_ideal)
    gain_df = pd.DataFrame(rows_gain)

    metrics = ["recall@20", "hit@1", "hit@5", "mrr", "ndcg@20"]

    # ── per-query summary ──────────────────────────────────────────────────
    print("\n" + "═"*70)
    print(f"  DATASET : {dump_path.parent.name}")
    print(f"  Queries evaluated: {len(cur_df)}")
    print("═"*70)

    header = f"{'Metric':<15} {'Current':>10} {'Ideal*':>10} {'Gap':>10} {'Gap%':>8}"
    print(header)
    print("─"*70)
    for m in metrics:
        cur_val  = cur_df[m].mean()
        best_val = best_df[m].mean()
        gap      = best_val - cur_val
        gap_pct  = (gap / best_val * 100) if best_val > 0 else 0.0
        print(f"  {m:<13} {cur_val:>10.4f} {best_val:>10.4f} {gap:>+10.4f} {gap_pct:>7.1f}%")

    print("─"*70)
    print("  * Ideal = GT nodes promoted to top of the SAME candidate list")
    print("    (upper bound achievable purely by reranking, no new candidates)")

    # ── how many queries can reranking ACTUALLY help ───────────────────────
    # A query can benefit from reranking only if ≥1 GT is in the top-20
    # but is NOT already at rank 1
    gt_in_top20_not_rank1 = (
        (best_df["hit@1"] > cur_df["hit@1"])        # GT could be at rank 1 but isn't
    ).sum()
    gt_in_top20 = (best_df["recall@20"] > 0).sum()  # GT reachable at all
    gt_nowhere  = (best_df["recall@20"] == 0).sum() # GT not in top-20 at all

    print(f"\n  Queries where GT is in top-20 ............ {gt_in_top20} / {len(cur_df)}")
    print(f"  Queries where GT is NOT in top-20 ........ {gt_nowhere} / {len(cur_df)}")
    print(f"    └─ reranking CANNOT help these at all")
    print(f"  Queries where reranking could improve H@1: {gt_in_top20_not_rank1} / {len(cur_df)}")

    # ── MRR gain breakdown ─────────────────────────────────────────────────
    print(f"\n  MRR gain distribution (per query):")
    gain_bins = [0, 0.01, 0.1, 0.25, 0.5, 1.01]
    labels    = ["=0 (no gain)", "0–0.01", "0.01–0.1", "0.1–0.25", "0.25–0.5", ">0.5"]
    counts, _ = np.histogram(gain_df["mrr"], bins=gain_bins)
    zero_gain  = (gain_df["mrr"] == 0).sum()
    print(f"    {'No gain (0)':.<35} {zero_gain:>5}")
    for i, (lo, hi) in enumerate(zip(gain_bins[:-1], gain_bins[1:])):
        mask = (gain_df["mrr"] > lo) if lo == 0.5 else \
               (gain_df["mrr"] > lo) & (gain_df["mrr"] <= hi)
        print(f"    {labels[i+1]:.<35} {mask.sum():>5}")

    # ── current vs ideal recall@20 breakdown ──────────────────────────────
    print(f"\n  Recall@20 breakdown:")
    print(f"    Queries with recall@20 = 1.0 (current):  {(cur_df['recall@20']==1.0).sum()}")
    print(f"    Queries with recall@20 = 1.0 (ideal):    {(best_df['recall@20']==1.0).sum()}")
    print(f"    [Note: recall@20 is UNCHANGED by reranking — shows retrieval ceiling]")

    print("═"*70 + "\n")


if __name__ == "__main__":
    main()

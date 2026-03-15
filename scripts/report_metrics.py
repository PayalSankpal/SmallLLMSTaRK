"""
report_metrics.py — Print a full metrics breakdown for a pipeline run folder.

Usage:
    python report_metrics.py <folder_name_or_path>

Examples:
    python report_metrics.py PRIME_NEW_PIPELINE
    python report_metrics.py world/PRIME_NEW_PIPELINE
    python report_metrics.py /raid/adityasd314/BTechProject/world/PRIME_NEW_PIPELINE
"""

import sys
import os
import glob
import pandas as pd

DEFAULT_BASE = os.path.join(os.path.dirname(__file__), "world")


def load_results(folder: str) -> pd.DataFrame:
    """Load and deduplicate the pipeline results CSV, merging all per-process files."""
    # Resolve path
    if not os.path.isabs(folder):
        # Try relative to cwd first, then relative to world/
        if os.path.exists(folder):
            folder = os.path.abspath(folder)
        else:
            folder = os.path.join(DEFAULT_BASE, folder)

    if not os.path.exists(folder):
        print(f"ERROR: folder not found: {folder}")
        sys.exit(1)

    main_csv = os.path.join(folder, "pipeline_results.csv")
    per_proc = glob.glob(os.path.join(folder, "pipeline_results_process_*.csv"))

    dfs = []
    if os.path.exists(main_csv):
        dfs.append(pd.read_csv(main_csv))
    for f in per_proc:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass

    if not dfs:
        print(f"ERROR: no pipeline_results*.csv found in {folder}")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["query_id"], keep="last")
    return df, folder


def fmt(val: float) -> str:
    return f"{val:.4f}"


def report(folder: str):
    df, resolved = load_results(folder)
    n = len(df)

    # --- Column presence checks ---
    has_grounding   = "recall@20"           in df.columns
    has_vss_merged  = "recall@20_vss_merged" in df.columns
    has_reranked    = "recall@20_reranked"   in df.columns

    # ── gather metrics ──────────────────────────────────────────────────────

    def safe_mean(col):
        if col in df.columns:
            return df[col].fillna(0).mean()
        return None

    grnd_r20  = safe_mean("recall@20")
    grnd_mrr  = safe_mean("mrr")
    grnd_h1   = safe_mean("hit@1")
    grnd_h5   = safe_mean("hit@5")
    grnd_r50  = safe_mean("recall@50")

    vss_r20   = safe_mean("recall@20_vss_merged")

    rer_r20   = safe_mean("recall@20_reranked")
    rer_mrr   = safe_mean("mrr_reranked")
    rer_h1    = safe_mean("hit@1_reranked")

    # Per-query improvement table (top 10 and bottom 10 by reranking delta)
    if has_vss_merged and has_reranked:
        df["rerank_delta"] = df["recall@20_reranked"].fillna(0) - df["recall@20_vss_merged"].fillna(0)

    # ── print ───────────────────────────────────────────────────────────────
    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print(f"  METRICS REPORT")
    print(f"  Folder : {resolved}")
    print(f"  Queries: {n}")
    print(sep)

    print(f"\n{'Stage':<35} {'R@20':>8} {'R@50':>8} {'MRR':>8} {'H@1':>8} {'H@5':>8}")
    print(thin)

    if has_grounding:
        print(f"{'Grounding (priority queue)':<35} "
              f"{fmt(grnd_r20):>8} {fmt(grnd_r50) if grnd_r50 is not None else 'N/A':>8} "
              f"{fmt(grnd_mrr):>8} {fmt(grnd_h1):>8} {fmt(grnd_h5):>8}")

    if has_vss_merged:
        print(f"{'After VSS merge (pre-rerank)':<35} "
              f"{fmt(vss_r20):>8} {'':>8} {'':>8} {'':>8} {'':>8}")

    if has_reranked:
        delta_r20 = (rer_r20 - vss_r20) if (vss_r20 is not None) else None
        delta_str = f"  ({'+' if delta_r20 >= 0 else ''}{fmt(delta_r20)})" if delta_r20 is not None else ""
        print(f"{'After LLM reranking':<35} "
              f"{fmt(rer_r20):>8}{delta_str}  MRR={fmt(rer_mrr)}  H@1={fmt(rer_h1)}")

    print(thin)

    # ── per-stage histogram (recall buckets) ──────────────────────────────
    if has_vss_merged:
        print(f"\nVSS-merged recall@20 distribution (pre-rerank, N={n}):")
        bins   = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.001]
        labels = ["0 (miss)", "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0", "1.0 (full)"]
        col    = df["recall@20_vss_merged"].fillna(0)
        counts = pd.cut(col, bins=bins, labels=labels, right=False).value_counts().sort_index()
        for label, cnt in counts.items():
            bar = "█" * int(cnt / n * 40)
            print(f"  {label:<15} {cnt:>4}  {bar}")

    if has_reranked:
        print(f"\nReranked recall@20 distribution (N={n}):")
        col    = df["recall@20_reranked"].fillna(0)
        counts = pd.cut(col, bins=bins, labels=labels, right=False).value_counts().sort_index()
        for label, cnt in counts.items():
            bar = "█" * int(cnt / n * 40)
            print(f"  {label:<15} {cnt:>4}  {bar}")

    # ── queries where reranking helped / hurt ─────────────────────────────
    if has_vss_merged and has_reranked and "rerank_delta" in df.columns:
        improved = (df["rerank_delta"] > 0).sum()
        hurt     = (df["rerank_delta"] < 0).sum()
        same     = (df["rerank_delta"] == 0).sum()
        avg_gain  = df.loc[df["rerank_delta"] > 0, "rerank_delta"].mean()
        avg_loss  = df.loc[df["rerank_delta"] < 0, "rerank_delta"].mean()
        print(f"\nReranking impact vs VSS-merged:")
        print(f"  Improved : {improved:>4} queries  avg gain = {fmt(avg_gain) if improved else 'N/A'}")
        print(f"  Hurt     : {hurt:>4} queries  avg loss = {fmt(avg_loss) if hurt else 'N/A'}")
        print(f"  No change: {same:>4} queries")

        # Worst hurt by reranking
        if hurt > 0:
            print(f"\n  Top 5 queries most hurt by reranking:")
            worst = df.nsmallest(5, "rerank_delta")[
                ["query_id", "query_text", "recall@20_vss_merged", "recall@20_reranked", "rerank_delta"]
            ]
            for _, row in worst.iterrows():
                print(f"    Q{int(row['query_id']):<6}  vss={fmt(row['recall@20_vss_merged'])}  "
                      f"reranked={fmt(row['recall@20_reranked'])}  "
                      f"delta={fmt(row['rerank_delta'])}  "
                      f"{str(row['query_text'])[:60]}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    report(sys.argv[1])

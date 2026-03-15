"""
Graph-Path LLM Reranker (GPLR) for PRIME.

Unlike DREQ (which asks "does this text mention query entities?"),
GPLR asks the LLM: "Given these KB graph paths connecting the query entities
to each candidate, which candidate best answers the query?"

This is orthogonal to DREQ:
  - DREQ uses: document text + entity names (fails for multi-hop PRIME)
  - GPLR uses: KB topology — actual graph edges from anchors → candidates

Algorithm (1 LLM call per query):
  1. Parse `entities` + `relations` from the dump row.
  2. For each anchor node (top-1 per constant entity), do BFS up to max_hops
     toward each candidate in vss_merged[:top_k].
  3. For each candidate, format the shortest found path(s) as a readable string:
       RTL10 -[ppi]→ EZH2 -[associated with]→ chromatin organization
  4. Prompt the LLM with the query + all (candidate, path) pairs.
  5. LLM returns a JSON array of candidate indices sorted by fitness.
  6. Fuse LLM rank with original rank:
       final = alpha * llm_rank_score + (1 - alpha) * orig_rank_score
     using --alpha (default 0.7).

Usage:
  python graph_path_reranker.py \
    --dump world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/full_data_dump.csv \
    --out  world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/gplr_scores.csv \
    --params params_new.json \
    --alpha 0.7 --top_k 10 --max_hops 2 --max_paths_per_cand 2 \
    --fixable_only --workers 4 --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import queue
import re
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd
from stark_qa import load_skb

sys.path.insert(0, "/raid/adityasd314/BTechProject")
from custom_pipeline.llm_bridge import LlmBridge

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a biomedical knowledge graph expert. "
    "You reason over graph paths in a biomedical KB (PRIME) to identify the "
    "most likely answer to a query."
)

RANKING_PROMPT = """You are given a biomedical query and a list of candidate answers.
For each candidate, I have found paths in the PRIME knowledge graph that connect
the query's known entities to the candidate via relevant relations.

Query: {query}

Candidates with graph paths (format: id | candidate name | path):
{candidate_lines}

Instructions:
- Rank ALL candidates by how well the graph paths support them as the answer.
- A candidate with a direct, relevant path (matching the query's intent) should rank higher.
- A candidate with no path or only tangential paths should rank lower.
- Return ONLY a JSON array of candidate ids (integers) from best to worst, e.g. [3, 1, 7, ...]
- Include every candidate id exactly once.

Ranking (JSON array only):"""

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


def compute_mrr(ranked: list[int], gt: set[int]) -> float:
    for i, n in enumerate(ranked):
        if n in gt:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Node name cache
# ---------------------------------------------------------------------------

_name_cache: dict[int, str] = {}
_name_lock = threading.Lock()
_name_pattern = re.compile(r"- name: (.+)")


def node_name(kb, nid: int) -> str:
    if nid not in _name_cache:
        with _name_lock:
            if nid not in _name_cache:
                try:
                    info = kb.get_doc_info(int(nid), add_rel=False)
                    m = _name_pattern.search(info)
                    _name_cache[nid] = m.group(1).strip() if m else str(nid)
                except Exception:
                    _name_cache[nid] = str(nid)
    return _name_cache[nid]


# ---------------------------------------------------------------------------
# BFS path finder (anchor → target, up to max_hops)
# ---------------------------------------------------------------------------

_bfs_cache: dict[tuple[int, int, int], Optional[list[tuple[int, str]]]] = {}
_bfs_lock = threading.Lock()


def bfs_path(kb, src: int, tgt: int, max_hops: int,
             allowed_rels: Optional[set[str]] = None
             ) -> Optional[list[tuple[int, str]]]:
    """
    BFS from src to tgt up to max_hops.
    Returns list of (node, edge_label_from_prev) steps, or None.
    Edge label is the relation type (or "?" if unknown / any).
    Since sparse_adj doesn't expose edge types per neighbor easily,
    we do a two-pass: first check rel-filtered, then fall back to any.
    """
    key = (src, tgt, max_hops)
    if key in _bfs_cache:
        return _bfs_cache[key]

    rel_types = kb.rel_type_lst() if allowed_rels is None else list(allowed_rels)

    # BFS: state = (node, path_so_far)
    # path_so_far: list of (node_id, edge_label)
    frontier = deque()
    frontier.append((src, []))  # (current_node, path_to_current)
    visited = {src}

    found_path = None
    while frontier:
        curr, path = frontier.popleft()
        if len(path) >= max_hops:
            continue

        # Get all neighbors with their edge types
        # We try each relation type to find edges
        for rel in rel_types:
            try:
                nbrs = kb.get_neighbor_nodes(curr, edge_type=rel)
            except Exception:
                continue
            for nbr in nbrs:
                if nbr == tgt:
                    found_path = path + [(nbr, rel)]
                    break
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.append((nbr, path + [(nbr, rel)]))
            if found_path:
                break
        if found_path:
            break

    with _bfs_lock:
        _bfs_cache[key] = found_path
    return found_path


def format_path(kb, anchor_id: int, anchor_name: str,
                path: list[tuple[int, str]]) -> str:
    """Convert BFS path to readable string."""
    parts = [anchor_name]
    for node_id, rel in path:
        parts.append(f"-[{rel}]→")
        parts.append(node_name(kb, node_id))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Fast 1-hop path (used when max_hops=1 to avoid BFS overhead)
# ---------------------------------------------------------------------------

def one_hop_paths(kb, anchor_id: int, anchor_name: str,
                  cand_id: int, rel_types: list[str]
                  ) -> list[str]:
    """Return list of 1-hop path strings for anchor→cand via any matching rel."""
    paths = []
    for rel in rel_types:
        try:
            nbrs = kb.get_neighbor_nodes(anchor_id, edge_type=rel)
        except Exception:
            continue
        if cand_id in set(nbrs):
            cname = node_name(kb, cand_id)
            paths.append(f"{anchor_name} -[{rel}]→ {cname}")
    return paths


# ---------------------------------------------------------------------------
# Per-query processing
# ---------------------------------------------------------------------------

def process_query(kb, row: pd.Series, llm_bridge: LlmBridge,
                  top_k: int, max_hops: int, max_paths_per_cand: int,
                  alpha: float, verbose: bool, rel_types: list[str]) -> dict:
    qid      = row["id"]
    query    = row["query"]
    orig_vmc = [int(x) for x in safe_parse(row["vss_merged_candidates"])][:top_k]
    gt       = set(int(x) for x in safe_parse(row["ground_truths"]))
    orig_mrr = compute_mrr(orig_vmc, gt)
    N = len(orig_vmc)
    if N == 0:
        return {"qid": qid, "orig_mrr": orig_mrr, "gplr_mrr": orig_mrr,
                "improved": 0, "hurt": 0, "same": 1, "n_anchors": 0}

    # ── Anchor nodes: top-1 per constant entity ──────────────────────────
    ents = safe_parse(row["entities"])
    isc  = safe_parse(row["initial_symbol_candidates"])
    anchors: list[tuple[str, int, str]] = []  # (entity_key, node_id, name)
    if isinstance(ents, dict) and isinstance(isc, dict):
        for role, edata in ents.items():
            if role == "ANSWER":
                continue
            if not (isinstance(edata, dict) and edata.get("constant", False)):
                continue  # only constant entities have reliable anchors
            cands = isc.get(role, [])
            if not cands:
                continue
            best = max(cands, key=lambda c: c.get("score", 0.0))
            nid = best["node_id"]
            anchors.append((role, nid, node_name(kb, nid)))

    if not anchors:
        # Fall back to all entities, non-constant included
        if isinstance(isc, dict):
            for role, cands in isc.items():
                if role == "ANSWER" or not cands:
                    continue
                best = max(cands, key=lambda c: c.get("score", 0.0))
                nid = best["node_id"]
                anchors.append((role, nid, node_name(kb, nid)))

    # ── Build path strings per candidate ─────────────────────────────────
    cand_paths: dict[int, list[str]] = {c: [] for c in orig_vmc}

    for anchor_role, anchor_id, anchor_nm in anchors:
        for cand_id in orig_vmc:
            if len(cand_paths[cand_id]) >= max_paths_per_cand:
                continue
            if max_hops == 1:
                paths = one_hop_paths(kb, anchor_id, anchor_nm, cand_id, rel_types)
                cand_paths[cand_id].extend(paths[:max_paths_per_cand - len(cand_paths[cand_id])])
            else:
                p = bfs_path(kb, anchor_id, cand_id, max_hops=max_hops)
                if p:
                    s = format_path(kb, anchor_id, anchor_nm, p)
                    cand_paths[cand_id].append(s)

    # ── Build prompt ──────────────────────────────────────────────────────
    lines = []
    for i, cand_id in enumerate(orig_vmc):
        nm = node_name(kb, cand_id)
        paths = cand_paths[cand_id]
        if paths:
            path_str = "; ".join(paths[:max_paths_per_cand])
        else:
            path_str = "no direct path found"
        lines.append(f"  {i+1} | {nm} | {path_str}")

    prompt = RANKING_PROMPT.format(
        query=query,
        candidate_lines="\n".join(lines),
    )

    if verbose:
        print(f"\n[GPLR] qid={qid} orig_mrr={orig_mrr:.4f} anchors={[a[2] for a in anchors]}")
        print(f"  Prompt snippet: {prompt[:400]}…")

    # ── LLM call ─────────────────────────────────────────────────────────
    try:
        answers, _ = llm_bridge.ask_llm_batch([prompt])
        raw = answers[0] if answers else ""
    except Exception as e:
        if verbose:
            print(f"  [GPLR] LLM error qid={qid}: {e}")
        return {"qid": qid, "orig_mrr": orig_mrr, "gplr_mrr": orig_mrr,
                "improved": 0, "hurt": 0, "same": 1, "n_anchors": len(anchors)}

    # ── Parse LLM ranking ────────────────────────────────────────────────
    llm_order = _parse_ranking(raw, N)

    # ── Fuse with original rank ───────────────────────────────────────────
    orig_rank  = {c: 1.0 - i / max(N - 1, 1) for i, c in enumerate(orig_vmc)}
    llm_rank   = {orig_vmc[idx]: 1.0 - i / max(N - 1, 1)
                  for i, idx in enumerate(llm_order)}
    final_rank = {c: alpha * llm_rank.get(c, 0.0) + (1 - alpha) * orig_rank[c]
                  for c in orig_vmc}
    reranked   = sorted(orig_vmc, key=lambda c: -final_rank[c])
    gplr_mrr   = compute_mrr(reranked, gt)

    if verbose:
        d = "↑" if gplr_mrr > orig_mrr else ("↓" if gplr_mrr < orig_mrr else "=")
        print(f"  orig={orig_mrr:.4f}  gplr={gplr_mrr:.4f}  {d}")
        if gplr_mrr < orig_mrr:
            gtd = [c for c in orig_vmc if c in gt]
            gtn = node_name(kb, gtd[0]) if gtd else "N/A"
            print(f"  HURT: GT={gtn}, LLM put rank-1={node_name(kb, reranked[0])}")

    return {
        "qid":        qid,
        "orig_mrr":   orig_mrr,
        "gplr_mrr":   gplr_mrr,
        "improved":   int(gplr_mrr > orig_mrr),
        "hurt":       int(gplr_mrr < orig_mrr),
        "same":       int(gplr_mrr == orig_mrr),
        "n_anchors":  len(anchors),
        "llm_raw":    raw[:200],
    }


def _parse_ranking(raw: str, N: int) -> list[int]:
    """
    Parse LLM response to a list of 0-based indices.
    Handles: [3,1,2], [2, 4, 1], or any list of integers 1..N.
    Returns 0-based indices sorted by the LLM ranking.
    """
    # Find JSON array
    m = re.search(r"\[([0-9,\s]+)\]", raw)
    if not m:
        return list(range(N))  # fallback: keep original order
    try:
        ids = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        # Convert 1-based to 0-based
        result = []
        seen = set()
        for x in ids:
            idx = x - 1  # LLM uses 1-based
            if 0 <= idx < N and idx not in seen:
                result.append(idx)
                seen.add(idx)
        # Append any missing indices
        for i in range(N):
            if i not in seen:
                result.append(i)
        return result
    except Exception:
        return list(range(N))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Graph-Path LLM Reranker (GPLR)")
    p.add_argument("--dump",     default="world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/full_data_dump.csv")
    p.add_argument("--out",      default="world/PRIME_NEW_PIPELINE_NEW_EXAMPLES_TRAIN_FIXED/gplr_scores.csv")
    p.add_argument("--params",   default="params_new.json",
                   help="Pipeline params JSON (for llm_name and config)")
    p.add_argument("--alpha",    type=float, default=0.7,
                   help="LLM rank weight (0=keep original, 1=trust LLM fully)")
    p.add_argument("--top_k",    type=int,   default=10,
                   help="Number of top candidates to rerank (and include in prompt)")
    p.add_argument("--max_hops", type=int,   default=2,
                   help="Max BFS hops for path finding (1=fast 1-hop only)")
    p.add_argument("--max_paths_per_cand", type=int, default=2,
                   help="Max path strings shown per candidate in prompt")
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--fixable_only", action="store_true",
                   help="Only score queries where GT is in top-20")
    p.add_argument("--max_queries", type=int, default=0,
                   help="Limit total queries (0=all; for quick tests)")
    p.add_argument("--verbose",  action="store_true")
    args = p.parse_args()

    # ── Load KB ────────────────────────────────────────────────────────────
    print("Loading PRIME KB …")
    kb = load_skb("prime", download_processed=True,
                  root="/raid/adityasd314/BTechProject/data")
    rel_types = kb.rel_type_lst()
    print(f"KB loaded. {len(rel_types)} relation types.\n")

    # ── Load params + LLM bridge ───────────────────────────────────────────
    with open(args.params) as f:
        params = json.load(f)
    llm_name    = params["models"]["llm_name"]
    llm_cfg_path = params["models"].get("llm_config_path", "configs.json")
    dataset     = params["experiment"]["dataset"]
    print(f"LLM: {llm_name}  config: {llm_cfg_path}")

    # Build a pool of LLM bridges for parallel workers
    bridge_pool: queue.Queue[LlmBridge] = queue.Queue()
    for _ in range(args.workers):
        bridge_pool.put(LlmBridge(
            model_name=llm_name,
            dataset=dataset,
            configs_path=llm_cfg_path,
            verbose=False,
        ))
    print(f"Created {args.workers} LLM bridge(s).\n")

    # ── Load dump ──────────────────────────────────────────────────────────
    dump = pd.read_csv(args.dump)
    print(f"Loaded {len(dump)} rows from {args.dump}")

    rows = []
    for _, row in dump.iterrows():
        orig_mrr = get_mrr(row)
        vmc  = [int(x) for x in safe_parse(row["vss_merged_candidates"])][:20]
        gt   = set(int(x) for x in safe_parse(row["ground_truths"]))
        gt_in_top = bool(gt.intersection(set(vmc)))
        if args.fixable_only and not gt_in_top:
            continue
        rows.append(row)

    if args.max_queries > 0:
        rows = rows[:args.max_queries]
    print(f"Evaluating {len(rows)} queries\n")

    # ── Process ────────────────────────────────────────────────────────────
    results = []
    lock = threading.Lock()

    def worker(row):
        bridge = bridge_pool.get()
        try:
            rec = process_query(
                kb, row, bridge,
                top_k=args.top_k,
                max_hops=args.max_hops,
                max_paths_per_cand=args.max_paths_per_cand,
                alpha=args.alpha,
                verbose=args.verbose,
                rel_types=rel_types,
            )
            with lock:
                results.append(rec)
                n = len(results)
                if n % 10 == 0:
                    imp = sum(r["improved"] for r in results)
                    hurt = sum(r["hurt"] for r in results)
                    print(f"  [{n}/{len(rows)}]  improved={imp}  hurt={hurt}", flush=True)
        finally:
            bridge_pool.put(bridge)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, row) for row in rows]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"  Worker exception: {exc}")

    # ── Stats ──────────────────────────────────────────────────────────────
    improved = sum(r["improved"] for r in results)
    hurt     = sum(r["hurt"]     for r in results)
    same_    = sum(r["same"]     for r in results)
    orig_avg = sum(r["orig_mrr"] for r in results) / max(len(results), 1)
    gplr_avg = sum(r["gplr_mrr"] for r in results) / max(len(results), 1)

    print(f"\n{'─'*60}")
    print(f"Queries evaluated : {len(results)}")
    print(f"Improved          : {improved}")
    print(f"Same              : {same_}")
    print(f"Hurt              : {hurt}")
    print(f"Orig avg MRR      : {orig_avg:.4f}")
    print(f"GPLR avg MRR      : {gplr_avg:.4f}")
    print(f"MRR delta         : {gplr_avg - orig_avg:+.4f}")
    print(f"{'─'*60}\n")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

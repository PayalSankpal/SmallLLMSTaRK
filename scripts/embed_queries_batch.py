"""
embed_queries_batch.py — Embed PRIME train queries via OpenAI Batch API.

Submits all 6162 train queries in ONE batch job to /v1/embeddings,
polls for completion, then saves:
    emb/prime/{emb_model}/query/prime_train_embeddings.pt
        format: {qid (int): torch.Tensor shape [D]}

Stages (resumable):
  1. Load queries from PRIME train split
  2. Write requests.jsonl  (one line per query)
  3. Upload + create batch job  →  batch_id saved to batch_scratch/batch_id.txt
  4. Poll until complete
  5. Download results.jsonl, parse, save .pt

Usage:
    python embed_queries_batch.py
    python embed_queries_batch.py --emb_model text-embedding-3-small
    python embed_queries_batch.py --batch_id batch_abc123   # resume polling
"""

import os, sys, json, time, argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from stark_qa import load_qa

load_dotenv()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Embed PRIME train queries via OpenAI Batch API")
    p.add_argument("--emb_model",     default="text-embedding-ada-002")
    p.add_argument("--poll_interval", type=int, default=30,
                   help="Seconds between status polls")
    p.add_argument("--batch_id",      default=None,
                   help="Resume an already-submitted batch (skip upload/submit)")
    return p.parse_args()


# ── Paths ─────────────────────────────────────────────────────────────────────

def get_paths(emb_model: str):
    base    = Path(f"emb/prime/{emb_model}/query")
    scratch = base / "batch_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return {
        "out":          base / "prime_train_embeddings.pt",
        "jsonl":        scratch / "requests.jsonl",
        "results":      scratch / "results.jsonl",
        "batch_id_txt": scratch / "batch_id.txt",
        "failed":       scratch / "failed.json",
    }


# ── Stage 1: load queries ─────────────────────────────────────────────────────

def load_train_queries() -> dict:
    print("[1/5] Loading PRIME train queries ...")
    qa = load_qa("prime", human_generated_eval=False)
    queries = {}
    for item in qa.get_subset("train"):
        text, qid, *_ = item
        queries[int(qid)] = str(text)
    print(f"      {len(queries):,} queries loaded.")
    return queries


# ── Stage 2: build JSONL ──────────────────────────────────────────────────────

def build_jsonl(queries: dict, emb_model: str, path: Path) -> None:
    print(f"[2/5] Writing {len(queries):,} requests to {path} ...")
    with path.open("w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({
                "custom_id": f"qid-{qid}",
                "method":    "POST",
                "url":       "/v1/embeddings",
                "body": {
                    "input":           text,
                    "model":           emb_model,
                    "encoding_format": "float",
                },
            }) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"      Written: {size_mb:.1f} MB")


# ── Stage 3: upload + submit ──────────────────────────────────────────────────

def submit(client: OpenAI, jsonl_path: Path, batch_id_txt: Path) -> str:
    print("[3/5] Uploading JSONL and submitting batch job ...")
    with jsonl_path.open("rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"      File uploaded : {upload.id}")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    batch_id_txt.write_text(batch.id)
    print(f"      Batch created : {batch.id}  status={batch.status}")
    print(f"      [tip] To resume: python {Path(__file__).name} --batch_id {batch.id}")
    return batch.id


# ── Stage 4: poll ─────────────────────────────────────────────────────────────

def poll(client: OpenAI, batch_id: str, interval: int) -> str:
    print(f"\n[4/5] Polling batch {batch_id} every {interval}s ...")
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"      [{time.strftime('%H:%M:%S')}] status={b.status:<12} "
              f"completed={c.completed}/{c.total}  failed={c.failed}", flush=True)
        if b.status in terminal:
            break
        time.sleep(interval)

    if b.status != "completed":
        sys.exit(f"\nBatch ended with status '{b.status}'. error_file={b.error_file_id}")

    print(f"      Batch complete. output_file={b.output_file_id}")
    return b.output_file_id


# ── Stage 5: download + parse + save ─────────────────────────────────────────

def save_results(client: OpenAI, file_id: str,
                 results_path: Path, out_path: Path, failed_path: Path) -> None:
    print(f"\n[5/5] Downloading results ...")
    data = client.files.content(file_id)
    results_path.write_bytes(data.content)
    print(f"      Downloaded: {results_path.stat().st_size / 1e6:.1f} MB")

    print("      Parsing embeddings ...")
    emb_dict = {}
    failures = []

    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec   = json.loads(line)
            qid   = int(rec["custom_id"].removeprefix("qid-"))
            error = rec.get("error")
            resp  = rec.get("response", {})

            if error or resp.get("status_code") != 200:
                failures.append({"qid": qid, "error": error,
                                  "status": resp.get("status_code")})
                continue

            vec = resp["body"]["data"][0]["embedding"]
            emb_dict[qid] = torch.tensor(vec, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, out_path)

    print(f"\n{'='*60}")
    print(f"  Embeddings saved : {len(emb_dict):,}")
    print(f"  Failed           : {len(failures)}")
    print(f"  Output file      : {out_path}")
    print(f"{'='*60}")

    if failures:
        failed_path.write_text(json.dumps(failures, indent=2))
        print(f"  Failed entries   -> {failed_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)
    paths  = get_paths(args.emb_model)

    # Guard: already done?
    if paths["out"].exists():
        existing = torch.load(paths["out"], map_location="cpu")
        print(f"Output already exists with {len(existing):,} embeddings: {paths['out']}")
        print("Delete the file or use a different --emb_model to re-run.")
        return

    batch_id = args.batch_id

    if not batch_id:
        queries  = load_train_queries()
        build_jsonl(queries, args.emb_model, paths["jsonl"])
        batch_id = submit(client, paths["jsonl"], paths["batch_id_txt"])
    else:
        print(f"[1-3/5] Skipping submit -- resuming batch {batch_id}")

    output_file_id = poll(client, batch_id, args.poll_interval)
    save_results(client, output_file_id,
                 paths["results"], paths["out"], paths["failed"])


if __name__ == "__main__":
    main()

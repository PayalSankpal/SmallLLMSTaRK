import argparse
import textwrap
from typing import List, Tuple

import cohere

try:
    from reranking_script_cohere import API_KEYS as DEFAULT_KEYS
except Exception:  # pragma: no cover - fallback when module unavailable
    DEFAULT_KEYS = []


def load_keys(explicit_keys: List[str], key_file: str) -> List[str]:
    keys: List[str] = []

    if key_file:
        with open(key_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    keys.append(stripped)

    if explicit_keys:
        keys.extend(explicit_keys)

    if not keys and DEFAULT_KEYS:
        keys.extend(DEFAULT_KEYS)

    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for key in keys:
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)
    return unique_keys


def check_key(key: str, model: str, query: str, documents: List[str], top_n: int, timeout: float) -> Tuple[bool, str]:
    try:
        client = cohere.ClientV2(key, timeout=timeout)
        response = client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            max_tokens_per_doc=2048,
        )
        scored = len(response.results) if hasattr(response, "results") else 0
        return True, f"Success ({scored} docs scored)"
    except Exception as exc:  # pragma: no cover - network errors are diverse
        return False, str(exc)


def format_key(key: str, reveal: bool) -> str:
    if reveal:
        return key
    if len(key) <= 8:
        return key
    return f"{key[:4]}...{key[-4:]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly verify which Cohere API keys are currently valid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Key resolution order:
              1. --keys values (highest priority)
              2. --key-file entries (one key per line)
              3. API_KEYS imported from reranking_script_cohere.py
            """
        ),
    )
    parser.add_argument("--keys", nargs="*", help="Explicit Cohere API keys (space separated)")
    parser.add_argument("--key-file", help="Path to a text file with one Cohere key per line")
    parser.add_argument("--model", default="rerank-v3.5", help="Cohere model to ping (default: rerank-v3.5)")
    parser.add_argument("--query", default="Test rerank query", help="Query text sent to Cohere")
    parser.add_argument(
        "--documents",
        nargs="*",
        default=[
            "Document about targeted cancer therapies and gene signatures.",
            "A short abstract describing hematological disorders.",
            "Random filler text for Cohere rerank smoke test.",
        ],
        help="Sample documents passed to the rerank endpoint.",
    )
    parser.add_argument("--top-n", type=int, default=3, help="Number of documents to request from rerank")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout per request (seconds)")
    parser.add_argument("--show-full", action="store_true", help="Print full keys instead of masked values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = load_keys(args.keys, args.key_file)

    if not keys:
        raise SystemExit("No API keys provided. Use --keys, --key-file, or define API_KEYS in reranking_script_cohere.py")

    print(f"Checking {len(keys)} Cohere key(s) against model {args.model}...")

    for key in keys:
        ok, message = check_key(
            key=key,
            model=args.model,
            query=args.query,
            documents=args.documents,
            top_n=args.top_n,
            timeout=args.timeout,
        )
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {format_key(key, args.show_full)} -> {message}")

    print("Done.")


if __name__ == "__main__":
    main()

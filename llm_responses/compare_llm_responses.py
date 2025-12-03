import csv
import sys
import argparse

def load_and_sort_csv(path):
    csv.field_size_limit(sys.maxsize)

    required_columns = ["id", "query", "entities", "relations"]

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        missing = [c for c in required_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        rows = [
            {col: row.get(col, "") for col in required_columns}
            for row in reader
        ]

    # Sort by query
    return sorted(rows, key=lambda r: r["query"])


def diff_rows(rows1, rows2):
    diffs = []

    max_len = max(len(rows1), len(rows2))
    for i in range(max_len):
        r1 = rows1[i] if i < len(rows1) else None
        r2 = rows2[i] if i < len(rows2) else None

        if r1 != r2:
            diffs.append((r1, r2))

    return diffs


def print_diff(differences):
    if not differences:
        print("✔ No differences found.")
        return

    print("\n❌ Differences found:\n")
    for i, (r1, r2) in enumerate(differences, start=1):
        print(f"--- Difference #{i} ---")
        print("FILE 1:", r1)
        print("FILE 2:", r2)
        print()


def main():
    parser = argparse.ArgumentParser(description="Diff two llm_response_dataset CSV files.")
    parser.add_argument("file1", help="First CSV file")
    parser.add_argument("file2", help="Second CSV file")
    args = parser.parse_args()

    rows1 = load_and_sort_csv(args.file1)
    rows2 = load_and_sort_csv(args.file2)

    differences = diff_rows(rows1, rows2)
    print_diff(differences)


if __name__ == "__main__":
    main()


import csv
import argparse
import sys
csv.field_size_limit(sys.maxsize)

def main():
    parser = argparse.ArgumentParser(description="Extract specific columns.")
    parser.add_argument("input", help="Path to input CSV file (positional).")
    parser.add_argument("--dataset", default="prime", help="Dataset name (default: mag)")
    args = parser.parse_args()

    input_file = args.input
    output_file = f"llm_response_dataset_{args.dataset}.csv"

    # FINAL output columns you want
    required_columns = ["id", "query", "entities", "relations"]

    with open(input_file, "r", newline="", encoding="utf-8") as infile, \
         open(output_file, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)

        # Ensure columns exist
        missing = [c for c in required_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        writer = csv.DictWriter(outfile, fieldnames=required_columns)
        writer.writeheader()

        for row in reader:
            writer.writerow({col: row.get(col, "") for col in required_columns})

    print(f"✔ Output saved to: {output_file}")

if __name__ == "__main__":
    main()

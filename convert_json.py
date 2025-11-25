import pandas as pd
import ast
import json
import argparse

def convert_csv_to_json(input_csv, output_json):
    # Read CSV
    df = pd.read_csv(input_csv)

    # Dictionary to store the output
    output = {}

    for _, row in df.iterrows():
        qid = str(row['q_id'])                  # convert to string for JSON keys
        top20_str = row['top_20_vss_array']     # list stored as string

        # Parse list safely
        try:
            top20_list = ast.literal_eval(top20_str)
        except Exception:
            top20_list = []

        output[qid] = top20_list

    # Save output JSON
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Saved JSON to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    convert_csv_to_json(args.input, args.output)


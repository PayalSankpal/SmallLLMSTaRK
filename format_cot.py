import pandas as pd
import ast
import json

df = pd.read_csv('teacher_cot_evaluations.csv')

def parse_reasoning(text):
    if not isinstance(text, str): return "", []
    try:
        reasoning = text.split('<reasoning>')[1].split('</reasoning>')[0].strip()
        ranking_str = text.split('<ranking>')[1].split('</ranking>')[0].strip()
        ranking = ast.literal_eval(ranking_str)
        return reasoning, ranking
    except:
        return text, []

print(f"Loaded {len(df)} teacher evaluations.")
for i, row in df.iterrows():
    r, rank = parse_reasoning(row['teacher_response'])
    print(f"\n--- QUERY: {row['query']} ---\n")
    print("REASONING SNEAK PEEK:", r[:200], "...")
    print("RANKING EXTRACTED:", rank[:5], "...")
    print("GROUND TRUTH:", row['ground_truths'])

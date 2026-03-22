import pandas as pd
import ast
import traceback

df = pd.read_csv('experiments/prime/8_TRAIN_DIVERSE/full_data_dump.csv')
print(f"Loaded {len(df)} rows.")

for idx, row in df.iterrows():
    qid = row['id']
    query = row['query']
    gt = ast.literal_eval(row['ground_truths'])
    cands = []
    if 'vss_merged_candidates' in row and not pd.isna(row['vss_merged_candidates']):
        try:
            cands = ast.literal_eval(row['vss_merged_candidates'])[:20]
        except:
            pass
    print(f"[{qid}] GT: {gt} | Candidates: {len(cands)}")

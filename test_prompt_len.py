import pandas as pd
import ast
from stark_qa import load_skb

kb = load_skb('prime', download_processed=True)
df = pd.read_csv('experiments/prime/8_TRAIN_DIVERSE/full_data_dump.csv')

for idx, row in df.iterrows():
    cands = ast.literal_eval(row['vss_merged_candidates'])[:20]
    docs_info = []
    for d in cands:
        d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
        n_type = str(kb.get_node_type_by_id(d))
        docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")

    prompt = chr(10).join(docs_info)
    print(f"Query {row['id']} Prompt Length (chars): {len(prompt)}")

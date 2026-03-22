import tiktoken
import pandas as pd
import ast
from stark_qa import load_skb

print("Loading KB...")
kb = load_skb('prime', download_processed=True)

df = pd.read_csv("experiments/prime/LLM_SAVED_RESPONSES/full_data_dump.csv")
sample_row = df.iloc[0]
sample_query = sample_row['query']
docs_str = sample_row['results']
try:
    docs = ast.literal_eval(docs_str).get('answer_list', [])[:20]
except:
    docs = []

if len(docs) >= 2:
    doc1, doc2 = docs[0], docs[1]
    d1_info = kb.get_doc_info(doc1, add_rel=True, compact=True)
    d2_info = kb.get_doc_info(doc2, add_rel=True, compact=True)
    n1_type = kb.get_node_type_by_id(doc1)
    n2_type = kb.get_node_type_by_id(doc2)
    
    enc = tiktoken.get_encoding("o200k_base") # standard encoding for gpt-4* models
    
    doc1_str = f"{doc1}, {n1_type}, {d1_info}."
    doc1_tokens = len(enc.encode(doc1_str))
    
    full_prompt = (
        f"The following two elements consist of an ID number, a type and a corresponding descriptive text:\n \n"
        f"{doc1_str} \n"
        f"{doc2}, {n2_type}, {d2_info}. \n\n"
        f"Find out which of the elements satisfies the following query better: \n"
        f"{sample_query} \n"
        f"Return ONLY the corresponding ID number which corresponds to the element that satisfies "
        f"the given query best. Nothing else."
    )
    full_tokens = len(enc.encode(full_prompt))
    
    print("\n--- STATISTICS ---")
    print(f"Characters in Document 1: {len(doc1_str):,}")
    print(f"Tokens for strictly ONE Document (ID + Type + Info): {doc1_tokens:,}")
    print(f"Tokens for a FULL Single Comparison Query (2 Docs + Instruction + Query): {full_tokens:,}")
else:
    print("Could not parse enough docs.")

import os
import ast
import json

def update_file():
    with open('reranking/rankgpt_nvidia.py', 'r') as f:
        content = f.read()

    # Find where to put the generation logic. Maybe right after `kb = load_skb('prime', download_processed=True)`
    
    new_logic = """kb = load_skb('prime', download_processed=True)

print("Building Few-Shot Prompt from teacher_cot_evaluations.csv...")
try:
    df_train = pd.read_csv('../teacher_cot_evaluations.csv')
    few_shot_examples = ""
    for idx, row in df_train.iterrows():
        try:
            resp = row['teacher_response']
            if not isinstance(resp, str): continue
            
            reasoning = resp.split('<reasoning>')[1].split('</reasoning>')[0].strip()
            ranking_str = resp.split('<ranking>')[1].split('</ranking>')[0].strip()
            
            candidates = ast.literal_eval(row['candidates'])
            
            docs_info = []
            for d in candidates:
                d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
                n_type = str(kb.get_node_type_by_id(d))
                docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")
            docs_str = '\\n'.join(docs_info)
            
            example = f\"\"\"
Example {idx+1}:
Query: {row['query']}

Candidate Documents:
{docs_str}

Response:
<reasoning>
{reasoning}
</reasoning>
<ranking>
{ranking_str}
</ranking>
\"\"\"
            few_shot_examples += example + "\\n"
        except Exception as e:
            continue
    print("Few-Shot Prompt built successfully.")
except Exception as e:
    print("Could not build few-shot prompt:", e)
    few_shot_examples = ""

DATA_DUMP_PATH
"""
    
    content = content.replace("kb = load_skb('prime', download_processed=True)\nDATA_DUMP_PATH", new_logic)

    # Let's replace the prompt inside `rerank_sublist`
    old_prompt = """        
    prompt = f\"\"\"You are an advanced search relevance ranker. Below is a search 
query and a list of candidate documents.                                        Please rank the candidate documents by their relevance to the search query, from
 the most relevant to the least relevant.                                       
Query: {query_text}

Candidate Documents:
{docs_str}

Output your response strictly as a JSON array containing ONLY the document IDs i
n the sorted order. Do not include any explanations, markdown syntax, or other text.                                                                            Example format:
[1234, 5678, 91011]
\"\"\""""
    new_prompt = """
    prompt = f\"\"\"You are an advanced search relevance ranker. Below are some examples followed by a new search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.
For your response to the final query, output your response in the same format: first provide a <reasoning> block with your reasoning, then provide a <ranking> block containing ONLY a JSON array of the document IDs in the sorted order.

{few_shot_examples}

=== NEW QUERY TO RANK ===
Query: {query_text}

Candidate Documents:
{docs_str}

Response:
\"\"\"
"""
    # Just in case whitespace issues, we will replace differently
    pass

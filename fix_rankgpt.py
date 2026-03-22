import re
with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "r") as f:
    code = f.read()

# Make sure we truncate nodes intelligently just in case
new_func = """def rerank_sublist(query_text, docs_list, kb, retries=3):
    docs_info = []
    for d in docs_list:
        d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
        # Hard truncate the node info to 3000 chars natively just to be completely safe
        if len(d_info) > 3000:
            d_info = d_info[:3000] + "... [TRUNCATED]"
        n_type = str(kb.get_node_type_by_id(d))
        docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")
        
    docs_str = "\\n".join(docs_info)
    
    prompt = f\"\"\"You are an advanced search relevance ranker. Below is a search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.

Query: {query_text}

Candidate Documents:
{docs_str}

Output your response strictly as a JSON array containing ONLY the document IDs in the sorted order. Do not include any explanations, markdown syntax, or other text.
Example format:
[1234, 5678, 91011]
\"\"\"
"""
code = re.sub(r'def rerank_sublist\(.*?\"\"\"\n', new_func, code, flags=re.DOTALL)

with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "w") as f:
    f.write(code)

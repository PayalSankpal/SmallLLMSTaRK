import os

with open('reranking/rankgpt_nvidia.py', 'r') as f:
    lines = f.readlines()

out_lines = []
in_prompt = False
in_api = False
for line in lines:
    if "kb = load_skb('prime', download_processed=True)" in line:
        out_lines.append(line)
        out_lines.append("""
print("Building Few-Shot Prompt from teacher_cot_evaluations.csv...")
few_shot_examples = ""
try:
    import pandas as pd
    import ast
    df_train = pd.read_csv('../teacher_cot_evaluations.csv')
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
            fs_docs_str = '\\n'.join(docs_info)
            
            example = f\"\"\"
Example {idx+1}:
Query: {row['query']}

Candidate Documents:
{fs_docs_str}

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
""")
        continue
        
    if 'prompt = f"""You are an advanced' in line:
        in_prompt = True
        out_lines.append('''    prompt = f"""You are an advanced search relevance ranker. Below are some examples followed by a new search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.
For your response to the final query, output your response in the same format: first provide a <reasoning> block with your reasoning, then provide a <ranking> block containing ONLY a JSON array of the document IDs in the sorted order.

{few_shot_examples}

=== NEW QUERY TO RANK ===
Query: {query_text}

Candidate Documents:
{docs_str}

Response:
"""\n''')
        continue
        
    if in_prompt:
        if '"""' in line and 'Example format:' not in line: # End of prompt
            # Actually, `"""` might appear. It's better to just skip until `for attempt in range(retries):`
            pass
        if 'for attempt in range(retries):' in line:
            in_prompt = False
            out_lines.append(line)
        continue

    if '            response = client.chat.completions.create(' in line:
        in_api = True
        out_lines.append('''            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            
            ranking_str = raw
            try:
                if '<ranking>' in raw and '</ranking>' in raw:
                    ranking_str = raw.split('<ranking>')[1].split('</ranking>')[0].strip()
            except Exception:
                pass
                
            # Clean possible markdown
            if ranking_str.startswith("```json"): ranking_str = ranking_str[7:]
            if ranking_str.startswith("```"): ranking_str = ranking_str[3:]
            if ranking_str.endswith("```"): ranking_str = ranking_str[:-3]
            ranking_str = ranking_str.strip()
            
            parsed = json.loads(ranking_str)\n''')
        continue
        
    if in_api:
        if '            if isinstance(parsed, list):' in line:
            in_api = False
            out_lines.append(line)
        continue

    out_lines.append(line)

with open('reranking/rankgpt_nvidia.py', 'w') as f:
    f.writelines(out_lines)

print("Done with iterative replacement!")

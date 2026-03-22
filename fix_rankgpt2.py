import re

with open('reranking/rankgpt_nvidia.py', 'r') as f:
    original_code = f.read()

# 1. Add few-shot loading
insertion_point = "kb = load_skb('prime', download_processed=True)"
new_code = f"""{insertion_point}

print("Building Few-Shot Prompt from teacher_cot_evaluations.csv...")
few_shot_examples = ""
try:
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
                docs_info.append(f"[{d}] Type: {{n_type}} | Info: {{d_info}}")
            fs_docs_str = '\\n'.join(docs_info)
            
            example = f\"\"\"
Example {{idx+1}}:
Query: {{row['query']}}

Candidate Documents:
{{fs_docs_str}}

Response:
<reasoning>
{{reasoning}}
</reasoning>
<ranking>
{{ranking_str}}
</ranking>
\"\"\"
            few_shot_examples += example + "\\n"
        except Exception as e:
            continue
    print("Few-Shot Prompt built successfully.")
except Exception as e:
    print("Could not build few-shot prompt:", e)
"""
modified_code = original_code.replace(insertion_point, new_code)

# 2. Modify prompt and API call settings
old_prompt_section = '''    prompt = f"""You are an advanced search relevance ranker. Below is a search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.

Query: {query_text}

Candidate Documents:
{docs_str}

Output your response strictly as a JSON array containing ONLY the document IDs in the sorted order. Do not include any explanations, markdown syntax, or other text.
Example format:
[1234, 5678, 91011]
"""'''

new_prompt_section = '''    prompt = f"""You are an advanced search relevance ranker. Below are some examples followed by a new search query and a list of candidate documents.
Please rank the candidate documents by their relevance to the search query, from the most relevant to the least relevant.
For your response to the final query, output your response in the same format: first provide a <reasoning> block with your reasoning, then provide a <ranking> block containing ONLY a JSON array of the document IDs in the sorted order.
{few_shot_examples}
=== NEW QUERY TO RANK ===
Query: {query_text}

Candidate Documents:
{docs_str}

Response:
"""'''
modified_code = modified_code.replace(old_prompt_section, new_prompt_section)

# 3. Modify generation parameters and parsing logic
old_api_call = '''            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            
            # Clean possible markdown
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]
            raw = raw.strip()
            
            parsed = json.loads(raw)'''

new_api_call = '''            response = client.chat.completions.create(
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
            
            parsed = json.loads(ranking_str)'''
modified_code = modified_code.replace(old_api_call, new_api_call)

with open('reranking/rankgpt_nvidia.py', 'w') as f:
    f.write(modified_code)

print("Replacement complete.")

import json

with open("/raid/adityasd314/BTechProject/reranking/batched_reranking.ipynb", "r") as f:
    nb = json.load(f)

NEW_FUNC = """import tiktoken

def submit_and_poll_batch(client, jsonl_path, batch_txt, interval=15):
    print(f"\\n-- Uploading {jsonl_path.name}...")
    with jsonl_path.open("rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    
    while True:
        try:
            b = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window="24h")
            batch_txt.write_text(b.id)
            print(f"-- Batch created: {b.id}. Polling...")
        except Exception as e:
            if "token_limit" in str(e).lower():
                print(f"   [RATE LIMIT] Enqueue limit hit on create. Sleeping 60s...")
                time.sleep(60)
                continue
            raise e
            
        terms = {"completed", "failed", "expired", "cancelled"}
        while True:
            curr = client.batches.retrieve(b.id)
            c = curr.request_counts
            done, tot, fail = (c.completed, c.total, c.failed) if c else (0,0,0)
            print(f"   [{time.strftime('%H:%M:%S')}] {curr.status} | {done}/{tot} (Failed {fail})", end='\\r')
            if curr.status in terms: 
                print() 
                break
            time.sleep(interval)
            
        if curr.status == 'failed':
            err_msg = str(curr.errors).lower()
            if 'token_limit_exceeded' in err_msg or 'limit' in err_msg:
                print(f"   [RATE LIMIT] Batch failed due to enqueued token limit. Sleeping 60s and retrying...")
                time.sleep(60)
                continue # Retry creating the batch
            else:
                raise RuntimeError(f"Batch Failed. Errors: {curr.errors}")
        elif curr.status != 'completed':
            raise RuntimeError(f"Batch ended with status: {curr.status}")
            
        return curr.output_file_id

def map_winner_from_response(ans, node1_id, node2_id):
    try: ans = ans.replace("'", "").replace('"', "").strip() 
    except: pass
    
    if "A" in ans or str(node1_id) in ans: return node1_id
    if "B" in ans or str(node2_id) in ans: return node2_id
    return node1_id # default to pivot tie if invalid

def run_tournament_sort_experiment(queries, experiment_name="batch_reranking_sort"):
    base = Path(f"../experiments/{experiment_name}")
    base.mkdir(parents=True, exist_ok=True)
    
    print("Loading token encoder for accurate batch estimation...")
    try: enc = tiktoken.encoding_for_model(MODEL)
    except: enc = tiktoken.get_encoding("o200k_base")
    
    states = [QuerySortState(q['id'], q['query'], q['top_20_nodes']) for q in queries]
    round_id = 0
    
    while True:
        round_id += 1
        comps = []
        for s in states:
            comps.extend(s.get_pending_comparisons())
            
        if not comps:
            print("All queries fully sorted!")
            break
            
        # Target 2M limit. Use 1.8M to be very safe and ensure we never exceed queue sizes.
        MAX_BATCH_TOKENS = 1_800_000
        comp_chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        print(f"\\n>>> ROUND {round_id} - Dynamically packing {len(comps)} comparisons within {MAX_BATCH_TOKENS:,} tokens/batch...")
        
        # Prepare and pack dynamically based on literal token sizes
        for qid, query, doc1, doc2 in comps:
            d1_info = kb.get_doc_info(doc1, add_rel=True, compact=True)
            d2_info = kb.get_doc_info(doc2, add_rel=True, compact=True)
            n1_type = kb.get_node_type_by_id(doc1)
            n2_type = kb.get_node_type_by_id(doc2)
            
            prompt = (
                f"The following two elements consist of an ID number, a type and a corresponding descriptive text:\\n \\n"
                f"{doc1}, {n1_type}, {d1_info}. \\n"
                f"{doc2}, {n2_type}, {d2_info}. \\n\\n"
                f"Find out which of the elements satisfies the following query better: \\n"
                f"{query} \\n"
                f"Return ONLY the corresponding ID number which corresponds to the element that satisfies "
                f"the given query best. Nothing else."
            )
            req = {
                "custom_id": f"q_{qid}__p_{doc1}__x_{doc2}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 10}
            }
            
            req_str = json.dumps(req)
            req_tokens = len(enc.encode(req_str))
            
            if current_chunk_tokens + req_tokens > MAX_BATCH_TOKENS and current_chunk:
                comp_chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = 0
                
            current_chunk.append(req_str)
            current_chunk_tokens += req_tokens
            
        if current_chunk:
            comp_chunks.append(current_chunk)
            
        results_dict = {}
        for part_idx, comp_chunk in enumerate(comp_chunks):
            suffix = f"_part_{part_idx+1}" if len(comp_chunks) > 1 else ""
            req_path = base / f"round_{round_id}{suffix}_req.jsonl"
            res_path = base / f"round_{round_id}{suffix}_res.jsonl"
            batch_txt = base / f"round_{round_id}{suffix}.txt"
            
            if len(comp_chunks) > 1:
                print(f"\\n  --- Submitting part {part_idx+1}/{len(comp_chunks)} ({len(comp_chunk)} queries) ---")
            
            with open(req_path, 'w', encoding='utf-8') as f:
                for req_str in comp_chunk:
                    f.write(req_str + "\\n")
                    
            out_id = submit_and_poll_batch(client, req_path, batch_txt)
            client.files.content(out_id).write_to_file(res_path)
            
            with open(res_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    cid = data["custom_id"]
                    parts = cid.replace("q_", "").split("__")
                    qid = int(parts[0])
                    d1 = int(parts[1].replace("p_",""))
                    d2 = int(parts[2].replace("x_",""))
                    
                    try:
                        ans = data["response"]["body"]["choices"][0]["message"]["content"]
                        winner = map_winner_from_response(ans, d1, d2)
                    except:
                        winner = d1
                    results_dict[(qid, d1, d2)] = winner
                    
        # Apply Results to the State
        for s in states:
            s.apply_results(results_dict)
            
    # Wrap up outputs
    final_queries = []
    state_dict = {s.qid: s.get_sorted_list() for s in states}
    for q in queries:
        output = q.copy()
        output['reranked'] = state_dict[q['id']]
        final_queries.append(output)
        
    return final_queries"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def run_tournament_sort_experiment" in source and "def submit_and_poll_batch" in source:
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in NEW_FUNC.split("\n")]
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"].pop()
                if cell["source"]: cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("/raid/adityasd314/BTechProject/reranking/batched_reranking.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook updated! Batches will now strictly pack up to token boundaries.")

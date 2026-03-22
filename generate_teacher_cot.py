import asyncio
import pandas as pd
import ast
from stark_qa import load_skb
from copilot import CopilotClient, PermissionHandler
import time

print("Loading Knowledge Base...")
kb = load_skb('prime', download_processed=True)
df = pd.read_csv('experiments/prime/8_TRAIN_DIVERSE/full_data_dump.csv')

async def ask_model(client, prompt):
    session = await client.create_session(
        model="claude-haiku-4.5",
        on_permission_request=PermissionHandler.approve_all,
    )
    
    done = asyncio.Event()
    response_text = ""
    
    def on_event(event):
        nonlocal response_text
        event_type = getattr(event.type, 'value', event.type) if hasattr(event, 'type') else None
        
        if event_type == "assistant.message":
            response_text = event.data.content
            done.set()
        elif event_type == "session.idle":
            done.set()
        elif event_type == "error":
            print(f"Error:", event)
            done.set()

    session.on(on_event)
    await session.send(prompt)
    
    try:
        await asyncio.wait_for(done.wait(), timeout=60.0)
    except asyncio.TimeoutError:
        print("Timeout waiting for response!")
        
    await session.disconnect()
    return response_text

async def main():
    print("Starting Copilot Client...", flush=True)
    client = CopilotClient()
    await client.start()
    
    results = []

    for idx, row in df.iterrows():
        qid = row['id']
        query_text = row['query']
        gt = ast.literal_eval(row['ground_truths'])
        try:
            cands = ast.literal_eval(row['vss_merged_candidates'])[:20]
        except:
            continue
            
        print(f"\n[{idx+1}/{len(df)}] Processing Query {qid} with {len(cands)} candidates...")
        
        docs_info = []
        for d in cands:
            d_info = str(kb.get_doc_info(d, add_rel=False, compact=True))
            n_type = str(kb.get_node_type_by_id(d))
            docs_info.append(f"[{d}] Type: {n_type} | Info: {d_info}")

        prompt = f"""You are an expert medical knowledge ranking system.

Below is a search query and a list of candidate documents.
Please analyze the candidates, identify the ones that perfectly answer the query, and provide your step-by-step reasoning for ranking them. 
Focus your reasoning on why particular candidates match the query requirements precisely.
After your reasoning, output a strictly formatted list containing the exact ranked order of document IDs.

Here is the context:
Query: {query_text}

Candidate Documents:
{chr(10).join(docs_info)}

Please structure your response exactly like this template:
<reasoning>
Your detailed rationale here. Evaluate the query constraints against the provided candidates.
</reasoning>
<ranking>
[123, 456, 789]
</ranking>
"""
        
        resp = await ask_model(client, prompt)
        print(f"Response: {resp[:150]}...")
        
        results.append({
            'qid': qid,
            'query': query_text,
            'ground_truths': gt,
            'candidates': cands,
            'teacher_response': resp
        })
        
        # Keep limits in check
        await asyncio.sleep(2)
        
    out_df = pd.DataFrame(results)
    out_df.to_csv('teacher_cot_evaluations.csv', index=False)
    print("\nSaved Teacher CoT queries to teacher_cot_evaluations.csv")

    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())

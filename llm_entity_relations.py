import argparse
from stark_qa import load_qa
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv(override=True)

from custom_pipeline.llm_bridge import LlmBridge
from custom_pipeline.query import Query
from custom_pipeline.entity_parsing import *
from custom_pipeline.relation_parsing import *
from custom_pipeline.prompt_generator import get_entity_extraction_prompt, get_relation_extraction_prompt

# Argument parser setup
parser = argparse.ArgumentParser(description="Run LLM entity/relation extraction for specified dataset.")
parser.add_argument('--dataset', type=str, default='mag', help="Dataset name, e.g., 'mag'")
parser.add_argument('--data_split', type=str, default='test', help="Data split, e.g., 'test'")
parser.add_argument('--model_name', type=str, default='meta/llama-3.3-70b-instruct', help="Model name")
args = parser.parse_args()

# Step 1 : getting Entities from LLM
def generate_entity_identification_prompt(query_str):
    return get_entity_extraction_prompt(query_str, args.dataset)

def get_llm_response(prompt, llm_bridge):
    response = llm_bridge.ask_llm_batch([prompt])
    return response[0][0]

def step1_identify_entities(query, llm_bridge):
    prompt = generate_entity_identification_prompt(query.query)
    response_string = get_llm_response(prompt, llm_bridge)
    query.entity_id_response = response_string
    if response_string == '':
        query.status = "FAILED"
        return
    else:
        try:
            identified_entities = parse_entity_response(response_string)
            query.entities = identified_entities
        except ValueError as e:
            query.status = "FAILED"
            print(f"Error parsing response for query {e}")

# Step 2 : identifying the relations between entities
def generate_relation_identification_prompt(query_string, entities_string):
    return get_relation_extraction_prompt(args.dataset, query_string, entities_string)

def step2_identify_relations(query, llm_bridge):
    prompt = generate_relation_identification_prompt(query.query, str(query.entities))
    response_string = get_llm_response(prompt, llm_bridge)
    query.relations_id_response = response_string

    if response_string == '':
        query.status = "FAILED"
        return
    else:
        try:
            identified_relations = parse_relation_string(response_string)
            query.relations = identified_relations
        except ValueError as e:
            query.status = "FAILED"
            print(f"Error parsing response for query {e}")

# Main
def main():
    qa_dataset = load_qa(args.dataset)
    qa = qa_dataset.split_indices[args.data_split].reshape(-1).tolist()
    qa = qa[:int(len(qa) * 0.1)]
    test_queries = [qa_dataset[i] for i in qa]

    llm_bridge = LlmBridge(model_name=args.model_name, configs_path="configs.json", verbose=False)

    rows = []
    def get_llm_responses(current_query):
        step1_identify_entities(current_query, llm_bridge)
        if current_query.status == "FAILED":
            return 0
        step2_identify_relations(current_query, llm_bridge)

    for q in tqdm(test_queries, desc="Processing Queries", unit=" query"):
        current_query = Query(id=q[1], query=q[0], ground_truths=q[2])
        get_llm_responses(current_query)

        rows.append([
            current_query.id,
            current_query.query,
            current_query.entity_id_response,
            current_query.relations_id_response
        ])

    df = pd.DataFrame(rows, columns=["id", "query", "entities", "relations"])
    df.to_csv(f"{args.dataset}_llm_reponses_new.csv", index=False)
    print(f"Saved Successfully to {args.dataset}_llm_reponses_new.csv")

if __name__ == '__main__':
    main()

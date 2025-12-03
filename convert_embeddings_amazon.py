import torch
from pathlib import Path
from stark_qa import load_skb
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--emb_model", default="text-embedding-ada-002",
                        help="Embedding model to use")
    parser.add_argument("--dataset", default="amazon")
    return parser.parse_args()

args = parse_args()
model_name = args.emb_model
dataset = args.dataset

# Load the original candidate embeddings
candidate_emb_dict = torch.load(f"../stark/emb/{dataset}/{model_name}/doc/candidate_emb_dict.pt")

# Assuming you have access to your kb object and node_ids_by_type
# If not, you'll need to determine node types from the candidate_emb_dict keys
nodes_emb_dir = Path(f"emb/{dataset}/{model_name}/nodes/")
nodes_emb_dir.mkdir(parents=True, exist_ok=True)

# Method 1: If you have kb and node_ids_by_type available
def convert_with_kb(candidate_emb_dict, kb, node_ids_by_type, nodes_emb_dir):
    node_emb_dict = {}
    
    node_type = "product"
        # Get node IDs for this type
    node_ids = node_ids_by_type[node_type]
        
        # Extract embeddings for nodes of this type
    type_embeddings = []
    for node_id in node_ids:
        if node_id in candidate_emb_dict:
            type_embeddings.append(candidate_emb_dict[node_id])
        elif str(node_id) in candidate_emb_dict:
            type_embeddings.append(candidate_emb_dict[str(node_id)])
        else:
            print(f"Warning: Node ID {node_id} not found in candidate_emb_dict")
        
        # Convert to tensor if it's a list
    if type_embeddings:
        node_emb_dict[node_type] = torch.stack(type_embeddings) if isinstance(type_embeddings[0], torch.Tensor) else torch.tensor(type_embeddings)
    else:
        node_emb_dict[node_type] = torch.empty(0)
        
        # Save to file
    node_emb_path = nodes_emb_dir / f'{node_type.replace("/", "")}_embeddings.pt'
    torch.save(node_emb_dict[node_type], node_emb_path)
        
        # Verify the assertion
    assert len(node_emb_dict[node_type]) == len(node_ids_by_type[node_type]), \
        (f"number of node embeddings ({len(node_emb_dict[node_type])}) does not match number of nodes "
        f"in the SKB ({len(node_ids_by_type[node_type])}). {node_type=}.")
    
    print(f'Converted and saved embeddings of nodes to {nodes_emb_dir}!')
    return node_emb_dict

dataset_name = f'{dataset}'
kb = load_skb(dataset_name, download_processed=True)

node_ids_by_type = {}
for node_type in kb.node_type_lst():
    # Get node IDs for this type
    node_ids_by_type[node_type] = kb.get_node_ids_by_type(node_type)
convert_with_kb(candidate_emb_dict, kb, node_ids_by_type, nodes_emb_dir)


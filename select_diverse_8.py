import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from stark_qa import load_qa
import pandas as pd

print("Loading embeddings...")
emb_dict = torch.load('emb/prime/text-embedding-ada-002/query/prime_train_embeddings.pt', map_location='cpu')

qids = list(emb_dict.keys())
embeddings = torch.stack([emb_dict[qid] for qid in qids]).numpy()

print(f"Loaded {len(embeddings)} embeddings. Running K-Means...")
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(embeddings)

print("Finding closest queries to centroids...")
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

selected_qids = [qids[idx] for idx in closest_indices]

print("Loading dataset to extract query texts...")
qa_dataset = load_qa('prime')

data = []
for qid in selected_qids:
    q = qa_dataset[qid]
    data.append({
        'qid': qid,
        'query': q[0],
        'answer_ids': q[1]
    })

df = pd.DataFrame(data)
df.to_csv('selected_8_train_queries.csv', index=False)
print("Saved selected queries to selected_8_train_queries.csv")
print(df)

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import ast

print("Loading CSV...")
df = pd.read_csv('diverse_train_samples.csv')
# qid might be int, let's just make a list of qids
qids_in_csv = df['qid'].tolist()

print("Loading embeddings...")
emb_dict = torch.load('emb/prime/text-embedding-ada-002/query/prime_train_embeddings.pt', map_location='cpu')

# filter embeddings to only those in the CSV
csv_qids = []
embeddings = []
for qid in qids_in_csv:
    if qid in emb_dict:
        csv_qids.append(qid)
        embeddings.append(emb_dict[qid])

embeddings = torch.stack(embeddings).numpy()

print(f"Loaded {len(embeddings)} embeddings from the CSV. Running K-Means...")
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(embeddings)

print("Finding closest queries to centroids...")
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

selected_qids = [csv_qids[idx] for idx in closest_indices]

selected_df = df[df['qid'].isin(selected_qids)]
selected_df.to_csv('selected_8_train_queries.csv', index=False)
print("Saved selected 8 queries:")
for _, row in selected_df.iterrows():
    print(f"- {row['qid']}: {row['query']}")


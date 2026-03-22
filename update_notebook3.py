import json

with open('reranking/batched_reranking.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['source'] and "## 6. Run on All Queries" in cell['source'][0]:
        break
    if cell['cell_type'] == 'code' and "all_queries =" in "".join(cell.get('source', [])):
        cell['source'] = [
            "# Get all queries (280 total usually for prime dataset test splits etc)\n",
            "all_queries = get_target_queries(df, target_type=\"all\")\n",
            "\n",
            "# EXPERIMENT 1: Pairwise (All)\n",
            "# final_queries, df_metrics = run_experiment(all_queries, method=\"pairwise\", experiment_name=\"pairwise_all_280\")\n",
            "\n",
            "# EXPERIMENT 2: Pointwise (All) - if implemented\n",
            "# final_queries, df_metrics = run_experiment(all_queries, method=\"pointwise\", experiment_name=\"pointwise_all_280\")\n"
        ]

with open('reranking/batched_reranking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)


import json

with open('reranking/batched_reranking.ipynb', 'r') as f:
    nb = json.load(f)

wrapper_code = [
    "def run_experiment(queries, method=\"pairwise\", experiment_name=\"experiment_1\"):\n",
    "    print(f\"\\n{'='*50}\")\n",
    "    print(f\"Starting Experiment: {experiment_name} | Method: {method.upper()} | Queries: {len(queries)}\")\n",
    "    print(f\"{'='*50}\\n\")\n",
    "    \n",
    "    if method == \"pairwise\":\n",
    "        outputs = run_tournament_sort_experiment(queries)\n",
    "    elif method == \"pointwise\":\n",
    "        # Assuming a pointwise function exists or will be added\n",
    "        if 'run_pointwise_experiment' in globals():\n",
    "            outputs = run_pointwise_experiment(queries)\n",
    "        else:\n",
    "            raise NotImplementedError(\"run_pointwise_experiment is not defined yet.\")\n",
    "    else:\n",
    "        raise ValueError(f\"Unknown method: {method}\")\n",
    "        \n",
    "    analysis_df = analyze_metrics(outputs, description=experiment_name)\n",
    "    return outputs, analysis_df\n"
]

# Insert wrapper before "## 6. Run on All Queries" markdown if it exists
insert_idx = len(nb['cells']) - 2
for i, cell in enumerate(nb['cells']):
    if cell['source'] and "## 6." in cell['source'][0]:
        insert_idx = i
        break

nb['cells'].insert(insert_idx, {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": wrapper_code
})

with open('reranking/batched_reranking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)


import json

with open('reranking/batched_reranking.ipynb', 'r') as f:
    nb = json.load(f)

# Modify the last cell to include saving functions
last_cell = nb['cells'][-1]
last_cell['source'] = [
    "import datetime\n",
    "def analyze_metrics(final_queries, save_dir=\"../experiments/batch_reranking_results\", description=\"pairwise_tournament_sort\"):\n",
    "    metrics = []\n",
    "    def mrr(lst, gts):\n",
    "        for i, v in enumerate(lst): \n",
    "            if v in gts: return 1.0/(i+1)\n",
    "        return 0.0\n",
    "        \n",
    "    for q in final_queries:\n",
    "        gts = set(q['ground_truths_list'])\n",
    "        orig, rrk = q['top_20_nodes'], q['reranked']\n",
    "        \n",
    "        metrics.append({\n",
    "            'id': q['id'],\n",
    "            'orig_mrr': mrr(orig, gts), 'new_mrr': mrr(rrk, gts),\n",
    "            'orig_h1': 1.0 if orig and orig[0] in gts else 0.0,\n",
    "            'new_h1': 1.0 if rrk and rrk[0] in gts else 0.0,\n",
    "            'orig_h5': 1.0 if set(orig[:5]) & gts else 0.0,\n",
    "            'new_h5': 1.0 if set(rrk[:5]) & gts else 0.0\n",
    "        })\n",
    "        \n",
    "    df_m = pd.DataFrame(metrics)\n",
    "    df_m['mrr_diff'] = df_m['new_mrr'] - df_m['orig_mrr']\n",
    "    \n",
    "    out_str = \"=== RERANKING RESULTS ===\\n\"\n",
    "    out_str += f\"Total Queries Run: {len(df_m)}\\n\"\n",
    "    out_str += f\"Method: {description}\\n\"\n",
    "    out_str += f\"MRR:      {df_m['orig_mrr'].mean():.3f} -> {df_m['new_mrr'].mean():.3f} (Δ {df_m['mrr_diff'].mean():+.3f})\\n\"\n",
    "    out_str += f\"Hit@1:    {df_m['orig_h1'].mean():.3f} -> {df_m['new_h1'].mean():.3f}\\n\"\n",
    "    out_str += f\"Hit@5:    {df_m['orig_h5'].mean():.3f} -> {df_m['new_h5'].mean():.3f}\\n\"\n",
    "    print(out_str)\n",
    "    \n",
    "    # Save results to folders\n",
    "    os.makedirs(save_dir, exist_ok=True)\n",
    "    ts = datetime.datetime.now().strftime('%Y%md_%H%M%S')\n",
    "    csv_path = os.path.join(save_dir, f\"detailed_metrics_{description}_{ts}.csv\")\n",
    "    txt_path = os.path.join(save_dir, f\"result_summary_{description}_{ts}.txt\")\n",
    "    \n",
    "    df_m.to_csv(csv_path, index=False)\n",
    "    with open(txt_path, 'w') as f:\n",
    "        f.write(out_str)\n",
    "    print(f\"Saved detailed metrics to: {csv_path}\")\n",
    "    print(f\"Saved summary to: {txt_path}\")\n",
    "    return df_m\n"
]

# Add a cell for the 280 queries run
nb['cells'].append({
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Run on All Queries (Batch)"
   ]
})
nb['cells'].append({
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all queries (280 total usually for prime dataset test splits etc)\n",
    "all_queries = get_target_queries(df, target_type=\"all\")\n",
    "print(f\"Running tournament sort on {len(all_queries)} queries...\")\n",
    "\n",
    "# outputs_all = run_tournament_sort_experiment(all_queries)\n",
    "# analysis_df_all = analyze_metrics(outputs_all, description=\"pairwise_tournament_all_280\")\n"
   ]
})

with open('reranking/batched_reranking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)


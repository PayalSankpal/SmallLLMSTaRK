import json

with open('reranking/batched_reranking.ipynb', 'r') as f:
    nb = json.load(f)

# Find where to insert (before test cell)
insert_idx = len(nb['cells']) - 1
for i, cell in enumerate(nb['cells']):
    if cell['source'] and "## 7. Run on Test Queries" in cell['source'][0]:
        insert_idx = i
        break

calc_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import tiktoken\n",
        "\n",
        "def estimate_token_cost(queries, method=\"pairwise\", model=\"gpt-4o-mini\"):\n",
        "    \"\"\"\n",
        "    Estimates the absolute maximum token count and cost for a batch operation.\n",
        "    \"\"\"\n",
        "    try: enc = tiktoken.encoding_for_model(model)\n",
        "    except: enc = tiktoken.get_encoding(\"o200k_base\") # Default to standard modern encoding\n",
        "    \n",
        "    total_prompt_tokens = 0\n",
        "    \n",
        "    # For Pairwise we estimate N(N-1)/2 max comparisons * token sizes\n",
        "    # even though tournament averages O(N log N), batch limits apply to what gets queued.\n",
        "    if method == \"pairwise\":\n",
        "        for q in queries:\n",
        "            docs = q['top_20_nodes']\n",
        "            if not docs: continue\n",
        "            \n",
        "            # Estimate size of average two docs for this query\n",
        "            d1 = docs[0]\n",
        "            d1_info = kb.get_doc_info(d1, add_rel=True, compact=True)\n",
        "            \n",
        "            sample_prompt = (\n",
        "                f\"The following two elements consist of an ID number, a type and a corresponding descriptive text:\\n \\n\"\n",
        "                f\"{d1}, x, {d1_info}. \\n\"\n",
        "                f\"{d1}, x, {d1_info}. \\n\\n\"\n",
        "                f\"Find out which of the elements satisfies the following query better: \\n\"\n",
        "                f\"{q['query']} \\n\"\n",
        "                f\"Return ONLY the corresponding ID number which corresponds to the element that satisfies \"\n",
        "                f\"the given query best. Nothing else.\"\n",
        "            )\n",
        "            \n",
        "            # QuickSort average depth is ~4.3 for 20 elements, so roughly ~60 pairs instead of 190. \n",
        "            # But we'll estimate upper bound for a single batch round comparing pivot vs everything (N-1 = 19).\n",
        "            expected_comparisons_first_round = len(docs) - 1\n",
        "            \n",
        "            base_tokens = len(enc.encode(sample_prompt))\n",
        "            total_prompt_tokens += (base_tokens * expected_comparisons_first_round)\n",
        "            \n",
        "    print(f\"--- Token Estimate for {method.upper()} ({len(queries)} queries) ---\")\n",
        "    print(f\"Estimated tokens per batch round: {total_prompt_tokens:,}\")\n",
        "    # Assume mini $0.15 / 1M tokens, standard gpt-4.5 is $75 / 1M\n",
        "    rate = 0.15 if 'mini' in model else 2.50  \n",
        "    print(f\"Estimated cost per round: ${(total_prompt_tokens / 1_000_000) * rate:.4f}\")\n",
        "    print(\"Note: The Batch API gives you a 50% discount on these rates.\")\n",
        "    return total_prompt_tokens\n",
        "\n",
        "test_queries_bw = get_target_queries(df, target_type=\"best_worst\", num_bad=1, num_good=1)\n",
        "estimate_token_cost(test_queries_bw, method=\"pairwise\", model=MODEL)\n"
    ]
}

nb['cells'].insert(insert_idx, calc_cell)

with open('reranking/batched_reranking.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)


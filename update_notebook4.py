import json

with open("/raid/adityasd314/BTechProject/reranking/batched_reranking.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def run_tournament_sort_experiment(queries):" in source:
            source = source.replace('def run_tournament_sort_experiment(queries):', 'def run_tournament_sort_experiment(queries, experiment_name="batch_reranking_sort"):')
            source = source.replace('base = Path("../experiments/batch_reranking_sort")', 'base = Path(f"../experiments/{experiment_name}")')
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
            # remove last spurious empty line
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"].pop()
                if cell["source"]: cell["source"][-1] = cell["source"][-1].rstrip("\n")

        if "outputs = run_tournament_sort_experiment(queries)" in source:
            source = source.replace('outputs = run_tournament_sort_experiment(queries)', 'outputs = run_tournament_sort_experiment(queries, experiment_name=experiment_name)')
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"].pop()
                if cell["source"]: cell["source"][-1] = cell["source"][-1].rstrip("\n")

        if "outputs_all = run_tournament_sort_experiment(test_queries_bw)" in source:
            source = source.replace('outputs_all = run_tournament_sort_experiment(test_queries_bw)', 'outputs_all = run_tournament_sort_experiment(test_queries_bw, experiment_name="pairwise_tournament_all_280")')
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
            if cell["source"] and cell["source"][-1] == "\n":
                cell["source"].pop()
                if cell["source"]: cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("/raid/adityasd314/BTechProject/reranking/batched_reranking.ipynb", "w") as f:
    json.dump(nb, f, indent=1)


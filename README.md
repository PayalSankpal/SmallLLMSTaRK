# StarkQA Pipeline

The primary entry point for this project is `main.py`.

## How to Run

1. Edit the configuration file located at `configs/params.json`.
2. Execute the pipeline by running:
   ```bash
   python main.py configs/params.json
   ```

## Folder Structure

- **`custom_pipeline/`**: Contains the core modules and logic for the pipeline, including entity and relation parsing, prompt generation, grounding algorithms, and the integration with language models (`llm_bridge.py`).
- **`experiments/`**: Stores the outputs from various pipeline runs. Each experiment gets its own directory containing the final dataset dumps (`full_data_dump.csv`), aggregate results, and subfolders for archived configs and execution logs.
- **`llm_responses/`**: Houses the generated datasets containing the responses obtained from the LLMs (skip repeated API calls during reruns via the batch scripts).
- **`reranking/`**: Contains specific scripts, notebooks, and algorithms (such as graph path rerankers, cross-encoder rerankers) dedicated to re-evaluating and refining the final list of candidate answers. 
- **`scripts/`**: Miscellaneous helper scripts and utilities for tasks such as batch embedding generation and metric reporting.
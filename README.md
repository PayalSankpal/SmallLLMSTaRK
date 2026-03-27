# Semi-Structured IR Pipeline (SLM Optimized)

This branch focuses on our optimized Semi-Structured Information Retrieval pipeline heavily tailored for Small Language Models (SLMs) such as Qwen 2.5 (7B), Gemma (7B), and Mistral. It adapts standard IR by introducing iterative sub-tasks to cope with the reasoning constraints of smaller models.

## Pipeline Overview
The `slm_pipeline.py` script executes the following stages to improve reliability:
1. **Fact Deconstruction**: Breaks down complex user queries into atomic factual propositions.
2. **Entity Extraction**: Recognizes and extracts named entities and constraints from the query.
3. **Iterative Relation Validation (Self-Correction)**: Uses the SLM to self-correct and strictly filter out invalid generated relationships (Schema Alignment).
4. **Knowledge Graph Grounding (Entity Disambiguation)**: Fetches top-K dense retrieval candidates and uses the SLM for final anchor validation before priority queue graph traversal.
5. **Dense VSS Fallback**: An aggressive failover mechanism that defaults to pure Vector Similarity Search when the exact strict graph mapping logic yields an empty set.

## Directory Structure
- `configs/`: Pipeline settings, LLM configurations, and hyper-parameters.
- `custom_pipeline/`: Core backend modules including entity and relation parsing, knowledge graph grounding algorithms, VSS Retrieval, and the `LlmBridge` client multiplexer.
- `output/`: Directory where executed experiment models generate their metrics (`aggregate_results.csv`) and output datasets (`full_data_dump.csv`).
- `reranking/`: Includes `reranking_notebook.ipynb` for experimenting with neural models (Cohere `rerank-v3.5`) and classical score-based/pairwise sorting techniques.
- `slm_pipeline.py`: The main runtime script for executing the standard approach.
- `ablation_pipeline.py`: A mirrored pipeline populated with explicit `--disable` flags specifically intended for rapid ablation studies.

## How to Run

Activate your Python virtual environment and run the main pipeline syntax:

```bash
python slm_pipeline.py --dataset prime --model qwen/qwen2.5-7b-instruct --exp_name my_qwen_run
```

To run a specific ablation study using the ablation pipeline, simply append the relevant toggle:

```bash
python ablation_pipeline.py \
    --dataset prime \
    --model qwen/qwen2.5-7b-instruct \
    --exp_name run_without_verification \
    --disable_anchor_verification
```

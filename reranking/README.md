# Reranking

**Target Full Data Dump Path:**  
`../experiments/prime/LLM_SAVED_RESPONSES/full_data_dump.csv`

## Overview

This folder contains the logic for re-scoring candidates obtained from the grounding steps.

You can use the script `reranking_analysis.py` to test and evaluate various rerankers directly on a target experiments folder.

`python reranking/reranking_analysis.py experiments/prime/LLM_SAVED_RESPONSES/full_data_dump.csv`
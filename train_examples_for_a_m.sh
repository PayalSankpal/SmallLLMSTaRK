#!/bin/bash

# --- Configuration ---
SPLIT="train"
CONFIG_FILE="./params/train_run_for_examples.json"
ALPHA=14

# Activate your environment if needed
# source venv/bin/activate

run_experiment() {
    local DATASET="$1"
    local EXP_NAME="${DATASET}_${SPLIT}_alpha_${ALPHA}"

    echo ">>> Running dataset: $DATASET | alpha=$ALPHA | exp=$EXP_NAME"

    python parallel_final.py "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --alpha "$ALPHA" \
        --exp_name "$EXP_NAME"

    echo ">>> Finished dataset: $DATASET"
    echo "--------------------------------------------------"
}

# ---- Run 2 datasets ----
# run_experiment "mag"
sleep 2
run_experiment "amazon"

echo "All runs completed."

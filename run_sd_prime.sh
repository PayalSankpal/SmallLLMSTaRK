#!/bin/bash

# --- Configuration ---
DATASET="prime"
SPLIT="val"
CONFIG_FILE="./params/prime_alpha_params.json"
ALPHA=14

# Score decay values from 0.7 to 1.0
SCORE_DECAYS=(0.7 0.75 0.8 0.85 0.9 0.95 1.0)

echo "=================================================="
echo "Running score_decay sweep for Dataset: $DATASET ($SPLIT)"
echo "Alpha: $ALPHA"
echo "Score decays: ${SCORE_DECAYS[*]}"
echo "=================================================="

for SCORE_DECAY in "${SCORE_DECAYS[@]}"; do
    
    EXP_NAME="${DATASET}_${SPLIT}_alpha_${ALPHA}_decay_${SCORE_DECAY}"

    echo ""
    echo ">>> Running: score_decay=$SCORE_DECAY | exp=$EXP_NAME"

    python parallel_pipeline.py "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --alpha "$ALPHA" \
        --score_decay "$SCORE_DECAY" \
        --exp_name "$EXP_NAME"

    echo ">>> Finished score_decay=$SCORE_DECAY"
    echo "--------------------------------------------------"

    sleep 2
done

echo "All runs completed."

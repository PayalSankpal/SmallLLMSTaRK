#!/bin/bash

# --- Configuration ---
DATASET="mag"
SPLIT="val"       # Change to 'test' if needed later
CONFIG_FILE="./params/mag_alpha_params.json"

# Alpha values: 5, then 10 through 18
ALPHAS=({1..20})

# Activate your environment if needed
# source venv/bin/activate 

echo "=================================================="
echo "Starting Grid Search for Dataset: $DATASET ($SPLIT)"
echo "Alphas to run: ${ALPHAS[*]}"
echo "=================================================="

for ALPHA in "${ALPHAS[@]}"; do
    # Create a unique experiment name for this run
    EXP_NAME="${DATASET}_${SPLIT}_alpha_${ALPHA}"
    
    echo ""
    echo ">>> Running Alpha: $ALPHA | Exp Name: $EXP_NAME"
    
    # Run the Python script with overrides
    python parallel_final.py "$CONFIG_FILE" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --alpha "$ALPHA" \
        --exp_name "$EXP_NAME"
        
    echo ">>> Finished Alpha: $ALPHA"
    echo "--------------------------------------------------"
    
    # Optional: Sleep briefly between runs to let OS clean up ports/processes
    sleep 2
done

echo "All runs completed."

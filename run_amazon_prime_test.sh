#!/bin/bash

CONFIG_FILE_AMAZON="./params/amazon_test_params.json"
CONFIG_FILE_PRIME="./params/prime_test_params.json"

echo "=================================================="
echo "Starting Evaluation Runs"
echo "=================================================="

run_experiment() {
    local CONFIG_FILE="$1"
    local NAME="$2"

    echo ""
    echo ">>> Running experiment for: $NAME"
    echo ">>> Config: $CONFIG_FILE"
    echo "--------------------------------------------------"

    python parallel_final.py "$CONFIG_FILE"

    echo ">>> Finished: $NAME"
    echo "--------------------------------------------------"
}

# Run tests
run_experiment "$CONFIG_FILE_AMAZON" "AMAZON"
sleep 2
run_experiment "$CONFIG_FILE_PRIME" "PRIME"

echo ""
echo "=================================================="
echo "All runs completed successfully."
echo "=================================================="

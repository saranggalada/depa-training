#!/bin/bash
# Complete Pipeline for Oncology Scenario
# Runs: Preprocessing -> Feature Engineering -> Data Joining -> Training

set -e  # Exit on error

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="oncology"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data

# Feature Engineering
echo "Step 2: Running Feature Engineering..."
echo "------------------------------------------------------------"
export DATA_DIR=$DATA_DIR
mkdir -p $DATA_DIR/features

docker compose -f docker-compose-feature-engineering.yml up --remove-orphans
if [ $? -ne 0 ]; then
    echo "ERROR: Feature engineering failed!"
    exit 1
fi
echo ""

# Step 3: Verify feature engineering output
echo "Step 3: Verifying Feature Engineering Output..."
echo "------------------------------------------------------------"
if [ ! -f "$DATA_DIR/features/cell_features.csv" ]; then
    echo "ERROR: cell_features.csv not found!"
    exit 1
fi
if [ ! -f "$DATA_DIR/features/drug_response_long.csv" ]; then
    echo "ERROR: drug_response_long.csv not found!"
    exit 1
fi
echo "Feature files created successfully:"
ls -lh $DATA_DIR/features/
echo ""


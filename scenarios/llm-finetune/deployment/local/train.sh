#!/bin/bash

# Run LLM fine-tuning locally with GPU.
#
# Prerequisites (run once, by separate parties):
#   1. Modeller:       bash save-model.sh
#   2. Data providers: bash preprocess.sh
#   3. Build:          cd ../../../../ && ./ci/build.sh

set -e

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="llm-finetune"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export MARKET_DATA_INPUT_PATH=$DATA_DIR/market_data_provider
export FINTECH_INPUT_PATH=$DATA_DIR/fintech_provider
export BANK_INPUT_PATH=$DATA_DIR/bank_provider

export MODEL_INPUT_PATH=$MODEL_DIR/models
export MODEL_OUTPUT_PATH=$MODEL_DIR/output

if [ ! -d "$MODEL_INPUT_PATH" ] || [ -z "$(ls -A $MODEL_INPUT_PATH 2>/dev/null)" ]; then
    echo "ERROR: Model not found at $MODEL_INPUT_PATH"
    echo "Run 'bash save-model.sh' first (modeller step)."
    exit 1
fi

if [ ! -f "$MARKET_DATA_INPUT_PATH/financial_phrasebank.jsonl" ]; then
    echo "ERROR: Datasets not found."
    echo "Run 'bash preprocess.sh' first (data provider step)."
    exit 1
fi

rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config

# Convert YAML config to pipeline JSON
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

echo "=== Starting GPU fine-tuning ==="
docker compose -f docker-compose-train.yml up --remove-orphans

#!/bin/bash

# In a real deployment each provider runs only their own container
# on their own machine.  This script runs all three for demo convenience.
#
# Usage:
#   bash preprocess.sh                      # all samples
#   MAX_SAMPLES=50 bash preprocess.sh       # cap at 50 per TDP

set -e

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="llm-finetune"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data

export MARKET_DATA_OUTPUT_PATH=$DATA_DIR/market_data_provider
export FINTECH_OUTPUT_PATH=$DATA_DIR/fintech_provider
export BANK_OUTPUT_PATH=$DATA_DIR/bank_provider
export MAX_SAMPLES="${MAX_SAMPLES:-0}"

rm -rf $MARKET_DATA_OUTPUT_PATH $FINTECH_OUTPUT_PATH $BANK_OUTPUT_PATH
mkdir -p $MARKET_DATA_OUTPUT_PATH $FINTECH_OUTPUT_PATH $BANK_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans

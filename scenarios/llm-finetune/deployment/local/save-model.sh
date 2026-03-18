#!/bin/bash

# The model is saved to modeller/models/<local-name>/ which gets
# mounted as /mnt/remote/model/<local-name> inside the CCR.
# The YAML config's base_model.name_or_path must match.
#
# Usage:
#   bash save-model.sh                                                                      # default: phi-2
#   HF_MODEL_ID=HuggingFaceTB/SmolLM2-135M MODEL_LOCAL_NAME=smollm2-135m bash save-model.sh # CPU test model

set -e

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="llm-finetune"
export MODEL_OUTPUT_PATH=$REPO_ROOT/scenarios/$SCENARIO/modeller/models
export HF_MODEL_ID="${HF_MODEL_ID:-microsoft/phi-2}"
export MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-base_llm_hf}"

rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

echo "=== Modeller: Downloading base model ==="
echo "Model:      $HF_MODEL_ID"
echo "Local path:     $MODEL_OUTPUT_PATH/$MODEL_LOCAL_NAME"
echo ""

docker compose -f docker-compose-modelsave.yml up --remove-orphans
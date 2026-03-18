#!/bin/bash

# CPU-only sanity test for LLM fine-tuning.
# Uses a scaled-down config (fp32, tiny batch, 20 steps) so it runs
# on any machine without a GPU.  The model output will be garbage;
# this purely validates the end-to-end pipeline.

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

rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config

# convert YAML config to pipeline JSON
python3 -c "
import yaml, json

with open('$CONFIGURATION_PATH/llm_finetune_config_cpu.yaml', 'r') as f:
    llm_config = yaml.safe_load(f)

pipeline = {
    'pipeline': [
        {
            'name': 'LLM_Finetune',
            'config': llm_config
        }
    ]
}

with open('$CONFIGURATION_PATH/pipeline_config.json', 'w') as f:
    json.dump(pipeline, f, indent=2)

print('CPU pipeline config written.')
"

echo "=== Starting CPU sanity test ==="
echo "Config: llm_finetune_config_cpu.yaml"
echo "max_steps=20, batch_size=1, max_samples=50, fp32"
echo ""

docker compose -f docker-compose-train-cpu.yml up --remove-orphans

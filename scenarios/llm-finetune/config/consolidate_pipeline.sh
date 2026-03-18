#!/bin/bash

# Consolidate LLM fine-tuning config into a pipeline_config.json
# that the pytrain pipeline executor can consume.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCENARIO=llm-finetune

template_path="$REPO_ROOT/scenarios/$SCENARIO/config/templates"
llm_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/llm_finetune_config.yaml"
pipeline_config_path="$REPO_ROOT/scenarios/$SCENARIO/config/pipeline_config.json"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# Convert YAML config to JSON and wrap in pipeline format
python3 -c "
import yaml, json, sys

with open('$llm_config_path', 'r') as f:
    llm_config = yaml.safe_load(f)

pipeline = {
    'pipeline': [
        {
            'name': 'LLM_Finetune',
            'config': llm_config
        }
    ]
}

with open('$pipeline_config_path', 'w') as f:
    json.dump(pipeline, f, indent=2)

print(f'Pipeline config written to $pipeline_config_path')
"

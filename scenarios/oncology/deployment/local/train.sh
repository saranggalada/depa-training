#!/bin/bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="oncology"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export GENOMICS_LAB_INPUT_PATH=$DATA_DIR/genomics_lab/preprocessed
export PHARMACEUTICAL_COMPANY_INPUT_PATH=$DATA_DIR/pharmaceutical_company/preprocessed
export COMPUTATIONAL_BIOLOGY_LAB_INPUT_PATH=$DATA_DIR/computational_biology_lab/preprocessed
export CANCER_INSTITUTE_INPUT_PATH=$DATA_DIR/cancer_institute/preprocessed

export MODEL_OUTPUT_PATH=$MODEL_DIR/output
sudo rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config
export FEATURE_ENGINEERING_PATH=$REPO_ROOT/scenarios/$SCENARIO/src

$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

docker compose -f docker-compose-train.yml up --remove-orphans

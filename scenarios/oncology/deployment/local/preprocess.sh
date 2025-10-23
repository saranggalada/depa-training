#!/bin/bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="oncology"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export GENOMICS_LAB_INPUT_PATH=$DATA_DIR/genomics_lab
export GENOMICS_LAB_OUTPUT_PATH=$DATA_DIR/genomics_lab/preprocessed
export PHARMACEUTICAL_COMPANY_INPUT_PATH=$DATA_DIR/pharmaceutical_company
export PHARMACEUTICAL_COMPANY_OUTPUT_PATH=$DATA_DIR/pharmaceutical_company/preprocessed
export COMPUTATIONAL_BIOLOGY_LAB_INPUT_PATH=$DATA_DIR/computational_biology_lab
export COMPUTATIONAL_BIOLOGY_LAB_OUTPUT_PATH=$DATA_DIR/computational_biology_lab/preprocessed
export CANCER_INSTITUTE_INPUT_PATH=$DATA_DIR/cancer_institute
export CANCER_INSTITUTE_OUTPUT_PATH=$DATA_DIR/cancer_institute/preprocessed
mkdir -p $GENOMICS_LAB_OUTPUT_PATH
mkdir -p $PHARMACEUTICAL_COMPANY_OUTPUT_PATH
mkdir -p $COMPUTATIONAL_BIOLOGY_LAB_OUTPUT_PATH
mkdir -p $CANCER_INSTITUTE_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans

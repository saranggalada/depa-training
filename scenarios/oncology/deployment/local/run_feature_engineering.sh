#!/bin/bash
# Feature Engineering Step for Oncology Scenario
# This script runs the feature engineering to transform raw TDP data into joinable features

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="oncology"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data

# Input paths (from preprocessed TDP data)
export GENOMICS_LAB_PATH=$DATA_DIR/genomics_lab/preprocessed
export PHARMACEUTICAL_COMPANY_PATH=$DATA_DIR/pharmaceutical_company/preprocessed
export COMPUTATIONAL_BIOLOGY_LAB_PATH=$DATA_DIR/computational_biology_lab/preprocessed
export CANCER_INSTITUTE_PATH=$DATA_DIR/cancer_institute/preprocessed

# Output path (for engineered features)
export FEATURE_OUTPUT_PATH=$DATA_DIR/features
mkdir -p $FEATURE_OUTPUT_PATH

echo "Running feature engineering for oncology scenario..."
echo "Input directories:"
echo "  Genomics Lab: $GENOMICS_LAB_PATH"
echo "  Pharmaceutical Company: $PHARMACEUTICAL_COMPANY_PATH"
echo "  Computational Biology Lab: $COMPUTATIONAL_BIOLOGY_LAB_PATH"
echo "  Cancer Institute: $CANCER_INSTITUTE_PATH"
echo "Output directory: $FEATURE_OUTPUT_PATH"
echo ""

# Run feature engineering
python3 $REPO_ROOT/scenarios/$SCENARIO/src/feature_engineering.py \
    $GENOMICS_LAB_PATH \
    $PHARMACEUTICAL_COMPANY_PATH \
    $COMPUTATIONAL_BIOLOGY_LAB_PATH \
    $CANCER_INSTITUTE_PATH \
    $FEATURE_OUTPUT_PATH

if [ $? -eq 0 ]; then
    echo ""
    echo "Feature engineering completed successfully!"
    echo "Generated files:"
    ls -lh $FEATURE_OUTPUT_PATH
else
    echo ""
    echo "Feature engineering failed!"
    exit 1
fi



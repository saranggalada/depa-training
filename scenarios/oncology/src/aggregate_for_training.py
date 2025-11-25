"""
Patient-level aggregation for oncology training
Aggregates cell-level features to patient-level for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys


def aggregate_to_patient_level(joined_df):
    """
    Aggregate cell-level features to patient-level
    Groups by patient and drug, averages all cell features
    """
    # Identify feature columns (excluding identifiers)
    id_cols = ['patient', 'drug', 'cell_type', 'drug_response']
    feature_cols = [c for c in joined_df.columns if c not in id_cols and c != 'cell_id']
    
    print(f"Aggregating {len(feature_cols)} features from cell-level to patient-level")
    print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    # Group by patient and drug, compute mean of all features
    agg_dict = {col: 'mean' for col in feature_cols}
    # Keep drug_response (should be same for all cells of same patient)
    if 'drug_response' in joined_df.columns:
        agg_dict['drug_response'] = 'first'
    
    patient_features = joined_df.groupby(['patient', 'drug']).agg(agg_dict).reset_index()
    
    print(f"Aggregated shape: {patient_features.shape}")
    print(f"Patients: {patient_features['patient'].nunique()}")
    print(f"Drugs: {patient_features['drug'].nunique()}")
    
    return patient_features


def normalize_features(df, exclude_cols=['patient', 'drug', 'drug_response']):
    """
    Normalize feature columns using MinMaxScaler
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"Normalized {len(feature_cols)} features")
    return df


def prepare_training_data(patient_df, output_dir):
    """
    Prepare final training dataset
    Splits features and target, handles any missing values
    """
    # Separate features and target
    X_cols = [c for c in patient_df.columns if c not in ['patient', 'drug', 'drug_response']]
    
    X = patient_df[['patient', 'drug'] + X_cols].copy()
    y = patient_df[['patient', 'drug', 'drug_response']].copy()
    
    # Handle missing values (if any)
    if X.isnull().any().any():
        print(f"Warning: Found {X.isnull().sum().sum()} missing values in features, filling with 0")
        X.fillna(0, inplace=True)
    
    # Save
    X.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    
    print(f"\nTraining data prepared:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    
    return X, y


def main():
    """Main execution function"""
    # Get input/output paths
    input_path = os.getenv('JOINED_DATASET_PATH', '/tmp/oncology_joined_dataset.csv')
    output_dir = os.getenv('TRAINING_DATA_PATH', '/mnt/remote/training')
    
    # Allow command line override
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Loading joined dataset from: {input_path}")
    
    # Load joined data
    joined_df = pd.read_csv(input_path)
    print(f"Joined dataset shape: {joined_df.shape}")
    print(f"Columns: {list(joined_df.columns)}")
    
    # Aggregate to patient level
    print("\nAggregating to patient level...")
    patient_df = aggregate_to_patient_level(joined_df)
    
    # Normalize features
    print("\nNormalizing features...")
    patient_df = normalize_features(patient_df)
    
    # Prepare training data
    print("\nPreparing training data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save patient-level aggregated data
    patient_agg_path = os.path.join(output_dir, 'patient_aggregated.csv')
    patient_df.to_csv(patient_agg_path, index=False)
    print(f"Saved patient-aggregated data to: {patient_agg_path}")
    
    # Prepare X, y splits
    X, y = prepare_training_data(patient_df, output_dir)
    
    print("\n" + "="*60)
    print("Aggregation complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - patient_aggregated.csv: Full patient-level dataset")
    print(f"  - X_train.csv: Feature matrix")
    print(f"  - y_train.csv: Target values")


if __name__ == '__main__':
    main()



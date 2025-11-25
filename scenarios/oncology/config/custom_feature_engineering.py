"""
Feature Engineering for Oncology Scenario
Transforms raw preprocessed data into cell-level features suitable for SQL joining
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import MinMaxScaler


def load_data(data_dirs):
    """Load all datasets from TDP directories"""
    # Genomics Lab
    expr_fp = os.path.join(data_dirs['genomics_lab'], 'preprocessed', 'sc_expr_matrix.csv')
    expr = pd.read_csv(expr_fp, index_col=0)
    
    # Pharmaceutical Company
    drug_resp_fp = os.path.join(data_dirs['pharmaceutical_company'], 'preprocessed', 'drug_response.csv')
    drug_targets_fp = os.path.join(data_dirs['pharmaceutical_company'], 'preprocessed', 'drug_targets.json')
    drug_resp = pd.read_csv(drug_resp_fp, index_col=0)
    with open(drug_targets_fp) as f:
        drug_targets = json.load(f)
    
    # Computational Biology Lab
    cell_markers_fp = os.path.join(data_dirs['computational_biology_lab'], 'preprocessed', 'cell_markers.json')
    with open(cell_markers_fp) as f:
        cell_markers = json.load(f)
    
    # Cancer Institute
    cell_meta_fp = os.path.join(data_dirs['cancer_institute'], 'preprocessed', 'cell_metadata.csv')
    cell_meta = pd.read_csv(cell_meta_fp)
    
    return expr, drug_resp, drug_targets, cell_markers, cell_meta


def compute_gene_set_enrichment(expr, drug_targets):
    """
    Compute gene set enrichment scores for each drug target set per cell
    Returns: DataFrame with cells as rows, drugs as columns
    """
    enrichment = {}
    for drug_name, target_genes in drug_targets.items():
        intersected = [g for g in target_genes if g in expr.columns]
        if len(intersected) == 0:
            gene_avg = pd.Series([0]*expr.shape[0], index=expr.index)
        else:
            gene_avg = expr[intersected].mean(axis=1)
        enrichment[drug_name] = gene_avg
    
    enrichment_df = pd.DataFrame(enrichment)
    enrichment_df.index.name = 'cell_id'
    enrichment_df.reset_index(inplace=True)
    return enrichment_df


def compute_sctype_scores(expr, cell_markers):
    """
    Compute scType scores for each cell type per cell
    Returns: DataFrame with cells as rows, cell types as columns
    """
    scores = pd.DataFrame(index=expr.index, columns=cell_markers.keys())
    
    for ctype, markers in cell_markers.items():
        intersected_markers = expr.columns.intersection(markers)
        if len(intersected_markers) > 0:
            pos_score = expr[intersected_markers].mean(axis=1)
        else:
            pos_score = pd.Series([0]*expr.shape[0], index=expr.index)
        scores[ctype] = pos_score
    
    scores.index.name = 'cell_id'
    scores.reset_index(inplace=True)
    # Add suffix to avoid confusion with metadata cell types
    scores.columns = [c if c == 'cell_id' else f'{c}_score' for c in scores.columns]
    return scores


def prepare_cell_features(expr, drug_targets, cell_markers, cell_meta):
    """
    Create a comprehensive cell-level feature table
    """
    # Compute enrichment and scores
    enrichment_df = compute_gene_set_enrichment(expr, drug_targets)
    sctype_scores = compute_sctype_scores(expr, cell_markers)
    
    # Merge cell metadata with features
    cell_features = cell_meta.rename(columns={'cell': 'cell_id'})
    cell_features = cell_features.merge(enrichment_df, on='cell_id', how='left')
    cell_features = cell_features.merge(sctype_scores, on='cell_id', how='left')
    
    return cell_features


def compute_tnse_scores(enrichment_df, cell_meta):
    """
    Compute tNSE scores for each cell type
    Returns: DataFrame with cell types as rows, single tNSE column
    """
    # Merge enrichment with cell metadata to get cell types
    enrichment_with_types = enrichment_df.merge(cell_meta.rename(columns={'cell': 'cell_id'}), on='cell_id', how='left')
    
    tnse_scores = {}
    drug_cols = [col for col in enrichment_df.columns if col != 'cell_id']
    
    for cell_type in cell_meta['type'].unique():
        # Filter cells of this type
        type_cells = enrichment_with_types[enrichment_with_types['type'] == cell_type]
        
        if len(type_cells) == 0:
            tnse_scores[cell_type] = 0
        else:
            # Compute aggregate score: sum / sqrt(n_cells)
            n_cells = len(type_cells)
            # Get all drug enrichment scores for this cell type
            all_drug_scores = type_cells[drug_cols].values.flatten()
            # Normalize enrichment scores and compute aggregate
            normalized_scores = (all_drug_scores - all_drug_scores.min()) / (all_drug_scores.max() - all_drug_scores.min() + 1e-8)
            tnse_score = np.sum(normalized_scores) / np.sqrt(n_cells)
            tnse_scores[cell_type] = tnse_score
    
    tnse_df = pd.DataFrame(list(tnse_scores.items()), columns=['cell_type', 'tNSE'])
    return tnse_df


def aggregate_to_patient_level(cell_features):
    """
    Aggregate cell-level features to patient level by averaging
    """
    # Group by patient and compute mean for all feature columns
    feature_cols = [col for col in cell_features.columns if col not in ['cell_id', 'patient', 'type']]
    patient_features = cell_features.groupby('patient')[feature_cols].mean()
    patient_features.index.name = 'patient_id'
    patient_features.reset_index(inplace=True)
    return patient_features


def normalize_features(patient_features):
    """
    Normalize patient features using MinMax scaling
    """
    feature_cols = [col for col in patient_features.columns if col != 'patient_id']
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize features
    normalized_features = patient_features.copy()
    normalized_features[feature_cols] = scaler.fit_transform(patient_features[feature_cols])
    
    return normalized_features, scaler


def align_training_dataset(patient_features, drug_resp):
    """
    Align patient features with drug response data
    """
    # Find common patients
    common_patients = list(set(patient_features['patient_id']) & set(drug_resp.index))
    
    # Filter to common patients
    features_aligned = patient_features[patient_features['patient_id'].isin(common_patients)].set_index('patient_id')
    response_aligned = drug_resp.loc[common_patients]
    
    return features_aligned, response_aligned


def create_flattened_training_dataset(features_aligned, response_aligned):
    """
    Create flattened training dataset where each patient's features are repeated for each drug
    Returns: Single DataFrame with all feature columns and drug_response column
    """
    # Get feature columns (exclude patient_id if it's in the index)
    feature_cols = features_aligned.columns.tolist()
    
    # Create repeated features: each patient's features repeated 14 times (once per drug)
    training_data_list = []
    
    for patient_id in features_aligned.index:
        patient_features = features_aligned.loc[patient_id].values
        patient_responses = response_aligned.loc[patient_id].values
        
        # Repeat patient features for each drug
        for drug_idx, drug_name in enumerate(response_aligned.columns):
            row_data = list(patient_features) + [patient_responses[drug_idx]]
            training_data_list.append(row_data)
    
    # Create combined DataFrame with feature columns and drug_response column
    all_cols = feature_cols + ['drug_response']
    training_data = pd.DataFrame(training_data_list, columns=all_cols)
    
    return training_data


def prepare_drug_response(drug_resp):
    """
    Convert drug response to long format for easier joining
    """
    drug_resp.index.name = 'patient_id'
    drug_resp_long = drug_resp.reset_index().melt(
        id_vars=['patient_id'],
        var_name='drug',
        value_name='drug_response'
    )
    return drug_resp_long


def main():
    """Main execution function"""
    # Get input paths from environment variables or command line
    genomics_lab_path = os.getenv('GENOMICS_LAB_PATH', '/mnt/remote/genomics_lab')
    pharma_path = os.getenv('PHARMACEUTICAL_COMPANY_PATH', '/mnt/remote/pharmaceutical_company')
    comp_bio_path = os.getenv('COMPUTATIONAL_BIOLOGY_LAB_PATH', '/mnt/remote/computational_biology_lab')
    cancer_inst_path = os.getenv('CANCER_INSTITUTE_PATH', '/mnt/remote/cancer_institute')
    output_path = os.getenv('FEATURE_OUTPUT_PATH', '/mnt/remote/features')
    
    # Allow command line override
    if len(sys.argv) > 1:
        genomics_lab_path = sys.argv[1]
    if len(sys.argv) > 2:
        pharma_path = sys.argv[2]
    if len(sys.argv) > 3:
        comp_bio_path = sys.argv[3]
    if len(sys.argv) > 4:
        cancer_inst_path = sys.argv[4]
    if len(sys.argv) > 5:
        output_path = sys.argv[5]
    
    data_dirs = {
        'genomics_lab': genomics_lab_path,
        'pharmaceutical_company': pharma_path,
        'computational_biology_lab': comp_bio_path,
        'cancer_institute': cancer_inst_path
    }
    
    print("Loading data from TDP directories...")
    print(f"  Genomics Lab: {genomics_lab_path}")
    print(f"  Pharmaceutical Company: {pharma_path}")
    print(f"  Computational Biology Lab: {comp_bio_path}")
    print(f"  Cancer Institute: {cancer_inst_path}")
    
    expr, drug_resp, drug_targets, cell_markers, cell_meta = load_data(data_dirs)
    
    print(f"\nComputing cell-level features...")
    print(f"  Expression matrix: {expr.shape}")
    print(f"  Cell metadata: {cell_meta.shape}")
    print(f"  Drug targets: {len(drug_targets)} drugs")
    print(f"  Cell markers: {len(cell_markers)} cell types")
    
    # Step 1: Gene Set Enrichment
    print("\nStep 1: Computing gene set enrichment...")
    enrichment_df = compute_gene_set_enrichment(expr, drug_targets)
    print(f"  Gene set enrichment: {enrichment_df.shape}")
    
    # Step 2: Cell Type Scoring
    print("\nStep 2: Computing cell type scores...")
    sctype_scores = compute_sctype_scores(expr, cell_markers)
    print(f"  Cell type scores: {sctype_scores.shape}")
    
    # Step 3: tNSE Scores
    print("\nStep 3: Computing tNSE scores...")
    tnse_scores = compute_tnse_scores(enrichment_df, cell_meta)
    print(f"  tNSE scores: {tnse_scores.shape}")
    
    # Step 4: Combine Cell-Level Features
    print("\nStep 4: Combining cell-level features...")
    cell_features = prepare_cell_features(expr, drug_targets, cell_markers, cell_meta)
    print(f"  Combined cell features: {cell_features.shape}")
    
    # Step 5: Aggregate to Patient Level
    print("\nStep 5: Aggregating to patient level...")
    patient_features = aggregate_to_patient_level(cell_features)
    print(f"  Patient-level features: {patient_features.shape}")
    
    # Step 6: Normalize Features
    print("\nStep 6: Normalizing features...")
    normalized_features, scaler = normalize_features(patient_features)
    print(f"  Normalized features: {normalized_features.shape}")
    
    # Step 7: Align Training Dataset
    print("\nStep 7: Aligning training dataset...")
    features_aligned, response_aligned = align_training_dataset(normalized_features, drug_resp)
    print(f"  Aligned features: {features_aligned.shape}")
    print(f"  Aligned response: {response_aligned.shape}")
    
    # Step 8: Create Flattened Training Dataset
    print("\nStep 8: Creating flattened training dataset...")
    training_dataset = create_flattened_training_dataset(features_aligned, response_aligned)
    print(f"  Training dataset: {training_dataset.shape}")
    print(f"  Feature columns: {len(training_dataset.columns) - 1}")
    print(f"  Target column: drug_response")

    return training_dataset
    
    '''
    # Save all outputs
    os.makedirs(output_path, exist_ok=True)
    
    # Save intermediate outputs
    enrichment_path = os.path.join(output_path, 'gene_set_enrichment.csv')
    sctype_path = os.path.join(output_path, 'cell_type_scores.csv')
    tnse_path = os.path.join(output_path, 'tnse_scores.csv')
    cell_features_path = os.path.join(output_path, 'cell_features.csv')
    patient_features_path = os.path.join(output_path, 'patient_features.csv')
    normalized_features_path = os.path.join(output_path, 'normalized_features.csv')
    features_aligned_path = os.path.join(output_path, 'features_aligned.csv')
    response_aligned_path = os.path.join(output_path, 'response_aligned.csv')
    X_repeated_path = os.path.join(output_path, 'X_repeated.csv')
    y_flattened_path = os.path.join(output_path, 'y_flattened.csv')
    drug_resp_path = os.path.join(output_path, 'drug_response_long.csv')
    
    # Save all datasets
    enrichment_df.to_csv(enrichment_path, index=False)
    sctype_scores.to_csv(sctype_path, index=False)
    tnse_scores.to_csv(tnse_path, index=False)
    cell_features.to_csv(cell_features_path, index=False)
    patient_features.to_csv(patient_features_path, index=False)
    normalized_features.to_csv(normalized_features_path, index=False)
    features_aligned.to_csv(features_aligned_path)
    response_aligned.to_csv(response_aligned_path)
    X_repeated.to_csv(X_repeated_path, index=False)
    y_flattened.to_csv(y_flattened_path, index=False)
    drug_resp_long.to_csv(drug_resp_path, index=False)
    
    # Save scaler for later use
    import pickle
    scaler_path = os.path.join(output_path, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nFeature engineering complete!")
    print(f"  All outputs saved to: {output_path}")
    print(f"\nSaved files:")
    print(f"  - Gene set enrichment: {enrichment_path}")
    print(f"  - Cell type scores: {sctype_path}")
    print(f"  - tNSE scores: {tnse_path}")
    print(f"  - Cell features: {cell_features_path}")
    print(f"  - Patient features: {patient_features_path}")
    print(f"  - Normalized features: {normalized_features_path}")
    print(f"  - Features aligned: {features_aligned_path}")
    print(f"  - Response aligned: {response_aligned_path}")
    print(f"  - X_repeated (flattened): {X_repeated_path}")
    print(f"  - y_flattened (flattened): {y_flattened_path}")
    print(f"  - Drug response (long): {drug_resp_path}")
    print(f"  - Feature scaler: {scaler_path}")
    print(f"\nFinal training dataset:")
    print(f"  Original features: {features_aligned.shape}")
    print(f"  Original response: {response_aligned.shape}")
    print(f"  Flattened features (X_repeated): {X_repeated.shape}")
    print(f"  Flattened response (y_flattened): {y_flattened.shape}")
    print(f"  Feature columns: {list(features_aligned.columns)}")
    print(f"  Response columns: {list(response_aligned.columns)}")
    '''

if __name__ == '__main__':
    main()



"""
Feature Engineering for Oncology Scenario
Transforms raw preprocessed data into cell-level features suitable for SQL joining
"""

import pandas as pd
import numpy as np
import json
import os
import sys


def load_data(data_dirs):
    """Load all datasets from TDP directories"""
    # Genomics Lab
    expr_fp = os.path.join(data_dirs['genomics_lab'], 'sc_expr_matrix.csv')
    expr = pd.read_csv(expr_fp, index_col=0)
    
    # Pharmaceutical Company
    drug_resp_fp = os.path.join(data_dirs['pharmaceutical_company'], 'drug_response.csv')
    drug_targets_fp = os.path.join(data_dirs['pharmaceutical_company'], 'drug_targets.json')
    drug_resp = pd.read_csv(drug_resp_fp, index_col=0)
    with open(drug_targets_fp) as f:
        drug_targets = json.load(f)
    
    # Computational Biology Lab
    cell_markers_fp = os.path.join(data_dirs['computational_biology_lab'], 'cell_markers.json')
    with open(cell_markers_fp) as f:
        cell_markers = json.load(f)
    
    # Cancer Institute
    cell_meta_fp = os.path.join(data_dirs['cancer_institute'], 'cell_metadata.csv')
    cell_meta = pd.read_csv(cell_meta_fp)
    
    return expr, drug_resp, drug_targets, cell_markers, cell_meta


def compute_gene_set_enrichment(expr, drug_targets):
    """
    Compute gene set enrichment scores for each drug target set per cell
    Returns: DataFrame with cells as rows, drugs as columns
    """
    enrichment = {}
    for drug_name, target_genes in drug_targets.items():
        intersected = [g for g in target_genes if g in expr.index]
        if len(intersected) == 0:
            gene_avg = pd.Series([0]*expr.shape[1], index=expr.columns)
        else:
            gene_avg = expr.loc[intersected].mean(axis=0)
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
    scores = pd.DataFrame(index=expr.columns, columns=cell_markers.keys())
    
    for ctype, markers in cell_markers.items():
        intersected_markers = expr.index.intersection(markers)
        if len(intersected_markers) > 0:
            pos_score = expr.loc[intersected_markers].mean(axis=0)
        else:
            pos_score = pd.Series([0]*expr.shape[1], index=expr.columns)
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
    
    # Prepare cell features
    cell_features = prepare_cell_features(expr, drug_targets, cell_markers, cell_meta)
    print(f"  Cell features computed: {cell_features.shape}")
    
    # Prepare drug response (long format)
    drug_resp_long = prepare_drug_response(drug_resp)
    print(f"  Drug response (long format): {drug_resp_long.shape}")
    
    # Save outputs
    os.makedirs(output_path, exist_ok=True)
    
    cell_features_path = os.path.join(output_path, 'cell_features.csv')
    drug_resp_path = os.path.join(output_path, 'drug_response_long.csv')
    
    cell_features.to_csv(cell_features_path, index=False)
    drug_resp_long.to_csv(drug_resp_path, index=False)
    
    print(f"\nFeature engineering complete!")
    print(f"  Cell features saved to: {cell_features_path}")
    print(f"  Drug response saved to: {drug_resp_path}")
    print(f"\nOutput columns:")
    print(f"  Cell features: {list(cell_features.columns)}")
    print(f"  Drug response: {list(drug_resp_long.columns)}")


if __name__ == '__main__':
    main()



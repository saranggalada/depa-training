#Working Code

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import os


def gene_sets_prepare(expr_matrix, gene_sets):
    enrichment = {}
    for set_name, genes in gene_sets.items():
        intersected = [g for g in genes if g in expr_matrix.index]
        if len(intersected) == 0:
            gene_avg = pd.Series([0]*expr_matrix.shape[1], index=expr_matrix.columns)
        else:
            gene_avg = expr_matrix.loc[intersected].mean(axis=0)
        enrichment[set_name] = gene_avg
    return pd.DataFrame(enrichment)

def sctype_score_(expr_matrix, positive_markers, negative_markers=None):
    scores = pd.DataFrame(index=expr_matrix.columns, columns=positive_markers.keys())
    for ctype, markers in positive_markers.items():
        pos_score = expr_matrix.loc[expr_matrix.index.intersection(markers)].mean()
        neg_score = 0
        if negative_markers and ctype in negative_markers:
            neg_score = expr_matrix.loc[expr_matrix.index.intersection(negative_markers[ctype])].mean()
        scores[ctype] = pos_score - neg_score
    return scores

def model_tnse(enrichment_matrix, cell_types, groupby='type'):
    tnse_scores = {}
    for ctype in cell_types[groupby].unique():
        idx = cell_types[cell_types[groupby] == ctype].index
        # Select rows from enrichment_matrix using the index from cell_types
        submat = enrichment_matrix.loc[idx]
        scaler = MinMaxScaler()
        normed = scaler.fit_transform(submat)
        overall_score = np.sum(normed) / np.sqrt(len(idx))
        tnse_scores[ctype] = overall_score
    return tnse_scores

def fit_model(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model


def set_load_paths(fp):
  expr_fp = f'{fp}/sc_expr_matrix.csv'
  drug_resp_fp = f'{fp}/drug_response.csv'
  cell_markers_fp = f'{fp}/cell_markers.json'
  drug_targets_fp = f'{fp}/drug_targets.json'
  cell_meta_fp = f'{fp}/cell_metadata.csv'

# ----------- Main Execution Script -----------

if __name__ == '__main__':
    # current we are using type2 synthetic data
#    data_fp='/content/drive/MyDrive/depa_oncology/ml_relapse_aml/data/synt_data/type2'
    data_fp='/content/Data'


    #set_load_paths(data_fp)
    expr_fp = f'{data_fp}/sc_expr_matrix.csv'
    drug_resp_fp = f'{data_fp}/drug_response.csv'
    cell_markers_fp = f'{data_fp}/cell_markers.json'
    drug_targets_fp = f'{data_fp}/drug_targets.json'
    cell_meta_fp = f'{data_fp}/cell_metadata.csv'


    #set model process paths
    model_output_fp=f'{data_fp}/output'
    os.makedirs(model_output_fp, exist_ok=True)
    gene_set_enr_fp = f'{model_output_fp}/gene_set_expr.csv'
    sctype_scores_fp = f'{model_output_fp}/sctype_scores.csv'
    tnse_scores_fp = f'{model_output_fp}/tnse_scores.csv'
    preds_fp = f'{model_output_fp}/predicted_dss.csv'

    # Load Data
    expr = pd.read_csv(expr_fp, index_col=0)
    drug_resp = pd.read_csv(drug_resp_fp, index_col=0)
    with open(cell_markers_fp) as f:
        cell_markers = json.load(f)
    with open(drug_targets_fp) as f:
        drug_targets = json.load(f)
    cell_meta = pd.read_csv(cell_meta_fp)
    cell_meta.index = cell_meta['cell']

    # Gene set enrichment (per cell)
    gene_set_enr = gene_sets_prepare(expr, drug_targets)
    print("Gene Set Enrichment Matrix:")
    print(gene_set_enr.head())
    # Save the gene_set_enr
    gene_set_enr.to_csv(gene_set_enr_fp)

    # Cell type scoring
    sctype_scores = sctype_score_(expr, cell_markers)
    print("\nscType Scores (first 5 cells):")
    print(sctype_scores.head())
    # Save the sctype_score
    sctype_scores.to_csv(sctype_scores_fp)


    # tNSE score (by patient type, example)
    tnse_scores = model_tnse(gene_set_enr, cell_meta)
    print("\ntNSE Scores:")
    print(tnse_scores)

    # Save the tnse_score
    tnse_scores_df = pd.DataFrame.from_dict(tnse_scores, orient='index', columns=['tNSE'])
    tnse_scores_df.to_csv(tnse_scores_fp)

    # Aggregate cell-level features to patient level
    patient_features = gene_set_enr.join(sctype_scores).join(cell_meta[['patient']]).groupby('patient').mean()

    # Normalize patient features
    scaler = MinMaxScaler()
    patient_features = pd.DataFrame(scaler.fit_transform(patient_features), columns=patient_features.columns, index=patient_features.index)

    # Fit XGBoost model (example: DSS vs aggregated features per patient)
    # Align patient features with drug response data
    # Find common patients
    common_patients = patient_features.index.intersection(drug_resp.index)

    # Filter both dataframes to keep only common patients
    X = patient_features.loc[common_patients].values
    y = drug_resp.loc[common_patients].values.flatten() # Flatten drug response to match the number of samples


    model = fit_model(X, y)
    preds = model.predict(X)
    print("\nPredicted DSS values (for all patients and drugs):")
    print(preds)

    # Reshape predictions to match the shape of drug_resp for common patients
    num_drugs = drug_resp.shape[1]
    preds_reshaped = preds.reshape(-1, num_drugs)

    preds_df = pd.DataFrame(preds_reshaped, index=common_patients, columns=drug_resp.columns)
    preds_df.to_csv(preds_fp)
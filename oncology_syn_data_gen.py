import pandas as pd
import numpy as np
import json
import os

def make_synthetic_data(output_dir='./data/'):
    os.makedirs(output_dir, exist_ok=True)

    # Setting file paths
    expr_fp = f'{output_dir}/sc_expr_matrix.csv'
    drug_resp_fp = f'{output_dir}/drug_response.csv'
    cell_markers_fp = f'{output_dir}/cell_markers.json'
    drug_targets_fp = f'{output_dir}/drug_targets.json'
    cell_meta_fp = f'{output_dir}/cell_metadata.csv'

    print("make_synthetic_data fp:",output_dir)
    '''
    gene_names = [
        'CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT',
        'CD14', 'CD11b', 'CD3', 'CD4', 'CD8', 'CD56', 'CD19',
        'BCL2', 'DNMT1', 'IDH2', 'AKT1', 'AKT2', 'AKT3', 'MDM2',
        'BRD4', 'CHEK1', 'CDK2', 'CDK7', 'CDK9', 'PDK1',
        'BCL2L1', 'FLT3', 'SRC', 'ABL1'
    ] + [f'GENE_A{i}' for i in range(10)]
    '''
    gene_names = [
      # Core AML/Leukemic blast markers and differentiation genes
      'CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT', 'CD14', 'CD11b',
      # Hematopoietic and lymphoid lineage
      'CD3', 'CD4', 'CD8', 'CD56', 'CD19', 'CD117', 'CD45',
      # Apoptosis and cell cycle (therapy targets)
      'BCL2', 'MCL1', 'BCL2L1', 'TP53', 'MDM2', 'CCND1', 'CCNB1', 'CDK2', 'CDK7', 'CDK9', 'CHEK1', 'BRD4',
      # Signal transduction & kinases (often mutated/targeted in AML)
      'FLT3', 'KIT', 'IDH1', 'IDH2', 'NRAS', 'KRAS', 'AKT1', 'AKT2', 'AKT3', 'PDK1', 'SRC', 'ABL1', 'STAT3', 'STAT5A', 'JAK2', 'PIK3CA', 'PIK3CB',
      # Epigenetic regulators and DNA methylation (commonly mutated/targeted)
      'DNMT3A', 'DNMT1', 'TET2', 'EZH2', 'ASXL1',
      # Additional cell fate/differentiation and drug resistance genes
      'RUNX1', 'CEBPA', 'GATA2', 'NPM1', 'WT1'
    ]

    n_patients = 50
    n_cells = 250  # 5 cells per patient on average
    patients = [f'patient{i+1}' for i in range(n_patients)]
    cell_types_master = ['AML_blast', 'Monocyte', 'Tcell', 'NKcell', 'Bcell']
    np.random.seed(42)
    cell_types = list(np.random.choice(cell_types_master, n_cells, p=[0.4, 0.2, 0.15, 0.13, 0.12]))
    cell_ids = [f'cell_{i+1}' for i in range(n_cells)]
    cell_patients = list(np.random.choice(patients, n_cells))

    # Simulate gene expression
    marker_dict = {
        'AML_blast': ['CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT', 'BCL2', 'DNMT1', 'AKT1', 'MDM2',
                      'BRD4', 'CHEK1', 'CDK2', 'CDK7', 'CDK9', 'PDK1', 'FLT3', 'SRC', 'ABL1'],
        'Monocyte': ['CD14', 'CD11b', 'AKT1', 'AKT2', 'AKT3', 'PDK1'],
        'Tcell': ['CD3', 'CD4', 'CD8'],
        'NKcell': ['CD56'],
        'Bcell': ['CD19']
    }
    expr_array = np.zeros((len(gene_names), n_cells))
    for idx, ctype in enumerate(cell_types):
        for gidx, gene in enumerate(gene_names):
            if gene in marker_dict.get(ctype, []):
                expr_array[gidx, idx] = np.random.uniform(8, 15)
            elif (ctype == 'AML_blast' and gene in ['CD14', 'CD11b']) or (ctype == 'Monocyte' and gene in ['CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT']):
                expr_array[gidx, idx] = np.random.uniform(0, 2)
            else:
                expr_array[gidx, idx] = np.random.uniform(1, 5)
    expr_df = pd.DataFrame(expr_array, index=gene_names, columns=cell_ids)
    expr_df.to_csv(expr_fp)

    # Cell metadata
    meta_df = pd.DataFrame({
        "cell": cell_ids,
        "patient": cell_patients,
        "type": cell_types
    })
    meta_df.to_csv(cell_meta_fp, index=False)

    # Marker and target definitions
    cell_markers = {
        "AML_blast": ['CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT'],
        "Monocyte":  ['CD14', 'CD11b'],
        "Tcell":     ['CD3', 'CD4', 'CD8'],
        "NKcell":    ['CD56'],
        "Bcell":     ['CD19']
    }
    with open(cell_markers_fp, "w") as f:
        json.dump(cell_markers, f, indent=2)
    drug_targets = {
        "Venetoclax": ["BCL2"],
        "Azacitidine": ["DNMT1", "IDH2"],
        "MK-2206": ["AKT1", "AKT2", "AKT3"],
        "SAR405838": ["MDM2"],
        "Molibresib": ["BRD4"],
        "MK-8776": ["CHEK1"],
        "Seliciclib": ["CDK2", "CDK7", "CDK9"],
        "BX-912": ["PDK1"],
        "Navitoclax": ["BCL2", "BCL2L1"],
        "Sorafenib": ["FLT3", "KIT"],
        "Midostaurin": ["FLT3", "KIT"],
        "Dasatinib": ["SRC", "ABL1", "KIT"],
        "Gilteritinib": ["FLT3"],
        "Idasanutlin": ["MDM2"]
    }
    with open(drug_targets_fp, "w") as f:
        json.dump(drug_targets, f, indent=2)

    # Drug response matrix with patient IDs from cell_metadata
    # Use all patients defined by n_patients
    all_patients = [f'patient{i+1}' for i in range(n_patients)]
    drugs = list(drug_targets.keys())
    resp = np.random.uniform(10, 30, size=(len(all_patients), len(drugs)))
    resp_df = pd.DataFrame(resp, index=all_patients, columns=drugs)
    resp_df.to_csv(drug_resp_fp)


if __name__ == '__main__':
#    data_fp='/content/drive/MyDrive/depa_oncology/ml_relapse_aml/data/synt_data/type2'
    data_fp='/content/Data'
    print(data_fp)
    # Generate Synthetic Data
    make_synthetic_data(data_fp)
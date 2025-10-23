import pandas as pd
import numpy as np
import os

input_dir = "/mnt/input/data"
output_dir = "/mnt/output/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Generate single-cell expression matrix
print("Generating sc_expr_matrix.csv for genomics_lab...")

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
n_cells = 250
cell_types_master = ['AML_blast', 'Monocyte', 'Tcell', 'NKcell', 'Bcell']
np.random.seed(42)
cell_types = list(np.random.choice(cell_types_master, n_cells, p=[0.4, 0.2, 0.15, 0.13, 0.12]))
cell_ids = [f'cell_{i+1}' for i in range(n_cells)]

# Simulate gene expression based on cell types
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
expr_df.to_csv(f'{output_dir}/sc_expr_matrix.csv')

print(f"Generated sc_expr_matrix.csv with shape {expr_df.shape}")
print("Done!")

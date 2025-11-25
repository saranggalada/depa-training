import pandas as pd
import numpy as np
import os

input_dir = "/mnt/input/data"
output_dir = "/mnt/output/preprocessed"
os.makedirs(output_dir, exist_ok=True)

print("Generating cell_metadata.csv for cancer_institute...")

n_patients = 50
n_cells = 250
patients = [f'patient{i+1}' for i in range(n_patients)]
cell_types_master = ['AML_blast', 'Monocyte', 'Tcell', 'NKcell', 'Bcell']

np.random.seed(42)
cell_types = list(np.random.choice(cell_types_master, n_cells, p=[0.4, 0.2, 0.15, 0.13, 0.12]))
cell_ids = [f'cell_{i+1}' for i in range(n_cells)]
cell_patients = list(np.random.choice(patients, n_cells))

# Cell metadata
meta_df = pd.DataFrame({
    "cell": cell_ids,
    "patient": cell_patients,
    "type": cell_types
})
meta_df.to_csv(f'{output_dir}/cell_metadata.csv', index=False)

print(f"Generated cell_metadata.csv with shape {meta_df.shape}")
print("Done!")

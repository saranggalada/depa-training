import json
import os

input_dir = "/mnt/input/data"
output_dir = "/mnt/output/preprocessed"
os.makedirs(output_dir, exist_ok=True)

print("Generating cell_markers.json for computational_biology_lab...")

# Cell type marker definitions
cell_markers = {
    "AML_blast": ['CD34', 'CD33', 'CD38', 'PROM1', 'ENG', 'CD99', 'KIT'],
    "Monocyte":  ['CD14', 'CD11b'],
    "Tcell":     ['CD3', 'CD4', 'CD8'],
    "NKcell":    ['CD56'],
    "Bcell":     ['CD19']
}

with open(f'{output_dir}/cell_markers.json', "w") as f:
    json.dump(cell_markers, f, indent=2)

print(f"Generated cell_markers.json with {len(cell_markers)} cell types")
print("Done!")

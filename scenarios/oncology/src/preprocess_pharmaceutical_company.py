import pandas as pd
import numpy as np
import json
import os

input_dir = "/mnt/input/data"
output_dir = "/mnt/output/preprocessed"
os.makedirs(output_dir, exist_ok=True)

print("Generating drug_response.csv and drug_targets.json for pharmaceutical_company...")

# Drug targets definition
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

with open(f'{output_dir}/drug_targets.json', "w") as f:
    json.dump(drug_targets, f, indent=2)

# Drug response matrix
n_patients = 50
all_patients = [f'patient{i+1}' for i in range(n_patients)]
drugs = list(drug_targets.keys())

np.random.seed(42)
resp = np.random.uniform(10, 30, size=(len(all_patients), len(drugs)))
resp_df = pd.DataFrame(resp, index=all_patients, columns=drugs)
resp_df.to_csv(f'{output_dir}/drug_response.csv')

print(f"Generated drug_targets.json with {len(drug_targets)} drugs")
print(f"Generated drug_response.csv with shape {resp_df.shape}")
print("Done!")

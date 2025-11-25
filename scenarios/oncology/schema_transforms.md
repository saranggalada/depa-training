# Data Processing Pipeline: Schema Transformations

## Raw Input Datasets

### 1. Expression Matrix (`sc_expr_matrix.csv`)
```
Shape: (genes, cells) = (56, 250)
Schema:
         cell_1    cell_2    cell_3    ...
CD34     12.4      2.1       11.8      ...
CD33     13.2      1.5       10.9      ...
BCL2     9.7       3.2       8.8       ...
AKT1     11.1      4.5       9.2       ...
...
```
**Source**: Single-cell RNA-seq / Flow cytometry  
**PII**: Low (no direct identifiers)

---

### 2. Cell Metadata (`cell_metadata.csv`)
```
Shape: (250, 3)
Schema:
   cell     | patient   | type
   ---------|-----------|----------
   cell_1   | patient1  | AML_blast
   cell_2   | patient1  | Monocyte
   cell_3   | patient2  | AML_blast
   ...
```
**Source**: Clinical annotation  
**PII**: HIGH (contains patient identifiers)

---

### 3. Drug Response (`drug_response.csv`)
```
Shape: (patients, drugs) = (50, 14)
Schema:
            Venetoclax | Azacitidine | MK-2206 | ...
patient1    25.3       | 18.7        | 22.1    | ...
patient2    15.8       | 20.3        | 19.5    | ...
patient3    28.9       | 16.2        | 24.7    | ...
...
```
**Source**: Clinical trials / Ex-vivo drug screening  
**PII**: HIGH (patient outcomes)

---

### 4. Cell Markers (`cell_markers.json`)
```json
{
  "AML_blast": ["CD34", "CD33", "CD38", "PROM1", "ENG", "CD99", "KIT"],
  "Monocyte": ["CD14", "CD11b"],
  "Tcell": ["CD3", "CD4", "CD8"],
  "NKcell": ["CD56"],
  "Bcell": ["CD19"]
}
```
**Source**: Scientific literature / Public databases  
**PII**: None (public knowledge)

---

### 5. Drug Targets (`drug_targets.json`)
```json
{
  "Venetoclax": ["BCL2"],
  "Azacitidine": ["DNMT1", "IDH2"],
  "MK-2206": ["AKT1", "AKT2", "AKT3"],
  "SAR405838": ["MDM2"],
  ...
}
```
**Source**: DrugBank / FDA / Pharmacological databases  
**PII**: None (public knowledge)

---

## 🔧 Feature Engineering Steps

### Step 1: Gene Set Enrichment
**Function**: `gene_sets_prepare(expr_matrix, drug_targets)`

**Input**: 
- Expression Matrix (56 genes × 250 cells)
- Drug Targets (14 drugs → gene lists)

**Process**:
```python
For each drug:
  1. Find target genes in expression matrix
  2. Average expression across target genes per cell
  3. Result: drug enrichment score per cell
```

**Output**: Gene Set Enrichment Matrix
```
Shape: (250 cells, 14 drugs)
Schema:
         Venetoclax | Azacitidine | MK-2206 | ...
cell_1   8.5        | 12.3        | 10.1    | ...
cell_2   3.1        | 4.2         | 5.7     | ...
cell_3   9.8        | 11.7        | 9.5     | ...
...
```
**Meaning**: How much each cell expresses the targets of each drug

---

### Step 2: Cell Type Scoring
**Function**: `sctype_score_(expr_matrix, cell_markers)`

**Input**:
- Expression Matrix (56 genes × 250 cells)
- Cell Markers (5 cell types → marker gene lists)

**Process**:
```python
For each cell type:
  1. Find marker genes in expression matrix
  2. Average expression of markers per cell
  3. Result: cell type score per cell
```

**Output**: Cell Type Scores Matrix
```
Shape: (250 cells, 5 cell_types)
Schema:
         AML_blast | Monocyte | Tcell | NKcell | Bcell
cell_1   10.2      | 2.1      | 1.5   | 1.2    | 1.8
cell_2   2.3       | 8.7      | 1.9   | 1.4    | 1.6
cell_3   11.5      | 1.8      | 1.3   | 1.1    | 1.7
...
```
**Meaning**: Likelihood that each cell belongs to each cell type

---

### Step 3: tNSE Scores (Optional)
**Function**: `model_tnse(gene_set_enr, cell_meta)`

**Input**:
- Gene Set Enrichment (250 cells × 14 drugs)
- Cell Metadata (cell types)

**Process**:
```python
For each cell type:
  1. Filter cells of that type
  2. Normalize enrichment scores
  3. Compute aggregate score: sum / sqrt(n_cells)
```

**Output**: tNSE Scores
```
Shape: (5 cell_types, 1)
Schema:
   cell_type  | tNSE
   -----------|-------
   AML_blast  | 45.2
   Monocyte   | 38.7
   Tcell      | 32.1
   NKcell     | 29.8
   Bcell      | 31.5
```
**Meaning**: Aggregate drug target expression per cell type

---

## Integration Steps

### Step 4: Combine Cell-Level Features
**Function**: Join gene_set_enr + sctype_scores + cell_meta

**Input**:
- Gene Set Enrichment (250 × 14)
- Cell Type Scores (250 × 5)
- Cell Metadata (250 × 3)

**Output**: Combined Cell Features
```
Shape: (250 cells, 20 features)
Schema:
         Venetoclax | Azacitidine | ... | AML_blast | Monocyte | ... | patient
cell_1   8.5        | 12.3        | ... | 10.2      | 2.1      | ... | patient1
cell_2   3.1        | 4.2         | ... | 2.3       | 8.7      | ... | patient1
cell_3   9.8        | 11.7        | ... | 11.5      | 1.8      | ... | patient2
...
```

---

### Step 5: Aggregate to Patient Level
**Function**: `groupby('patient').mean()`

**Input**: Combined Cell Features (250 cells × 19 features + patient)

**Process**:
```python
For each patient:
  1. Find all cells belonging to that patient
  2. Average all feature values across cells
  3. Result: one feature vector per patient
```

**Output**: Patient-Level Features
```
Shape: (50 patients, 19 features)
Schema:
           Venetoclax_avg | Azacitidine_avg | ... | AML_blast_avg | Monocyte_avg
patient1   5.8            | 8.25            | ... | 6.25          | 5.4
patient2   7.1            | 9.8             | ... | 8.7           | 3.2
patient3   6.3            | 7.9             | ... | 7.1           | 4.8
...
```
**Meaning**: Average cellular features per patient

---

### Step 6: Normalize Features
**Function**: `MinMaxScaler()`

**Input**: Patient Features (50 × 19)

**Process**:
```python
For each feature column:
  normalized = (value - min) / (max - min)
```

**Output**: Normalized Patient Features
```
Shape: (50 patients, 19 features)
Schema: All values scaled to [0, 1]
           Venetoclax_norm | Azacitidine_norm | ... 
patient1   0.42            | 0.67             | ...
patient2   0.58            | 0.83             | ...
...
```

---

## Final Training Dataset

### Step 7: Align with Drug Response
**Function**: Match patients in features and targets

**Input**:
- Normalized Features (50 patients × 19 features)
- Drug Response (50 patients × 14 drugs)

**Process**:
```python
common_patients = features.index ∩ drug_response.index
X = features.loc[common_patients]
y = drug_response.loc[common_patients]
```

**Output**: Final Training Dataset
```
X (Features):
Shape: (50 patients, 19 features)
All drug enrichment + cell type scores, normalized

y (Targets):
Shape: (50 patients, 14 drugs)
Drug sensitivity scores (DSS)

ALIGNED ON: patient ID
```

---

## Training Format

The model is trained with a flattened approach:

```
Original:
- X: 50 patients × 19 features
- y: 50 patients × 14 drugs

Flattened for training:
- X_repeated: (50 × 14) = 700 samples × 19 features
- y_flattened: 700 drug response values

Each patient's feature vector is repeated 14 times (once per drug)
Model learns: features → drug response (treating each drug independently)
```

---

## Key Transformations Summary

| Stage | Input Dim | Output Dim | Operation | Key |
|-------|-----------|------------|-----------|-----|
| Expression → Enrichment | 56×250 | 250×14 | Average target genes | Drug targets |
| Expression → Cell Types | 56×250 | 250×5 | Average marker genes | Cell markers |
| Cell Features → Patient | 250×19 | 50×19 | Group by patient, mean | Patient ID |
| Patient Features + Response | 50×19, 50×14 | 50×(19+14) | Inner join | Patient ID |
| Training Format | 50×19, 50×14 | 700×19, 700×1 | Repeat & flatten | Patient×Drug |


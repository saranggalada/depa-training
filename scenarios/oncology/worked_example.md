# Worked Example: Feature Engineering Pipeline

This document walks through the feature engineering with **concrete example data** to illustrate exactly what happens at each step.

## Sample Input Data (Simplified)

### 1. Expression Matrix (3 genes × 3 cells)
```
gene  | cell_1 | cell_2 | cell_3
------|--------|--------|--------
BCL2  | 8.5    | 3.1    | 9.8
CD34  | 10.1   | 4.9    | 11.8
CD14  | 1.4    | 1.1    | 4.6
```

### 2. Cell Metadata
```
cell   | patient   | type
-------|-----------|----------
cell_1 | patient1  | AML_blast
cell_2 | patient1  | Monocyte
cell_3 | patient2  | AML_blast
```

### 3. Drug Response
```
         | Venetoclax | Azacitidine
---------|------------|-------------
patient1 | 17.5       | 29.0
patient2 | 13.6       | 13.7
```

### 4. Drug Targets (JSON)
```json
{
  "Venetoclax": ["BCL2"],
  "Azacitidine": ["DNMT1", "IDH2"]
}
```

### 5. Cell Markers (JSON)
```json
{
  "AML_blast": ["CD34", "CD33"],
  "Monocyte": ["CD14", "CD11b"]
}
```

---

## Transformation Steps

### STEP 1: Unpivot Expression Matrix

**Operation:** `stack(3, 'cell_1', cell_1, 'cell_2', cell_2, 'cell_3', cell_3)`

**Result (expr_long):** 9 rows
```
gene  | cell_id | expression
------|---------|------------
BCL2  | cell_1  | 8.5
BCL2  | cell_2  | 3.1
BCL2  | cell_3  | 9.8
CD34  | cell_1  | 10.1
CD34  | cell_2  | 4.9
CD34  | cell_3  | 11.8
CD14  | cell_1  | 1.4
CD14  | cell_2  | 1.1
CD14  | cell_3  | 4.6
```

### STEP 2: Parse Drug Targets JSON

**Operation:** `explode(map(...)) + LATERAL VIEW explode(...)`

**Result (drug_targets_exploded):** 1 row (simplified; normally 25)
```
drug        | target_gene
------------|------------
Venetoclax  | BCL2
```

### STEP 3: Parse Cell Markers JSON

**Operation:** `explode(map(...)) + LATERAL VIEW explode(...)`

**Result (cell_markers_exploded):** 2 rows (simplified)
```
cell_type  | marker_gene
-----------|------------
AML_blast  | CD34
Monocyte   | CD14
```

---

## Feature Engineering

### STEP 4: Gene Set Enrichment (Drug Target Expression)

**Operation:** 
```sql
SELECT el.cell_id, dte.drug, AVG(el.expression) AS drug_score
FROM expr_long el
INNER JOIN drug_targets_exploded dte ON el.gene = dte.target_gene
GROUP BY el.cell_id, dte.drug
```

**Join Details:**
```
expr_long (BCL2 rows) + drug_targets (Venetoclax → BCL2)
--------------------------------------------------------------
BCL2, cell_1, 8.5  + Venetoclax, BCL2  →  cell_1, Venetoclax, 8.5
BCL2, cell_2, 3.1  + Venetoclax, BCL2  →  cell_2, Venetoclax, 3.1
BCL2, cell_3, 9.8  + Venetoclax, BCL2  →  cell_3, Venetoclax, 9.8
```

**Result (gene_set_enrichment):** 3 rows
```
cell_id | drug       | drug_score
--------|------------|------------
cell_1  | Venetoclax | 8.5         ← avg(BCL2) for cell_1
cell_2  | Venetoclax | 3.1         ← avg(BCL2) for cell_2
cell_3  | Venetoclax | 9.8         ← avg(BCL2) for cell_3
```

**Interpretation:** 
- cell_1 expresses BCL2 at 8.5 → Venetoclax (BCL2 inhibitor) may be effective
- cell_2 expresses BCL2 at 3.1 → Lower target expression, may respond less

### STEP 5: Cell Type Scores (Marker Gene Expression)

**Operation:**
```sql
SELECT el.cell_id, cme.cell_type, AVG(el.expression) AS cell_type_score
FROM expr_long el
INNER JOIN cell_markers_exploded cme ON el.gene = cme.marker_gene
GROUP BY el.cell_id, cme.cell_type
```

**Join Details:**
```
expr_long (CD34 rows) + cell_markers (AML_blast → CD34)
--------------------------------------------------------
CD34, cell_1, 10.1  + AML_blast, CD34  →  cell_1, AML_blast, 10.1
CD34, cell_2, 4.9   + AML_blast, CD34  →  cell_2, AML_blast, 4.9
CD34, cell_3, 11.8  + AML_blast, CD34  →  cell_3, AML_blast, 11.8

expr_long (CD14 rows) + cell_markers (Monocyte → CD14)
-------------------------------------------------------
CD14, cell_1, 1.4   + Monocyte, CD14   →  cell_1, Monocyte, 1.4
CD14, cell_2, 1.1   + Monocyte, CD14   →  cell_2, Monocyte, 1.1
CD14, cell_3, 4.6   + Monocyte, CD14   →  cell_3, Monocyte, 4.6
```

**Result (cell_type_scores):** 6 rows
```
cell_id | cell_type  | cell_type_score
--------|------------|----------------
cell_1  | AML_blast  | 10.1           ← avg(CD34) for cell_1
cell_1  | Monocyte   | 1.4            ← avg(CD14) for cell_1
cell_2  | AML_blast  | 4.9
cell_2  | Monocyte   | 1.1
cell_3  | AML_blast  | 11.8
cell_3  | Monocyte   | 4.6
```

**Interpretation:**
- cell_1: High AML_blast score (10.1), low Monocyte (1.4) → Likely AML blast cell
- cell_2: Low AML_blast (4.9), low Monocyte (1.1) → Ambiguous
- cell_3: High AML_blast (11.8), moderate Monocyte (4.6) → Likely AML blast

---

## Pivot & Integrate

### STEP 6-7: Pivot to Wide Format

**gene_set_wide:**
```
cell_id | Venetoclax_expr
--------|----------------
cell_1  | 8.5
cell_2  | 3.1
cell_3  | 9.8
```

**cell_type_wide:**
```
cell_id | AML_blast_score | Monocyte_score
--------|-----------------|----------------
cell_1  | 10.1            | 1.4
cell_2  | 4.9             | 1.1
cell_3  | 11.8            | 4.6
```

### STEP 8: Combine Cell-Level Features

**Operation:** Join gene_set_wide + cell_type_wide + cell_metadata

**Result (cell_features):** 3 rows
```
cell_id | patient  | cell_type  | Venetoclax_expr | AML_blast_score | Monocyte_score
--------|----------|------------|-----------------|-----------------|----------------
cell_1  | patient1 | AML_blast  | 8.5             | 10.1            | 1.4
cell_2  | patient1 | Monocyte   | 3.1             | 4.9             | 1.1
cell_3  | patient2 | AML_blast  | 9.8             | 11.8            | 4.6
```

---

## Patient-Level Aggregation

### STEP 9: Aggregate to Patient Level

**Operation:** `GROUP BY patient`, `AVG(all features)`

**Calculation for patient1:**
```
Venetoclax_avg  = (8.5 + 3.1) / 2 = 5.8
AML_blast_avg   = (10.1 + 4.9) / 2 = 7.5
Monocyte_avg    = (1.4 + 1.1) / 2 = 1.25
```

**Calculation for patient2:**
```
Venetoclax_avg  = 9.8 / 1 = 9.8
AML_blast_avg   = 11.8 / 1 = 11.8
Monocyte_avg    = 4.6 / 1 = 4.6
```

**Result (patient_features):** 2 rows
```
patient  | Venetoclax_avg | AML_blast_avg | Monocyte_avg
---------|----------------|---------------|---------------
patient1 | 5.8            | 7.5           | 1.25
patient2 | 9.8            | 11.8          | 4.6
```

**🔒 Privacy Achieved:** Individual cell data is now aggregated!

---

## Final Output (ML-Ready)

### STEP 10: Unpivot Drug Response

**drug_response_long:** 4 rows
```
patient  | drug        | response
---------|-------------|----------
patient1 | Venetoclax  | 17.5
patient1 | Azacitidine | 29.0
patient2 | Venetoclax  | 13.6
patient2 | Azacitidine | 13.7
```

### STEP 11: Final Join

**Operation:** `patient_features JOIN drug_response_long ON patient`

**Final Dataset:** 4 rows × 6 columns
```
patient  | drug        | response | Venetoclax_avg | AML_blast_avg | Monocyte_avg
---------|-------------|----------|----------------|---------------|---------------
patient1 | Venetoclax  | 17.5     | 5.8            | 7.5           | 1.25
patient1 | Azacitidine | 29.0     | 5.8            | 7.5           | 1.25
patient2 | Venetoclax  | 13.6     | 9.8            | 11.8          | 4.6
patient2 | Azacitidine | 13.7     | 9.8            | 11.8          | 4.6
```

---

## Machine Learning Interpretation

Each row represents a **patient-drug pair** with:
- **Target:** `response` (drug efficacy to predict)
- **Features:** Patient's cellular profile

### Example: Predicting patient1 response to Venetoclax

**Input Features:**
- Venetoclax_avg = 5.8 (average BCL2 expression across patient's cells)
- AML_blast_avg = 7.5 (average AML blast marker expression)
- Monocyte_avg = 1.25 (average monocyte marker expression)

**Target:**
- response = 17.5 (observed drug efficacy)

**ML Model Task:** Learn pattern like:
```
response = f(Venetoclax_avg, AML_blast_avg, Monocyte_avg, ...)
```

**Hypothesis:**
- Higher Venetoclax_avg (BCL2 expression) → Better response to BCL2 inhibitor
- Higher AML_blast_avg → More aggressive disease, may need higher doses
- Monocyte_avg → Immune context, may modulate response

### Comparison: patient1 vs patient2 for Venetoclax

```
patient2 has:
- Higher Venetoclax_avg (9.8 vs 5.8) → More BCL2 expression
- Higher AML_blast_avg (11.8 vs 7.5) → More blast cells
- Higher Monocyte_avg (4.6 vs 1.25) → More immune cells

But patient1 has better response (17.5 vs 13.6) ❓

ML model will learn from these patterns across all 50 patients!
```

---

## Scaling to Real Data

**Example used:** 3 genes, 3 cells, 2 patients, 2 drugs
**Actual data:** 56 genes, 250 cells, 50 patients, 14 drugs

The **same SQL query structure** processes:
- 14,000 expression values (56 × 250)
- 3,500 drug enrichment scores (250 × 14)
- 1,250 cell type scores (250 × 5)
- 50 patient profiles (aggregated from 5 cells each on average)
- 700 final rows (50 × 14 patient-drug pairs)

**All in a single SQL query!** 🚀















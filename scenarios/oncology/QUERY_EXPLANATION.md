# Feature Engineering SQL Query Explanation

## Overview
This document explains the comprehensive SQL query in `join_config_copy.json` that performs all feature engineering purely within a single Spark SQL query, ensuring privacy-preserving federated computation.

## Challenge
Transform 5 distributed, privacy-sensitive datasets into a patient-drug feature matrix through SQL only, without exposing raw data.

## Input Datasets
1. **sc_expr_matrix.csv** (Genomics Lab) - 56 genes × 250 cells
2. **cell_metadata.csv** (Cancer Institute) - 250 rows with cell, patient, type
3. **drug_response.csv** (Pharmaceutical Co.) - 50 patients × 14 drugs
4. **drug_targets.json** (Pharmaceutical Co.) - Maps 14 drugs → target genes
5. **cell_markers.json** (Biology Lab) - Maps 5 cell types → marker genes

## Query Pipeline (11 CTEs)

### 1. `expr_long` - Unpivot Expression Matrix
```sql
SELECT gene, cell_id, expression 
FROM (SELECT _c0 AS gene, 
      stack(250, 'cell_1', cell_1, 'cell_2', cell_2, ..., 'cell_250', cell_250) 
      AS (cell_id, expression) 
      FROM expr_matrix)
```
**Transforms:** (56 genes × 250 cells) → 14,000 rows (gene, cell_id, expression)
**Purpose:** Convert wide format to long format for aggregation

### 2. `drug_targets_exploded` - Parse Drug Targets JSON
```sql
SELECT drug, target_gene 
FROM (SELECT explode(map('Venetoclax', array('BCL2'), ...)) 
      AS (drug, target_genes)) 
LATERAL VIEW explode(target_genes) AS target_gene
```
**Output:** 25 rows mapping drugs to individual target genes
**Example:** Venetoclax → BCL2, Azacitidine → DNMT1, Azacitidine → IDH2

### 3. `cell_markers_exploded` - Parse Cell Markers JSON
```sql
SELECT cell_type, marker_gene 
FROM (SELECT explode(map('AML_blast', array('CD34', ...), ...)) 
      AS (cell_type, marker_genes)) 
LATERAL VIEW explode(marker_genes) AS marker_gene
```
**Output:** 18 rows mapping cell types to individual marker genes
**Example:** AML_blast → CD34, AML_blast → CD33, Monocyte → CD14

### 4. `gene_set_enrichment` - Drug Target Expression (Feature Engineering Step 1)
```sql
SELECT cell_id, drug, AVG(expression) AS drug_score
FROM expr_long el
INNER JOIN drug_targets_exploded dte ON el.gene = dte.target_gene
GROUP BY cell_id, drug
```
**Purpose:** For each cell, compute average expression of each drug's target genes
**Output:** 250 cells × 14 drugs = 3,500 rows
**Example:** 
- cell_1, Venetoclax → avg(BCL2 expression) = 8.5
- cell_1, Azacitidine → avg(DNMT1, IDH2 expression) = 12.3

### 5. `cell_type_scores` - Cell Type Scoring (Feature Engineering Step 2)
```sql
SELECT cell_id, cell_type, AVG(expression) AS cell_type_score
FROM expr_long el
INNER JOIN cell_markers_exploded cme ON el.gene = cme.marker_gene
GROUP BY cell_id, cell_type
```
**Purpose:** For each cell, compute average expression of each cell type's marker genes
**Output:** 250 cells × 5 cell types = 1,250 rows
**Example:**
- cell_1, AML_blast → avg(CD34, CD33, ...) = 10.2
- cell_1, Monocyte → avg(CD14, CD11b) = 2.1

### 6. `gene_set_wide` - Pivot Drug Scores
```sql
SELECT cell_id,
  MAX(CASE WHEN drug = 'Venetoclax' THEN drug_score END) AS Venetoclax_expr,
  MAX(CASE WHEN drug = 'Azacitidine' THEN drug_score END) AS Azacitidine_expr,
  ...
FROM gene_set_enrichment
GROUP BY cell_id
```
**Purpose:** Convert drug scores from long to wide format
**Output:** 250 rows × 14 drug columns

### 7. `cell_type_wide` - Pivot Cell Type Scores
```sql
SELECT cell_id,
  MAX(CASE WHEN cell_type = 'AML_blast' THEN cell_type_score END) AS AML_blast_score,
  MAX(CASE WHEN cell_type = 'Monocyte' THEN cell_type_score END) AS Monocyte_score,
  ...
FROM cell_type_scores
GROUP BY cell_id
```
**Purpose:** Convert cell type scores from long to wide format
**Output:** 250 rows × 5 cell type columns

### 8. `cell_features` - Join Cell-Level Features (Step 4)
```sql
SELECT cm.cell AS cell_id, cm.patient, cm.type AS cell_type,
  gsw.Venetoclax_expr, ..., gsw.Idasanutlin_expr,
  ctw.AML_blast_score, ..., ctw.Bcell_score
FROM cell_metadata cm
INNER JOIN gene_set_wide gsw ON cm.cell = gsw.cell_id
INNER JOIN cell_type_wide ctw ON cm.cell = ctw.cell_id
```
**Purpose:** Combine all cell-level features with metadata
**Output:** 250 rows × 22 columns (patient, 14 drug expr, 5 cell type scores, cell_type metadata)

### 9. `patient_features` - Aggregate to Patient Level (Step 5)
```sql
SELECT patient,
  AVG(Venetoclax_expr) AS Venetoclax_avg,
  AVG(Azacitidine_expr) AS Azacitidine_avg,
  ...
  AVG(AML_blast_score) AS AML_blast_avg,
  AVG(Monocyte_score) AS Monocyte_avg,
  ...
FROM cell_features
GROUP BY patient
```
**Purpose:** Average all cell-level features per patient
**Output:** 50 patients × 19 features (14 drug expr + 5 cell type scores)
**Meaning:** Each patient's cellular "signature" - average drug target expression and cell type composition

### 10. `drug_response_long` - Unpivot Drug Response
```sql
SELECT patient, drug, response
FROM (SELECT _c0 AS patient,
      stack(14, 'Venetoclax', Venetoclax, 'Azacitidine', Azacitidine, ...)
      AS (drug, response)
      FROM drug_response)
```
**Purpose:** Convert drug response from wide to long format for joining
**Output:** 50 patients × 14 drugs = 700 rows

### 11. Final Join - Patient-Drug Feature Matrix
```sql
SELECT pf.patient, drl.drug, drl.response,
  pf.Venetoclax_avg, ..., pf.Idasanutlin_avg,
  pf.AML_blast_avg, ..., pf.Bcell_avg
FROM patient_features pf
INNER JOIN drug_response_long drl ON pf.patient = drl.patient
```
**Purpose:** Create final dataset with one row per patient-drug combination
**Output:** 700 rows (50 patients × 14 drugs) × 22 columns

## Final Schema
```
patient         | drug        | response | Venetoclax_avg | ... | Bcell_avg
----------------|-------------|----------|----------------|-----|----------
patient1        | Venetoclax  | 17.49    | 5.8           | ... | 3.2
patient1        | Azacitidine | 29.01    | 5.8           | ... | 3.2
...
```

## Privacy Preservation
1. **No Raw Cell Data Exposed**: Cell-level data is aggregated at patient level
2. **Federated Computation**: Each party only exposes what they mount; joins happen in secure compute
3. **Statistical Aggregation**: Individual cell identities are lost through averaging
4. **Minimum Data Movement**: Only final aggregated results leave the secure compute environment

## Feature Engineering Achieved
✅ **Gene Set Enrichment** - Drug target expression per cell (CTE 4)
✅ **Cell Type Scoring** - Marker gene expression per cell (CTE 5)
✅ **Cell-Level Integration** - Combined features per cell (CTE 8)
✅ **Patient-Level Aggregation** - Average features per patient (CTE 9)
✅ **Drug Response Integration** - Final patient-drug matrix (CTE 11)

## Key SQL Techniques Used
- **STACK()** - Unpivot wide tables (expr_matrix, drug_response)
- **MAP() + EXPLODE()** - Parse JSON into relational format
- **LATERAL VIEW** - Flatten nested arrays
- **CASE WHEN + MAX()** - Pivot long tables to wide format
- **Multiple CTEs** - Break complex logic into readable steps
- **AVG() GROUP BY** - Aggregate cell-level to patient-level

## Result
A fully engineered dataset with **19 feature columns** (14 drug target expressions + 5 cell type scores) per patient-drug pair, ready for machine learning, computed entirely within the privacy-preserving joining query without exposing raw single-cell data.



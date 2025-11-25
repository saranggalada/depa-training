# Data Pipeline Flow Visualization

## Input → Feature Engineering → Output

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: 5 DISTRIBUTED DATASETS                         │
└──────────────────────────────────────────────────────────────────────────────┘

📊 Expression Matrix          📋 Cell Metadata          💊 Drug Response
   (Genomics Lab)                (Cancer Institute)        (Pharmaceutical)
   ┌──────────────┐              ┌──────────────┐          ┌──────────────┐
   │ 56 genes     │              │ cell_id      │          │ patient      │
   │ × 250 cells  │              │ patient      │          │ × 14 drugs   │
   │              │              │ cell_type    │          │              │
   │ CD34, CD33,  │              │              │          │ Venetoclax,  │
   │ BCL2, AKT1,  │              │ 250 rows     │          │ Azacitidine, │
   │ ...          │              │              │          │ ...          │
   └──────────────┘              └──────────────┘          └──────────────┘
                                                            50 patients

🧬 Drug Targets JSON          🔬 Cell Markers JSON
   (Pharmaceutical)              (Biology Lab)
   ┌──────────────┐              ┌──────────────┐
   │ drug →       │              │ cell_type →  │
   │   [genes]    │              │   [markers]  │
   │              │              │              │
   │ Venetoclax:  │              │ AML_blast:   │
   │   [BCL2]     │              │   [CD34,     │
   │              │              │    CD33, ...] │
   │ Azacitidine: │              │              │
   │   [DNMT1,    │              │ Monocyte:    │
   │    IDH2]     │              │   [CD14,     │
   │ ...          │              │    CD11b]    │
   └──────────────┘              └──────────────┘


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: UNPIVOT & PARSE JSON                          │
└──────────────────────────────────────────────────────────────────────────────┘

   Expression Matrix                Drug Targets              Cell Markers
   (56×250 → 14,000 rows)          (exploded)                (exploded)
   ┌──────────────────┐            ┌──────────────┐          ┌──────────────┐
   │ gene  | cell | exp│            │ drug | gene │          │ type | gene │
   │-------|------|----│            │------|------│          │------|------│
   │ CD34  | c_1  |10.1│            │ Vene | BCL2 │          │ AML  | CD34 │
   │ CD34  | c_2  | 4.9│            │ Azac |DNMT1 │          │ AML  | CD33 │
   │ CD33  | c_1  |10.7│            │ Azac | IDH2 │          │ Mono | CD14 │
   │ CD33  | c_2  | 3.1│            │ ...  | ...  │          │ ...  | ...  │
   │ ...   | ...  | ...│            └──────────────┘          └──────────────┘
   └──────────────────┘


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│               STEP 2: GENE SET ENRICHMENT (Drug Targets)                     │
└──────────────────────────────────────────────────────────────────────────────┘

   JOIN expr_long + drug_targets ON gene = target_gene
   GROUP BY cell_id, drug → AVG(expression)

   ┌───────────────────────────────────────┐
   │ cell_id  | drug        | drug_score  │
   │----------|-------------|-------------│
   │ cell_1   | Venetoclax  | 8.5         │  ← avg(BCL2 expr in cell_1)
   │ cell_1   | Azacitidine | 12.3        │  ← avg(DNMT1, IDH2 in cell_1)
   │ cell_1   | MK-2206     | 10.1        │  ← avg(AKT1, AKT2, AKT3 in cell_1)
   │ cell_2   | Venetoclax  | 3.1         │
   │ ...      | ...         | ...         │
   └───────────────────────────────────────┘
   3,500 rows (250 cells × 14 drugs)


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│               STEP 3: CELL TYPE SCORING (Marker Genes)                       │
└──────────────────────────────────────────────────────────────────────────────┘

   JOIN expr_long + cell_markers ON gene = marker_gene
   GROUP BY cell_id, cell_type → AVG(expression)

   ┌────────────────────────────────────────────┐
   │ cell_id  | cell_type  | cell_type_score   │
   │----------|------------|-------------------│
   │ cell_1   | AML_blast  | 10.2              │  ← avg(CD34,CD33,... in c_1)
   │ cell_1   | Monocyte   | 2.1               │  ← avg(CD14,CD11b in c_1)
   │ cell_1   | Tcell      | 1.5               │  ← avg(CD3,CD4,CD8 in c_1)
   │ cell_2   | AML_blast  | 2.3               │
   │ cell_2   | Monocyte   | 8.7               │
   │ ...      | ...        | ...               │
   └────────────────────────────────────────────┘
   1,250 rows (250 cells × 5 cell types)


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: PIVOT TO WIDE FORMAT                              │
└──────────────────────────────────────────────────────────────────────────────┘

   CASE WHEN + MAX() to pivot columns

   ┌──────────────────────────────────────────────────────────────────────────┐
   │ cell_id | Venetoclax | Azacitidine | ... | AML_blast | Monocyte | ...    │
   │---------|------------|-------------|-----|-----------|----------|--------│
   │ cell_1  | 8.5        | 12.3        | ... | 10.2      | 2.1      | ...    │
   │ cell_2  | 3.1        | 4.2         | ... | 2.3       | 8.7      | ...    │
   │ cell_3  | 9.8        | 11.7        | ... | 11.5      | 1.8      | ...    │
   │ ...     | ...        | ...         | ... | ...       | ...      | ...    │
   └──────────────────────────────────────────────────────────────────────────┘
   250 rows × 19 feature columns (14 drug expr + 5 cell type scores)


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│              STEP 5: JOIN WITH METADATA & AGGREGATE TO PATIENT               │
└──────────────────────────────────────────────────────────────────────────────┘

   JOIN with cell_metadata → GROUP BY patient → AVG(all features)

   Cell Features (250 rows)         Patient Features (50 rows)
   ┌──────────────────────┐         ┌──────────────────────────────┐
   │ cell_1  | patient21  │         │ patient1  | Venetoclax_avg  │
   │ cell_2  | patient32  │  ━━━━━> │ patient2  | Azacitidine_avg │
   │ cell_3  | patient23  │  AVG()  │ ...       | ...             │
   │ ...     | ...        │         │ patient50 | Bcell_avg       │
   └──────────────────────┘         └──────────────────────────────┘


                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│                STEP 6: JOIN WITH DRUG RESPONSE (UNPIVOTED)                   │
└──────────────────────────────────────────────────────────────────────────────┘

   Patient Features × Drug Response → Final Dataset

   ┌────────────────────────────────────────────────────────────────────────┐
   │ patient | drug       | response | Venetoclax_avg | ... | Bcell_avg    │
   │---------|------------|----------|----------------|-----|-------------│
   │ patient1| Venetoclax | 17.49    | 5.8            | ... | 3.2         │
   │ patient1| Azacitidine| 29.01    | 5.8            | ... | 3.2         │
   │ patient1| MK-2206    | 24.64    | 5.8            | ... | 3.2         │
   │ ...     | ...        | ...      | ...            | ... | ...         │
   │ patient2| Venetoclax | 13.64    | 7.1            | ... | 4.1         │
   │ patient2| Azacitidine| 13.67    | 7.1            | ... | 4.1         │
   │ ...     | ...        | ...      | ...            | ... | ...         │
   └────────────────────────────────────────────────────────────────────────┘
   700 rows (50 patients × 14 drugs) × 22 columns

                                ⬇️
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT: ML-READY DATASET                             │
└──────────────────────────────────────────────────────────────────────────────┘

🎯 Final Features per Patient-Drug Pair:
   • 2 Identifiers: patient, drug
   • 1 Target: response (drug efficacy)
   • 19 Features:
     ├─ 14 Drug Target Expression Features
     │  └─ Average cellular expression of each drug's target genes
     └─ 5 Cell Type Composition Features
        └─ Average cellular marker expression for each cell type

📊 Shape: 700 rows × 22 columns
📈 Ready for: Regression, Classification, Feature Selection
🔒 Privacy: No individual cell data exposed; all aggregated at patient level
```

## Key Transformations Summary

| Step | Input Shape | Output Shape | Transformation |
|------|-------------|--------------|----------------|
| 1. Unpivot Expression | 56×250 | 14,000×3 | Wide → Long (stack) |
| 2. Parse JSONs | 2 JSON files | 43 rows | JSON → Relational (explode) |
| 3. Gene Set Enrichment | 14,000 rows | 3,500 rows | Join + Aggregate |
| 4. Cell Type Scores | 14,000 rows | 1,250 rows | Join + Aggregate |
| 5. Pivot Features | 4,750 rows | 250×19 | Long → Wide (CASE WHEN) |
| 6. Join Metadata | 250 + 250 | 250×22 | Inner Join |
| 7. Patient Aggregation | 250×22 | 50×19 | Group By + Average |
| 8. Unpivot Drug Response | 50×14 | 700×3 | Wide → Long (stack) |
| 9. Final Join | 50 + 700 | 700×22 | Inner Join |

## Privacy Properties

✅ **Cell-level data never leaves secure compute**
✅ **Only aggregated patient-level statistics in output**
✅ **Minimum 5 cells per patient** (statistical anonymization)
✅ **No raw gene expression values exposed**
✅ **Each data provider only exposes their dataset**
✅ **Feature engineering happens in federated SQL engine**

## Machine Learning Use Case

This dataset enables training models to predict:
- **Drug response prediction**: Given a patient's cellular profile, predict efficacy of each drug
- **Patient stratification**: Cluster patients by cellular signatures
- **Biomarker discovery**: Identify which cell types/drug targets correlate with response
- **Personalized medicine**: Recommend optimal drugs per patient profile















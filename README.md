# Cancer-RPCA (Cancer RNA-Protein Correlation Analyzer)

## Overview

Cancer-RPCA is a Python-based bioinformatics tool designed to investigate the relationship between mRNA abundance and protein abundance across human cancers.

The project utilizes publicly available proteogenomic datasets from the Clinical Proteomic Tumor Analysis Consortium (CPTAC) and provides a framework for comparing mRNA-protein concordance across multiple cancer types, normal adjacent tissues, tumor progression stages, and individual genes of interest.

---

## Scientific Background

Large-scale transcriptomic studies have transformed cancer research by enabling genome-wide characterization of gene expression patterns. However, mRNA abundance is often an insufficient predictor of protein abundance.

Protein levels are influenced by multiple regulatory mechanisms beyond transcription, including translational regulation, protein degradation, protein complex assembly, and post-translational modifications, resulting in mRNA-protein discrepancies.

Understanding when transcript abundance accurately reflects protein abundance is important for interpreting RNA-sequencing studies, identifying pathways regulated at the protein level, and improving biomarker discovery.

---

## Scientific Motivation

Many cancer studies rely heavily on transcriptomic measurements because RNA sequencing is widely available and relatively inexpensive. However, biological functions are ultimately carried out by proteins rather than transcripts.

This project aims to investigate:

1. How mRNA-protein correlation differs between cancer types.
2. Whether tumors exhibit different RNA-protein relationships compared to normal adjacent tissues.
3. Whether mRNA-protein correlation changes during tumor progression and across pathological stages.
4. Which cancer types maintain stronger or weaker transcript-to-protein coupling.
5. Which specific genes show the largest change in mRNA-protein coupling between tumor and normal tissue ("switching genes").

The resulting analyses may provide insights into post-transcriptional regulation and proteogenomic differences across cancers.

---

## Project Objectives

The program performs four main analyses.

### 1. Cancer-Type Correlation Analysis

For each CPTAC cancer cohort:

- Match transcriptomic and proteomic measurements from the exact same samples (to eliminate biological variation).
- Identify genes measured in both datasets.
- Handle missing values (NaNs) before computing correlations.
- Calculate gene-wise mRNA-protein Spearman correlations across patients.
- Compute summary statistics describing the overall mRNA-protein correlation of the cohort.

**Outputs:**

- Per-gene Spearman correlation tables (CSV)
- Correlation distribution histograms (mean and median annotated)
- Cross-cancer boxplot
- Summary heatmap

---

### 2. Tumor vs. Normal Tissue Comparison

When normal adjacent tissue (NAT) samples are available:

- Split samples into tumor and normal groups using clinical metadata.
- Calculate mRNA-protein correlations in tumor samples.
- Calculate mRNA-protein correlations in normal samples.
- Compare global concordance between the two groups.

**Note:** In CPTAC, normal adjacent tissue is collected from the same patients as the tumor samples, making this a particularly reliable within-patient comparison.

**Outputs:**

- Per-gene Spearman correlation tables for tumor samples and normal samples separately (CSV)
- Delta table showing the change in Spearman r (Δr = tumor_r − normal_r) per gene (CSV)
- Summary table comparing mean and median Spearman r between tumor and normal groups (CSV)
- Violin plots comparing tumor vs. normal Spearman r distributions

---

### 3. Stage-Specific Analysis

For cancer types with pathological stage annotations:

- Separate samples according to tumor stage.
- Calculate mRNA-protein correlations within each stage.
- Compare correlation distributions across stages.

**Note:** Stage comparisons are cross-sectional - different patients at different stages - not longitudinal tracking of the same patient over time.

**Outputs:**

- Per-gene Spearman correlation tables for each tumor stage separately (CSV)
- Summary table comparing mean and median Spearman r across stages (CSV)
- Stage comparison boxplots

---

### 4. Switching Genes Analysis

Identifies genes whose mRNA-protein correlation changes most between tumor and normal tissue and between stage progression:

**Tumor vs. Normal:**

- Computes the delta (Δr = tumor_r - normal_r) for each gene.
- Selects the top N genes with the largest absolute Δr, split between genes that gain and lose coupling in tumor.
- Generates scatter plots (mRNA vs. protein) for each switching gene, shown side-by-side for tumor and normal.
  
**Stage Progression:**
- Computes the delta (Δr = late_stage_r - early_stage_r) for each gene.
- Selects the top N genes with the largest absolute Δr, split between genes that gain and lose coupling as the disease progresses.
- Generates scatter plots across all available stages.

Genes are selected by largest |Δr| rather than highest r, because a gene that gains or loses coupling between contexts is more biologically interesting than one that is simply consistently correlated.

**Outputs:**
- Switching genes table with Δr for tumor vs. normal (CSV)
- Switching genes table with Δr  for stage progression (CSV)
- Multi-panel scatter plots for top switching genes - tumor vs. normal (one row per gene, Tumor | Normal columns)
- Multi-panel scatter plots for top switching genes - stage progression (one row per gene, one column per stage)

---

## Input Data

The project uses publicly available CPTAC datasets, accessed through the `cptac` Python package (no manual download required).
 
| Data type | Source |
|---|---|
| Transcriptomics | BCM (Baylor College of Medicine) |
| Proteomics | UMich (University of Michigan) |
| Clinical metadata | MSSM (Icahn School of Medicine at Mount Sinai) |
 
**Supported cancer types:**
 
| Short name | Cancer type |
|---|---|
| brca | Breast cancer |
| luad | Lung adenocarcinoma |
| gbm | Glioblastoma |
| hnscc | Head and neck squamous cell carcinoma |
| lscc | Lung squamous cell carcinoma |
| ov | Ovarian cancer |
| pdac | Pancreatic ductal adenocarcinoma |
| ucec | Uterine corpus endometrial carcinoma |
| ccrcc | Clear cell renal cell carcinoma |

---

## Output

All results are organized into subfolders:
 
```
figures/
├── correlations/     ← per-cancer mRNA-protein correlation histograms
├── comparisons/      ← cross-cancer boxplot and summary heatmap
├── tumor_normal/     ← violin plots and switching genes scatter plots
└── stages/           ← stage comparison boxplots and switching genes scatter plots
 
results/
├── correlations/     ← per-cancer gene correlation CSVs
├── summaries/        ← cohort-level summary statistics
├── tumor_normal/     ← tumor/normal correlation and delta CSVs
└── stages/           ← per-stage correlation and delta CSVs
```

### Summary Tables (CSV)

- Mean and median mRNA-protein correlation (Spearman r) per cancer type
- Number of matched genes 
- Number of analysed samples
- Percentage of genes with positive correlation (r > 0)
- Percentage of genes with strong correlation (r > 0.5)

### Visualizations

- Histograms of correlation distributions (mean and median annotated) per cancer
- Summary boxplot comparing all cancer types
- Heatmap summarizing concordance statistics across all cancer types
- Violin plots comparing tumor vs. normal tissue
- Stage comparison boxplots
- Multi-panel scatter plots for top switching genes - tumor vs. normal 
- Multi-panel scatter plots for top switching genes - stage progression

---

## Program Architecture

```text
Data Loader (data.loader.py)
      │
      ▼
Sample Matching + Gene Filtering
      │
      ▼
Gene Filtering
      │
      ▼
Correlation Engine (correlation.py)
      │
      ├── Cancer-Type Analysis
      ├── Tumor vs. Normal Analysis  (normal_comparison.py)
      ├── Stage-Specific Analysis    (stage_analysis.py)
      └── Switching Genes Analysis   (gene_analysis.py)
      │
      ▼
Visualization Module  (visualization.py)
      │
      ▼
Results Export  (results/ and figures/)
```

---

## Dependencies

The project uses several third-party libraries that need to be installed before running it:

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `cptac`
- `pytest`

Install all dependencies:
 
```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/rotemvazana/RNA-Protein-Correlation-Analyzer.git

cd RNA-Protein-Correlation-Analyzer

pip install -r requirements.txt
```

---

## Running the Project
 
**Basic usage — analyze one cancer type (BRCA by default):**
 
```bash
python main.py
```
 
**Analyze a specific cancer type (for example - luad):**
 
```bash
python main.py --cancer luad
```
 
**Analyze multiple cancer types:**
 
```bash
python main.py --cancer brca luad ucec
```
 
**Run all cancer types:**
 
```bash
python main.py --all-cancers
```
 
**Add tumor vs. normal comparison:**
 
```bash
python main.py --all-cancers --compare-normal
```
 
**Add stage-specific analysis:**
 
```bash
python main.py --all-cancers --stage-analysis
```
 
**Full pipeline — all analyses:**
 
```bash
python main.py --all-cancers --compare-normal --stage-analysis --switching-genes
```
 
**Optional flags:**
 
| Flag | Description | Default |
|---|---|---|
| `--cancer` | One or more cancer type short names | brca |
| `--all-cancers` | Run all 9 supported cancer types | — |
| `--compare-normal` | Tumor vs. normal comparison | off |
| `--stage-analysis` | Stage-specific analysis | off |
| `--switching-genes` | Top switching genes scatter plots | off |
| `--n-genes` | Number of switching genes to plot | 5 |
| `--min-samples` | Minimum paired samples per gene | 10 |
| `--rna-source` | Lab source for transcriptomics | bcm |
| `--protein-source` | Lab source for proteomics | umich |
 
---

## Running the Tests

```bash
pytest tests/
```

---

## Repository Structure

```text
project/

├── src/
│   ├── data_loader.py
│   ├── correlation.py
│   ├── normal_comparison.py
│   ├── stage_analysis.py
│   └── visualization.py
│
├── results/
├── figures/
├── tests/
├── requirements.txt
└── README.md
```

---

## Future Extensions

Potential future improvements include:

- Pathway-level correlation analysis
- Mutation-specific analyses
- Integration with TCGA datasets
- Interactive dashboard
- Functional enrichment analysis of highly discordant genes

---

This project was written as part of the [Python course 20263062](https://github.com/Code-Maven/wis-python-course-2026-03/) at the Weizmann Institute of Science taught by [Gábor Szabó](https://github.com/szabgab).

# Cancer-RPCA (Cancer RNA-Protein Correlation Analyzer)

**Note:** This document serves as the initial project proposal. As implementation progresses, the architecture, features, and specific pipeline components may be refined.
The README will be continuously updated to reflect the current state of the project.

## Overview

Cancer-RPCA (Cancer RNA-Protein Correlation Analyzer) is a Python-based bioinformatics tool designed to investigate the relationship between mRNA abundance and protein abundance across human cancers.

The project utilizes publicly available proteogenomic datasets from the Clinical Proteomic Tumor Analysis Consortium (CPTAC) and provides a framework for comparing RNA-protein concordance across multiple cancer types, normal adjacent tissues, and tumor progression stages.

The goal of the project is to explore how transcript-to-protein relationships vary across biological and clinical contexts and to provide a reusable research tool for proteogenomic data analysis.

---

## Scientific Background

Large-scale transcriptomic studies have transformed cancer research by enabling genome-wide characterization of gene expression patterns. However, mRNA abundance is often an insufficient predictor of protein abundance.

Protein levels are influenced by multiple regulatory mechanisms beyond transcription, including translational regulation, protein degradation, protein complex assembly, and post-translational modifications, resulting in RNA-protein discrepancies.

Understanding when transcript abundance accurately reflects protein abundance is important for interpreting RNA-sequencing studies, identifying pathways regulated at the protein level, and improving biomarker discovery.

---

## Scientific Motivation

Many cancer studies rely heavily on transcriptomic measurements because RNA sequencing is widely available and relatively inexpensive. However, biological functions are ultimately carried out by proteins rather than transcripts.

This project aims to investigate:

1. How RNA-protein correlation differs between cancer types.
2. Whether tumors exhibit different RNA-protein relationships compared to normal adjacent tissues.
3. Whether RNA-protein correlation changes during tumor progression and across pathological stages.
4. Which cancer types maintain stronger or weaker transcript-to-protein coupling.

The resulting analyses may provide insights into post-transcriptional regulation and proteogenomic differences across cancers.

---

## Project Objectives

The program will perform three main analyses.

### 1. Cancer-Type Correlation Analysis

For each CPTAC cancer cohort:

- Match transcriptomic and proteomic measurements from the exact same samples (to eliminate biological variation).
- Identify genes measured in both datasets.
- Handle missing values (NaNs) before computing correlations.
- Calculate gene-wise RNA-protein Spearman correlations across patients.
- Compute summary statistics describing the overall RNA-protein correlation of the cohort.

Outputs:

- Correlation distributions
- Mean and median correlation values
- Ranked lists of genes by correlation

---

### 2. Tumor vs. Normal Tissue Comparison

When normal adjacent tissue samples are available:

- Calculate RNA-protein correlations in tumor samples.
- Calculate RNA-protein correlations in normal samples.
- Compare global concordance between the two groups.

Outputs:

- Tumor-versus-normal correlation distributions
- Differences in average correlation
- Comparative visualizations

---

### 3. Stage-Specific Analysis

For cancer types with pathological stage annotations:

- Separate samples according to tumor stage.
- Calculate RNA-protein correlations within each stage.
- Compare correlation distributions across stages.

Outputs:

- Stage-specific correlation profiles
- Trends in RNA-protein concordance during disease progression
- Comparative visualizations

---

## Input Data

The project uses publicly available CPTAC datasets.

Expected input data include:

- Transcriptomics (RNA-seq)
- Proteomics (mass spectrometry)
- Clinical metadata

Examples of metadata used by the analysis:

- Cancer type
- Sample status (tumor or normal)
- Pathological stage

Data will be accessed through the CPTAC Python package.

---

## Output

The program will generate:

### Summary Tables

- Mean RNA-protein correlation
- Median RNA-protein correlation
- Number of matched genes
- Number of analyzed samples

### Visualizations

- Histograms of correlation distributions
- Boxplots comparing cancer types
- Tumor-versus-normal comparison plots
- Stage comparison plots
- Heatmaps summarizing cohort-level concordance

### Exported Results

- CSV tables
- figures
- Analysis summaries

---

## Program Architecture

```text
Data Loader
      │
      ▼
Sample Matching
      │
      ▼
Gene Filtering
      │
      ▼
Correlation Engine
      │
      ├── Cancer-Type Analysis
      ├── Tumor vs Normal Analysis
      └── Stage Analysis
      │
      ▼
Visualization Module
      │
      ▼
Results Export
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

Optional future extensions:

- `streamlit`
- `plotly`

---

## Installation

```bash
git clone https://github.com/rotemvazana/RNA-Protein-Correlation-Analyzer.git

cd RNA-Protein-Correlation-Analyzer

pip install -r requirements.txt
```

---

## Running the Project

Example commands (the names will be changed after writing the program):

```bash
python main.py
```

Analyze a specific cancer type:

```bash
python main.py --cancer brca
```

Run tumor-versus-normal analysis:

```bash
python main.py --compare-normal
```

Run stage-specific analysis:

```bash
python main.py --stage-analysis
```

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

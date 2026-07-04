"""
normal_comparison.py
--------------------
Tumor vs. Normal Tissue Comparison Analysis.

In CPTAC datasets, some samples are tumor tissue and others are normal
adjacent tissue (NAT) — healthy tissue taken from near the tumor.

By splitting samples by tissue type and computing RNA-protein correlations
separately in each group, we can ask:
  "Is the RNA-protein relationship different in cancer cells vs. healthy cells?"

This module handles:
  - Splitting samples into tumor and normal groups using clinical metadata.
  - Running correlation analysis on each group independently.
  - Returning structured results for comparison and visualization.
"""

import pandas as pd
from .correlation import compute_gene_correlations, summarize_correlations


# Keywords used to identify the sample type column in clinical metadata.
# CPTAC datasets use different column names across cancer types.
SAMPLE_TYPE_KEYWORDS = ["sample_type", "tissue_type", "tumor_or_normal", "type"]

# Values that indicate a tumor sample (case-insensitive substring match).
TUMOR_KEYWORDS = ["tumor", "tumour", "cancer", "primary"]

# Values that indicate a normal sample (case-insensitive substring match).
NORMAL_KEYWORDS = ["normal", "adjacent", "nat", "healthy"]


def find_sample_type_column(clinical: pd.DataFrame) -> str | None:
    """
    Search the clinical metadata table for a column that encodes sample type
    (tumor vs. normal).

    Parameters
    ----------
    clinical : pd.DataFrame
        Clinical metadata table (rows = samples, columns = clinical variables).

    Returns
    -------
    str or None
        The name of the sample-type column if found, otherwise None.
    """

    for col in clinical.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(keyword in col_lower for keyword in SAMPLE_TYPE_KEYWORDS):
            return col

    # If no column matched by name, search column values for tumor/normal terms.
    for col in clinical.columns:
        unique_vals = clinical[col].dropna().astype(str).str.lower().unique()
        has_tumor = any(any(kw in v for kw in TUMOR_KEYWORDS) for v in unique_vals)
        has_normal = any(any(kw in v for kw in NORMAL_KEYWORDS) for v in unique_vals)
        if has_tumor and has_normal:
            return col

    return None


def split_tumor_normal(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    clinical: pd.DataFrame,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame],
]:
    """
    Split matched RNA and protein tables into tumor and normal groups
    based on clinical metadata.

    Parameters
    ----------
    rna : pd.DataFrame
        Matched RNA table (rows = samples).
    protein : pd.DataFrame
        Matched protein table (rows = samples).
    clinical : pd.DataFrame
        Clinical metadata table (rows = samples).

    Returns
    -------
    (rna_tumor, protein_tumor) : tuple of DataFrames
        Subset of RNA and protein tables for tumor samples.
    (rna_normal, protein_normal) : tuple of DataFrames
        Subset of RNA and protein tables for normal samples.

    Raises
    ------
    ValueError
        If no sample-type column is found in clinical metadata,
        or if fewer than 5 samples are found in either group.
    """

    sample_type_col = find_sample_type_column(clinical)

    if sample_type_col is None:
        raise ValueError(
            "Could not find a sample-type column in clinical metadata. "
            "Tumor vs. normal comparison is not possible for this dataset."
        )

    print(f"[NormalComparison] Using column '{sample_type_col}' to split tumor vs. normal.")

    # Align the clinical index with our matched RNA/protein samples.
    # CPTAC clinical tables often use a MultiIndex (Patient_ID, Sample_ID).
    # We flatten it to a simple index first so .loc[] returns a Series, not a DataFrame.
    clinical_flat = clinical.copy()
    if isinstance(clinical_flat.index, pd.MultiIndex):
        clinical_flat = clinical_flat.reset_index(level=0, drop=True)

    shared_samples = rna.index.intersection(clinical_flat.index)
    if len(shared_samples) == 0:
        raise ValueError(
            "No overlapping samples between RNA/protein tables and clinical metadata. "
            "Tumor vs. normal comparison cannot be performed."
        )

    sample_col_data = clinical_flat.loc[shared_samples, sample_type_col]
    # If we still got a DataFrame (duplicate index values), take the first column.
    if isinstance(sample_col_data, pd.DataFrame):
        sample_col_data = sample_col_data.iloc[:, 0]

    # Convert to string, then replace "nan" (stringified NaN) with empty string
    # so NaN entries don't accidentally match any keyword.
    clinical_aligned = sample_col_data.fillna("").astype(str).str.lower().str.strip()
    clinical_aligned = clinical_aligned.replace("nan", "")

    # Classify each sample as tumor or normal.
    is_tumor = clinical_aligned.apply(
        lambda v: bool(v) and any(kw in v for kw in TUMOR_KEYWORDS)
    )
    is_normal = clinical_aligned.apply(
        lambda v: bool(v) and any(kw in v for kw in NORMAL_KEYWORDS)
    )

    tumor_samples = clinical_aligned[is_tumor].index
    normal_samples = clinical_aligned[is_normal].index

    print(f"[NormalComparison] Tumor samples: {len(tumor_samples)}, Normal samples: {len(normal_samples)}")

    if len(tumor_samples) < 5:
        raise ValueError(f"Too few tumor samples found ({len(tumor_samples)}). Cannot compute reliable correlations.")
    if len(normal_samples) < 5:
        raise ValueError(f"Too few normal samples found ({len(normal_samples)}). Cannot compute reliable correlations.")

    rna_tumor = rna.loc[rna.index.isin(tumor_samples)]
    protein_tumor = protein.loc[protein.index.isin(tumor_samples)]

    rna_normal = rna.loc[rna.index.isin(normal_samples)]
    protein_normal = protein.loc[protein.index.isin(normal_samples)]

    return (rna_tumor, protein_tumor), (rna_normal, protein_normal)


def run_tumor_normal_analysis(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    clinical: pd.DataFrame,
    cancer_type: str,
    min_samples: int = 5,
) -> dict:
    """
    Full tumor vs. normal comparison pipeline.

    Steps:
      1. Split samples into tumor and normal groups.
      2. Compute per-gene Spearman correlations in each group.
      3. Compute summary statistics for each group.
      4. Compute per-gene difference in correlation (tumor minus normal).

    Parameters
    ----------
    rna : pd.DataFrame
        Matched RNA table.
    protein : pd.DataFrame
        Matched protein table.
    clinical : pd.DataFrame
        Clinical metadata.
    cancer_type : str
        Cancer type label (used in output summaries).
    min_samples : int
        Minimum valid sample pairs per gene.

    Returns
    -------
    results : dict with keys:
        - "tumor_corr"   : per-gene correlation DataFrame for tumor
        - "normal_corr"  : per-gene correlation DataFrame for normal
        - "tumor_summary": summary dict for tumor group
        - "normal_summary": summary dict for normal group
        - "delta_corr"   : DataFrame of per-gene (tumor_r - normal_r)
    """

    print(f"\n=== Tumor vs. Normal Analysis: {cancer_type.upper()} ===")

    (rna_tumor, protein_tumor), (rna_normal, protein_normal) = split_tumor_normal(
        rna, protein, clinical
    )

    print("[NormalComparison] Computing correlations in tumor samples ...")
    tumor_corr = compute_gene_correlations(rna_tumor, protein_tumor, min_samples=min_samples)
    tumor_summary = summarize_correlations(tumor_corr, label=f"{cancer_type}_tumor")

    print("[NormalComparison] Computing correlations in normal samples ...")
    normal_corr = compute_gene_correlations(rna_normal, protein_normal, min_samples=min_samples)
    normal_summary = summarize_correlations(normal_corr, label=f"{cancer_type}_normal")

    print(f"\n  Tumor  — Mean r: {tumor_summary['mean_r']}, Median r: {tumor_summary['median_r']}")
    print(f"  Normal — Mean r: {normal_summary['mean_r']}, Median r: {normal_summary['median_r']}")

    # Compute per-gene delta (tumor r minus normal r) for genes present in both.
    tumor_r = tumor_corr.set_index("gene")["spearman_r"]
    normal_r = normal_corr.set_index("gene")["spearman_r"]
    common_genes = tumor_r.index.intersection(normal_r.index)

    delta_corr = pd.DataFrame({
        "gene": common_genes,
        "tumor_r": tumor_r.loc[common_genes].values,
        "normal_r": normal_r.loc[common_genes].values,
        "delta_r": (tumor_r.loc[common_genes] - normal_r.loc[common_genes]).values,
    }).sort_values("delta_r", ascending=False).reset_index(drop=True)

    return {
        "tumor_corr": tumor_corr,
        "normal_corr": normal_corr,
        "tumor_summary": tumor_summary,
        "normal_summary": normal_summary,
        "delta_corr": delta_corr,
    }

"""
correlation.py
--------------
The core analysis engine.

For each gene measured in both RNA and protein datasets, this module computes
the Spearman correlation between RNA abundance and protein abundance across
all matched patient samples.

Spearman correlation is preferred over Pearson here because:
  - Proteomics and RNA-seq data are often not normally distributed.
  - Spearman is rank-based and more robust to outliers.
  - It captures monotonic (not just linear) relationships.

The output is a table of per-gene correlations and a summary of global statistics.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_gene_correlations(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Compute Spearman RNA-protein correlation for each gene across samples.

    For each gene column, this function:
      1. Drops samples where either the RNA or protein value is missing (NaN).
      2. Skips the gene if too few samples remain after dropping NaNs.
      3. Computes the Spearman correlation between the RNA and protein vectors.

    Parameters
    ----------
    rna : pd.DataFrame
        RNA expression table. Rows = samples, columns = genes.
        Must be pre-matched and filtered (same samples and genes as `protein`).
    protein : pd.DataFrame
        Protein abundance table. Rows = samples, columns = genes.
        Must be pre-matched and filtered (same samples and genes as `rna`).
    min_samples : int
        Minimum number of non-NaN paired samples required to compute a
        correlation for a gene. Genes with fewer valid pairs are skipped.
        Default is 10.

    Returns
    -------
    results : pd.DataFrame
        A table with one row per gene, containing:
          - gene          : gene name
          - spearman_r    : Spearman correlation coefficient (−1 to +1)
          - p_value       : two-sided p-value for the correlation
          - n_samples     : number of paired samples used in this gene's correlation
    """

    gene_names = rna.columns.tolist()
    records = []  # We'll build a list of dicts, then convert to DataFrame.

    print(f"[Correlation] Computing Spearman correlations for {len(gene_names)} genes ...")

    for gene in gene_names:
        # Extract the RNA and protein vectors for this gene.
        rna_vec = rna[gene].values.astype(float)
        protein_vec = protein[gene].values.astype(float)

        # Find positions where BOTH RNA and protein values are non-NaN.
        valid_mask = ~np.isnan(rna_vec) & ~np.isnan(protein_vec)
        n_valid = valid_mask.sum()

        # Skip genes with too few valid pairs.
        if n_valid < min_samples:
            continue

        # Compute Spearman correlation on the valid (non-NaN) pairs.
        r, p = stats.spearmanr(rna_vec[valid_mask], protein_vec[valid_mask])

        records.append({
            "gene": gene,
            "spearman_r": round(r, 4),
            "p_value": p,
            "n_samples": n_valid,
        })

    results = pd.DataFrame(records)

    if results.empty:
        raise ValueError(
            "No genes passed the minimum-sample threshold. "
            "Try lowering min_samples or check that RNA and protein tables overlap correctly."
        )

    # Sort by correlation (descending) so the most correlated genes appear first.
    results = results.sort_values("spearman_r", ascending=False).reset_index(drop=True)

    print(f"[Correlation] Successfully computed correlations for {len(results)} genes.")

    return results


def summarize_correlations(correlation_df: pd.DataFrame, label: str = "") -> dict:
    """
    Compute summary statistics from a per-gene correlation table.

    Parameters
    ----------
    correlation_df : pd.DataFrame
        Output of compute_gene_correlations().
    label : str
        Optional label (e.g. cancer type or tissue group) to include in the summary.

    Returns
    -------
    summary : dict
        Dictionary containing:
          - label           : provided label string
          - n_genes         : number of genes analyzed
          - mean_r          : mean Spearman correlation across genes
          - median_r        : median Spearman correlation
          - std_r           : standard deviation
          - pct_positive    : percentage of genes with positive correlation
          - pct_high        : percentage of genes with r > 0.5 (strong coupling)
    """

    r_values = correlation_df["spearman_r"].dropna()

    summary = {
        "label": label,
        "n_genes": len(r_values),
        "mean_r": round(r_values.mean(), 4),
        "median_r": round(r_values.median(), 4),
        "std_r": round(r_values.std(), 4),
        "pct_positive": round((r_values > 0).mean() * 100, 1),
        "pct_high": round((r_values > 0.5).mean() * 100, 1),
    }

    return summary


def run_cancer_type_analysis(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    cancer_type: str,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Full cancer-type correlation analysis pipeline.

    Combines compute_gene_correlations() and summarize_correlations()
    into a single convenience function.

    Parameters
    ----------
    rna : pd.DataFrame
        Matched, filtered RNA table.
    protein : pd.DataFrame
        Matched, filtered protein table.
    cancer_type : str
        Name of the cancer type (used as a label in the summary).
    min_samples : int
        Minimum valid samples required per gene.

    Returns
    -------
    corr_df : pd.DataFrame
        Per-gene correlation table.
    summary : dict
        Summary statistics dictionary.
    """

    print(f"\n=== Cancer-Type Analysis: {cancer_type.upper()} ===")

    corr_df = compute_gene_correlations(rna, protein, min_samples=min_samples)
    summary = summarize_correlations(corr_df, label=cancer_type)

    print(f"  Mean Spearman r  : {summary['mean_r']}")
    print(f"  Median Spearman r: {summary['median_r']}")
    print(f"  Genes analyzed   : {summary['n_genes']}")
    print(f"  % positively correlated: {summary['pct_positive']}%")
    print(f"  % highly correlated (r>0.4): {summary['pct_high']}%")

    return corr_df, summary

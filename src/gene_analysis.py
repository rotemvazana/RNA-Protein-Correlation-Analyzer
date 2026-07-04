"""
gene_analysis.py
----------------
Identifies the most interesting "switching genes" — genes whose mRNA-protein
correlation changes the most between two biological contexts — and prepares
their raw expression data for scatter plot visualization.

Two comparison types are supported:
  1. Tumor vs. Normal: genes that gain or lose mRNA-protein coupling in cancer.
  2. Stage progression: genes that change coupling from early to late stage.

Why "switching genes"?
  A gene with r=0.8 in normal tissue but r=0.1 in tumor means that in healthy
  cells, protein abundance tracks mRNA closely — but in cancer, something
  (translational repression, protein degradation, miRNA regulation, etc.)
  breaks that relationship. These genes are candidates for post-transcriptional
  regulation that is specifically activated or deactivated by cancer.

  Picking by largest |delta_r| is more biologically meaningful than picking
  by highest r, because we care about *change*, not just coupling strength.
"""

import pandas as pd
import numpy as np


def find_switching_genes_tumor_normal(
    tumor_normal_results: dict,
    n_genes: int = 5,
) -> pd.DataFrame:
    """
    Find genes whose mRNA-protein correlation changes most between
    tumor and normal tissue.

    Uses the pre-computed delta_corr table (tumor_r - normal_r) from
    run_tumor_normal_analysis(), and picks the top n_genes by absolute delta.
    We take the top n_genes/2 most increased in tumor and top n_genes/2
    most decreased in tumor, to show both directions of switching.

    Parameters
    ----------
    tumor_normal_results : dict
        Output of run_tumor_normal_analysis().
    n_genes : int
        Total number of switching genes to return.

    Returns
    -------
    pd.DataFrame
        Subset of delta_corr for the top switching genes, with a column
        'direction' indicating whether coupling increased or decreased in tumor.
    """
    delta = tumor_normal_results["delta_corr"].copy()

    delta["abs_delta"] = delta["delta_r"].abs()
    delta = delta.sort_values("abs_delta", ascending=False)

    gained = delta[delta["delta_r"] > 0].head(n_genes // 2)
    lost   = delta[delta["delta_r"] < 0].head(n_genes - n_genes // 2)

    top = pd.concat([gained, lost]).sort_values("delta_r", ascending=False)
    top["direction"] = top["delta_r"].apply(
        lambda d: "Higher in Tumor" if d > 0 else "Lower in Tumor"
    )

    return top.reset_index(drop=True)


def find_switching_genes_stages(
    stage_results: dict,
    n_genes: int = 5,
) -> pd.DataFrame:
    """
    Find genes whose mRNA-protein correlation changes most between
    the earliest and latest available tumor stage.

    Parameters
    ----------
    stage_results : dict
        Output of run_stage_analysis().
    n_genes : int
        Total number of switching genes to return.

    Returns
    -------
    pd.DataFrame or None
        Top switching genes with delta_r, or None if fewer than 2 stages exist.
    """
    stage_corrs = stage_results["stage_corrs"]
    stages = sorted(stage_corrs.keys())

    if len(stages) < 2:
        return None

    early_stage = stages[0]
    late_stage  = stages[-1]

    early_r = stage_corrs[early_stage].set_index("gene")["spearman_r"]
    late_r  = stage_corrs[late_stage].set_index("gene")["spearman_r"]

    common_genes = early_r.index.intersection(late_r.index)

    delta = pd.DataFrame({
        "gene":        common_genes,
        "early_r":     early_r.loc[common_genes].values,
        "late_r":      late_r.loc[common_genes].values,
        "delta_r":     (late_r.loc[common_genes] - early_r.loc[common_genes]).values,
        "early_stage": early_stage,
        "late_stage":  late_stage,
    })

    delta["abs_delta"] = delta["delta_r"].abs()
    delta = delta.sort_values("abs_delta", ascending=False)

    gained = delta[delta["delta_r"] > 0].head(n_genes // 2)
    lost   = delta[delta["delta_r"] < 0].head(n_genes - n_genes // 2)

    top = pd.concat([gained, lost]).sort_values("delta_r", ascending=False)
    top["direction"] = top["delta_r"].apply(
        lambda d: f"Higher in Stage {late_stage}" if d > 0 else f"Lower in Stage {late_stage}"
    )

    return top.reset_index(drop=True)


def extract_gene_vectors(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    gene: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract paired mRNA and protein vectors for a single gene,
    dropping any samples where either value is NaN.

    Parameters
    ----------
    rna : pd.DataFrame
        mRNA expression table (rows = samples, columns = genes).
    protein : pd.DataFrame
        Protein abundance table (rows = samples, columns = genes).
    gene : str
        Gene name to extract.

    Returns
    -------
    rna_vals : np.ndarray
        mRNA values for valid (non-NaN) samples.
    protein_vals : np.ndarray
        Protein values for the same samples.
    """
    if gene not in rna.columns or gene not in protein.columns:
        return np.array([]), np.array([])

    r = rna[gene].values.astype(float)
    p = protein[gene].values.astype(float)

    valid = ~np.isnan(r) & ~np.isnan(p)
    return r[valid], p[valid]

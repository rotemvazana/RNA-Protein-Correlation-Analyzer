"""
stage_analysis.py
-----------------
Stage-Specific RNA-Protein Correlation Analysis.

Pathological stage describes how advanced a tumor is (Stage I → IV).
Early-stage tumors are generally less aggressive; late-stage tumors have
often spread beyond the primary site.

This module asks: "Does the RNA-protein relationship change as cancer progresses?"

For each stage group, we compute Spearman RNA-protein correlations and
then compare the distributions across stages.
"""

import re
import pandas as pd
from .correlation import compute_gene_correlations, summarize_correlations


# Keywords used to locate the stage column in clinical metadata.
STAGE_KEYWORDS = ["stage", "pathologic_stage", "tumor_stage", "ajcc_pathologic_stage"]

# Mapping of raw stage strings to simplified stage labels.
# This handles the many ways CPTAC encodes stage (e.g. "Stage IIA" → "II").
STAGE_NORMALIZATION = {
    "i": "I",
    "ia": "I",
    "ib": "I",
    "ii": "II",
    "iia": "II",
    "iib": "II",
    "iii": "III",
    "iiia": "III",
    "iiib": "III",
    "iiic": "III",
    "iv": "IV",
    "iva": "IV",
    "ivb": "IV",
}


def find_stage_column(clinical: pd.DataFrame) -> str | None:
    """
    Search clinical metadata for a column containing tumor stage information.

    Parameters
    ----------
    clinical : pd.DataFrame
        Clinical metadata table.

    Returns
    -------
    str or None
        Column name if found, else None.
    """

    for col in clinical.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(kw in col_lower for kw in STAGE_KEYWORDS):
            return col

    return None


def normalize_stage(raw_value: str) -> str | None:
    """
    Convert a raw stage string (e.g. "Stage IIa", "IIIB") to a
    simplified Roman numeral label (I, II, III, or IV).

    Parameters
    ----------
    raw_value : str
        Raw stage value from clinical metadata.

    Returns
    -------
    str or None
        Simplified stage label (e.g. "II"), or None if unrecognizable.
    """

    if pd.isna(raw_value):
        return None

    # Strip whitespace, lowercase, and remove the word "stage" and spaces.
    cleaned = str(raw_value).strip().lower()
    cleaned = re.sub(r"\bstage\b", "", cleaned).strip().replace(" ", "")

    return STAGE_NORMALIZATION.get(cleaned, None)


def split_by_stage(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    clinical: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Partition the matched RNA and protein tables by tumor stage.

    Parameters
    ----------
    rna : pd.DataFrame
        Matched RNA table.
    protein : pd.DataFrame
        Matched protein table.
    clinical : pd.DataFrame
        Clinical metadata.

    Returns
    -------
    stage_groups : dict
        Keys are simplified stage labels (e.g. "I", "II", "III", "IV").
        Values are (rna_subset, protein_subset) tuples for that stage.

    Raises
    ------
    ValueError
        If no stage column is found, or if no recognizable stages are present.
    """

    stage_col = find_stage_column(clinical)

    if stage_col is None:
        raise ValueError(
            "Could not find a stage column in clinical metadata. "
            "Stage-specific analysis is not available for this dataset."
        )

    print(f"[StageAnalysis] Using column '{stage_col}' for stage annotation.")

    # Align clinical metadata to samples in our matched RNA/protein tables.
    # CPTAC clinical tables often use a MultiIndex (Patient_ID, Sample_ID).
    # We flatten it to a simple index first so .loc[] returns a Series, not a DataFrame.
    clinical_flat = clinical.copy()
    if isinstance(clinical_flat.index, pd.MultiIndex):
        clinical_flat = clinical_flat.reset_index(level=0, drop=True)

    shared_samples = rna.index.intersection(clinical_flat.index)
    if len(shared_samples) == 0:
        raise ValueError(
            "No overlapping samples between RNA/protein tables and clinical metadata. "
            "Stage analysis cannot be performed."
        )

    stage_col_data = clinical_flat.loc[shared_samples, stage_col]
    # If we still got a DataFrame (duplicate index values), take the first column.
    if isinstance(stage_col_data, pd.DataFrame):
        stage_col_data = stage_col_data.iloc[:, 0]

    stage_series = stage_col_data

    # Normalize each raw stage value to a simplified label.
    normalized_stages = stage_series.apply(normalize_stage)

    # Drop samples with unrecognizable stage annotations.
    valid_mask = normalized_stages.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[StageAnalysis] Dropped {n_dropped} samples with unrecognizable stage values.")

    normalized_stages = normalized_stages[valid_mask]

    if normalized_stages.empty:
        raise ValueError(
            "No samples with recognizable stage annotations found. "
            "Stage values may use an unexpected format."
        )

    # Build a group dict: stage → (rna_subset, protein_subset).
    stage_groups = {}

    for stage_label in sorted(normalized_stages.unique()):
        sample_ids = normalized_stages[normalized_stages == stage_label].index
        stage_groups[stage_label] = (
            rna.loc[rna.index.isin(sample_ids)],
            protein.loc[protein.index.isin(sample_ids)],
        )
        print(f"[StageAnalysis] Stage {stage_label}: {len(sample_ids)} samples")

    return stage_groups


def run_stage_analysis(
    rna: pd.DataFrame,
    protein: pd.DataFrame,
    clinical: pd.DataFrame,
    cancer_type: str,
    min_samples: int = 5,
) -> dict:
    """
    Full stage-specific correlation analysis pipeline.

    For each tumor stage, computes per-gene Spearman correlations and
    summarizes the global RNA-protein concordance of that stage.

    Parameters
    ----------
    rna : pd.DataFrame
        Matched RNA table.
    protein : pd.DataFrame
        Matched protein table.
    clinical : pd.DataFrame
        Clinical metadata.
    cancer_type : str
        Cancer type label for output annotation.
    min_samples : int
        Minimum valid sample pairs per gene.

    Returns
    -------
    results : dict with keys:
        - "stage_corrs"    : dict mapping stage label → per-gene correlation DataFrame
        - "stage_summaries": dict mapping stage label → summary statistics dict
    """

    print(f"\n=== Stage-Specific Analysis: {cancer_type.upper()} ===")

    stage_groups = split_by_stage(rna, protein, clinical)

    stage_corrs = {}
    stage_summaries = {}

    for stage_label, (rna_stage, protein_stage) in stage_groups.items():

        print(f"\n[StageAnalysis] Computing correlations for Stage {stage_label} ({len(rna_stage)} samples) ...")

        # Skip stages with fewer samples than the threshold.
        if len(rna_stage) < min_samples:
            print(f"[StageAnalysis] Skipping Stage {stage_label}: fewer than {min_samples} samples.")
            continue

        try:
            corr_df = compute_gene_correlations(rna_stage, protein_stage, min_samples=min_samples)
            summary = summarize_correlations(corr_df, label=f"{cancer_type}_stage{stage_label}")

            stage_corrs[stage_label] = corr_df
            stage_summaries[stage_label] = summary

            print(f"  Mean r: {summary['mean_r']}, Median r: {summary['median_r']}, Genes: {summary['n_genes']}")

        except ValueError as e:
            print(f"[StageAnalysis] Could not compute correlations for Stage {stage_label}: {e}")

    if not stage_corrs:
        raise ValueError(
            "No stages had enough samples for correlation analysis. "
            "Try lowering min_samples or check your data."
        )

    return {
        "stage_corrs": stage_corrs,
        "stage_summaries": stage_summaries,
    }

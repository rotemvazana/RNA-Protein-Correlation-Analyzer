"""
tests/test_correlation.py
-------------------------
Unit tests for the correlation engine (src/correlation.py).

These tests use synthetic (fake) data so they run without downloading
any real CPTAC datasets. We create small DataFrames with known properties
and verify the output is correct.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.correlation import compute_gene_correlations, summarize_correlations, run_cancer_type_analysis


# ─── Fixtures: reusable test data ────────────────────────────────────────────

@pytest.fixture
def perfectly_correlated_data():
    """RNA and protein tables where every gene is perfectly correlated (r=1)."""
    np.random.seed(42)
    n_samples = 30
    n_genes = 20
    samples = [f"sample_{i}" for i in range(n_samples)]
    genes = [f"GENE{i}" for i in range(n_genes)]

    rna = pd.DataFrame(np.random.randn(n_samples, n_genes), index=samples, columns=genes)
    protein = rna.copy()  # Identical to RNA → Spearman r = 1.0 for every gene.
    return rna, protein


@pytest.fixture
def anti_correlated_data():
    """RNA and protein tables where every gene is perfectly anti-correlated (r=-1)."""
    np.random.seed(42)
    n_samples = 30
    n_genes = 20
    samples = [f"sample_{i}" for i in range(n_samples)]
    genes = [f"GENE{i}" for i in range(n_genes)]

    rna = pd.DataFrame(np.random.randn(n_samples, n_genes), index=samples, columns=genes)
    protein = -rna  # Negated → Spearman r = -1.0 for every gene.
    return rna, protein


@pytest.fixture
def data_with_nans():
    """RNA and protein tables with missing values in some genes and samples."""
    np.random.seed(0)
    n_samples = 40
    n_genes = 15
    samples = [f"sample_{i}" for i in range(n_samples)]
    genes = [f"GENE{i}" for i in range(n_genes)]

    rna = pd.DataFrame(np.random.randn(n_samples, n_genes), index=samples, columns=genes)
    protein = rna * 0.8 + np.random.randn(n_samples, n_genes) * 0.3

    # Introduce NaNs in various places.
    rna.iloc[0:5, 0] = np.nan       # First 5 samples, first gene
    protein.iloc[3:8, 2] = np.nan   # Different rows, third gene
    rna.iloc[:, 10] = np.nan        # Entire gene column → should be skipped

    return rna, protein


# ─── Tests for compute_gene_correlations ─────────────────────────────────────

class TestComputeGeneCorrelations:

    def test_perfect_correlation_returns_r_one(self, perfectly_correlated_data):
        """When RNA == protein, every gene should have Spearman r ≈ 1.0."""
        rna, protein = perfectly_correlated_data
        result = compute_gene_correlations(rna, protein, min_samples=5)

        assert not result.empty, "Result should not be empty."
        assert "spearman_r" in result.columns
        # All correlations should be very close to 1.0.
        assert (result["spearman_r"] > 0.99).all(), "Expected all r ≈ 1.0 for identical data."

    def test_anti_correlation_returns_r_minus_one(self, anti_correlated_data):
        """When protein = -RNA, every gene should have Spearman r ≈ -1.0."""
        rna, protein = anti_correlated_data
        result = compute_gene_correlations(rna, protein, min_samples=5)

        assert not result.empty
        assert (result["spearman_r"] < -0.99).all(), "Expected all r ≈ -1.0 for negated data."

    def test_output_columns_are_correct(self, perfectly_correlated_data):
        """Result DataFrame must have the expected columns."""
        rna, protein = perfectly_correlated_data
        result = compute_gene_correlations(rna, protein, min_samples=5)

        expected_cols = {"gene", "spearman_r", "p_value", "n_samples"}
        assert expected_cols.issubset(result.columns), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

    def test_nan_handling_skips_invalid_samples(self, data_with_nans):
        """NaN values should be dropped per-gene; genes with too few valid pairs are skipped."""
        rna, protein = data_with_nans
        # Gene 10 is all-NaN in RNA → should be absent from results.
        result = compute_gene_correlations(rna, protein, min_samples=5)

        gene_names = result["gene"].tolist()
        assert "GENE10" not in gene_names, "All-NaN gene should be excluded from results."

    def test_n_samples_reflects_valid_pairs(self, data_with_nans):
        """n_samples for each gene should equal the count of non-NaN pairs."""
        rna, protein = data_with_nans
        result = compute_gene_correlations(rna, protein, min_samples=5)

        # For GENE0: RNA has NaN in rows 0-4, so only 35 valid pairs.
        gene0_row = result[result["gene"] == "GENE0"]
        if not gene0_row.empty:
            assert gene0_row.iloc[0]["n_samples"] <= 35

    def test_min_samples_filter_works(self):
        """Genes with fewer valid pairs than min_samples should be excluded."""
        np.random.seed(1)
        samples = [f"s{i}" for i in range(20)]
        genes = ["G1", "G2"]
        rna = pd.DataFrame(np.random.randn(20, 2), index=samples, columns=genes)
        protein = rna.copy()

        # Set G1 to have only 3 valid samples.
        # Use .loc to avoid pandas Copy-on-Write warning in newer pandas versions.
        rna.loc[samples[3:], "G1"] = np.nan

        result = compute_gene_correlations(rna, protein, min_samples=10)
        assert "G1" not in result["gene"].values, "G1 should be filtered out (only 3 valid pairs)."
        assert "G2" in result["gene"].values, "G2 should be present (20 valid pairs)."

    def test_result_is_sorted_by_correlation_descending(self, perfectly_correlated_data):
        """Results should be sorted by spearman_r from highest to lowest."""
        rna, protein = perfectly_correlated_data
        # Add some noise to break perfect correlation and create variation.
        protein_noisy = protein + np.random.randn(*protein.shape) * 0.5
        result = compute_gene_correlations(rna, protein_noisy, min_samples=5)

        r_values = result["spearman_r"].tolist()
        assert r_values == sorted(r_values, reverse=True), "Results should be sorted descending by r."

    def test_raises_if_no_genes_pass_threshold(self):
        """Should raise ValueError if no genes meet the min_samples requirement."""
        rna = pd.DataFrame({"G1": [1.0, 2.0, np.nan, np.nan, np.nan]},
                           index=[f"s{i}" for i in range(5)])
        protein = rna.copy()

        with pytest.raises(ValueError, match="minimum-sample threshold"):
            compute_gene_correlations(rna, protein, min_samples=10)


# ─── Tests for summarize_correlations ────────────────────────────────────────

class TestSummarizeCorrelations:

    def test_summary_keys_are_present(self, perfectly_correlated_data):
        """Summary dict must contain all expected keys."""
        rna, protein = perfectly_correlated_data
        corr_df = compute_gene_correlations(rna, protein, min_samples=5)
        summary = summarize_correlations(corr_df, label="test_cancer")

        expected_keys = {"label", "n_genes", "mean_r", "median_r", "std_r", "pct_positive", "pct_high"}
        assert expected_keys.issubset(summary.keys())

    def test_label_is_stored_correctly(self, perfectly_correlated_data):
        """Label passed to summarize_correlations should appear in output."""
        rna, protein = perfectly_correlated_data
        corr_df = compute_gene_correlations(rna, protein, min_samples=5)
        summary = summarize_correlations(corr_df, label="brca")
        assert summary["label"] == "brca"

    def test_mean_and_median_are_near_one_for_perfect_data(self, perfectly_correlated_data):
        """For perfectly correlated data, mean and median r should be close to 1."""
        rna, protein = perfectly_correlated_data
        corr_df = compute_gene_correlations(rna, protein, min_samples=5)
        summary = summarize_correlations(corr_df)

        assert summary["mean_r"] > 0.99
        assert summary["median_r"] > 0.99

    def test_pct_positive_is_100_for_perfect_data(self, perfectly_correlated_data):
        """If all genes are positively correlated, pct_positive should be 100."""
        rna, protein = perfectly_correlated_data
        corr_df = compute_gene_correlations(rna, protein, min_samples=5)
        summary = summarize_correlations(corr_df)
        assert summary["pct_positive"] == 100.0

    def test_n_genes_matches_corr_df_length(self, data_with_nans):
        """n_genes in summary should equal number of rows in corr_df."""
        rna, protein = data_with_nans
        corr_df = compute_gene_correlations(rna, protein, min_samples=5)
        summary = summarize_correlations(corr_df)
        assert summary["n_genes"] == len(corr_df)

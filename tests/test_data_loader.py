"""
tests/test_data_loader.py
-------------------------
Unit tests for data loading and preprocessing functions (src/data_loader.py).

All tests use synthetic DataFrames — no CPTAC download required.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    match_samples,
    filter_common_genes,
    list_available_cancers,
    _get_gene_names,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_rna_protein(samples, rna_genes, protein_genes):
    """Helper: create minimal RNA and protein DataFrames with given samples and genes."""
    rna = pd.DataFrame(
        np.random.randn(len(samples), len(rna_genes)),
        index=samples,
        columns=rna_genes,
    )
    protein = pd.DataFrame(
        np.random.randn(len(samples), len(protein_genes)),
        index=samples,
        columns=protein_genes,
    )
    return rna, protein


# ─── Tests for list_available_cancers ─────────────────────────────────────────

class TestListAvailableCancers:

    def test_returns_non_empty_list(self):
        result = list_available_cancers()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_brca_is_included(self):
        assert "brca" in list_available_cancers()

    def test_all_entries_are_lowercase_strings(self):
        for name in list_available_cancers():
            assert isinstance(name, str)
            assert name == name.lower()


# ─── Tests for match_samples ──────────────────────────────────────────────────

class TestMatchSamples:

    def test_keeps_only_shared_samples(self):
        """Only samples present in both RNA and protein tables should remain."""
        rna_samples     = ["s1", "s2", "s3", "s4"]
        protein_samples = ["s2", "s3", "s5"]

        # Create each table with its own sample list so no KeyError on .loc
        rna     = pd.DataFrame(np.random.randn(4, 2),
                               index=rna_samples, columns=["G1", "G2"])
        protein = pd.DataFrame(np.random.randn(3, 2),
                               index=protein_samples, columns=["G1", "G2"])

        rna_m, protein_m = match_samples(rna, protein)

        assert set(rna_m.index) == {"s2", "s3"}
        assert set(protein_m.index) == {"s2", "s3"}

    def test_output_indices_are_equal(self):
        """RNA and protein matched tables must have the same sample index."""
        rna_samples     = ["s1", "s2", "s3"]
        protein_samples = ["s2", "s3", "s4"]

        rna     = pd.DataFrame(np.random.randn(3, 1),
                               index=rna_samples, columns=["G1"])
        protein = pd.DataFrame(np.random.randn(3, 1),
                               index=protein_samples, columns=["G1"])

        rna_m, protein_m = match_samples(rna, protein)
        assert list(rna_m.index) == list(protein_m.index)

    def test_raises_if_no_shared_samples(self):
        """Should raise ValueError when RNA and protein share no samples."""
        rna, _ = make_rna_protein(["s1", "s2"], ["G1"], ["G1"])
        _, protein = make_rna_protein(["s3", "s4"], ["G1"], ["G1"])

        with pytest.raises(ValueError, match="No overlapping samples"):
            match_samples(rna, protein)

    def test_both_tables_same_shape(self):
        """After matching, both tables should have the same number of rows."""
        rna, protein = make_rna_protein(["s1", "s2", "s3"], ["G1"], ["G1"])
        rna_m, protein_m = match_samples(rna, protein)
        assert rna_m.shape[0] == protein_m.shape[0]


# ─── Tests for filter_common_genes ────────────────────────────────────────────

class TestFilterCommonGenes:

    def test_keeps_only_shared_genes(self):
        """Only genes present in both tables should remain after filtering."""
        rna, protein = make_rna_protein(
            ["s1", "s2"],
            ["G1", "G2", "G3"],
            ["G2", "G3", "G4"],
        )
        rna_f, protein_f = filter_common_genes(rna, protein)

        assert set(rna_f.columns) == {"G2", "G3"}
        assert set(protein_f.columns) == {"G2", "G3"}

    def test_column_order_is_identical(self):
        """RNA and protein filtered tables must have columns in the same order."""
        rna, protein = make_rna_protein(
            ["s1", "s2"],
            ["G1", "G2", "G3"],
            ["G3", "G1", "G4"],
        )
        rna_f, protein_f = filter_common_genes(rna, protein)
        assert list(rna_f.columns) == list(protein_f.columns)

    def test_raises_if_no_shared_genes(self):
        """Should raise ValueError when RNA and protein share no gene names."""
        rna, _ = make_rna_protein(["s1"], ["G1", "G2"], ["G1", "G2"])
        _, protein = make_rna_protein(["s1"], ["G3", "G4"], ["G3", "G4"])

        with pytest.raises(ValueError, match="No overlapping gene names"):
            filter_common_genes(rna, protein)

    def test_values_are_not_altered(self):
        """Filtering should not change any data values, only which columns are present."""
        rna, protein = make_rna_protein(
            ["s1", "s2"],
            ["G1", "G2"],
            ["G1", "G2"],
        )
        rna_f, protein_f = filter_common_genes(rna, protein)

        # Values in the shared gene G1 should be unchanged.
        pd.testing.assert_series_equal(
            rna["G1"].sort_index(),
            rna_f["G1"].sort_index(),
        )

    def test_handles_multiindex_columns(self):
        """filter_common_genes should work even if columns are a MultiIndex."""
        samples = ["s1", "s2", "s3"]
        # Create a MultiIndex column: (gene_name, database_name)
        rna_cols = pd.MultiIndex.from_tuples([("G1", "db1"), ("G2", "db1"), ("G3", "db1")])
        protein_cols = pd.MultiIndex.from_tuples([("G2", "db2"), ("G3", "db2"), ("G4", "db2")])

        rna = pd.DataFrame(np.random.randn(3, 3), index=samples, columns=rna_cols)
        protein = pd.DataFrame(np.random.randn(3, 3), index=samples, columns=protein_cols)

        # Should not raise — it should handle MultiIndex transparently.
        rna_f, protein_f = filter_common_genes(rna, protein)
        assert rna_f.shape[1] == 2  # G2 and G3 overlap
        assert protein_f.shape[1] == 2


# ─── Tests for _get_gene_names helper ─────────────────────────────────────────

class TestGetGeneNames:

    def test_regular_index_returned_as_is(self):
        df = pd.DataFrame({"G1": [1], "G2": [2]})
        result = _get_gene_names(df)
        assert list(result) == ["G1", "G2"]

    def test_multiindex_returns_first_level(self):
        cols = pd.MultiIndex.from_tuples([("G1", "db1"), ("G2", "db1")])
        df = pd.DataFrame([[1, 2]], columns=cols)
        result = _get_gene_names(df)
        assert list(result) == ["G1", "G2"]

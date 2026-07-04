"""
tests/test_normal_comparison.py
--------------------------------
Unit tests for tumor vs. normal tissue comparison (src/normal_comparison.py).
All tests use synthetic DataFrames — no CPTAC download required.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.normal_comparison import (
    find_sample_type_column,
    split_tumor_normal,
    run_tumor_normal_analysis,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_matched_data(n_tumor=20, n_normal=10, n_genes=15, seed=0):
    """
    Create synthetic matched RNA, protein, and clinical DataFrames
    with both tumor and normal samples.
    """
    np.random.seed(seed)
    n_total = n_tumor + n_normal
    samples = [f"sample_{i}" for i in range(n_total)]
    genes = [f"GENE{i}" for i in range(n_genes)]

    rna = pd.DataFrame(np.random.randn(n_total, n_genes), index=samples, columns=genes)
    protein = rna * 0.7 + np.random.randn(n_total, n_genes) * 0.5

    # Clinical: first n_tumor samples are "Tumor", rest are "Normal".
    sample_type = ["Tumor"] * n_tumor + ["Normal"] * n_normal
    clinical = pd.DataFrame({"sample_type": sample_type}, index=samples)

    return rna, protein, clinical


# ─── Tests for find_sample_type_column ───────────────────────────────────────

class TestFindSampleTypeColumn:

    def test_finds_column_by_name(self):
        """Should detect a column with 'sample_type' in its name."""
        clinical = pd.DataFrame({"sample_type": ["Tumor", "Normal"]})
        col = find_sample_type_column(clinical)
        assert col == "sample_type"

    def test_finds_column_by_values(self):
        """Should detect a column whose values contain 'tumor' and 'normal' keywords."""
        clinical = pd.DataFrame({"random_col": ["Tumor tissue", "Normal adjacent"]})
        col = find_sample_type_column(clinical)
        assert col == "random_col"

    def test_returns_none_if_not_found(self):
        """Should return None if no sample-type column can be identified."""
        clinical = pd.DataFrame({"age": [45, 60], "sex": ["M", "F"]})
        col = find_sample_type_column(clinical)
        assert col is None


# ─── Tests for split_tumor_normal ─────────────────────────────────────────────

class TestSplitTumorNormal:

    def test_correct_split_sizes(self):
        """Tumor and normal subsets should match the annotated groups."""
        rna, protein, clinical = make_matched_data(n_tumor=20, n_normal=10)
        (rna_t, prot_t), (rna_n, prot_n) = split_tumor_normal(rna, protein, clinical)

        assert len(rna_t) == 20
        assert len(rna_n) == 10

    def test_no_sample_overlap_between_groups(self):
        """Tumor and normal groups must not share any sample IDs."""
        rna, protein, clinical = make_matched_data()
        (rna_t, _), (rna_n, _) = split_tumor_normal(rna, protein, clinical)

        overlap = set(rna_t.index) & set(rna_n.index)
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_raises_if_no_sample_type_column(self):
        """Should raise ValueError if clinical metadata has no sample-type column."""
        rna, protein, _ = make_matched_data()
        clinical_no_type = pd.DataFrame(
            {"age": np.random.randint(40, 80, 30)},
            index=rna.index,
        )
        with pytest.raises(ValueError, match="sample-type column"):
            split_tumor_normal(rna, protein, clinical_no_type)

    def test_raises_if_too_few_normal_samples(self):
        """Should raise ValueError if fewer than 5 normal samples are found."""
        rna, protein, clinical = make_matched_data(n_tumor=28, n_normal=2)
        with pytest.raises(ValueError, match="Too few normal samples"):
            split_tumor_normal(rna, protein, clinical)


# ─── Tests for run_tumor_normal_analysis ─────────────────────────────────────

class TestRunTumorNormalAnalysis:

    def test_returns_expected_keys(self):
        """Result dict must contain all expected keys."""
        rna, protein, clinical = make_matched_data(n_tumor=20, n_normal=15)
        result = run_tumor_normal_analysis(rna, protein, clinical, cancer_type="test")

        expected_keys = {"tumor_corr", "normal_corr", "tumor_summary", "normal_summary", "delta_corr"}
        assert expected_keys.issubset(result.keys())

    def test_delta_corr_contains_difference(self):
        """delta_r should equal tumor_r minus normal_r for each gene."""
        rna, protein, clinical = make_matched_data(n_tumor=20, n_normal=15)
        result = run_tumor_normal_analysis(rna, protein, clinical, cancer_type="test")

        delta_df = result["delta_corr"]
        for _, row in delta_df.iterrows():
            expected_delta = round(row["tumor_r"] - row["normal_r"], 4)
            assert abs(row["delta_r"] - expected_delta) < 1e-3


"""
tests/test_stage_analysis.py
-----------------------------
Unit tests for stage-specific analysis (src/stage_analysis.py).
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stage_analysis import (
    find_stage_column,
    normalize_stage,
    split_by_stage,
    run_stage_analysis,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_staged_data(n_per_stage=15, n_genes=10, seed=1):
    """Create synthetic RNA, protein, and clinical DataFrames with stage annotations."""
    np.random.seed(seed)
    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    n_total = n_per_stage * len(stages)
    samples = [f"sample_{i}" for i in range(n_total)]
    genes = [f"GENE{i}" for i in range(n_genes)]

    rna = pd.DataFrame(np.random.randn(n_total, n_genes), index=samples, columns=genes)
    protein = rna * 0.6 + np.random.randn(n_total, n_genes) * 0.4

    # Assign each sample to a stage (repeating the stage list).
    stage_labels = []
    for s in stages:
        stage_labels.extend([s] * n_per_stage)

    clinical = pd.DataFrame({"stage": stage_labels}, index=samples)
    return rna, protein, clinical


# ─── Tests for normalize_stage ────────────────────────────────────────────────

class TestNormalizeStage:

    def test_roman_numeral_i(self):
        assert normalize_stage("Stage I") == "I"
        assert normalize_stage("i") == "I"
        assert normalize_stage("Stage IA") == "I"

    def test_roman_numeral_ii(self):
        assert normalize_stage("Stage IIa") == "II"
        assert normalize_stage("IIB") == "II"

    def test_roman_numeral_iii(self):
        assert normalize_stage("Stage IIIC") == "III"

    def test_roman_numeral_iv(self):
        assert normalize_stage("Stage IVB") == "IV"

    def test_unrecognized_returns_none(self):
        assert normalize_stage("Unknown") is None
        assert normalize_stage("N/A") is None

    def test_nan_returns_none(self):
        assert normalize_stage(np.nan) is None


# ─── Tests for find_stage_column ──────────────────────────────────────────────

class TestFindStageColumn:

    def test_finds_column_named_stage(self):
        clinical = pd.DataFrame({"stage": ["Stage I", "Stage II"]})
        assert find_stage_column(clinical) == "stage"

    def test_finds_pathologic_stage_column(self):
        clinical = pd.DataFrame({"pathologic_stage": ["I", "II"]})
        assert find_stage_column(clinical) == "pathologic_stage"

    def test_returns_none_if_not_found(self):
        clinical = pd.DataFrame({"age": [50], "sex": ["F"]})
        assert find_stage_column(clinical) is None


# ─── Tests for split_by_stage ─────────────────────────────────────────────────

class TestSplitByStage:

    def test_returns_four_stages(self):
        """With data covering all four stages, should return four groups."""
        rna, protein, clinical = make_staged_data()
        groups = split_by_stage(rna, protein, clinical)
        assert set(groups.keys()) == {"I", "II", "III", "IV"}

    def test_each_group_has_correct_sample_count(self):
        """Each stage group should have exactly n_per_stage samples."""
        rna, protein, clinical = make_staged_data(n_per_stage=15)
        groups = split_by_stage(rna, protein, clinical)
        for stage, (rna_s, _) in groups.items():
            assert len(rna_s) == 15, f"Stage {stage} has {len(rna_s)} samples, expected 15."

    def test_raises_if_no_stage_column(self):
        """Should raise ValueError when no stage column is found."""
        rna, protein, clinical = make_staged_data()
        clinical_no_stage = pd.DataFrame(
            {"age": np.random.randint(40, 80, len(rna))},
            index=rna.index,
        )
        with pytest.raises(ValueError, match="stage column"):
            split_by_stage(rna, protein, clinical_no_stage)


# ─── Tests for run_stage_analysis ─────────────────────────────────────────────

class TestRunStageAnalysis:

    def test_returns_expected_keys(self):
        """Result dict should contain 'stage_corrs' and 'stage_summaries'."""
        rna, protein, clinical = make_staged_data(n_per_stage=15)
        result = run_stage_analysis(rna, protein, clinical, cancer_type="test", min_samples=5)
        assert "stage_corrs" in result
        assert "stage_summaries" in result

    def test_all_stages_analyzed(self):
        """With enough samples in all stages, all four should appear in results."""
        rna, protein, clinical = make_staged_data(n_per_stage=15)
        result = run_stage_analysis(rna, protein, clinical, cancer_type="test", min_samples=5)
        assert set(result["stage_corrs"].keys()) == {"I", "II", "III", "IV"}

    def test_summary_labels_are_correct(self):
        """Each stage summary label should start with the cancer type name."""
        rna, protein, clinical = make_staged_data(n_per_stage=15)
        result = run_stage_analysis(rna, protein, clinical, cancer_type="mytest", min_samples=5)
        for summary in result["stage_summaries"].values():
            assert summary["label"].startswith("mytest"), (
                f"Unexpected label: {summary['label']}"
            )

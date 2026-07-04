"""
main.py
-------
Entry point for the Cancer RNA-Protein Correlation Analyzer (Cancer-RPCA).

Run from the project root directory:

    python main.py                                # BRCA, default sources (bcm)
    python main.py --cancer luad                  # Specific cancer type
    python main.py --cancer brca luad             # Multiple cancer types
    python main.py --compare-normal               # Add tumor vs. normal
    python main.py --stage-analysis               # Add stage-specific analysis
    python main.py --all-cancers                  # Run all available cancer types
    python main.py --rna-source broad             # Change RNA data source
    python main.py --protein-source umich         # Change proteomics source

Results are saved to results/ (CSV tables) and figures/ (PNG plots).
"""

import argparse
import os
import sys
import pandas as pd

# Add the project root to the path so we can import from src/.
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    load_cancer_data,
    match_samples,
    filter_common_genes,
    list_available_cancers,
    DEFAULT_RNA_SOURCE,
    DEFAULT_PROTEIN_SOURCE,
)
from src.correlation import run_cancer_type_analysis
from src.normal_comparison import run_tumor_normal_analysis
from src.stage_analysis import run_stage_analysis
from src.gene_analysis import (
    find_switching_genes_tumor_normal,
    find_switching_genes_stages,
)
from src.visualization import (
    plot_correlation_distribution,
    plot_cancer_type_boxplot,
    plot_tumor_normal_comparison,
    plot_stage_comparison,
    plot_summary_heatmap,
    plot_switching_genes_tumor_normal,
    plot_switching_genes_stages,
)


# Default output directories — organized into subfolders by analysis type.
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

# Subfolder names within results/ and figures/
SUBDIR_CORRELATIONS = "correlations"
SUBDIR_SUMMARIES    = "summaries"
SUBDIR_TUMOR_NORMAL = "tumor_normal"
SUBDIR_STAGES       = "stages"
SUBDIR_COMPARISONS  = "comparisons"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Cancer RNA-Protein Correlation Analyzer (Cancer-RPCA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --cancer brca
  python main.py --cancer brca luad coad --compare-normal --stage-analysis
  python main.py --all-cancers
  python main.py --rna-source broad --protein-source umich
        """
    )

    parser.add_argument(
        "--cancer",
        nargs="+",
        default=["brca"],
        metavar="CANCER",
        help=(
            "One or more cancer type short names to analyze. "
            f"Available: {list_available_cancers()}. "
            "Default: brca"
        ),
    )

    parser.add_argument(
        "--all-cancers",
        action="store_true",
        help="Run analysis on all available CPTAC cancer types.",
    )

    parser.add_argument(
        "--compare-normal",
        action="store_true",
        help="Run tumor vs. normal tissue comparison (requires clinical metadata).",
    )

    parser.add_argument(
        "--stage-analysis",
        action="store_true",
        help="Run stage-specific correlation analysis (requires clinical metadata with stage).",
    )

    parser.add_argument(
        "--rna-source",
        default=DEFAULT_RNA_SOURCE,
        help=f"Lab source for transcriptomics data (e.g. bcm, broad, washu). Default: {DEFAULT_RNA_SOURCE}",
    )

    parser.add_argument(
        "--protein-source",
        default=DEFAULT_PROTEIN_SOURCE,
        help=f"Lab source for proteomics data (e.g. umich, bcm). Default: {DEFAULT_PROTEIN_SOURCE}",
    )

    parser.add_argument(
        "--switching-genes",
        action="store_true",
        help="Find and plot top switching genes (requires --compare-normal or --stage-analysis).",
    )

    parser.add_argument(
        "--n-genes",
        type=int,
        default=5,
        help="Number of top switching genes to plot per cancer type. Default: 5.",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum number of paired samples required per gene. Default: 10.",
    )

    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Directory to save CSV result tables. Default: {RESULTS_DIR}",
    )

    parser.add_argument(
        "--figures-dir",
        default=FIGURES_DIR,
        help=f"Directory to save figure files. Default: {FIGURES_DIR}",
    )

    return parser.parse_args()


def save_csv(df: pd.DataFrame, filename: str, results_dir: str) -> str:
    """Save a DataFrame to a CSV file inside results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[Export] Saved CSV: {filepath}")
    return filepath


def analyze_one_cancer(
    cancer_type: str,
    compare_normal: bool,
    stage_analysis: bool,
    switching_genes: bool,
    n_genes: int,
    min_samples: int,
    rna_source: str,
    protein_source: str,
    results_dir: str,
    figures_dir: str,
) -> dict:
    """
    Run the full analysis pipeline for a single cancer type.

    Steps:
      1. Download and load RNA, protein, and clinical data.
      2. Match samples and filter to common genes.
      3. Compute per-gene correlations for the full cohort.
      4. Optionally run tumor vs. normal comparison.
      5. Optionally run stage-specific analysis.
      6. Save all results to disk (CSV + figures).
    """

    print(f"\n{'='*60}")
    print(f"  Analyzing: {cancer_type.upper()}")
    print(f"{'='*60}")

    # Define subfolders for this run
    fig_dist    = os.path.join(figures_dir, SUBDIR_CORRELATIONS)
    fig_tn      = os.path.join(figures_dir, SUBDIR_TUMOR_NORMAL)
    fig_stages  = os.path.join(figures_dir, SUBDIR_STAGES)
    res_corr    = os.path.join(results_dir, SUBDIR_CORRELATIONS)
    res_summary = os.path.join(results_dir, SUBDIR_SUMMARIES)
    res_tn      = os.path.join(results_dir, SUBDIR_TUMOR_NORMAL)
    res_stages  = os.path.join(results_dir, SUBDIR_STAGES)

    for d in [fig_dist, fig_tn, fig_stages,
              res_corr, res_summary, res_tn, res_stages]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────
    rna_raw, protein_raw, clinical = load_cancer_data(
        cancer_type,
        rna_source=rna_source,
        protein_source=protein_source,
    )

    # ── Step 2: Match samples and filter to common genes ──────────
    rna_matched, protein_matched = match_samples(rna_raw, protein_raw)
    rna_filtered, protein_filtered = filter_common_genes(rna_matched, protein_matched)

    # ── Step 3: Cancer-type correlation analysis ───────────────────
    corr_df, summary = run_cancer_type_analysis(
        rna_filtered, protein_filtered,
        cancer_type=cancer_type,
        min_samples=min_samples,
    )

    save_csv(corr_df, f"{cancer_type}_gene_correlations.csv", res_corr)
    save_csv(pd.DataFrame([summary]), f"{cancer_type}_summary.csv", res_summary)
    plot_correlation_distribution(corr_df, cancer_type, output_dir=fig_dist)

    # ── Step 4 (optional): Tumor vs. Normal comparison ────────────
    if compare_normal:
        if clinical.empty:
            print("[Main] Skipping tumor vs. normal: no clinical metadata available.")
        else:
            try:
                tn_results = run_tumor_normal_analysis(
                    rna_filtered, protein_filtered, clinical,
                    cancer_type=cancer_type,
                    min_samples=min_samples,
                )

                save_csv(tn_results["tumor_corr"],
                         f"{cancer_type}_tumor_correlations.csv", res_tn)
                save_csv(tn_results["normal_corr"],
                         f"{cancer_type}_normal_correlations.csv", res_tn)
                save_csv(tn_results["delta_corr"],
                         f"{cancer_type}_delta_correlations.csv", res_tn)

                tn_summary_df = pd.DataFrame([
                    tn_results["tumor_summary"],
                    tn_results["normal_summary"],
                ])
                save_csv(tn_summary_df, f"{cancer_type}_tumor_normal_summary.csv", res_tn)

                plot_tumor_normal_comparison(tn_results, cancer_type, output_dir=fig_tn)

            except ValueError as e:
                # Some cancers don't have normal samples — that's OK, just skip.
                print(f"[Main] Tumor vs. normal analysis skipped: {e}")

    # ── Step 4b (optional): Switching genes — Tumor vs. Normal ────────────
    if compare_normal and switching_genes:
        tn_tumor_csv  = os.path.join(results_dir, SUBDIR_TUMOR_NORMAL, f"{cancer_type}_tumor_correlations.csv")
        tn_normal_csv = os.path.join(results_dir, SUBDIR_TUMOR_NORMAL, f"{cancer_type}_normal_correlations.csv")
        tn_delta_csv  = os.path.join(results_dir, SUBDIR_TUMOR_NORMAL, f"{cancer_type}_delta_correlations.csv")

        if all(os.path.exists(p) for p in [tn_tumor_csv, tn_normal_csv, tn_delta_csv]):
            try:
                tn_results_loaded = {
                    "tumor_corr":  pd.read_csv(tn_tumor_csv),
                    "normal_corr": pd.read_csv(tn_normal_csv),
                    "delta_corr":  pd.read_csv(tn_delta_csv),
                }

                top_genes_tn = find_switching_genes_tumor_normal(
                    tn_results_loaded, n_genes=n_genes
                )

                print(f"[Main] Top switching genes (tumor vs. normal):")
                for _, row in top_genes_tn.iterrows():
                    print(f"  {row['gene']:10s}  tumor_r={row['tumor_r']:+.3f}  "
                          f"normal_r={row['normal_r']:+.3f}  Δr={row['delta_r']:+.3f}  "
                          f"({row['direction']})")

                if not clinical.empty:
                    from src.normal_comparison import split_tumor_normal
                    try:
                        (rna_t, prot_t), (rna_n, prot_n) = split_tumor_normal(
                            rna_filtered, protein_filtered, clinical
                        )
                        plot_switching_genes_tumor_normal(
                            top_genes_tn,
                            rna_t, prot_t,
                            rna_n, prot_n,
                            cancer_type=cancer_type,
                            output_dir=fig_tn,
                        )
                    except Exception as e:
                        print(f"[Main] Switching genes (tumor/normal) plot failed: {e}")

                save_csv(top_genes_tn,
                         f"{cancer_type}_switching_genes_tumor_normal.csv",
                         res_tn)
            except Exception as e:
                print(f"[Main] Switching genes (tumor/normal) skipped: {e}")
        else:
            print("[Main] Switching genes (tumor/normal): run --compare-normal first.")

    # ── Step 5 (optional): Stage-specific analysis ─────────────────
    if stage_analysis:
        if clinical.empty:
            print("[Main] Skipping stage analysis: no clinical metadata available.")
        else:
            try:
                stage_results = run_stage_analysis(
                    rna_filtered, protein_filtered, clinical,
                    cancer_type=cancer_type,
                    min_samples=min_samples,
                )

                for stage_label, stage_corr_df in stage_results["stage_corrs"].items():
                    save_csv(
                        stage_corr_df,
                        f"{cancer_type}_stage{stage_label}_correlations.csv",
                        res_stages,
                    )

                stage_summary_df = pd.DataFrame(list(stage_results["stage_summaries"].values()))
                save_csv(stage_summary_df, f"{cancer_type}_stage_summaries.csv", res_stages)

                plot_stage_comparison(stage_results, cancer_type, output_dir=fig_stages)

            except ValueError as e:
                # Some cancers don't have stage annotations — that's OK, just skip.
                print(f"[Main] Stage analysis skipped: {e}")

    # ── Step 5b (optional): Switching genes — Stage progression ───────────
    if stage_analysis and switching_genes:
        from src.stage_analysis import split_by_stage
        if not clinical.empty:
            try:
                stage_groups = split_by_stage(rna_filtered, protein_filtered, clinical)
                # Reload per-stage correlation CSVs
                stage_corrs_loaded = {}
                stage_rna_dict    = {}
                stage_prot_dict   = {}
                for stage_label, (rna_s, prot_s) in stage_groups.items():
                    csv_path = os.path.join(
                        results_dir, SUBDIR_STAGES, f"{cancer_type}_stage{stage_label}_correlations.csv"
                    )
                    if os.path.exists(csv_path):
                        stage_corrs_loaded[stage_label] = pd.read_csv(csv_path)
                        stage_rna_dict[stage_label]     = rna_s
                        stage_prot_dict[stage_label]    = prot_s

                if len(stage_corrs_loaded) >= 2:
                    stage_results_for_genes = {"stage_corrs": stage_corrs_loaded}
                    top_genes_stage = find_switching_genes_stages(
                        stage_results_for_genes, n_genes=n_genes
                    )
                    if top_genes_stage is not None and not top_genes_stage.empty:
                        early = top_genes_stage["early_stage"].iloc[0]
                        late  = top_genes_stage["late_stage"].iloc[0]
                        print(f"[Main] Top switching genes (Stage {early} → Stage {late}):")
                        for _, row in top_genes_stage.iterrows():
                            print(f"  {row['gene']:10s}  early_r={row['early_r']:+.3f}  "
                                  f"late_r={row['late_r']:+.3f}  Δr={row['delta_r']:+.3f}  "
                                  f"({row['direction']})")
                        plot_switching_genes_stages(
                            top_genes_stage,
                            stage_rna=stage_rna_dict,
                            stage_protein=stage_prot_dict,
                            stage_corrs=stage_corrs_loaded,
                            cancer_type=cancer_type,
                            output_dir=fig_stages,
                        )
                        save_csv(top_genes_stage,
                                 f"{cancer_type}_switching_genes_stages.csv",
                                 res_stages)
            except Exception as e:
                print(f"[Main] Switching genes (stages) skipped: {e}")

    return summary


def main():
    """Main function: parse arguments, run all analyses, save outputs."""

    args = parse_args()

    if args.all_cancers:
        cancer_list = list_available_cancers()
    else:
        cancer_list = [c.lower().strip() for c in args.cancer]

    print(f"\nCancer-RPCA starting.")
    print(f"Cancer types to analyze: {cancer_list}")
    print(f"RNA source:        {args.rna_source}")
    print(f"Protein source:    {args.protein_source}")
    print(f"Tumor vs. normal:  {'Yes' if args.compare_normal else 'No'}")
    print(f"Stage analysis:    {'Yes' if args.stage_analysis else 'No'}")
    print(f"Min samples/gene:  {args.min_samples}")

    all_corr_dfs = {}
    all_summaries = []

    for cancer_type in cancer_list:
        try:
            summary = analyze_one_cancer(
                cancer_type=cancer_type,
                compare_normal=args.compare_normal,
                stage_analysis=args.stage_analysis,
                switching_genes=args.switching_genes,
                n_genes=args.n_genes,
                min_samples=args.min_samples,
                rna_source=args.rna_source,
                protein_source=args.protein_source,
                results_dir=args.results_dir,
                figures_dir=args.figures_dir,
            )
            all_summaries.append(summary)

            corr_path = os.path.join(args.results_dir, SUBDIR_CORRELATIONS,
                                     f"{cancer_type}_gene_correlations.csv")
            if os.path.exists(corr_path):
                all_corr_dfs[cancer_type] = pd.read_csv(corr_path)

        except Exception as e:
            print(f"\n[Main] ERROR analyzing {cancer_type}: {e}")
            print("[Main] Skipping this cancer type and continuing.\n")

    # ── Cross-cancer comparisons (only if more than one cancer type was analyzed) ──
    if len(all_corr_dfs) > 1:
        print("\n=== Cross-Cancer Comparison ===")

        fig_comp = os.path.join(args.figures_dir, SUBDIR_COMPARISONS)
        res_sum  = os.path.join(args.results_dir, SUBDIR_SUMMARIES)
        os.makedirs(fig_comp, exist_ok=True)
        os.makedirs(res_sum,  exist_ok=True)

        plot_cancer_type_boxplot(all_corr_dfs, output_dir=fig_comp)

        if all_summaries:
            plot_summary_heatmap(all_summaries, output_dir=fig_comp)

        combined_summary_df = pd.DataFrame(all_summaries)
        save_csv(combined_summary_df, "all_cancers_summary.csv", res_sum)

    print(f"\nAnalysis complete.")
    print(f"Results saved to: {args.results_dir}/")
    print(f"Figures saved to: {args.figures_dir}/")


if __name__ == "__main__":
    main()

"""
visualization.py
----------------
All plotting and visualization functions for Cancer-RPCA.

- "RNA" replaced with "mRNA" throughout all axis labels and titles.
- Histogram: shows median line only, no threshold shading.
- Cross-cancer boxplot: simple style, horizontal lines at -0.5, 0, +0.5.
- Tumor vs. normal: violin plot, mean annotated, no delta sentence.
- Stage comparison: simple boxplot style.
- Heatmap: original style (unsorted, centered at 0).
- Switching genes (tumor/normal): two-column scatter subplot layout.
- Switching genes (stages): multi-column scatter subplot layout.
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR = "figures"
COLOR_TUMOR  = "#D9534F"
COLOR_NORMAL = "#5BC0DE"
PALETTE_MAIN = "muted"


def _ensure_figures_dir(output_dir: str = FIGURES_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ── 1. Per-cancer correlation histogram ──────────────────────────────────────

def plot_correlation_distribution(
    corr_df: pd.DataFrame,
    cancer_type: str,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Histogram of Spearman r values for one cancer type."""
    _ensure_figures_dir(output_dir)

    r = corr_df["spearman_r"].dropna()
    median_r = r.median()
    mean_r   = r.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(median_r, color="#DD4444", linestyle="--", linewidth=1.8,
               label=f"Median r = {median_r:.3f}")
    ax.axvline(mean_r, color="#FF8800", linestyle=":", linewidth=1.8,
               label=f"Mean r = {mean_r:.3f}")
    ax.set_xlabel("Spearman Correlation (mRNA vs. Protein)", fontsize=12)
    ax.set_ylabel("Number of Genes", fontsize=12)
    ax.set_title(f"mRNA–Protein Correlation Distribution\n{cancer_type.upper()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-1, 1)
    sns.despine(ax=ax)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{cancer_type}_correlation_distribution.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 2. Cross-cancer boxplot ───────────────────────────────────────────────────

def plot_cancer_type_boxplot(
    all_corr_dfs: dict,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Boxplot comparing mRNA-protein correlations across cancer types."""
    _ensure_figures_dir(output_dir)

    records = []
    for cancer, corr_df in all_corr_dfs.items():
        for r in corr_df["spearman_r"].dropna():
            records.append({"cancer_type": cancer.upper(), "spearman_r": r})
    long_df = pd.DataFrame(records)

    order = (
        long_df.groupby("cancer_type")["spearman_r"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.2), 5))

    sns.boxplot(
        data=long_df,
        x="cancer_type", y="spearman_r",
        order=order,
        hue="cancer_type", palette=PALETTE_MAIN,
        legend=False,
        width=0.55,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )

    for y_val, alpha in [(-0.5, 0.5), (0, 0.7), (0.5, 0.5)]:
        ax.axhline(y_val, color="gray", linestyle="--", linewidth=1, alpha=alpha)

    ax.set_xlabel("Cancer Type", fontsize=12)
    ax.set_ylabel("Spearman Correlation (mRNA vs. Protein)", fontsize=12)
    ax.set_title("mRNA–Protein Correlation by Cancer Type", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "cancer_type_comparison_boxplot.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 3. Tumor vs. Normal violin plot ──────────────────────────────────────────

def plot_tumor_normal_comparison(
    tumor_normal_results: dict,
    cancer_type: str,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Violin plot comparing mRNA-protein correlations in tumor vs. normal tissue."""
    _ensure_figures_dir(output_dir)

    tumor_r  = tumor_normal_results["tumor_corr"]["spearman_r"].dropna()
    normal_r = tumor_normal_results["normal_corr"]["spearman_r"].dropna()

    df = pd.DataFrame({
        "spearman_r": pd.concat([tumor_r, normal_r], ignore_index=True),
        "tissue":     (["Tumor"] * len(tumor_r)) + (["Normal"] * len(normal_r)),
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=df, x="tissue", y="spearman_r",
        hue="tissue",
        palette={"Tumor": COLOR_TUMOR, "Normal": COLOR_NORMAL},
        legend=False,
        inner="box", cut=0, ax=ax,
    )

    for i, (label, vals) in enumerate(zip(["Tumor", "Normal"], [tumor_r, normal_r])):
        ax.text(i, vals.max() + 0.03,
                f"mean={vals.mean():.3f}",
                ha="center", fontsize=9, color="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Tissue Type", fontsize=12)
    ax.set_ylabel("Spearman Correlation (mRNA vs. Protein)", fontsize=12)
    ax.set_title(f"Tumor vs. Normal mRNA–Protein Correlation\n{cancer_type.upper()}",
                 fontsize=13, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{cancer_type}_tumor_normal_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 4. Stage comparison boxplot ──────────────────────────────────────────────

def plot_stage_comparison(
    stage_results: dict,
    cancer_type: str,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Boxplot comparing mRNA-protein correlations across tumor stages."""
    _ensure_figures_dir(output_dir)

    stage_corrs = stage_results["stage_corrs"]

    records = []
    for stage, corr_df in stage_corrs.items():
        for r in corr_df["spearman_r"].dropna():
            records.append({"stage": f"Stage {stage}", "spearman_r": r})
    df = pd.DataFrame(records)

    stage_order = [f"Stage {s}" for s in ["I", "II", "III", "IV"]
                   if f"Stage {s}" in df["stage"].unique()]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=df, x="stage", y="spearman_r",
        order=stage_order,
        hue="stage", palette="Blues",
        legend=False,
        width=0.5,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Tumor Stage", fontsize=12)
    ax.set_ylabel("Spearman Correlation (mRNA vs. Protein)", fontsize=12)
    ax.set_title(f"mRNA–Protein Correlation by Tumor Stage\n{cancer_type.upper()}",
                 fontsize=13, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"{cancer_type}_stage_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 5. Summary heatmap ────────────────────────────────────────────────────────

def plot_summary_heatmap(
    summary_list: list,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Heatmap of summary statistics across cancer types."""
    _ensure_figures_dir(output_dir)

    df = pd.DataFrame(summary_list).set_index("label")

    display_cols = ["mean_r", "median_r", "std_r", "pct_positive", "pct_high"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].astype(float)

    df_display = df_display.rename(columns={
        "mean_r":       "Mean r",
        "median_r":     "Median r",
        "std_r":        "Std r",
        "pct_positive": "% Positive",
        "pct_high":     "% High (r>0.5)",
    })

    fig, ax = plt.subplots(figsize=(9, max(3, len(df_display) * 0.7)))

    sns.heatmap(
        df_display,
        annot=True, fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Value"},
    )

    ax.set_title("mRNA–Protein Concordance Summary Across Cancer Types",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Cancer Types", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "summary_heatmap.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 6. Switching genes scatter plots — Tumor vs. Normal ──────────────────────

def plot_switching_genes_tumor_normal(
    switching_genes: pd.DataFrame,
    rna_tumor: pd.DataFrame,
    protein_tumor: pd.DataFrame,
    rna_normal: pd.DataFrame,
    protein_normal: pd.DataFrame,
    cancer_type: str,
    output_dir: str = FIGURES_DIR,
) -> str:
    """
    Multi-panel scatter plot for the top switching genes (tumor vs. normal).

    Layout: one row per gene, two columns (Tumor | Normal).
    Each panel shows mRNA vs. protein abundance for all samples in that group,
    with a linear regression line and the Spearman r annotated.
    """
    from scipy import stats
    from src.gene_analysis import extract_gene_vectors

    _ensure_figures_dir(output_dir)

    genes   = switching_genes["gene"].tolist()
    n_genes = len(genes)

    fig, axes = plt.subplots(
        n_genes, 2,
        figsize=(10, 3.5 * n_genes),
        squeeze=False,
    )

    col_titles = ["Tumor", "Normal"]
    col_colors = [COLOR_TUMOR, COLOR_NORMAL]
    datasets   = [(rna_tumor, protein_tumor), (rna_normal, protein_normal)]

    for row, gene in enumerate(genes):
        gene_row  = switching_genes[switching_genes["gene"] == gene].iloc[0]
        tumor_r   = gene_row["tumor_r"]
        normal_r  = gene_row["normal_r"]
        delta_r   = gene_row["delta_r"]
        direction = gene_row["direction"]
        r_values  = [tumor_r, normal_r]

        for col, ((rna_df, prot_df), title, color, r_val) in enumerate(
            zip(datasets, col_titles, col_colors, r_values)
        ):
            ax = axes[row][col]
            x, y = extract_gene_vectors(rna_df, prot_df, gene)

            if len(x) < 3:
                ax.text(0.5, 0.5, "Insufficient data",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{gene} — {title}", fontsize=10)
                continue

            ax.scatter(x, y, color=color, alpha=0.55, s=18, edgecolors="none")

            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="black", linewidth=1.4, linestyle="--", alpha=0.7)

            ax.text(0.05, 0.93, f"r = {r_val:.3f}",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

            ax.set_xlabel("mRNA abundance", fontsize=8)
            ax.set_ylabel("Protein abundance", fontsize=8)
            ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_title(
                    f"{gene}   |   Δr = {delta_r:+.3f}   ({direction})\n{title}",
                    fontsize=9, fontweight="bold", loc="left"
                )
            else:
                ax.set_title(title, fontsize=9, loc="left")

            sns.despine(ax=ax)

    fig.suptitle(
        f"Top Switching Genes — Tumor vs. Normal\n{cancer_type.upper()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir,
                            f"{cancer_type}_switching_genes_tumor_normal.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath


# ── 7. Switching genes scatter plots — Stage progression ─────────────────────

def plot_switching_genes_stages(
    switching_genes: pd.DataFrame,
    stage_rna: dict,
    stage_protein: dict,
    stage_corrs: dict,
    cancer_type: str,
    output_dir: str = FIGURES_DIR,
) -> str:
    """
    Multi-panel scatter plot for the top switching genes across tumor stages.

    Layout: one row per gene, one column per available stage.
    """
    from scipy import stats
    from src.gene_analysis import extract_gene_vectors

    _ensure_figures_dir(output_dir)

    genes   = switching_genes["gene"].tolist()
    stages  = sorted(stage_rna.keys())
    n_genes = len(genes)
    n_cols  = len(stages)

    stage_colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_cols))

    fig, axes = plt.subplots(
        n_genes, n_cols,
        figsize=(4 * n_cols, 3.5 * n_genes),
        squeeze=False,
    )

    for row, gene in enumerate(genes):
        gene_row  = switching_genes[switching_genes["gene"] == gene].iloc[0]
        delta_r   = gene_row["delta_r"]
        direction = gene_row["direction"]

        for col, stage in enumerate(stages):
            ax    = axes[row][col]
            color = stage_colors[col]

            rna_df  = stage_rna[stage]
            prot_df = stage_protein[stage]

            corr_df   = stage_corrs[stage]
            gene_corr = corr_df[corr_df["gene"] == gene]
            r_val = gene_corr.iloc[0]["spearman_r"] if not gene_corr.empty else float("nan")

            x, y = extract_gene_vectors(rna_df, prot_df, gene)

            if len(x) < 3:
                ax.text(0.5, 0.5, "Insufficient\ndata",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(f"Stage {stage}", fontsize=9)
                continue

            ax.scatter(x, y, color=color, alpha=0.6, s=18, edgecolors="none")

            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="black", linewidth=1.4, linestyle="--", alpha=0.7)

            r_text = f"r = {r_val:.3f}" if not np.isnan(r_val) else "r = N/A"
            ax.text(0.05, 0.93, r_text,
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

            ax.set_xlabel("mRNA abundance", fontsize=8)
            ax.set_ylabel("Protein abundance", fontsize=8)
            ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_title(
                    f"{gene}   |   Δr = {delta_r:+.3f}   ({direction})\nStage {stage}",
                    fontsize=9, fontweight="bold", loc="left"
                )
            else:
                ax.set_title(f"Stage {stage}", fontsize=9, loc="left")

            sns.despine(ax=ax)

    early = switching_genes["early_stage"].iloc[0]
    late  = switching_genes["late_stage"].iloc[0]

    fig.suptitle(
        f"Top Switching Genes — Stage {early} → Stage {late}\n{cancer_type.upper()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir,
                            f"{cancer_type}_switching_genes_stages.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {filepath}")
    return filepath

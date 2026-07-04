"""
data_loader.py
--------------
Responsible for loading CPTAC cancer datasets and extracting three tables:
  - RNA (transcriptomics)
  - Protein (proteomics)
  - Clinical metadata

The CPTAC package handles authentication and downloading automatically.
Each cancer type is accessed by a short name (e.g. "brca", "luad").

IMPORTANT: The current cptac package (v1.5+) requires a "source" argument
for each data type, because multiple research labs (BCM, Broad, WashU, UMich)
each processed the same patient samples with their own pipelines.
We default to 'bcm' (Baylor College of Medicine), which provides both
transcriptomics and proteomics for most cancers.
"""

import cptac
import pandas as pd


# Map of short cancer-type names to their CPTAC dataset class names.
# These are the cancer types currently available in the cptac package.
CANCER_REGISTRY = {
    "brca": "Brca",      # Breast cancer
    "luad": "Luad",      # Lung adenocarcinoma
    "coad": "Coad",      # Colon adenocarcinoma (formerly "Colon")
    "gbm": "Gbm",        # Glioblastoma
    "hnscc": "Hnscc",    # Head and neck squamous cell carcinoma
    "lscc": "Lscc",      # Lung squamous cell carcinoma
    "ov": "Ov",          # Ovarian cancer (formerly "Ovarian")
    "pdac": "Pdac",      # Pancreatic ductal adenocarcinoma
    "ucec": "Ucec",      # Uterine corpus endometrial carcinoma
    "ccrcc": "Ccrcc",    # Clear cell renal cell carcinoma
}

# Default sources.
# In practice across CPTAC datasets: BCM provides transcriptomics,
# UMich provides proteomics. These defaults reflect that pattern.
DEFAULT_RNA_SOURCE = "bcm"
DEFAULT_PROTEIN_SOURCE = "umich"

# Fallback order to try if the preferred source fails, for each data type.
RNA_SOURCE_CANDIDATES     = ["bcm", "broad", "washu", "umich"]
PROTEIN_SOURCE_CANDIDATES = ["umich", "bcm", "washu", "broad"]

# Kept for backwards compatibility (used as the import name in main.py).
DEFAULT_SOURCE = DEFAULT_RNA_SOURCE

# Common sources to try for clinical metadata, in order of preference.
CLINICAL_SOURCE_CANDIDATES = ["mssm", "bcm", "washu", "broad", "umich"]


def list_available_cancers() -> list[str]:
    """Return the list of supported cancer type short names."""
    return list(CANCER_REGISTRY.keys())


def load_cancer_data(
    cancer_type: str,
    rna_source: str = DEFAULT_RNA_SOURCE,
    protein_source: str = DEFAULT_PROTEIN_SOURCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download (if needed) and load RNA, protein, and clinical data for a given cancer.

    Parameters
    ----------
    cancer_type : str
        Short name of the cancer (e.g. "brca"). Must be a key in CANCER_REGISTRY.
    rna_source : str
        Lab source for transcriptomics. Default 'bcm'.
        Common options: 'bcm', 'broad', 'washu'.
    protein_source : str
        Lab source for proteomics. Default 'bcm'.
        Common options: 'bcm', 'umich'.

    Returns
    -------
    rna : pd.DataFrame
        Transcriptomics table. Rows = samples, columns = genes.
    protein : pd.DataFrame
        Proteomics table. Rows = samples, columns = genes.
    clinical : pd.DataFrame
        Clinical metadata table. Rows = samples, columns = clinical variables.
        Will be empty if no clinical data could be loaded.

    Raises
    ------
    ValueError
        If the cancer_type string is not recognized.
    RuntimeError
        If the dataset itself fails to load from CPTAC.
    """

    cancer_type = cancer_type.lower().strip()

    if cancer_type not in CANCER_REGISTRY:
        raise ValueError(
            f"Unknown cancer type: '{cancer_type}'. "
            f"Available options are: {list(CANCER_REGISTRY.keys())}"
        )

    class_name = CANCER_REGISTRY[cancer_type]

    print(f"[DataLoader] Loading CPTAC dataset: {class_name} ...")

    try:
        # Dynamically grab the dataset class from the cptac module and instantiate it.
        # For example, cptac.Brca() downloads and loads breast cancer data.
        dataset_class = getattr(cptac, class_name)
        dataset = dataset_class()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CPTAC dataset '{class_name}'. "
            f"Make sure the cptac package is installed and you have internet access.\n"
            f"Original error: {e}"
        )

    # Show the user which sources are available for this cancer type.
    # This helps if any source we try below is missing.
    try:
        print(f"[DataLoader] Available sources for {cancer_type}:")
        print(dataset.list_data_sources())
    except Exception:
        # list_data_sources may not exist in some cptac versions — not critical.
        pass

    # --- Extract transcriptomics (RNA) table ---
    rna = _try_load_omics(
        dataset, "transcriptomics", preferred=rna_source,
        fallbacks=RNA_SOURCE_CANDIDATES, label="RNA"
    )

    # --- Extract proteomics table ---
    protein = _try_load_omics(
        dataset, "proteomics", preferred=protein_source,
        fallbacks=PROTEIN_SOURCE_CANDIDATES, label="Protein"
    )

    # --- Extract clinical metadata ---
    # Clinical data comes from a different source than omics data in most cancers.
    # We try multiple known sources and use the first that works.
    clinical = _try_load_clinical(dataset, cancer_type)

    return rna, protein, clinical


def _try_load_clinical(dataset, cancer_type: str) -> pd.DataFrame:
    """
    Try multiple clinical-data sources in order and return the first that works.

    Returns an empty DataFrame if none succeed (non-critical — analyses that
    require clinical info will be skipped automatically).
    """
    for source in CLINICAL_SOURCE_CANDIDATES:
        try:
            clinical = dataset.get_clinical(source)
            if clinical is not None and not clinical.empty:
                print(f"[DataLoader] Clinical table loaded from '{source}': "
                      f"{clinical.shape[0]} samples × {clinical.shape[1]} variables")
                return clinical
        except Exception:
            # This source didn't work — silently try the next one.
            continue

    print(f"[DataLoader] Warning: Could not load clinical metadata for {cancer_type} "
          f"from any of {CLINICAL_SOURCE_CANDIDATES}. "
          f"Tumor vs. normal and stage analyses will be skipped.")
    return pd.DataFrame()


def _try_load_omics(
    dataset,
    data_type: str,
    preferred: str,
    fallbacks: list[str],
    label: str,
) -> pd.DataFrame:
    """
    Try to load an omics table (transcriptomics or proteomics) from a preferred
    source, automatically falling back to other sources if that fails.

    Parameters
    ----------
    dataset : cptac dataset object
        The loaded CPTAC cancer dataset.
    data_type : str
        Either "transcriptomics" or "proteomics".
    preferred : str
        The source to try first (e.g. "bcm" or "umich").
    fallbacks : list[str]
        Ordered list of sources to try if the preferred one fails.
    label : str
        Human-readable label for log messages ("RNA" or "Protein").

    Returns
    -------
    pd.DataFrame
        The loaded omics table.

    Raises
    ------
    RuntimeError
        If no source succeeds.
    """
    getter = getattr(dataset, f"get_{data_type}")

    # Build the list: preferred source first, then the remaining fallbacks.
    sources_to_try = [preferred] + [s for s in fallbacks if s != preferred]

    last_error = None
    for source in sources_to_try:
        try:
            df = getter(source)
            print(f"[DataLoader] {label} table loaded from '{source}': "
                  f"{df.shape[0]} samples × {df.shape[1]} genes/proteins")
            return df
        except Exception as e:
            last_error = e
            # Only print a warning if this wasn't our first choice.
            if source != preferred:
                print(f"[DataLoader] Source '{source}' also failed for {data_type}: {e}")
            else:
                print(f"[DataLoader] Preferred source '{source}' failed for {data_type}, "
                      f"trying alternatives ...")

    raise RuntimeError(
        f"Could not load {data_type} from any source {sources_to_try}.\n"
        f"Last error: {last_error}"
    )


def match_samples(rna: pd.DataFrame, protein: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only samples (rows) that appear in BOTH the RNA and protein tables.

    Correlation must be computed on matched pairs — the RNA and protein
    measurement from the same patient/sample.

    Parameters
    ----------
    rna : pd.DataFrame
        Transcriptomics table (rows = samples).
    protein : pd.DataFrame
        Proteomics table (rows = samples).

    Returns
    -------
    rna_matched : pd.DataFrame
        RNA table restricted to shared samples, sorted by sample ID.
    protein_matched : pd.DataFrame
        Protein table restricted to shared samples, sorted by sample ID.
    """

    shared_samples = rna.index.intersection(protein.index)

    if len(shared_samples) == 0:
        raise ValueError(
            "No overlapping samples found between RNA and protein tables. "
            "Check that both datasets use compatible sample identifiers."
        )

    print(f"[DataLoader] Matched {len(shared_samples)} samples shared between RNA and protein.")

    rna_matched = rna.loc[shared_samples].sort_index()
    protein_matched = protein.loc[shared_samples].sort_index()

    return rna_matched, protein_matched


def filter_common_genes(rna: pd.DataFrame, protein: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only genes (columns) that appear in BOTH the RNA and protein tables.

    Not every gene measured by RNA-seq has a corresponding protein measurement,
    so we restrict the analysis to the intersection of both gene sets.

    Parameters
    ----------
    rna : pd.DataFrame
        Matched RNA table (rows = samples, columns = genes).
    protein : pd.DataFrame
        Matched protein table (rows = samples, columns = proteins).

    Returns
    -------
    rna_filtered : pd.DataFrame
        RNA table restricted to genes also present in proteomics.
    protein_filtered : pd.DataFrame
        Protein table restricted to genes also present in transcriptomics.
        Columns of both tables are aligned (same order).
    """

    # CPTAC column names are often a MultiIndex (gene, database_id) —
    # we collapse them to the gene name only for matching.
    rna_genes = _get_gene_names(rna)
    protein_genes = _get_gene_names(protein)

    shared_genes = rna_genes.intersection(protein_genes)

    if len(shared_genes) == 0:
        raise ValueError(
            "No overlapping gene names found between RNA and protein tables. "
            "Column naming may be inconsistent — check the raw data."
        )

    print(f"[DataLoader] Found {len(shared_genes)} genes measured in both RNA and protein datasets.")

    rna_filtered = _select_genes(rna, shared_genes)
    protein_filtered = _select_genes(protein, shared_genes)

    # If columns are still a MultiIndex (e.g. because some genes have multiple
    # database IDs), flatten by keeping only the first occurrence of each gene.
    rna_filtered = _flatten_multiindex_columns(rna_filtered)
    protein_filtered = _flatten_multiindex_columns(protein_filtered)

    # Now both tables have plain gene-name columns — align them in the same order.
    common_after_flatten = rna_filtered.columns.intersection(protein_filtered.columns)
    rna_filtered = rna_filtered[sorted(common_after_flatten)]
    protein_filtered = protein_filtered[sorted(common_after_flatten)]

    return rna_filtered, protein_filtered


def _get_gene_names(df: pd.DataFrame) -> pd.Index:
    """
    Extract gene names from a DataFrame, handling MultiIndex columns.

    CPTAC sometimes returns columns as a MultiIndex like (gene_name, database).
    We always want just the first level (gene name).
    """
    if isinstance(df.columns, pd.MultiIndex):
        return pd.Index(df.columns.get_level_values(0)).unique()
    return df.columns


def _select_genes(df: pd.DataFrame, genes: pd.Index) -> pd.DataFrame:
    """Select columns matching the given gene names, handling MultiIndex columns."""
    if isinstance(df.columns, pd.MultiIndex):
        return df.loc[:, df.columns.get_level_values(0).isin(genes)]
    return df.loc[:, df.columns.isin(genes)]


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has a MultiIndex column (gene, database_id), keep only the first
    occurrence of each gene and rename columns to just the gene name.

    This makes downstream code simpler: it can assume plain gene-name columns.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Walk through columns and keep only the first column per gene name.
    gene_names = df.columns.get_level_values(0)
    seen = set()
    keep_cols = []
    for i, gene in enumerate(gene_names):
        if gene not in seen:
            seen.add(gene)
            keep_cols.append(i)

    df_flat = df.iloc[:, keep_cols].copy()
    df_flat.columns = [df.columns[i][0] for i in keep_cols]
    return df_flat

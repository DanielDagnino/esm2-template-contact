import pandas as pd

from data_preparation.base_parameters import MMSEQS_COLUMNS


def load_mmseqs_tsv(
        tsv_path: str
) -> pd.DataFrame:
    """Load MMseqs2 TSV results into a DataFrame.
    Args:
        tsv_path: Path to the MMseqs2 TSV results file.
    Returns:
        pd.DataFrame: DataFrame containing the MMseqs2 results with appropriate column names.
    """

    # Load + assign column names
    df = pd.read_csv(tsv_path, sep="\t", header=None, usecols=range(len(MMSEQS_COLUMNS)))
    df.columns = MMSEQS_COLUMNS

    # Normalize names to match npz filenames (headers were ">filename.npz")
    #   removes leading '>' and trailing whitespace from sequence names
    def clean(x):
        return x.strip().lstrip(">").rstrip()

    # Apply cleaning to query and target columns
    df["query"] = df["query"].apply(clean)
    df["target"] = df["target"].apply(clean)
    return df


def topk_hits_for(
        query_name: str,
        df: pd.DataFrame,
        topk: int
) -> pd.DataFrame:
    """Get top-k hits for a given query from the MMseqs2 results DataFrame.
    Args:
        query_name: The name of the query sequence.
        df : DataFrame containing MMseqs2 results.
        topk: Number of top hits to return.
    Returns:
        pd.DataFrame: DataFrame containing the top-k hits for the query.
    """

    # Get all hits for the query
    sub = df[df["query"] == query_name].copy()

    # Sort by bitscore and percent identity
    sub = sub.sort_values(["bits", "pident"], ascending=[False, False])

    # Exclude self-hit if present
    sub = sub[sub["target"] != query_name]

    return sub.head(topk)

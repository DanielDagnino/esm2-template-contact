import argparse
import os
import subprocess

import numpy as np
from path import Path
from tqdm import tqdm

from data_preparation.base_parameters import TOP_K_SIMILAR, MMSEQS_COLUMNS


def run_mmseqs_search(
        seq_fasta: str,
        out_dir: str,
        topk: int,
) -> None:

    # Prepare directories
    db_path = out_dir / "mmseqs_db"
    tmp_dir = out_dir / "tmp"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Create database
    subprocess.run([
        "mmseqs",
        "createdb",
        seq_fasta,
        str(db_path)
    ], check=True)

    # Run search
    subprocess.run([
        "mmseqs",
        "search",
        str(db_path),
        str(db_path),
        str(out_dir / "result"),
        str(tmp_dir),
        "--max-seqs", str(topk),
        "-s", "7"  # Sensitivity parameter (High sensitivity). Higher values increase sensitivity but also runtime
    ], check=True)

    # Convert results to TSV
    subprocess.run([
        "mmseqs",
        "convertalis",
        str(db_path),
        str(db_path),
        str(out_dir / "result"),
        str(out_dir / "result.tsv"),
        "--format-output", ",".join(MMSEQS_COLUMNS)
    ], check=True)


if __name__ == "__main__":
    """
    Run MMseqs2 search on all sequences in the dataset to find top-k similar sequences for each.
    It allows to speed up template search during training and evaluation.
    Generates a TSV file with the search results.
    
    What it does:
        1. Dump all sequences from the processed NPZ files into a single FASTA file.
        2. Run MMseqs2 search using the generated FASTA file against itself to find similar sequences.
        3. Save the search results in a TSV file in the specified output directory.
        4. The results can be used later for template-based modeling or feature extraction.
    
    Considerations:
        1. Ensure MMseqs2 and Python requirements are installed (explained in the README.md).
        2. Adjust TOP_K_SIMILAR in base_parameters.py as needed.
    """

    parser = argparse.ArgumentParser(description='Preprocess PDB files to NPZ format.')
    parser.add_argument('--base_data_dir',
                        type=str,
                        default='~/MyData/esm2-residue_contact',
                        help='Base directory containing PDB files.'
                        )
    args = parser.parse_args()
    _base_data_dir = Path(args.base_data_dir).expanduser()

    # Paths
    npz_dir = _base_data_dir / Path("processed/train").expanduser()
    fasta_path = npz_dir.parent / "all_sequences.fasta"
    out_dir = npz_dir.parent / "mmseqs_results"

    # Dump FASTA
    with open(fasta_path, "w") as f:
        for fn in tqdm(list(npz_dir.walkfiles('*.npz')), desc="Writing FASTA"):
            seq = str(np.load(fn, allow_pickle=True)['seq'])
            f.write(f">{fn.name}\n{seq}\n")

    print(f"Written {fasta_path}, running MMseqs2 search...")
    run_mmseqs_search(fasta_path, out_dir, TOP_K_SIMILAR)

    print("Finished MMseqs2 search.")

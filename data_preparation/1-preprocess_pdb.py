import argparse
import numpy as np
from Bio import PDB
from path import Path
from tqdm import tqdm

from data_preparation.base_parameters import MIN_LEN_SEARCH, THREE_TO_ONE, UNKNOWN_AA


pdb_parser = PDB.PDBParser(QUIET=True)


def parse_pdb_to_npz(
        pdb_path: str,
        out_dir: str,
        cutoff: float = 8.
) -> None:

    pdb_path = Path(pdb_path)
    structure = pdb_parser.get_structure(pdb_path.name, pdb_path)

    # Iterate models in structure
    for model in structure:
        # Iterate chains in model
        for chain in model:
            res_list = []
            ca_coords = []
            res_ids = []
            # Iterate residues in chain
            for res in chain:
                resname = res.get_resname().strip()

                # Only consider standard amino acids with CA atom for contact map calculation
                if 'CA' in res:
                    ca = res['CA']

                    # Handle alternate locations
                    if ca.is_disordered():
                        ca = max(ca.child_dict.values(), key=lambda a: a.get_occupancy() or 0.0)  # choose altloc with highest occupancy
                    coord = ca.get_coord()

                    # Convert three-letter code to one-letter code
                    one = THREE_TO_ONE.get(resname, UNKNOWN_AA)
                    res_list.append(one)
                    ca_coords.append(coord)
                    res_ids.append(res.get_id())

            # Skip short chains
            if len(res_list) < MIN_LEN_SEARCH:
                continue

            seq = ''.join(res_list)
            ca_coords = np.array(ca_coords)  # (L, 3)

            # compute pairwise distances
            dist = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
            contact = (dist < cutoff).astype(np.uint8)
            np.fill_diagonal(contact, 0)  # avoid self-contacts

            # Save to NPZ
            npz_path = Path(out_dir) / f"{pdb_path.stem}_model{model.get_id()}_chain{chain.id}.npz"
            np.savez_compressed(
                npz_path,
                seq=seq,
                coords=ca_coords,
                contact=contact,
                dist=dist,
                res_ids=res_ids,
                pdb=pdb_path,
                chain=chain.id
            )


def run_pdb_preprocessing(
        base_data_dir: str
) -> None:

    for split_name in ['test', 'train']:
        print(f'Processing split: {split_name}')
        pdb_dir = base_data_dir / split_name
        pdb_files = sorted([
            fn for fn in pdb_dir.walkfiles()
            if fn.name.lower().endswith('.pdb')
        ])

        out_dir = base_data_dir / 'processed' / split_name
        out_dir.makedirs_p()
        for fn in tqdm(pdb_files, total=len(pdb_files), desc='Loading PDBs'):
            parse_pdb_to_npz(fn, out_dir)


if __name__ == '__main__':
    """
    Preprocess PDB files to extract sequences, CA coordinates, and contact maps, saving them as compressed NPZ files.
    It allows for efficient loading during model training and evaluation.
    
    What it does:
        1. Iterate over PDB files in the specified directories (train and test).
        2. For each PDB file, parse the structure and extract sequences and CA coordinates for each chain.
        3. Compute pairwise distance matrices and contact maps based on a distance cutoff.
        4. Save the processed data (sequence, coordinates, contact map, distance matrix, residue IDs) as 
            compressed NPZ files in the output directory.
        5. Skip chains shorter than the minimum length threshold.
        6. Output NPZ files will be saved in a structured directory under '<base_data_dir>/processed/<split_name>'.
        7. Each NPZ file is named as {pdb_stem}_model{model_id}_chain{chain_id}.npz
        8. Example usage: python 1-preprocess_pdb.py
    
    Considerations:
        1. Ensure Python requirements are installed (explained in the README.md).
        2. Adjust MIN_LEN_SEARCH and other parameters in base_parameters.py as needed.
    """

    parser = argparse.ArgumentParser(description='Preprocess PDB files to NPZ format.')
    parser.add_argument('--base_data_dir',
                        type=str,
                        default='~/MyData/esm2-residue_contact',
                        help='Base directory containing PDB files.'
                        )
    args = parser.parse_args()
    _base_data_dir = Path(args.base_data_dir).expanduser()

    run_pdb_preprocessing(_base_data_dir)

    print("PDB preprocessing completed.")

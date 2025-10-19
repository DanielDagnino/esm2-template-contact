
# This file contains base parameters used in data preparation scripts

# Define minimum length of protein sequences to consider
MIN_LEN_SEARCH = 2

# Mapping from three-letter amino acid codes to one-letter codes
THREE_TO_ONE = {'ALA': 'A', 'ARG': 'R', 'ASP': 'D', 'CYS': 'C', 'CYX': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                'HIS': 'H', 'HIE': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'ASN': 'N', 'PHE': 'F',
                'PRO': 'P', 'SEC': 'U', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
print(f'standard amino acids = {len(set(THREE_TO_ONE.values()))}')

# Define unknown amino acid representation
UNKNOWN_AA = 'X'

# Define the number of top similar sequences/templates to consider from MMseqs2 search
# NOTE:
#   1. This value should match the `max_candidates` parameter in the ContactDataset class.
#   2. In case of doubt, choose a larger value to avoid missing potential templates and **recompute** the MMseqs2 search.
TOP_K_SIMILAR = 200

# Define the columns to extract from MMseqs2 TSV output
MMSEQS_COLUMNS = [
    "query",  # the query sequence name
    "target",  # the target sequence name
    "qstart",  # the start position of the alignment in the query
    "qend", # the end position of the alignment in the query
    "tstart",  # the start position of the alignment in the target
    "tend",  # the end position of the alignment in the target
    "evalue",  # the e-value of the alignment
    "bits",  # the bitscore of the alignment
    "alnlen",  # the length of the alignment
    "qlen",  # the length of the query sequence
    "tlen",  # the length of the target sequence
    "pident"  # the percent identity of the alignment
]

# Mapping from one-letter amino acid codes to full names
AMINO_ACID_MAP = {
    "A": "Alanine",
    "C": "Cysteine",
    "D": "Aspartic acid",
    "E": "Glutamic acid",
    "F": "Phenylalanine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "K": "Lysine",
    "L": "Leucine",
    "M": "Methionine",
    "N": "Asparagine",
    "P": "Proline",
    "Q": "Glutamine",
    "R": "Arginine",
    "S": "Serine",
    "T": "Threonine",
    "V": "Valine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "O": "Pyrrolysine",
}
print(f"Amino acid map not in THREE_TO_ONE = {set(AMINO_ACID_MAP.keys()) - set(THREE_TO_ONE.values())}")

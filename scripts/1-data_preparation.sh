##/bin/bash
set -e
set -o pipefail
set -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:src

python3 data_preparation/1-preprocess_pdb.py \
    --base_data_dir '~/MyData/esm2-residue_contact'

python3 data_preparation/2-mmseqs_search.py \
    --base_data_dir '~/MyData/esm2-residue_contact'

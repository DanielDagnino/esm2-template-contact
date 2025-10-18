##/bin/bash
set -e
set -o pipefail
set -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:src

python3 src/cli/analysis.py \
  --pred_dir "${HOME}/MyTmp/esm2-residue_contact/outputs-v2/pred_test/npz" \
  --out_dir "${HOME}/MyTmp/esm2-residue_contact/outputs-v2/pred_test/analysis"

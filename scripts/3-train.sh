##/bin/bash
set -e
set -o pipefail
set -x

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:src

python3 src/cli/train.py \
  --config "src/cfg/config-v1.yaml"

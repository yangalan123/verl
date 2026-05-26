#!/bin/bash
# Preprocess all code-reasoning datasets into verl-compatible parquets.
# Outputs:
#   data/eurus2_code/train.parquet              -- RL training set
#   data/livecodebench/release_v2_test.parquet  -- in-training & final eval
#   data/humanevalplus/test.parquet             -- in-training & final eval
#
# Run from the verl repo root.
set -euxo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-10000}"
LCB_VERSION="${LCB_VERSION:-release_v2}"

mkdir -p "${DATA_ROOT}"

# 1) Training set: code subset of Eurus-2-RL-Data, capped at MAX_TRAIN_EXAMPLES.
python recipe/annealed_sampling/codeRL/prepare_train_eurus2_code.py \
    --hf_repo "PRIME-RL/Eurus-2-RL-Data" \
    --hf_split train \
    --local_dir "${DATA_ROOT}/eurus2_code" \
    --max_examples "${MAX_TRAIN_EXAMPLES}"

# 2) LiveCodeBench eval (stdin/stdout style; uses prime_code reward).
python recipe/annealed_sampling/codeRL/prepare_eval_livecodebench.py \
    --version "${LCB_VERSION}" \
    --local_dir "${DATA_ROOT}/livecodebench"

# 3) HumanEval+ eval (function-style; uses the new humanevalplus reward).
python recipe/annealed_sampling/codeRL/prepare_eval_humanevalplus.py \
    --local_dir "${DATA_ROOT}/humanevalplus"

echo "Done. Parquets written under ${DATA_ROOT}/{eurus2_code,livecodebench,humanevalplus}/"

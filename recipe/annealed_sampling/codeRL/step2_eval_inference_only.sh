#!/bin/bash
# "Day 0" sanity check for the rebuttal: inference-only EAD vs fixed-T on
# HumanEval+ and LiveCodeBench. No RL training required.
#
# Runs four configs:
#   (1) fixed temp 0.7   -- low-temperature competitive baseline
#   (2) fixed temp 1.0   -- high-entropy baseline
#   (3) fixed temp 1.2   -- matches EAD's exploration_temp upper bound
#   (4) EAD negexp 1.2 -> 0.1 with d_0 = 200 (paper default for small models)
#
# Costs ~30-60 min per config on 1 A100 for the 1.5B model.
set -euxo pipefail

# -- Cluster scaffolding (edit for your slurm setup) --
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your_env>

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./logs/inference_only_eval/$(basename ${MODEL})}"
N_SAMPLES="${N_SAMPLES:-8}"
TP="${TP:-1}"

mkdir -p "${OUT_DIR}"

for PARQUET in \
    "${DATA_ROOT}/humanevalplus/test.parquet" \
    "${DATA_ROOT}/livecodebench/release_v2_test.parquet"
do
    if [ ! -f "${PARQUET}" ]; then
        echo "Skipping ${PARQUET} (not found). Did you run step0_data_preprocess.sh?"
        continue
    fi

    for TEMP in 0.7 1.0 1.2; do
        python recipe/annealed_sampling/codeRL/inference_only_eval.py \
            --model_name_or_path "${MODEL}" \
            --eval_parquet "${PARQUET}" \
            --mode fixed --temperature "${TEMP}" \
            --n_samples "${N_SAMPLES}" \
            --tensor_parallel_size "${TP}" \
            --output_dir "${OUT_DIR}"
    done

    python recipe/annealed_sampling/codeRL/inference_only_eval.py \
        --model_name_or_path "${MODEL}" \
        --eval_parquet "${PARQUET}" \
        --mode ead \
        --start_temp 1.2 --end_temp 0.1 --decay_freq 200 \
        --decay_mode negexp --decay_freq_cap_large 40000 \
        --decay_freq_increase_factor 5 \
        --n_samples "${N_SAMPLES}" \
        --tensor_parallel_size "${TP}" \
        --output_dir "${OUT_DIR}"
done

echo "Done. Summaries in ${OUT_DIR}/summary__*.json"

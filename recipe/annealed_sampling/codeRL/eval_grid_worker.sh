#!/bin/bash
# Single-GPU eval worker. Pins one GPU and loops over a grid of sampling
# configs on ONE benchmark, calling inference_only_eval.py once per config.
#
# Usage:
#   bash eval_grid_worker.sh <gpu_id> <benchmark> <mode>
#     <gpu_id>     : integer CUDA device index (e.g. 0)
#     <benchmark>  : humanevalplus | livecodebench
#     <mode>       : fixed | ead
#
# Examples:
#   bash eval_grid_worker.sh 0 humanevalplus fixed   # loops over FIXED_TEMPS
#   bash eval_grid_worker.sh 1 humanevalplus ead     # loops over DECAY_FREQS
#   bash eval_grid_worker.sh 2 livecodebench fixed
#   bash eval_grid_worker.sh 3 livecodebench ead
#
# Grids (override via env vars):
#   FIXED_TEMPS   default "0.7 1.0 1.2"     -- fixed-temperature sweep
#   DECAY_FREQS   default "25 50 100 200"   -- EAD decay_freq (d_0) ablation
#   START_TEMP    default 1.2               -- EAD tau_max
#   END_TEMP      default 0.1               -- EAD tau_min
#
# Other env vars:
#   MODEL          default Qwen/Qwen2.5-Coder-1.5B-Instruct
#   DATA_ROOT      default ./data
#   OUT_DIR        default ./logs/inference_only_eval/<model-basename>
#   N_SAMPLES      default 8
#   MAX_PROMPTS    default -1 (all)
#   LCB_VERSION    default release_v2
#   GPU_MEM_UTIL   default 0.85
#   MAX_TOKENS     default 2048   -- generation budget; raise for reasoning models
#   ENABLE_THINKING default auto  -- auto | on | off (Qwen3 dual-mode templates)
#   MAX_MODEL_LEN  default -1     -- vLLM context window (-1 = model default)
#   TP             default 1      -- tensor_parallel_size (raise for big models)
#   SAVE_COMPLETIONS default 0    -- 1 = persist all completion texts for later
#                                    re-scoring (pass@k/worst@k for any k<=N are
#                                    already recomputable from 'successes')
#   COMPLETION_CHAR_CAP default 0 -- truncate saved completions (0 = no cap)
#
# NOTE: to later sweep pass@k / worst@k up to K, generate with N_SAMPLES>=K
# (e.g. N_SAMPLES=16), then use recompute_passk.py -- no regeneration needed.

set -euo pipefail

GPU_ID="${1:?need gpu_id, e.g. 0}"
BENCH="${2:?need benchmark: humanevalplus | livecodebench}"
MODE="${3:?need mode: fixed | ead}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:-./data}"
# Base output dir (per model). Each benchmark gets its own subdir below, so
# HumanEval+ and LiveCodeBench summaries never share a folder.
OUT_DIR_BASE="${OUT_DIR:-./logs/inference_only_eval_sample32/$(basename ${MODEL})}"
N_SAMPLES="${N_SAMPLES:-32}"
MAX_PROMPTS="${MAX_PROMPTS:--1}"
LCB_VERSION="${LCB_VERSION:-release_v2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
ENABLE_THINKING="${ENABLE_THINKING:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:--1}"
TP="${TP:-1}"
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-0}"
COMPLETION_CHAR_CAP="${COMPLETION_CHAR_CAP:-0}"

FIXED_TEMPS="${FIXED_TEMPS:-0.7 1.0 1.2}"
DECAY_FREQS="${DECAY_FREQS:-25 50 100 200}"
START_TEMP="${START_TEMP:-1.2}"
END_TEMP="${END_TEMP:-0.1}"

export TOKENIZERS_PARALLELISM=false

# Resolve benchmark -> parquet path and a per-benchmark output subdir.
case "${BENCH}" in
    humanevalplus)
        PARQUET="${DATA_ROOT}/humanevalplus/test.parquet" ;;
    livecodebench)
        PARQUET="${DATA_ROOT}/livecodebench/${LCB_VERSION}_test.parquet" ;;
    *)
        echo "Unknown benchmark '${BENCH}' (want humanevalplus | livecodebench)"; exit 1 ;;
esac

OUT_DIR="${OUT_DIR_BASE}/${BENCH}"

if [ ! -f "${PARQUET}" ]; then
    echo "Parquet not found: ${PARQUET}. Run step0_data_preprocess.sh first."
    exit 1
fi

mkdir -p "${OUT_DIR}"
echo "[worker] GPU=${GPU_ID} bench=${BENCH} mode=${MODE} model=${MODEL}"
echo "[worker] parquet=${PARQUET}"

SAVE_ARGS=()
if [ "${SAVE_COMPLETIONS}" = "1" ]; then
    SAVE_ARGS+=(--save_completions --completion_char_cap "${COMPLETION_CHAR_CAP}")
fi

run_one() {
    # args passed straight through to the python entry point
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python recipe/annealed_sampling/codeRL/inference_only_eval.py \
        --model_name_or_path "${MODEL}" \
        --eval_parquet "${PARQUET}" \
        --n_samples "${N_SAMPLES}" \
        --max_prompts "${MAX_PROMPTS}" \
        --max_tokens "${MAX_TOKENS}" \
        --enable_thinking "${ENABLE_THINKING}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --tensor_parallel_size "${TP}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        --output_dir "${OUT_DIR}" \
        ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} \
        "$@"
}

if [ "${MODE}" = "fixed" ]; then
    for T in ${FIXED_TEMPS}; do
        echo "[worker] >>> fixed T=${T}"
        run_one --mode fixed --temperature "${T}"
    done
elif [ "${MODE}" = "ead" ]; then
    for D in ${DECAY_FREQS}; do
        echo "[worker] >>> ead negexp ${START_TEMP}->${END_TEMP} decay_freq=${D}"
        run_one --mode ead \
            --start_temp "${START_TEMP}" --end_temp "${END_TEMP}" \
            --decay_freq "${D}" --decay_mode negexp \
            --decay_freq_cap_large 40000 --decay_freq_increase_factor 5
    done
else
    echo "Unknown mode '${MODE}' (want fixed | ead)"; exit 1
fi

echo "[worker] done. Summaries in ${OUT_DIR}/summary__*.json"

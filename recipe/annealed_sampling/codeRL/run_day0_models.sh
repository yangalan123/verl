#!/bin/bash
# Day-0 (inference-only, no RL training) sweep over a registry of newer Qwen
# models. For each model we run the full {humanevalplus, livecodebench} x
# {fixed, ead} eval grid via eval_grid_worker.sh, with the per-model output
# length and thinking flag set correctly.
#
# Why this script exists: long-reasoning models (Qwen3 dual-mode) emit <think>
# traces and need a much larger MAX_TOKENS / context than the code-specialized
# Instruct models, so a one-size-fits-all max length would either truncate the
# reasoning models or waste memory on the small ones. The registry below pins
# the right setting per model.
#
# Usage:
#   bash recipe/annealed_sampling/codeRL/run_day0_models.sh
#   # subset / quick smoke test:
#   MODELS_FILTER="Qwen3-4B" MAX_PROMPTS=40 N_SAMPLES=4 \
#       bash recipe/annealed_sampling/codeRL/run_day0_models.sh
#
# Models run SEQUENTIALLY (one at a time). Within each model, the 4 (benchmark,
# mode) tasks run in PARALLEL across GPUs 0-3 when TP=1; for TP>1 models the
# tasks run sequentially on the first TP GPUs.
#
# Registry row format (pipe-separated):
#   MODEL_ID | MAX_TOKENS | ENABLE_THINKING(auto|on|off) | TP | MAX_MODEL_LEN
#
# Env overrides (forwarded to eval_grid_worker.sh):
#   DATA_ROOT, OUT_DIR, N_SAMPLES, MAX_PROMPTS, LCB_VERSION, GPU_MEM_UTIL,
#   FIXED_TEMPS, DECAY_FREQS, START_TEMP, END_TEMP.
#   MODELS_FILTER : substring; only run registry rows whose MODEL_ID matches.

set -euo pipefail

WORKER="recipe/annealed_sampling/codeRL/eval_grid_worker.sh"
LOG_ROOT="${LOG_ROOT:-./logs/inference_only_eval/_worker_logs}"
mkdir -p "${LOG_ROOT}"

MODELS_FILTER="${MODELS_FILTER:-}"

# --- Model registry -------------------------------------------------------
# Code-specialized Instruct models are non-thinking and short-output; the
# Qwen3 general models are dual-mode and need a big generation budget.
# Qwen3-Coder-30B-A3B is a 30B (3B-active) MoE: it fits one A100-80GB in bf16
# but is safer with TP=2; if you only have TP=1, drop MAX_MODEL_LEN/N_SAMPLES.
MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct|2048|off|1|-1"
    "Qwen/Qwen2.5-Coder-7B-Instruct|2048|off|1|-1"
    "Qwen/Qwen3-4B|16384|on|1|20480"
    "Qwen/Qwen3-8B|16384|on|1|20480"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct|4096|off|2|16384"
)

# (benchmark, mode) tasks each model is evaluated on.
TASKS=(
    "humanevalplus fixed"
    "humanevalplus ead"
    "livecodebench fixed"
    "livecodebench ead"
)

run_model() {
    local model="$1" max_tokens="$2" think="$3" tp="$4" max_model_len="$5"
    local tag; tag="$(basename "${model}")"
    echo "============================================================"
    echo "[day0] model=${model}"
    echo "[day0]   max_tokens=${max_tokens} enable_thinking=${think} tp=${tp} max_model_len=${max_model_len}"
    echo "============================================================"

    export MODEL="${model}"
    export MAX_TOKENS="${max_tokens}"
    export ENABLE_THINKING="${think}"
    export MAX_MODEL_LEN="${max_model_len}"
    export TP="${tp}"

    if [ "${tp}" -le 1 ]; then
        # One task per GPU, in parallel across GPUs 0..3.
        local pids=() gpu=0
        for t in "${TASKS[@]}"; do
            set -- ${t}; local bench="$1" mode="$2"
            local logf="${LOG_ROOT}/${tag}__gpu${gpu}_${bench}_${mode}.log"
            echo "[day0] launch GPU=${gpu} ${bench} ${mode} -> ${logf}"
            bash "${WORKER}" "${gpu}" "${bench}" "${mode}" > "${logf}" 2>&1 &
            pids+=("$!")
            gpu=$((gpu + 1))
        done
        local fail=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || { echo "[day0] worker pid ${pid} failed"; fail=1; }
        done
        [ "${fail}" -eq 0 ] || echo "[day0] WARNING: ${tag} had failing workers (see logs)"
    else
        # TP>1: run tasks sequentially on the first TP GPUs (comma list).
        local gpu_list; gpu_list="$(seq -s, 0 $((tp - 1)))"
        for t in "${TASKS[@]}"; do
            set -- ${t}; local bench="$1" mode="$2"
            local logf="${LOG_ROOT}/${tag}__gpu${gpu_list}_${bench}_${mode}.log"
            echo "[day0] run GPU=${gpu_list} ${bench} ${mode} -> ${logf}"
            bash "${WORKER}" "${gpu_list}" "${bench}" "${mode}" > "${logf}" 2>&1 \
                || echo "[day0] WARNING: ${tag} ${bench} ${mode} failed (see ${logf})"
        done
    fi
    echo "[day0] done model=${model}. Summaries in OUT_DIR/${tag}/<benchmark>/"
}

ran=0
for row in "${MODELS[@]}"; do
    IFS='|' read -r m mt th tp mml <<< "${row}"
    if [ -n "${MODELS_FILTER}" ] && [[ "${m}" != *"${MODELS_FILTER}"* ]]; then
        continue
    fi
    run_model "${m}" "${mt}" "${th}" "${tp}" "${mml}"
    ran=$((ran + 1))
done

if [ "${ran}" -eq 0 ]; then
    echo "No models matched MODELS_FILTER='${MODELS_FILTER}'. Registry models:"
    for row in "${MODELS[@]}"; do IFS='|' read -r m _ _ _ _ <<< "${row}"; echo "  - ${m}"; done
    exit 1
fi
echo "[day0] all done. ${ran} model(s) evaluated."

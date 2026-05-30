#!/bin/bash
# Launch the full inference-only eval grid across 4 GPUs in parallel.
# Each GPU owns one (benchmark, mode) pair and loops over its grid internally:
#   GPU 0 -> humanevalplus, fixed   (loops FIXED_TEMPS = 0.7 1.0 1.2)
#   GPU 1 -> humanevalplus, ead     (loops DECAY_FREQS = 25 50 100 200)
#   GPU 2 -> livecodebench, fixed   (loops FIXED_TEMPS)
#   GPU 3 -> livecodebench, ead     (loops DECAY_FREQS)
#
# Usage:
#   bash recipe/annealed_sampling/codeRL/run_all_eval_parallel.sh
#
# All env vars understood by eval_grid_worker.sh are forwarded (MODEL,
# DATA_ROOT, OUT_DIR, N_SAMPLES, MAX_PROMPTS, LCB_VERSION, FIXED_TEMPS,
# DECAY_FREQS, START_TEMP, END_TEMP, GPU_MEM_UTIL).
#
# If you have fewer than 4 GPUs, run eval_grid_worker.sh directly with the
# (gpu, benchmark, mode) tuples you want, sequentially or however you like.

set -euo pipefail

WORKER="recipe/annealed_sampling/codeRL/eval_grid_worker.sh"
LOG_ROOT="${LOG_ROOT:-./logs/inference_only_eval/_worker_logs}"
mkdir -p "${LOG_ROOT}"

# (gpu, benchmark, mode) assignments. Edit to match your GPU count.
ASSIGNMENTS=(
    "0 humanevalplus fixed"
    "1 humanevalplus ead"
    "2 livecodebench fixed"
    "3 livecodebench ead"
)

pids=()
for a in "${ASSIGNMENTS[@]}"; do
    set -- ${a}
    gpu="$1"; bench="$2"; mode="$3"
    logf="${LOG_ROOT}/gpu${gpu}_${bench}_${mode}.log"
    echo "Launching: GPU=${gpu} ${bench} ${mode}  (log -> ${logf})"
    bash "${WORKER}" "${gpu}" "${bench}" "${mode}" > "${logf}" 2>&1 &
    pids+=("$!")
done

echo "Launched ${#pids[@]} workers; waiting..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "Worker pid ${pid} exited non-zero."
        fail=1
    fi
done

if [ "${fail}" -ne 0 ]; then
    echo "One or more workers failed; check ${LOG_ROOT}/*.log"
    exit 1
fi
echo "All workers finished. Summaries in your OUT_DIR (default ./logs/inference_only_eval/<model>/)."

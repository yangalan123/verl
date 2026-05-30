#!/bin/bash
# One-shot collector for Day-0 inference-only results. Wraps
# aggregate_day0_results.py: prints the full grid table + the EAD-best vs
# fixed-best headline summary, and dumps CSV + Markdown for the paper.
#
# Usage:
#   bash recipe/annealed_sampling/codeRL/collect_day0_results.sh
#   ROOT=./logs/inference_only_eval OUT=./logs/day0_tables \
#       bash recipe/annealed_sampling/codeRL/collect_day0_results.sh
#
# Env:
#   ROOT  default ./logs/inference_only_eval   -- where summary__*.json live
#   OUT   default ./logs/day0_tables           -- where CSV / MD are written
#   MODEL_FILTER     optional substring (e.g. Qwen3-4B)
#   BENCH_FILTER     optional substring (e.g. HumanEval)

set -euo pipefail

ROOT="${ROOT:-./logs/inference_only_eval}"
OUT="${OUT:-./logs/day0_tables}"
PY="recipe/annealed_sampling/codeRL/aggregate_day0_results.py"

mkdir -p "${OUT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${OUT}/day0_${STAMP}.csv"
MD="${OUT}/day0_${STAMP}.md"

ARGS=(--root "${ROOT}" --csv "${CSV}" --md "${MD}" --summary)
[ -n "${MODEL_FILTER:-}" ] && ARGS+=(--model "${MODEL_FILTER}")
[ -n "${BENCH_FILTER:-}" ] && ARGS+=(--benchmark "${BENCH_FILTER}")

echo "[collect] root=${ROOT}"
python "${PY}" "${ARGS[@]}"
echo "[collect] CSV -> ${CSV}"
echo "[collect] MD  -> ${MD}"

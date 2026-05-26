# COLM 2026 Rebuttal Runbook

This document is the single entry point for collaborators running the COLM rebuttal experiments. It explains *what* to run, *in what order*, *how long it takes*, and *which experiments can run in parallel*. Every script below was written so it can be invoked from the `verl` repo root on a single GPU node (4xA100-80GB) without Docker or any external sandbox service.

Anything marked **D0** is "Day 0" -- cheap, inference-only, and can be launched immediately in parallel with everything else.

---

## 0. One-time setup

```bash
# Repo root
cd /path/to/verl

# Conda env (or whatever your cluster uses)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <your_verl_env>

# Where parquets and checkpoints will live (edit if you have a scratch dir)
export ROOT_DIR=$(pwd)
export DATA_ROOT=${ROOT_DIR}/data
mkdir -p ${DATA_ROOT}
```

If your cluster mangles `\r` line-endings, run:

```bash
find recipe/annealed_sampling -name "*.sh" -exec dos2unix {} +
```

---

## 1. Math data (already used by the main paper)

If you already have `data/numina_math/train.parquet` and `data/math500/test.parquet` from previous runs, **skip this step**. Otherwise:

```bash
# Numina-Math training set (large; ~5-30 min depending on disk)
python recipe/annealed_sampling/minimal_rl_data_preprocess/numina_math.py \
    --local_dir ${DATA_ROOT}/numina_math
python recipe/annealed_sampling/minimal_rl_data_preprocess/math_dataset.py \
    --local_dir ${DATA_ROOT}/math500
```

These are the same parquets the original experiments used. No EAD-specific changes.

---

## 2. Code-reasoning data (new)

This builds three parquets used by the code experiments. All execution stays local (subprocess + `SIGALRM`), no Docker.

```bash
bash recipe/annealed_sampling/codeRL/step0_data_preprocess.sh
```

Outputs:

* `data/eurus2_code/train.parquet`            -- 10K-problem RL training set
* `data/livecodebench/release_v2_test.parquet` -- held-out eval, stdin/stdout
* `data/humanevalplus/test.parquet`            -- held-out eval, function-style

Time: ~5-15 min once the HuggingFace caches are warm.

Optional knobs (env vars):

| Var | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./data` | Where parquets go |
| `MAX_TRAIN_EXAMPLES` | `10000` | Cap on training rows |
| `LCB_VERSION` | `release_v2` | LiveCodeBench config tag |

If your network is fussy about HF, you can pre-`huggingface-cli download` the three repos and re-run.

---

## 3. The runs

### Priority matrix

| Tag | Experiment | Wall-clock (4xA100) | Independent? |
|----|----|----|----|
| **D0-a** | Inference-only eval on Qwen-2.5-Coder-1.5B-Instruct: HumanEval+ + LCB, 4 sampling configs each | ~3-4 h total | Yes (each config independent) |
| **D0-b** | Inference-only Maj@N eval on math (already in the paper -- only rerun if data tables go in) | ~2 h | Yes |
| **D1** | Code-RL: fixed-T baseline T=1.0 | ~10-14 h | Yes |
| **D1** | Code-RL: fixed-T baseline T=0.7 | ~10-14 h | Yes |
| **D1** | Code-RL: EAD | ~10-14 h | Yes |
| **D2** | Math ablation: `d_max` sweep (5 values) | ~10 h x 5 = 50 h | Sequential by default; parallelizable if you can spread runs across nodes |
| **D2** | Math ablation: schedule shapes (4 modes) | ~10 h x 4 = 40 h | Same -- sequential by default, parallelizable across nodes |

If you have only one box, run D0 first while you decide whether you can afford D1/D2. **D0 alone is enough to give the rebuttal a concrete code-reasoning result** (inference-only Pass@8) and to validate that the EAD schedule transfers to code without RL training.

### Parallelizing on a single 4xA100 node

The Day-0 inference-only configs are very small (a 1.5B model in vLLM at TP=1 uses one GPU). On a 4xA100 node you can run **four** Day-0 configs simultaneously, one per GPU, by setting `CUDA_VISIBLE_DEVICES`:

```bash
# Terminal/job 1
CUDA_VISIBLE_DEVICES=0 N_SAMPLES=8 TP=1 \
  bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh \
       2>&1 | tee logs/d0_gpu0.log &

# Terminal/job 2 (or job 3, 4) -- same command, different CUDA_VISIBLE_DEVICES
```

To split across configs rather than benchmarks, call the underlying Python directly:

```bash
# GPU 0 -- HumanEval+, fixed T=0.7
CUDA_VISIBLE_DEVICES=0 python recipe/annealed_sampling/codeRL/inference_only_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_parquet ./data/humanevalplus/test.parquet \
    --mode fixed --temperature 0.7 --n_samples 8 \
    --tensor_parallel_size 1 \
    --output_dir ./logs/inference_only_eval &

# GPU 1 -- HumanEval+, EAD
CUDA_VISIBLE_DEVICES=1 python recipe/annealed_sampling/codeRL/inference_only_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_parquet ./data/humanevalplus/test.parquet \
    --mode ead --start_temp 1.2 --end_temp 0.1 --decay_freq 200 \
    --n_samples 8 --tensor_parallel_size 1 \
    --output_dir ./logs/inference_only_eval &

# GPU 2 -- LiveCodeBench, fixed T=1.0
CUDA_VISIBLE_DEVICES=2 python recipe/annealed_sampling/codeRL/inference_only_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_parquet ./data/livecodebench/release_v2_test.parquet \
    --mode fixed --temperature 1.0 --n_samples 8 \
    --tensor_parallel_size 1 \
    --output_dir ./logs/inference_only_eval &

# GPU 3 -- LiveCodeBench, EAD
CUDA_VISIBLE_DEVICES=3 python recipe/annealed_sampling/codeRL/inference_only_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_parquet ./data/livecodebench/release_v2_test.parquet \
    --mode ead --start_temp 1.2 --end_temp 0.1 --decay_freq 200 \
    --n_samples 8 --tensor_parallel_size 1 \
    --output_dir ./logs/inference_only_eval &

wait
```

The training runs (D1 / D2) each need all 4 GPUs because they use FSDP, so they cannot be parallelized on a single node.

If you have multiple nodes, the D1/D2 sweeps are embarrassingly parallel -- just split the shell loops across nodes.

---

## 4. Day 0 (start here, runs in parallel)

```bash
# All-in-one script (sequential by default -- splits across configs and benchmarks)
bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh \
    2>&1 | tee logs/d0_run.log
```

This loops over:

* benchmarks: `HumanEval+`, `LiveCodeBench release_v2`
* sampling: fixed `T in {0.7, 1.0, 1.2}` and `EAD negexp 1.2 -> 0.1`

For each config it writes:

```
logs/inference_only_eval/<model>/summary__<bench>__<mode>.json
logs/inference_only_eval/<model>/per_prompt__<bench>__<mode>.jsonl
```

The summary files contain `pass@1`, `pass@K`, `worst@K`, the number of prompts, and the full sampling config -- copy these directly into Table 6 (inference-only) in the paper appendix.

You can run the *same* script against an RL-trained checkpoint by setting `MODEL=path/to/checkpoint` -- this is the cheapest way to populate Table 5 (RL-trained Pass@1) at the end.

---

## 5. Day 1 -- code-reasoning RL training

Each of these is a single 4xA100 job. They are independent and can be scheduled in any order.

```bash
# Fixed-temperature baselines
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 1.0
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 0.7
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 1.2

# EAD
bash recipe/annealed_sampling/codeRL/step1_train_ead_qwen_coder_1_5b.sh
```

Each script automatically:

* trains on `data/eurus2_code/train.parquet`
* validates on both `LiveCodeBench` and `HumanEval+` every 10 steps
* checkpoints to `checkpoints/codeRL_eurus2_qwen_coder_1_5b/<experiment_name>/`
* logs to W&B if a key is configured, otherwise just to console + file

Recommended minimum to fill in the rebuttal: at least one fixed-T baseline (T=1.0) + one EAD run. The extra baselines (T=0.7, T=1.2) are nice-to-have for the table but the inference-only comparison already establishes the baseline trend.

After training finishes, rerun Day-0 eval against the final checkpoint:

```bash
EXP=ead_negexp_explore_1.2_stable_0.1_d0_200_alpha_5_dmax_40000
MODEL=checkpoints/codeRL_eurus2_qwen_coder_1_5b/${EXP}/global_step_<last>/actor/huggingface
N_SAMPLES=8 MODEL=${MODEL} bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh
```

(`global_step_<last>` -- pick the highest-numbered subdirectory.)

---

## 6. Day 2 -- math ablations (only if compute allows)

Both scripts run a *sequential* sweep over hyperparameter / decay-mode values. Each individual run takes ~10 h. The scripts are written so you can comment out values you don't need.

```bash
# d_max sweep: 200, 1000, 5000, 40000, 200000 (the paper uses 40000)
bash recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh

# schedule shapes: negexp, linear, two_stage, mean_matched
bash recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh
```

The schedule-shape script depends on the new decay modes added to `verl/workers/rollout/vllm_rollout/annealed_sampling.py` (see the `# BEGIN COLM REBUTTAL EDIT` block). These modes are already in the repo; no extra setup needed.

To split a sweep across nodes, copy the `for` loop body into a per-value script and launch each on its own node.

---

## 7. Collecting results back into the paper

The placeholder cells live in `review_and_rebuttal/COLM/source_unzipped/colm_2026/tex/appendix.tex` inside the `# COLM REBUTTAL EDIT` blocks. Search the file for `TBD` to find them.

* `Table 5` ("Code-reasoning results, RL-trained Pass@1"): fill in the Pass@1 column with the final-step `pass@1` from the eval JSON of each RL-trained checkpoint.
* `Table 6` ("Inference-only Pass@8"): copy from the Day-0 summary files directly.
* `Fig. ablation_cap_extended` (d_max sweep): build a single matplotlib figure from the W&B "best@16" curves across the 5 runs.
* `Fig. schedule_shapes`: same, across the 4 modes.

A short Python helper to assemble Table 6 from the Day-0 JSON files:

```python
import glob, json, pandas as pd
rows = []
for f in glob.glob("logs/inference_only_eval/Qwen2.5-Coder-1.5B-Instruct/summary__*.json"):
    s = json.load(open(f))
    rows.append({
        "bench": "HumanEval+" if "humanevalplus" in s["eval_parquet"] else "LiveCodeBench",
        "mode": s["mode"],
        "T": s["config"]["temperature"] if s["mode"] == "fixed" else "1.2->0.1",
        "pass@1": round(s["pass@1"], 4),
        f"pass@{s['n_samples_per_prompt']}": round(s[f"pass@{s['n_samples_per_prompt']}"], 4),
    })
print(pd.DataFrame(rows).sort_values(["bench", "mode", "T"]).to_markdown(index=False))
```

---

## 8. Troubleshooting

* **`$'\r': command not found`** -- LF/CRLF mismatch. Run `dos2unix` on the script (see Section 0).
* **vLLM `OOM` on EAD eval** -- lower `--gpu_memory_utilization` from 0.85 to 0.7 in the inference-only eval call.
* **HumanEval+ scores all zero** -- usually the candidate is wrapped in extra prose. Check `per_prompt__*.jsonl`; if `successes` are mostly 0.0 with status `"failed: ..."` the model is just bad at the task. If status is `"timed out"` for many, raise the per-test timeout from 8s to 16s in `verl/utils/reward_score/humanevalplus.py:compute_score(..., timeout=...)`.
* **LCB scores all zero** -- usually a parsing issue. Make sure the model wraps the answer in a ```` ```python ... ``` ```` block; the system prompt in `prepare_eval_livecodebench.py` already asks for this. If you swap to a non-instruct base model, you may need to add a few-shot prefix.

---

## 9. Minimum viable rebuttal (if compute is tight)

If you can only afford one day on one node:

1. Run all of **D0** in parallel across the 4 GPUs (above) -- ~3 h.
2. Copy the 8 summary numbers into Table 6 of `tex/appendix.tex`.
3. Skip D1 / D2 and note "training-time code-reasoning runs are in progress; we will add them to the camera-ready" in the rebuttal text.

The inference-only result alone speaks to two reviewer concerns (generality beyond math; schedule-shape vs. entropy budget), and Tables 5 / 6 plus the new appendix sections already provide the structural changes the reviewers asked for.

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

The Day-0 inference-only configs are very small: a 1.5B model in vLLM at `TP=1` fits on one A100. On a 4xA100 node you can therefore run **four configs at once, one per GPU**, by launching `inference_only_eval.py` directly (one process per `CUDA_VISIBLE_DEVICES`) instead of using the wrapper script `step2_eval_inference_only.sh`.

> The wrapper script loops over *all* (benchmark, sampling-mode) configs sequentially in a single process. So running the wrapper four times in parallel would just do the same work four times. Either use the wrapper once (sequential, slow) **or** call the Python entry point directly (parallel, fast) -- not both.

Example: 4 configs in parallel, two benchmarks x {best fixed-T baseline, EAD}:

```bash
# GPU 0 -- HumanEval+, fixed T=1.0
CUDA_VISIBLE_DEVICES=0 python recipe/annealed_sampling/codeRL/inference_only_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_parquet ./data/humanevalplus/test.parquet \
    --mode fixed --temperature 1.0 --n_samples 8 \
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

This covers the four highest-priority cells of Table 6 in one pass. To also run the secondary fixed temperatures (`T=0.7`, `T=1.2`), launch a second round in the same shape after the first `wait`.

If you'd rather just "set and forget", run the wrapper script **once** on all 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TP=1 N_SAMPLES=8 \
  bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh \
       2>&1 | tee logs/d0_run.log
```

This is sequential (one config at a time, each using one GPU and leaving the others idle), so it is ~4x slower than the parallel pattern above but requires no manual orchestration.

The training runs (D1 / D2) each need all 4 GPUs because they use FSDP, so they cannot be parallelized on a single node. If you have multiple nodes, the D1/D2 sweeps are embarrassingly parallel -- just split the shell loops across nodes.

---

## 4. Day 0 (start here)

The Day-0 wrapper script runs **one config at a time** (sequentially) over:

* benchmarks: `HumanEval+`, `LiveCodeBench release_v2`
* sampling: fixed `T in {0.7, 1.0, 1.2}` and `EAD negexp 1.2 -> 0.1`

That is 8 configs in series, each using a single GPU. **Run it once**, not once-per-GPU.

```bash
# Set-and-forget option (sequential, ~3-4 h on 1 A100, ~same on 4 A100s since GPUs 1-3 sit idle)
bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh \
    2>&1 | tee logs/d0_run.log
```

To actually use all 4 GPUs in parallel, call `inference_only_eval.py` directly with one process per `CUDA_VISIBLE_DEVICES` -- see the "Parallelizing on a single 4xA100 node" block in Section 3. That cuts the wall-clock to ~1 h.

### Automated grid sweep (recommended)

Two helper scripts wrap the per-config calls so you don't orchestrate by hand:

* **`eval_grid_worker.sh <gpu_id> <benchmark> <mode>`** -- pins one GPU and loops over a grid on one benchmark:
  * `mode=fixed` loops over `FIXED_TEMPS` (default `0.7 1.0 1.2`).
  * `mode=ead` loops over the EAD decay-rate ablation `DECAY_FREQS` (default `25 50 100 200`, i.e. `d_0`), at `START_TEMP=1.2 -> END_TEMP=0.1`.
  ```bash
  # one GPU, HumanEval+, all three fixed temperatures
  bash recipe/annealed_sampling/codeRL/eval_grid_worker.sh 0 humanevalplus fixed
  # one GPU, HumanEval+, EAD decay_freq ablation
  bash recipe/annealed_sampling/codeRL/eval_grid_worker.sh 1 humanevalplus ead
  ```

* **`run_all_eval_parallel.sh`** -- launches four workers at once (one per GPU): `{humanevalplus, livecodebench} x {fixed, ead}`. Each worker loops over its own grid internally.
  ```bash
  bash recipe/annealed_sampling/codeRL/run_all_eval_parallel.sh
  ```
  Per-worker logs land in `logs/inference_only_eval/_worker_logs/`. All `eval_grid_worker.sh` env vars (`MODEL`, `DATA_ROOT`, `OUT_DIR`, `N_SAMPLES`, `MAX_PROMPTS`, `LCB_VERSION`, `FIXED_TEMPS`, `DECAY_FREQS`, `START_TEMP`, `END_TEMP`, `GPU_MEM_UTIL`, `MAX_TOKENS`, `ENABLE_THINKING`, `MAX_MODEL_LEN`, `TP`) are honored.

* **`run_day0_models.sh`** -- Day-0 only: sweeps a **registry of newer Qwen models** (edit the `MODELS` array at the top), running the full `{humanevalplus, livecodebench} x {fixed, ead}` grid for each. Models run sequentially; for `TP=1` models the four tasks run in parallel across GPUs 0-3. This is the script to use for the "does EAD transfer to newer / stronger / long-reasoning models?" comparison.
  ```bash
  bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  # quick subset (one model, subsampled):
  MODELS_FILTER="Qwen3-4B" MAX_PROMPTS=40 N_SAMPLES=4 \
      bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  # one dedicated session per (benchmark, mode) TASK; each session loops over
  # ALL registry models on one GPU (run each line in a separate terminal/pane):
  GPUS=0 TASK_FILTER="humanevalplus fixed" bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  GPUS=1 TASK_FILTER="humanevalplus ead"   bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  GPUS=2 TASK_FILTER="livecodebench fixed" bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  GPUS=3 TASK_FILTER="livecodebench ead"   bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  # (or dedicate a session per MODEL instead, with MODELS_FILTER):
  GPUS=0 MODELS_FILTER="Qwen2.5-Coder-7B"  bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  # a TP>1 MoE model needs >=TP GPUs in $GPUS:
  GPUS="0,1" MODELS_FILTER="Qwen3-Coder-30B" bash recipe/annealed_sampling/codeRL/run_day0_models.sh
  ```
  Two orthogonal filters let you slice the (model x benchmark x mode) grid however you like:
  * `TASK_FILTER` (substring on `"<benchmark> <mode>"`) restricts which `(benchmark, mode)` tasks run -- e.g. `"humanevalplus fixed"` for one cell, `"ead"` for both EAD tasks. Each session then loops that task over all models.
  * `MODELS_FILTER` (substring on the model id) restricts which models run.
  * `GPUS` (comma- or space-separated) restricts the invocation to those GPUs: for TP=1 models the selected tasks run up to `|GPUS|` at a time (so `GPUS=0` runs them sequentially on GPU 0, ideal for a dedicated per-task or per-model session); for TP>1 models the first `TP` ids are used and the model is skipped if `|GPUS| < TP`.
  The registry pins the right output budget per model. **Long-reasoning models matter here:** the Qwen3 dual-mode models (`Qwen3-4B`, `Qwen3-8B`) emit `<think>` traces, so they run with `ENABLE_THINKING=on` and a large `MAX_TOKENS` (16384) / `MAX_MODEL_LEN` (20480); the code-specialized Instruct models (`Qwen2.5-Coder-{1.5B,7B}-Instruct`) stay at `MAX_TOKENS=2048`, non-thinking. `Qwen3-Coder-30B-A3B-Instruct` is a 30B (3B-active) MoE set to `TP=2`; drop it to `TP=1` if you can spare only one GPU per model (it fits one A100-80GB with a smaller context). The EAD logits processor anneals temperature over the *full* token stream, including the thinking trace, which is exactly the regime we want to test.

`eval_grid_worker.sh` writes each benchmark into its **own subdirectory** so HumanEval+ and LiveCodeBench never share a folder:

```
logs/inference_only_eval/<model>/humanevalplus/summary__test__<mode>__<tag>.json
logs/inference_only_eval/<model>/livecodebench/summary__release_v2_test__<mode>__<tag>.json
```

Within a benchmark subdir, each config writes a distinct file (fixed configs tagged by temperature, EAD configs by `neg_<tmax>_<tmin>_d<decay_freq>`), so nothing overwrites and you can diff the whole grid afterward.

(The bare `inference_only_eval.py` entry point still writes flat into whatever `--output_dir` you pass; the per-benchmark subdir is added by the `eval_grid_worker.sh` wrapper.)

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
* **caps at 100 training steps by default** (`TOTAL_TRAINING_STEPS=100`) for a quick-and-dirty rebuttal signal; override with `TOTAL_TRAINING_STEPS=-1` for a full epoch
* validates on both `LiveCodeBench` and `HumanEval+` every 10 steps
* keeps only the most recent checkpoint (`max_*_ckpt_to_keep=1`) to bound disk usage
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

Both scripts run a *sequential* sweep over hyperparameter / decay-mode values. **Ablations are capped at 100 training steps by default** (`TOTAL_TRAINING_STEPS=100`) -- enough to reveal the relative ordering of configs without training to convergence, which is all an ablation needs. At ~100 steps each run is roughly 1-2 h instead of ~10 h. The scripts are written so you can comment out values you don't need.

```bash
# d_max sweep. Default DMAX_VALUES="200 300 500 40000" -- tuned for the 100-step
# cap: within 100 steps the decay rate only grows to d_s = d_0 + alpha*100 = 700,
# so any d_max >= ~700 behaves identically. We therefore sweep small caps that
# actually clamp the schedule plus one large "uncapped" reference (40000).
bash recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh

# schedule shapes: negexp, linear, two_stage, mean_matched
bash recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh
```

> If you want the paper's *wide* d_max robustness sweep (200 ... 200000), it only
> becomes meaningful with many more steps. Run that one at full length:
> ```bash
> TOTAL_TRAINING_STEPS=-1 DMAX_VALUES="200 1000 5000 40000 200000" \
>   bash recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh
> ```

To change the cap (e.g. a longer 200-step run, or a full epoch):

```bash
TOTAL_TRAINING_STEPS=200 bash recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh
TOTAL_TRAINING_STEPS=-1  bash recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh   # full epoch
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

**Aggregator (recommended).** `aggregate_day0_results.py` walks every
`summary__*.json` under the eval root and emits a console table, CSV, Markdown,
and an "EAD-best vs fixed-best" headline summary (the number for the rebuttal).
Pure stdlib, no pandas. The shell wrapper `collect_day0_results.sh` runs it with
timestamps and dumps CSV + MD into `./logs/day0_tables/`:

```bash
# everything at once: console table + CSV + MD + headline deltas
bash recipe/annealed_sampling/codeRL/collect_day0_results.sh

# or call the python directly with filters
python recipe/annealed_sampling/codeRL/aggregate_day0_results.py \
    --root ./logs/inference_only_eval --summary --model Qwen3-4B
```

The headline summary prints, per (model, benchmark), the best fixed-T vs. best
EAD config on Pass@1 / Pass@K / Worst@K plus the EAD delta -- copy the table
rows straight into Table 6 and quote the deltas in the rebuttal text.

**Sweeping pass@k / worst@k WITHOUT re-running generation.** Each
`per_prompt__*.jsonl` already stores the per-completion score vector
(`successes`), and pass@k / worst@k are a pure function of "c correct out of n".
So `recompute_passk.py` computes pass@k and worst@k for any list of k offline
(unbiased Codex-style estimators), no regeneration needed:

```bash
python recipe/annealed_sampling/codeRL/recompute_passk.py \
    --root ./logs/inference_only_eval --k 1,2,4,8,16 --csv passk.csv --md passk.md
```

The ONLY requirement is that generation used `n_samples >= max(k)`: to be able
to report pass@16 later, run the eval grid with `N_SAMPLES=16` (the script warns
and leaves cells blank for any k that exceeds the samples actually generated).
Re-scoring under a *different* reward/extraction is a separate need -- for that,
generate with `SAVE_COMPLETIONS=1` (optionally `COMPLETION_CHAR_CAP=N`) so the
full completion texts are persisted to the per-prompt JSONL; otherwise only the
first completion's (truncated) text is kept.

---

## 8. Troubleshooting

* **`$'\r': command not found`** -- LF/CRLF mismatch. Run `dos2unix` on the script (see Section 0).
* **vLLM `OOM` on EAD eval** -- lower `--gpu_memory_utilization` from 0.85 to 0.7 in the inference-only eval call.
* **`huggingface/tokenizers: The current process just got forked, after parallelism has already been used...`** -- harmless warning, not an error. It fires once per process when vLLM forks its workers after `AutoTokenizer.from_pretrained` has already initialised the Rust tokenizer thread pool. `inference_only_eval.py` already sets `TOKENIZERS_PARALLELISM=false` at the top of the file, so you should not see it from the eval entry point. If you do see it from a custom wrapper, prepend the env var:
  ```bash
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python my_script.py ...
  ```
  or once at the top of the shell:
  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```
* **`OSError: [Errno 16] Device or resource busy: '.nfsXXXX'`** during scoring -- triggered by NFS silly-rename on the `multiprocessing.Manager()` socket when `$TMPDIR` is NFS-backed. Both code rewards now avoid `Manager()` and use a `multiprocessing.Queue` (anonymous pipes, no on-disk artefact):
  * HumanEval+: `verl/utils/reward_score/humanevalplus.py` (also routes its scratch dir to a local root `/dev/shm` -> `/tmp` -> `/var/tmp`; override with `HUMANEVALPLUS_TMPDIR`).
  * LiveCodeBench / codecontests / apps / taco: `verl/utils/reward_score/prime_code/utils.py`.

  If you still hit it (e.g. an older checkout), the one-line workaround without code changes is to point `$TMPDIR` at a local disk:
  ```bash
  export TMPDIR=/tmp     # or /dev/shm
  ```
  Note: the traceback usually appears at interpreter exit, *after* the eval has already finished computing and written its summary JSON, so your scores were probably correct -- check `logs/.../summary__*.json` before re-running.
* **HumanEval+ scores 0% across the board** -- almost always a reward-path bug, not the model. Published Qwen2.5-Coder-1.5B-Instruct hits ~64% Pass@1, so a clean 0/164 means the candidate never reached the `check(...)` harness. Run the diagnostic:
  ```bash
  python recipe/annealed_sampling/codeRL/diagnose_humanevalplus.py \
      --eval_parquet ./data/humanevalplus/test.parquet \
      --per_prompt_jsonl ./logs/inference_only_eval/<model>/humanevalplus/per_prompt__test__fixed__T1.0.jsonl
  ```
  It (1) checks that every parquet row has non-empty `entry_point` and `tests`; (2) re-scores the canonical solution (must return 1.0); (3) bucketises the failure statuses from a previous run and prints a few example completions per bucket so you can see whether the model's output is mis-formatted, the function name doesn't match, the test harness raises, etc. The patched `inference_only_eval.py` now also saves `statuses` and `first_completion` to `per_prompt__*.jsonl`, so future runs are debuggable end-to-end.

  Common categories the diagnostic surfaces:
  * `"failed: NameError: name 'X' is not defined"` -- the model used a different function name than `entry_point`. Loosen the candidate by also feeding the original prompt header (set `prompt_header = ex["prompt"]` in `prepare_eval_humanevalplus.py`).
  * `"failed: SyntaxError: ..."` -- the extracted block is not Python (e.g., chat preamble leaked in). Inspect `first_completion`; you may want to switch to a non-instruct base model or tighten the system prompt.
  * `"timed out"` -- raise the per-test timeout from 8s to 16s in `humanevalplus.compute_score(..., timeout=16.0)`.
  * `"missing entry_point or tests"` from the parquet check -- the prepare script grabbed the wrong HF field. Re-run `prepare_eval_humanevalplus.py` after verifying the dataset schema (`evalplus/humanevalplus` uses field `test`).
* **LCB scores all zero** -- usually a parsing issue. Make sure the model wraps the answer in a ```` ```python ... ``` ```` block; the system prompt in `prepare_eval_livecodebench.py` already asks for this. If you swap to a non-instruct base model, you may need to add a few-shot prefix.

---

## 9. Minimum viable rebuttal (if compute is tight)

If you can only afford one day on one node:

1. Run all of **D0** in parallel across the 4 GPUs (above) -- ~3 h.
2. Copy the 8 summary numbers into Table 6 of `tex/appendix.tex`.
3. Skip D1 / D2 and note "training-time code-reasoning runs are in progress; we will add them to the camera-ready" in the rebuttal text.

The inference-only result alone speaks to two reviewer concerns (generality beyond math; schedule-shape vs. entropy budget), and Tables 5 / 6 plus the new appendix sections already provide the structural changes the reviewers asked for.

# ICLR 2027 rebuttal runs

One place to run the extra experiments for the ICLR 2027 revision.
These scripts live on `yangalan123/verl`, branch `annealed_sampling`.
Do not copy them into the public release repo `yangalan123/EAD-RLVR`.

Run order, cheapest useful result first:

1. Mean-matched temperature control (one new math run).
2. Dr. GRPO baseline, then EAD + Dr. GRPO (two math runs).
3. Code: inference-only check, then the short RL runs.
4. Pass@K training is not in this round. Leave it for the camera-ready version.

All math runs use the same data, model, and batch as the paper's 1.5B DAPO runs,
so the new curves drop onto the existing W&B figures.

## 0. From scratch

```bash
git clone git@github.com:yangalan123/verl.git
cd verl
git checkout annealed_sampling
```

The cluster copies of these scripts assume:

- repo root `ROOT_DIR=/fsx/zhuokai/verl`
- Slurm working directory `/fsx/zhuokai/verl`
- 8 GPUs, the existing verl environment
- `VLLM_USE_V1=0` (already set inside the scripts; the EAD logits processor is the vLLM v0 API)
- training data `$ROOT_DIR/data/numina_math/train.parquet`
- eval data `$ROOT_DIR/data/math500/test.parquet`

Those two parquets are the same ones used by `run_minimal_rl_dapo.sh`.
If they are missing, build them the way the other minimal-RL runs were built
before starting anything here.

Paper settings, already filled in the scripts:

| knob | value |
|---|---|
| model | `Qwen/Qwen2.5-Math-1.5B` (Llama-3.2-1B-Instruct is commented in each script) |
| algorithm | GRPO advantage, KL coefficient 0 |
| batch | train 1024, mini-batch 256, 1 epoch |
| context | prompt 1024, response 3072 |
| rollouts | `rollout_n=4`, validation `n=16`, lr `1e-6` |
| EAD | `decay_mode=negexp`, `decay_freq=25`, increase 5 per step, cap 40000, warmup 10, start 1.2, floor 0.1 |

`decay_freq` is the paper's `d_0`. The code divides the exponent by `20 * decay_freq`,
not by `decay_freq` alone. The temperature calculator below uses that same formula.
Do not "correct" it back to `d_0 = 25` in the exponent.

Launch with `sbatch` from the repo root. Each script writes

- W&B project `minimal_rl_numina_math`
- a console log at `logs/${project_name}/${experiment_name}.log`
- rollout dumps under `logs/${project_name}/${experiment_name}/`

## 1. Mean-matched control

An exact match is not something we can do inside the rollout. EAD's average
temperature depends on the response length, and that length is unknown before
sampling and shifts during training. A pilot rollout to estimate it is noisy,
costs an extra generation, and forces the batch to wait, which breaks the
parallel rollout.

The control we run is first-order. Take the mean response length at each step
of the existing EAD run, compute EAD's average temperature on a response of
that length, and average over training. Then train once at that fixed temperature.
Every other setting stays identical to the EAD run.

Export the EAD run from W&B as a CSV with a `Step` column and a
`response_length/mean` column (the trainer logs this). Then, on a CPU machine:

```bash
python scripts/mrs_h100/compute_mean_matched_temperature.py \
    --length_csv ead_response_length.csv \
    --length_col "response_length/mean" \
    --tau_max 1.2 --tau_min 0.1 --d0 25 --alpha 5 --d_max 40000 --warmup 10
```

The script prints `MEAN_MATCHED_TEMP=...`. Use that value unchanged
(it already accounts for the warmup, the floor, and the growing decay rate).
A constant-length fallback, if the per-step export is unavailable:

```bash
python scripts/mrs_h100/compute_mean_matched_temperature.py \
    --mean_length 800 --num_steps 400
```

For the paper's length range (about 400 to 1200 tokens) this lands near 1.13-1.17,
close to the existing temperature-1.2 baseline. That is expected: after a few
dozen steps the decay rate is large and most tokens are sampled near 1.2.

Submit the run:

```bash
MEAN_MATCHED_TEMP=1.15 sbatch scripts/mrs_h100/run_minimal_rl_dapo_mean_matched.sh
```

The script refuses to start if `MEAN_MATCHED_TEMP` is empty.
Use the fixed-temperature script. Do not turn on `decay_mode=mean_matched`
in the logits processor: that path still goes through the annealing hook and
is a different, cruder control.

Compare against the existing EAD run and the fixed-temperature DAPO runs
at 1.0 and 1.2. Metrics: Pass@1, Pass@16 (`best@16`), Worst@16.

## 2. Dr. GRPO

Dr. GRPO (Liu et al., 2025, arXiv:2503.20783) is GRPO with two normalizers
removed:

- no division of the advantage by the group's reward standard deviation
  (`algorithm.norm_adv_by_std_in_grpo=False`)
- no division of each response's loss by its own length. The token sum is
  divided by the constant max response length
  (`actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm`)

It is relevant because EAD changes response length, and GRPO's length
normalizer down-weights long responses. Running EAD on top of Dr. GRPO
removes that confound. EAD only changes sampling, so the two combine by
stacking the flags.

Both scripts use clip 0.2/0.2 and no KL, matching Dr. GRPO rather than the
asymmetric DAPO clip. They share model, lr, and batch, so compare them to
each other. The loss is smaller than token-mean GRPO because it is divided
by 3072; keep the shared lr at `1e-6` and do not retune one of them.

```bash
sbatch scripts/mrs_h100/run_minimal_rl_drgrpo.sh
sbatch scripts/mrs_h100/run_annealed_sampling_minimal_rl_negexp_drgrpo.sh
```

The baseline samples at temperature 1.0. The EAD script uses the paper schedule
(`negexp`, floor 0.1).

## 3. Code

The Docker-free code pipeline is `recipe/annealed_sampling/codeRL/`.
Its own `README.md` has the data layout. Short version, from the verl root:

```bash
# one-time, about 5 minutes
bash recipe/annealed_sampling/codeRL/step0_data_preprocess.sh

# cheap check before any RL: base model, no training
bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh

# RL. 4 GPUs is enough. TOTAL_TRAINING_STEPS defaults to 100.
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 1.0
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 1.2
bash recipe/annealed_sampling/codeRL/step1_train_ead_qwen_coder_1_5b.sh
```

Training set is the code subset of Eurus-2 (capped at 10k). Eval is HumanEval+
and LiveCodeBench release_v2. Rewards are local Python subprocesses
(`prime_code`, `humanevalplus`); nothing needs Docker or firejail.
Model is `Qwen/Qwen2.5-Coder-1.5B-Instruct`. The EAD script uses
`decay_freq=200` because the math default (`25`) is sized for math responses.
If EAD loses to the fixed-temperature runs, sweep that value with
`recipe/annealed_sampling/codeRL/ead_sweep.py` (`README_ead_sweep.md`).

## 4. What to send back

For each new run, the W&B run name plus:

- `val-aux/.../reward/best@16/mean`
- `val-aux/.../reward/worst@16/mean`
- `response_length/mean`

Export each metric the same way as the existing figure CSVs: one `Step` column,
then for every run a value column and `__MIN` / `__MAX` columns.
Those CSVs are not in git. They live in `visualization/` next to the verl
checkout (for example `visualization/qwen_2_5_math_1.5B/best_at_16_qwen_2_5_math.csv`).
`visualization/visualization_training_dynamic.py` and `consts.py` turn a CSV
of that shape into the paper figures. Add the new run as extra columns in a
copy of the relevant CSV; do not overwrite the columns that already match
the submitted figures.

A copy of the 2026-06-06 exports is in `visualization_backup_2026-06-06/`
(about 100 CSVs). Several W&B runs share a display name, so a re-export by
name can pick the wrong rerun. Match the run id behind the existing column,
not just the display name.

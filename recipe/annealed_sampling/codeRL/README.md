# Code-Reasoning EAD Recipe (no Docker / no firejail)

This folder contains a minimal, dependency-light recipe for running
**Exploratory Annealed Decoding (EAD)** on code-reasoning tasks. Everything
here uses the existing `verl.utils.reward_score.prime_code` reward, which runs
candidate solutions in a local Python subprocess (multiprocessing + signal
timeout). **No Docker, no firejail, no compiler service is required.**

## What is included

| File | Purpose |
|------|---------|
| `prepare_train_eurus2_code.py` | Build a small train parquet from the code subset of `PRIME-RL/Eurus-2-RL-Data` (stdin/stdout style) |
| `prepare_eval_livecodebench.py` | Build a LiveCodeBench eval parquet (stdin/stdout) routed to `prime_code` |
| `prepare_eval_humanevalplus.py` | Build a HumanEval+ eval parquet that uses a tiny function-call reward (no `evalplus` install needed for scoring; we just `exec` the candidate + reference asserts in a subprocess) |
| `step0_data_preprocess.sh` | Calls all three preprocessors in one shot |
| `step1_train_baseline_qwen_coder_1_5b.sh` | Fixed-temperature GRPO baseline on Qwen2.5-Coder-1.5B-Instruct (temp swept via `START_TEMP` arg) |
| `step1_train_ead_qwen_coder_1_5b.sh` | EAD GRPO on the same model |
| `step2_eval_inference_only.sh` | Pure inference + scoring on HumanEval+ / LiveCodeBench, both with vanilla temperature sampling and EAD (the "Day 0" sanity check) |

## Expected hardware / wall-clock budget

* 4xA100-80GB is sufficient for all the 1.5B runs.
* Training:  ~10-14 h per RL run (1 epoch, 10K problems, `n=4` rollouts).
* Inference-only eval: ~30-60 min for HumanEval+ (164 problems x `n=8`).

## How code execution works (and why no Docker is needed)

`prime_code.compute_score` (see `verl/utils/reward_score/prime_code/__init__.py`)
forks a Python subprocess via `multiprocessing.Process`, feeds each test case
on stdin, captures stdout, and enforces a per-test wall-clock timeout with
`signal.SIGALRM`. The HumanEval+ reward (`humanevalplus.py`, added below) uses
the same subprocess+timeout pattern but evaluates the candidate as a callable
and applies the reference asserts. This is the standard sandboxing approach
used by PRIME-RL and Eurus-2, and it is the path the public PRIME-RL
checkpoints were trained with.

## Quick start

```bash
# 1) preprocess (one-time, ~5 min)
bash recipe/annealed_sampling/codeRL/step0_data_preprocess.sh

# 2) inference-only sanity check (Day-0 result for the rebuttal)
bash recipe/annealed_sampling/codeRL/step2_eval_inference_only.sh

# 3) full RL runs
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 1.0   # baseline temp=1.0
bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh 0.7   # baseline temp=0.7
bash recipe/annealed_sampling/codeRL/step1_train_ead_qwen_coder_1_5b.sh
```

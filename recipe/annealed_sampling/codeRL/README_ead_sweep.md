# EAD parameter sweep (Day-0, inference-only)

Find an Exploratory Annealed Decoding (`negexp`) configuration that **beats the
fixed-temperature baseline** on the code benchmarks, on 4 GPUs, without RL
training / Docker.

## Why a sweep (and why the paper defaults fail on code)

For the `negexp` schedule at inference (`global_step=0`) the temperature is

```
tau(t) = 1 + tau_max - exp( t / (20 * decay_freq) ),   clamped to >= tau_min
```

so it only reaches the floor at `t* = 20 * decay_freq * ln(1 + tau_max - tau_min)`
**response tokens**. The paper's math defaults (`decay_freq=200, tau_max=1.2,
tau_min=0.1`) give `t* ~ 2968` tokens. Code completions are often 100-400 tokens,
so the temperature never anneals and EAD degenerates into high-temperature
sampling -> it underperforms. **The fix: size `decay_freq` from the actual code
response-length distribution.** That is what the smoke/analysis phase measures.

`warmup_period` keeps the first few (low-entropy, structural) response tokens at
the base temperature; we estimate it from the per-position entropy curve.

## Pieces

| File | Role |
|------|------|
| `inference_only_eval.py` | generation worker; `--generate_only` dumps completions + token lengths + neg-avg-loglik (entropy proxy) and skips scoring; `--logprobs 1` enables the entropy proxy |
| `score_dump.py` | CPU-only scorer: reads a `gen__*.jsonl` dump and writes the standard `summary__*.json` / `per_prompt__*.jsonl` |
| `analyze_smoke.py` | reads a dump -> recommends `warmup_period` + `decay_freq` candidates (negexp closed-form) + temp grids |
| `ead_sweep.py` | orchestrator: baseline+smoke -> analyze -> sweep -> report, with a 4-GPU pool and **decoupled CPU scoring** |

The entropy proxy (negative average log-likelihood of sampled tokens estimates
average per-token entropy for long sequences) follows
[`yangalan123/LLMBranchingFactor`](https://github.com/yangalan123/LLMBranchingFactor)
("How Alignment Shrinks the Generative Horizon").

## GPU/CPU split

`--generate_only` makes the GPU process exit as soon as generation is done; the
slow, sandbox-bound scoring runs in a separate CPU process (`score_dump.py`,
thread-pooled). The scheduler keeps all GPUs generating while scoring overlaps on
CPU, so neither resource idles.

## Usage

```bash
# 0) see the plan first (no GPU work)
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --dry_run

# 1) full pipeline (baseline+smoke -> analyze -> sweep -> report)
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3

# narrow to one model / benchmark
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 \
    --models Qwen2.5-Coder-1.5B --benchmarks humanevalplus

# run phases individually
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --phase data
python recipe/annealed_sampling/codeRL/ead_sweep.py --phase analyze   # CPU only
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --phase sweep
python recipe/annealed_sampling/codeRL/ead_sweep.py --phase report    # CPU only
```

Key flags: `--root` (default `./inference-eval-sample32`), `--data_root`,
`--n_samples` (32), `--smoke_prompts/--smoke_samples`, `--sweep_max_prompts`
(subsample the sweep for speed; default full set so comparison to the full
baseline is apples-to-apples), `--cpu_slots`/`--score_workers`, `--metric`
(`pass@1|pass@k|worst@k` for ranking winners), `--force`.

## What each phase does

1. **data** - for each `(model, benchmark)`: if the fixed-`T=1.0` baseline is
   missing, generate it (generate-only, `--logprobs 1`) and score it on CPU; the
   same dump feeds analysis. If a baseline exists but no dump is present, run a
   small smoke generate-only job with logprobs.
2. **analyze** (in-process) - writes `search_space.json` per `(model, benchmark)`
   with `warmup_period`, `decay_freqs`, `start_temps`, `end_temps`, length stats,
   and the entropy curve head.
3. **sweep** - grid = `start_temps x end_temps x decay_freqs` (negexp, fixed
   warmup); generate-only on GPU + score on CPU.
4. **report** - compares each EAD config to the baseline and lists winners;
   writes `ead_sweep_report.json` under `--root`.

## Output layout

```
<root>/<model>/<benchmark>/
    gen__<tag>.jsonl  genmeta__<tag>.json     # generate-only dumps
    summary__<tag>.json  per_prompt__<tag>.jsonl
    search_space.json
<root>/ead_sweep_report.json
<root>/_sweep_logs/*.log
```

`summary__*` / `per_prompt__*` are the same format as the rest of the codeRL
tooling, so `aggregate_day0_results.py` and `recompute_passk.py` work on them.

## Notes

- Long-reasoning models (`Qwen3-4B/8B`, thinking on) use a large `max_tokens`;
  the registry in `ead_sweep.py` pins per-model `max_tokens`/`tp`/`max_model_len`/
  `gpu_mem`. `Qwen3-Coder-30B-A3B` is `tp=2`.
- To later sweep `pass@k` at multiple k, the runs already use `n_samples=32`, so
  `recompute_passk.py --k 1,2,4,8,16,32` works without regeneration.
- Generate-only dumps with `--logprobs 1` and 32 samples can be sizable; they are
  written once per baseline and reused for analysis.

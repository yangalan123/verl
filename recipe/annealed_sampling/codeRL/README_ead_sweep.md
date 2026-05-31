# EAD parameter sweep (Day-0, inference-only)

Find an Exploratory Annealed Decoding (`negexp`) configuration that **beats the
best fixed-temperature baseline** on the code benchmarks, on 4 GPUs, without RL
training / Docker. To *confirm* EAD's success the baseline is a **grid** of fixed
temperatures (default `T=0.6,0.7,0.8,1.0,1.2`), and EAD must beat the *best* of
them — not just `T=1.0`.

The bar is intentionally **rough**: a config is *kept as promising* if it beats
the best fixed baseline at **any** `pass@K` (ideally `K>1`); configs that beat at
no `K` are discarded. This is a screening pass, not a final-number computation.

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

# two independent sessions across 8 GPUs (one dataset each) -- the two datasets
# do NOT share parameter setups, so split them and let each tune separately:
#   session A:
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --benchmarks humanevalplus
#   session B:
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 4,5,6,7 --benchmarks livecodebench

# run phases individually
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --phase data
python recipe/annealed_sampling/codeRL/ead_sweep.py --phase analyze   # CPU only
python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --phase sweep
python recipe/annealed_sampling/codeRL/ead_sweep.py --phase report    # CPU only
```

Key flags: `--root` (default `./inference-eval-sample32`), `--data_root`,
`--baseline_temps` (fixed-T grid, default `0.6,0.7,0.8,1.0,1.2`),
`--report_ks` (pass@K values to check; a config is kept if it beats best-fixed
at any K, default `1,2,4,8,16,32`), `--n_samples` (32),
`--smoke_prompts/--smoke_samples`, `--sweep_max_prompts` (subsample the sweep for
speed; default full set so comparison to the full baseline is apples-to-apples),
`--cpu_slots`/`--score_workers`, `--force`. (`--metric` is now unused — keep
decisions are per-K.)

## What each phase does

1. **data** - for each `(model, benchmark)`: generate+score every missing
   fixed-`T` in `--baseline_temps` (generate-only on GPU, scored on CPU). The
   `T=1.0` run additionally captures `--logprobs 1` and doubles as the analysis
   dump; if `T=1.0` already exists and no dump is present, a small smoke
   generate-only job (with logprobs) is run just for analysis.
2. **analyze** (in-process) - writes `search_space.json` per `(model, benchmark)`
   with `warmup_period`, `decay_freqs`, `start_temps`, `end_temps`, length stats,
   and the entropy curve head.
3. **sweep** - grid = `start_temps x end_temps x decay_freqs` (negexp, fixed
   warmup); generate-only on GPU + score on CPU.
4. **report** - recomputes unbiased `pass@K` (over the saved per-prompt
   `successes` vectors) at every `K` in `--report_ks` (default `1,2,4,8,16,32`),
   for the fixed-`T` grid and every EAD config. It **keeps** an EAD config if it
   beats the *best* fixed baseline at **any** `K` (a win at `K>1` is the strong
   case, flagged with `*`), and **discards** configs that beat at no `K`. Prints
   the best-fixed `pass@K` line (value + winning `T` per `K`) and each kept
   config's winning `K`s; writes `ead_sweep_report.json` under `--root`.

## Output layout

```
<root>/<model>/<benchmark>/
    gen__<tag>.jsonl  genmeta__<tag>.json     # generate-only dumps
    summary__<tag>.json  per_prompt__<tag>.jsonl
    search_space.json
<root>/ead_sweep_report__<benchmarks>.json   # per benchmark-filter, so
                                            # parallel sessions don't collide
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

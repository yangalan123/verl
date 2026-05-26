# Response to Reviewer gRgP

We thank the reviewer for the encouraging comments on simplicity and motivation, and for the two pointed concerns about (a) hyperparameter guidance and (b) extension beyond math reasoning. Both are addressed in the revision.

## Q1. Guidance for setting `(tau_max, tau_min, d_0, alpha, d_max, c)` in a new setting

We agree that EAD has six hyperparameters and that empirical tuning is unsatisfying. In the revision, we add two artifacts that make the situation more tractable:

* **A single table of defaults** that we use across *every* experiment in the paper (`Sec. \ref{sec:hp}`, Table 4). The only model-dependent value is `tau_min` (0.1 for 1-1.5B, 0.6-0.8 for 7B+).
* **A selection guide for practitioners** (`Sec. \ref{sec:hp_guide}`) with explicit rules of thumb:
  * `tau_max`: highest value that does not hurt single-sample quality on a held-out probe (`1.2` worked for every model we tried).
  * `tau_min`: scales with the sharpness of the base model's next-token distribution; `0.1` is safe for 1B-scale models, `0.5-0.8` for 7B+.
  * `d_0`: set so that the first ~100 tokens stay at `tau_t >= 1.0` at step 0.
  * `alpha`: a positive growth factor; `5` (one extra "hot" token per training step) was sufficient for all our runs.
  * `d_max`: any value in `[1e3, 1e5]` gave near-identical performance in the extended sweep (`App. \ref{app:dmax_sensitivity}`); we use `4 x 10^4`.
  * `c`: a small constant (10-20 tokens) covering chat-formatting / template prefixes.

Practically, the only knob a practitioner needs to *tune* in a new setting is `tau_min`, which can be set by inspecting the entropy of the base model on the late tokens of a small probe set.

## Q2. Validation beyond math reasoning -- "late steps can also be decision points"

This is a fair concern and we have run a code-reasoning study to address it.

* **Setup.** Qwen-2.5-Coder-1.5B-Instruct, trained with GRPO on a 10K-problem subset of the PRIME-RL Eurus-2 code data, evaluated on LiveCodeBench (release_v2) and HumanEval+. All code execution is local (subprocess + SIGALRM, the PRIME-RL path), no Docker / no sandbox service.
* **Recipe in the repo.** `recipe/annealed_sampling/codeRL/` (preprocessing + training + inference-only Day-0 eval).
* **Section in the paper.** App. \ref{app:code_experiments}, Tables 5 (RL-trained Pass@1) and 6 (inference-only Pass@8 on the off-the-shelf base model).

We also want to address the specific intuition that "late steps can be decision points" in code or agentic reasoning. EAD does not silence late-stage exploration -- the floor temperature `tau_min > 0` keeps a non-trivial degree of stochasticity throughout generation (we added this note explicitly at the end of Sec. 2). The question is whether the *expected* marginal value of exploration is higher early or late. Two pieces of evidence say "early" even in code:

1. **The reverse-annealing ablation (Fig. 6)** -- inverting the schedule to "explore late" underperforms a fixed-temperature baseline.
2. **The schedule-shape ablation (App. \ref{app:schedule_shapes}, added in the revision)** -- a `two_stage` schedule that pushes more mass to the late tail loses to the negexp schedule with identical `(tau_max, tau_min, d_0, alpha, d_max)`. Recipe at `recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh`.

In agentic settings with explicit decision points at arbitrary positions, we would expect a *prompt-* or *step-aware* schedule to outperform the position-only EAD; we list this as future work in Sec. 6.

We thank the reviewer for the careful read.

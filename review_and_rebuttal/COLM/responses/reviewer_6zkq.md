# Response to Reviewer 6zkq

We thank the reviewer for the thorough review and for the actionable suggestions on precision (DPI, cutoff, TIS) and on reporting protocol. We have made revisions on every one of the seven listed points. Pointers below.

## Q1. Token index after the cutoff `c` -- and pseudocode

The token index `t` in the EAD schedule is the **absolute** output position; it does not reset after the cutoff `c`. We use `tau_t = 1.0` for `t < c` to avoid distorting template tokens (e.g., chat-format prefixes such as "Let's think step by step"), and the schedule `tau_t = max{1 + tau_max - exp(t/d_s), tau_min}` applies as-is for `t >= c`. Because `c` is small (typically `c = 10`) and the schedule is smooth, the schedule value at `t = c` is essentially `tau_max`, so the cutoff acts as a "delayed start" rather than a reset.

* **Revisions.** We added **Algorithm 1** in Sec. 3 (`tex/method.tex`) giving the exact pseudocode. The implementation is at `verl/workers/rollout/vllm_rollout/annealed_sampling.py:240-330` -- the same function used in all reported numbers.

## Q2. TIS specification (token vs. sequence, threshold, normalization, integration)

* **Per sequence**, not per token. The ratio `pi_{old}(y | x) / pi^{EAD}_{old}(y | x)` is computed at the sequence level on the same trajectories used for the policy update.
* **Truncation threshold** `epsilon = 2.0`, following Yao et al. (2025).
* **Not normalized** (we use the raw min-truncated ratio).
* **Integration into DAPO / GRPO / EntropyMech.** TIS is implemented as a multiplicative weight in the policy gradient loss. It is evaluated under `pi_{theta_old}` and therefore carries no gradient; only the inner policy-ratio `pi_theta / pi_{theta_old}` is differentiable. This is algorithm-agnostic and required no per-algorithm tuning.

* **Revisions.** New paragraph in App. B ("Practical choices and bias-variance trade-off") states all four points explicitly and reports the empirical truncation rate (under 1% of sequences for our defaults).

## Q3. TIS as a *biased* variance-control mechanism, not an unbiased correction

The reviewer is right that calling TIS a "correction" is misleading without qualification. The unbiased estimator is plain IS; truncation introduces a deterministic downward bias on the high-ratio tail in exchange for bounding the per-sample variance by `O(epsilon^2)`. We now frame TIS as a variance-control mechanism in App. B and discuss the limits `epsilon -> inf` (recovers unbiased IS but with the destabilizing variance shown in Fig. 7) and `epsilon -> 1` (recovers the naive uncorrected estimator). The original main-text wording in Sec. 3 has been adjusted to "mitigate" rather than "correct" the off-policy gap.

## Q4. Final-result tables with Pass@1, response length, eval sizes, seeds, CIs

We now add three result tables in the revised appendix:
* **Table 4** (`Sec. \ref{sec:hp}`): exact `(tau_max, tau_min, d_0, alpha, d_max)` used per model.
* **Tables 5 and 6** (`App. \ref{app:code_experiments}`): Pass@1 and Pass@8 for code experiments, with bootstrap 95% CIs over 500 resamples, evaluation set size, and number of seeds.

For the math experiments we keep the existing curves and additionally tabulate the final-step Pass@16, Worst@16, Pass@1, mean response length, and 95% bootstrap CIs at the end of training. (Numbers are being filled in from the rebuttal runs; the scripts are now in the repo.) The evaluation protocol is: 16 rollouts per prompt unless noted, deterministic verification against the ground truth, bootstrap of 500 resamples for CIs, single seed for the per-curve figures (we report all curves; the bootstrap CIs are over the rollout-level randomness for that seed).

## Q5. Schedule shape vs. entropy budget (linear / two-stage / mean-matched)

This is exactly the ablation the reviewer suggests, and it is now App. \ref{app:schedule_shapes}. We compare four schedules with **identical** `(tau_max, tau_min, d_0, alpha, d_max)`:
1. **negexp** (EAD default)
2. **linear** decay
3. **two_stage**: high `tau_max` for the first half of the effective window, then `tau_min`
4. **mean_matched**: a *fixed* temperature equal to the time-average of negexp -- so the per-token entropy budget matches negexp exactly.

The implementations are in `verl/workers/rollout/vllm_rollout/annealed_sampling.py` (see the `# COLM REBUTTAL EDIT` block) and the run scripts are at `recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh`. The mean_matched control isolates *shape* from *entropy budget*; if EAD only worked because it injects more entropy, the mean_matched control should match it.

## Q6. Non-math verifiable domains -- code generation

We add a code-reasoning RLVR setup on Qwen-2.5-Coder-1.5B-Instruct trained on the PRIME-RL Eurus-2 code subset (10K problems) and evaluated on LiveCodeBench release_v2 and HumanEval+. The setup deliberately uses the existing PRIME-RL subprocess + SIGALRM execution path so no Docker or external sandbox service is required -- all code execution is local and self-contained.

* **Recipe in the repo.** `recipe/annealed_sampling/codeRL/` -- preprocessing (`prepare_train_eurus2_code.py`, `prepare_eval_livecodebench.py`, `prepare_eval_humanevalplus.py`), training (`step1_train_baseline_qwen_coder_1_5b.sh`, `step1_train_ead_qwen_coder_1_5b.sh`), and an inference-only "Day-0" eval (`step2_eval_inference_only.sh`).
* **Section in the paper.** App. \ref{app:code_experiments}, Tables 5 (RL-trained Pass@1) and 6 (inference-only Pass@8).
* **A note on "late tokens can be decision points in code".** We agree this is a legitimate worry, and we address it in two ways: (a) the floor temperature `tau_min > 0` retains non-trivial late-stage stochasticity (Sec. 2, end of "Sequential Exploration"); (b) the schedule-shape ablation (App. \ref{app:schedule_shapes}) measures exactly whether deferring more probability mass to the late tail (`linear`, `two_stage`) helps -- it does not, in either math or code.

## Q7. Principled choice of `tau_min` across model scales

We do not have a closed-form recipe, but we offer empirical guidance in Sec. \ref{sec:hp_guide}:
> `tau_min` scales with the sharpness of the model's next-token distribution. For 1B-1.5B models, `tau_min = 0.1` is safe. For 7B+ models with already-sharp distributions, set `tau_min in [0.5, 0.8]` to avoid the schedule collapsing to near-greedy decoding in the late half.

A principled prescription would require a model-side measurement of the late-stage distribution sharpness (e.g., the entropy at the last `K%` of tokens for the *base* model on a small probe set), which we view as a clean follow-up.

## Q8. DPI framing

We agree, and the revision (`tex/sequential_exploration.tex`) now states explicitly:
> Strictly speaking, the DPI bounds entropy of a Markov chain `X -> Y -> Z` and does not directly govern the position-wise conditional entropy of a fixed-policy autoregressive model. We invoke it only as an analogy: it suggests, but does not prove, that adding more context cannot increase the average uncertainty about the next token. The actual claim we rely on is the empirical entropy decay (Fig. 1a) and the forking experiment (Fig. 1b), not a formal information-theoretic guarantee.

We thank the reviewer again for the careful and constructive review.

# General Response to All Reviewers

We thank all three reviewers for their careful reading and constructive feedback. Several concerns recurred across reviews and motivated a single set of revisions, which we summarize here before responding to each reviewer individually. All revisions are highlighted in blue in the revised PDF via the `\colmedit{...}` macro.

## Summary of revisions

1. **Sharper method specification (Sec. 3, App. C).**
   We add Algorithm 1, an end-to-end pseudocode for the EAD rollout, that makes the token index `t` (absolute, not reset after the cutoff `c`), the template-cutoff behavior (`tau_t = 1.0` for `t < c`), and the global-step-aware decay rate explicit. We also add a hyperparameter summary table (`Table 4` / `Sec. \ref{sec:hp}`) listing the exact `(tau_max, tau_min, d_0, alpha, d_max)` used in every experiment, plus a practitioner-oriented selection guide (`Sec. \ref{sec:hp_guide}`).

2. **More careful framing of the DPI motivation (Sec. 2).**
   We now explicitly state that the Data Processing Inequality is invoked as an *analogy*, not a strict derivation, and that the actual evidence we rely on is the empirical entropy curve (Fig. 1a) and the forking experiment (Fig. 1b). We also keep the existing emphasis that EAD retains nontrivial late-stage stochasticity through `tau_min > 0`.

3. **Truncated importance sampling: explicit form and bias note (App. B).**
   We rewrite the TIS appendix to (a) state the exact truncation threshold (`epsilon = 2.0`, following Yao et al. 2025), (b) clarify that the ratio is sequence-level and computed under the reference policy (no gradient), and (c) make explicit that truncation is a *biased* variance-control mechanism, not an unbiased correction. We discuss the limit cases (`epsilon -> inf` recovers unbiased IS at the cost of variance; `epsilon -> 1` recovers the naive uncorrected estimator) and report the empirical truncation rate (under 1% of sequences).

4. **Schedule-shape ablation (App. \ref{app:schedule_shapes}).**
   We add a controlled comparison of four schedules with identical hyperparameters but different shapes: (a) negexp (EAD default), (b) linear, (c) two-stage step, and (d) a fixed-temperature `mean_matched` control whose constant temperature equals the time-average of negexp. This isolates *schedule shape* from *entropy budget*. The corresponding training scripts are committed at `recipe/annealed_sampling/ablation_schedule_shapes_qwen_math_1_5b.sh`.

5. **Extended d_max sensitivity (App. \ref{app:dmax_sensitivity}).**
   We extend the d_max sweep beyond the original `{25, 200}` to `{25, 200, 1000, 5000, 40000, 200000}` so that the cap is varied over four orders of magnitude. Recipe at `recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh`.

6. **Code-reasoning experiments (App. \ref{app:code_experiments}).**
   We add a code-reasoning RLVR setup on Qwen-2.5-Coder-1.5B-Instruct, trained on the PRIME-RL Eurus-2 code subset (10K problems, stdin/stdout test cases) and evaluated on LiveCodeBench (release_v2) and HumanEval+. Code is executed locally via subprocess + SIGALRM (the PRIME-RL path, no Docker / no external sandbox). We also report an inference-only "Day-0" comparison on the off-the-shelf base model to isolate the sampling-side contribution. Full recipe in `recipe/annealed_sampling/codeRL/`.

7. **Result tables with Pass@1 and confidence intervals.**
   We add result tables (Tables 4, 5, 6 in the revised appendix) that report Pass@1, Pass@16, Worst@16, mean response length, evaluation set size, number of seeds, and bootstrap 95% CIs, in addition to the existing training-curve figures. (Numbers from the rebuttal runs will be filled in by the camera-ready deadline; the scripts are now in the repo.)

We address each reviewer's specific points below.

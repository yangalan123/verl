# Response to Reviewer tz3H

We thank the reviewer for the thoughtful summary and for highlighting the practicality of EAD as a plug-and-play module. We address the two main concerns below.

## Q1. Sensitivity to the "maximum global step" / decay cap `d_max`

The decay cap `d_max` controls the largest value the global-step-aware decay rate `d_s` can grow to during training. We agree this is an important knob and we have added a much more thorough sensitivity analysis.

* **Where in the paper.** The original sweep (App. C, Fig. 8a) compared two values, `d_max in {25, 200}`. In the revision (App. \ref{app:dmax_sensitivity}, Fig. \ref{fig: ablation_cap_extended}) we extend this to `d_max in {25, 200, 1000, 5000, 40000, 200000}`, spanning four orders of magnitude.
* **Recipe in the repo.** `recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh`.
* **Expected message.** Our existing data already shows that performance is essentially flat for any `d_max >= 200`; the only failure mode is `d_max = 25 = d_0`, which collapses the schedule to a fixed-temperature run and removes the global-step adaptation. The extended sweep is designed to confirm that no upper-side failure mode exists either.

We also note that the *initial* decay rate `d_0` is the more sensitive of the two parameters: Fig. 8b in the current submission already shows that combining a small `d_0` with `alpha = 0` (no growth) hurts, while `d_0 in {50, 200, 500}` with `alpha = 5` all perform similarly. We expanded `Sec. \ref{sec:hp_guide}` to give explicit selection rules for both `d_0` and `d_max`.

## Q2. Worst@16 collapse for EAD-alone on Llama-3.2-1B-Instruct (Fig. 3, bottom-right)

The Worst@16 drop on Llama-3.2-1B-Instruct without TIS is, in our reading, exactly the off-policy failure mode that motivates Sec. 4 / App. B of the paper. When the floor temperature `tau_min` is small, a small fraction of trajectories end up with extreme importance ratios under the EAD behavior policy; this inflates the gradient variance and degrades the *worst* sample without affecting the *best*. Pass@16 (best-of-16) is therefore largely unchanged, but Worst@16 (worst-of-16) is sensitive to this tail effect.

* **Empirical evidence we already report.** Fig. 7 shows the corresponding clip-fraction surge and gradient-norm spike on Qwen-2.5-Math-1.5B for the same uncorrected EAD. Adding TIS removes both, and removes the Worst@16 drop.
* **Revisions.** We have rewritten the TIS appendix (App. B) to make explicit that (i) TIS is a *biased* variance-control mechanism with truncation threshold `epsilon = 2.0`; (ii) the truncated ratio is sequence-level; (iii) the empirical truncation rate is under 1% of sequences. We discuss the bias-variance trade-off in the new paragraph "Practical choices and bias-variance trade-off" in App. B.
* **Practical recommendation.** For 1B-scale models with small `tau_min`, we now explicitly recommend pairing EAD with TIS (Sec. \ref{sec:hp_guide}). For 7B+ models with `tau_min >= 0.5`, the off-policy gap is small and EAD-alone is stable.

## Q3. Interaction with PPO / GRPO

EAD is a sampling-side intervention -- it only changes the rollout distribution and is orthogonal to the policy-update rule. We test this empirically with three RL algorithms in the existing Sec. 5.3 / Fig. 4: **DAPO** (our main setting), **GRPO**, and **EntropyMech**. EAD improves all three across all measured checkpoints. PPO is a special case of GRPO with a learned value head; we have not run it directly because the Minimal-RL recipe we follow drops the critic for compute reasons, but no part of EAD's derivation or implementation is GRPO-specific.

We thank the reviewer again for the careful read; we would be glad to clarify any further point.

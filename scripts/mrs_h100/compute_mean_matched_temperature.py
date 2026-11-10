"""
First-order mean-matched temperature for the EAD (negexp) schedule.

An exact mean-matched control is not implementable in RLVR: the average
temperature of an EAD rollout depends on its length, which is unknown before
sampling and drifts during training. As a first-order approximation, we take
the mean response length logged by a previous training run, compute the
per-token average of the EAD temperature at every global step, and average
over training (token-weighted). The resulting single temperature is then used
for a fixed-temperature run (run_minimal_rl_dapo_mean_matched.sh).

The schedule mirrors `annealed_sampling_processor(decay_mode="negexp")` in
verl/workers/rollout/vllm_rollout/annealed_sampling.py:
    tau_t = 1.0                                              if t < warmup
    tau_t = max(1 + tau_max - exp(t / (20 * d_s)), tau_min)  otherwise
    d_s   = min(d_0 + alpha * s, d_max)

Usage:
    # lengths exported from W&B (columns: a step column + response_length/mean)
    python scripts/mrs_h100/compute_mean_matched_temperature.py \
        --length_csv wandb_export.csv --length_col "response_length/mean"

    # or a constant mean length over a fixed number of steps
    python scripts/mrs_h100/compute_mean_matched_temperature.py \
        --mean_length 600 --num_steps 400
"""

import argparse
import csv

import numpy as np


def ead_temperatures(length, global_step, tau_max, tau_min, d0, alpha, d_max, warmup):
    d_s = min(d0 + alpha * global_step, d_max)
    t = np.arange(int(length))
    temps = np.maximum(1.0 + tau_max - np.exp(t / (20.0 * d_s)), tau_min)
    temps[t < warmup] = 1.0
    return temps


def load_lengths(path, step_col, length_col):
    steps, lengths = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if step_col not in reader.fieldnames:
            raise KeyError(f"step column '{step_col}' not in {reader.fieldnames}")
        matches = [c for c in reader.fieldnames if c == length_col or c.endswith(f" - {length_col}")]
        if not matches:
            raise KeyError(f"length column '{length_col}' not in {reader.fieldnames}")
        col = matches[0]
        for row in reader:
            if row[col] in ("", "NaN", "nan"):
                continue
            steps.append(int(float(row[step_col])))
            lengths.append(float(row[col]))
    return np.array(steps), np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--length_csv", type=str, help="W&B CSV export with per-step mean response length")
    src.add_argument("--mean_length", type=float, help="constant mean response length (tokens)")
    parser.add_argument("--step_col", type=str, default="Step")
    parser.add_argument("--length_col", type=str, default="response_length/mean")
    parser.add_argument("--num_steps", type=int, default=400, help="only used with --mean_length")
    parser.add_argument("--tau_max", type=float, default=1.2)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--d0", type=float, default=25)
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--d_max", type=float, default=40000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--per_step_csv", type=str, default=None, help="optional output of per-step averages")
    args = parser.parse_args()

    if args.length_csv:
        steps, lengths = load_lengths(args.length_csv, args.step_col, args.length_col)
    else:
        steps = np.arange(args.num_steps)
        lengths = np.full(args.num_steps, args.mean_length)

    per_step_mean = np.array([
        ead_temperatures(L, s, args.tau_max, args.tau_min, args.d0, args.alpha, args.d_max, args.warmup).mean()
        for s, L in zip(steps, lengths)
    ])
    token_weighted = float((per_step_mean * lengths).sum() / lengths.sum())
    step_weighted = float(per_step_mean.mean())

    print(f"steps: {steps.min()}..{steps.max()} ({len(steps)} points), mean length: {lengths.mean():.1f} tokens")
    print(f"per-step mean temperature: first {per_step_mean[0]:.3f}, last {per_step_mean[-1]:.3f}")
    print(f"token-weighted mean temperature: {token_weighted:.3f}")
    print(f"step-weighted  mean temperature: {step_weighted:.3f}")
    print(f"MEAN_MATCHED_TEMP={token_weighted:.2f}")

    if args.per_step_csv:
        with open(args.per_step_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_length", "mean_temperature"])
            for s, L, m in zip(steps, lengths, per_step_mean):
                writer.writerow([int(s), f"{L:.1f}", f"{m:.4f}"])


if __name__ == "__main__":
    main()

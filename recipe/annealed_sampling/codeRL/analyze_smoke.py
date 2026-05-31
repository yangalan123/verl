"""
Analyze a generate-only smoke dump and RECOMMEND an EAD search space.

The core problem we solve: the paper's EAD defaults (decay_freq=200, etc.) were
tuned for *math* responses, which are long. For the `negexp` schedule at
inference (global_step=0) the temperature is

    tau(t) = 1 + tau_max - exp( t / (20 * decay_freq) ),  clamped to >= tau_min

so it only reaches the floor at

    t* = 20 * decay_freq * ln(1 + tau_max - tau_min)   [response tokens].

If code completions are ~150 tokens but t* ~ 3000, the temperature never anneals
and EAD just behaves like high-temperature sampling -> it underperforms. So the
right decay_freq must come from the ACTUAL response-length distribution.

This script reads the smoke dump (gen__*.jsonl with `token_lens` and, if
available, `neglogp_by_pos` -- the negative avg log-likelihood per position,
which is the long-sequence entropy proxy from "How Alignment Shrinks the
Generative Horizon", yangalan123/LLMBranchingFactor) and recommends:

  * warmup_period  -- length of the low-entropy structural prefix to leave at
                      base temperature (so we don't anneal template tokens).
  * decay_freq     -- candidates so that t* lands at a few fractions of the
                      observed response length (the schedule actually anneals
                      within a real completion).
  * start_temp / end_temp grids (sane code-friendly defaults).

Usage:
    python recipe/annealed_sampling/codeRL/analyze_smoke.py \\
        --gen_jsonl ./logs/.../gen__test__fixed__T1.0__smoke.jsonl \\
        --out ./logs/.../search_space__Qwen2.5-Coder-1.5B__HumanEval+.json
"""

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional

# Reference temps used only to turn a target anneal-length into a decay_freq.
# (The dependence on temps is weak -- it enters via ln(1 + tmax - tmin).)
_REF_TMAX = 1.2
_REF_TMIN = 0.3
# Default temperature grids for the sweep (start slightly >1, end <=1).
DEFAULT_START_TEMPS = [1.1, 1.2, 1.3]
DEFAULT_END_TEMPS = [0.3, 0.5, 0.7]
# Fractions of the reference response length at which tau should hit the floor.
DEFAULT_ANNEAL_FRACS = [0.25, 0.5, 0.75, 1.0]


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def decay_freq_for_anneal_len(target_len: float,
                              tmax: float = _REF_TMAX,
                              tmin: float = _REF_TMIN) -> int:
    """Invert t* = 20 d ln(1 + tmax - tmin) for d, given a target anneal length."""
    denom = 20.0 * math.log(max(1.0 + tmax - tmin, 1.0 + 1e-6))
    return max(5, int(round(target_len / denom)))


def load_dump(gen_jsonl: str) -> Dict[str, Any]:
    """Collect token lengths and (optional) per-position neg-logprob from a dump."""
    lens: List[int] = []
    # accumulate neg-logprob by absolute position: sum and count per position
    pos_sum: List[float] = []
    pos_cnt: List[int] = []
    seq_neglogp: List[float] = []
    with open(gen_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for tl in (r.get("token_lens") or []):
                if tl:
                    lens.append(int(tl))
            for mnl in (r.get("mean_neglogp") or []):
                if mnl is not None:
                    seq_neglogp.append(float(mnl))
            for arr in (r.get("neglogp_by_pos") or []):
                for pos, val in enumerate(arr):
                    if pos >= len(pos_sum):
                        pos_sum.extend([0.0] * (pos + 1 - len(pos_sum)))
                        pos_cnt.extend([0] * (pos + 1 - len(pos_cnt)))
                    pos_sum[pos] += float(val)
                    pos_cnt[pos] += 1
    entropy_curve = [pos_sum[i] / pos_cnt[i] if pos_cnt[i] else None
                     for i in range(len(pos_sum))]
    return {
        "lens": lens,
        "entropy_curve": entropy_curve,      # avg neg-logprob per absolute position
        "seq_mean_neglogp": seq_neglogp,     # per-completion average entropy proxy
    }


def recommend_warmup(entropy_curve: List[Optional[float]],
                     max_warmup: int = 64) -> int:
    """Length of the low-entropy structural prefix.

    Structural/template tokens at the very start are near-deterministic (low
    neg-logprob). We set warmup_period to the first position whose neg-logprob
    rises to >= 50% of the plateau (median of positions 0..max_warmup), capped.
    Falls back to a small constant if no logprobs were captured.
    """
    vals = [v for v in entropy_curve[:max_warmup] if v is not None]
    if len(vals) < 4:
        return 8  # no usable curve -> conservative default
    plateau = sorted(vals)[len(vals) // 2]  # median over the early window
    thresh = 0.5 * plateau
    for pos, v in enumerate(entropy_curve[:max_warmup]):
        if v is not None and v >= thresh:
            return max(0, min(pos, max_warmup))
    return min(8, max_warmup)


def recommend_search_space(gen_jsonl: str,
                           start_temps: List[float] = None,
                           end_temps: List[float] = None,
                           anneal_fracs: List[float] = None) -> Dict[str, Any]:
    start_temps = start_temps or DEFAULT_START_TEMPS
    end_temps = end_temps or DEFAULT_END_TEMPS
    anneal_fracs = anneal_fracs or DEFAULT_ANNEAL_FRACS

    data = load_dump(gen_jsonl)
    lens = sorted(data["lens"])
    stats = {
        "count": len(lens),
        "p10": _percentile(lens, 0.10),
        "p25": _percentile(lens, 0.25),
        "p50": _percentile(lens, 0.50),
        "p75": _percentile(lens, 0.75),
        "p90": _percentile(lens, 0.90),
        "max": lens[-1] if lens else 0,
        "mean": (sum(lens) / len(lens)) if lens else 0.0,
    }
    # Reference length for sizing decay: the median completion length.
    ref_len = max(stats["p50"], 1.0)
    decay_freqs = sorted({decay_freq_for_anneal_len(f * ref_len)
                          for f in anneal_fracs})
    warmup = recommend_warmup(data["entropy_curve"])

    seq_nl = data["seq_mean_neglogp"]
    avg_entropy = (sum(seq_nl) / len(seq_nl)) if seq_nl else None

    return {
        "gen_jsonl": gen_jsonl,
        "length_stats": stats,
        "avg_entropy_proxy_nats": avg_entropy,
        "warmup_period": warmup,
        "decay_freqs": decay_freqs,
        "start_temps": start_temps,
        "end_temps": end_temps,
        "anneal_fracs": anneal_fracs,
        "ref_len_for_decay": ref_len,
        "note": ("decay_freqs chosen so tau hits the floor at "
                 f"{anneal_fracs} x median_len({ref_len:.0f}); "
                 "t* = 20*d*ln(1+tmax-tmin)."),
        "entropy_curve_head": data["entropy_curve"][:64],
    }


def _print_report(space: Dict[str, Any]) -> None:
    s = space["length_stats"]
    print("=== smoke analysis ===")
    print(f"  completions analyzed : {s['count']}")
    print(f"  response length      : p10={s['p10']:.0f} p25={s['p25']:.0f} "
          f"p50={s['p50']:.0f} p75={s['p75']:.0f} p90={s['p90']:.0f} "
          f"max={s['max']:.0f} mean={s['mean']:.0f}")
    if space["avg_entropy_proxy_nats"] is not None:
        print(f"  avg entropy proxy    : {space['avg_entropy_proxy_nats']:.3f} nats/token "
              "(neg avg log-likelihood)")
    else:
        print("  avg entropy proxy    : (no logprobs captured; re-run smoke with --logprobs 1)")
    print("=== recommended EAD search space ===")
    print(f"  warmup_period : {space['warmup_period']}")
    print(f"  decay_freqs   : {space['decay_freqs']}")
    print(f"  start_temps   : {space['start_temps']}")
    print(f"  end_temps     : {space['end_temps']}")
    print(f"  ({space['note']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_jsonl", default=None,
                    help="A gen__*.jsonl smoke dump. If omitted, pass --glob.")
    ap.add_argument("--glob", default=None,
                    help="Glob for gen__*smoke*.jsonl (first match used).")
    ap.add_argument("--out", default=None, help="Write search-space JSON here.")
    ap.add_argument("--start_temps", default=None,
                    help="Comma list overriding default start temps.")
    ap.add_argument("--end_temps", default=None,
                    help="Comma list overriding default end temps.")
    args = ap.parse_args()

    gen = args.gen_jsonl
    if gen is None and args.glob:
        matches = sorted(glob.glob(args.glob, recursive=True))
        if not matches:
            print(f"No files match {args.glob}")
            return
        gen = matches[0]
    if gen is None:
        ap.error("provide --gen_jsonl or --glob")

    st = [float(x) for x in args.start_temps.split(",")] if args.start_temps else None
    et = [float(x) for x in args.end_temps.split(",")] if args.end_temps else None
    space = recommend_search_space(gen, start_temps=st, end_temps=et)
    _print_report(space)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(space, f, indent=2)
        print(f"[ok] wrote search space -> {args.out}")


if __name__ == "__main__":
    main()

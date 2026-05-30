"""
Recompute pass@k / worst@k at arbitrary k from already-saved Day-0 eval outputs.

These metrics are a pure function of "c correct out of n samples" per prompt, and
``inference_only_eval.py`` saves that vector (the ``successes`` field) in every
``per_prompt__*.jsonl``. So you can sweep k WITHOUT re-running generation -- the
only requirement is that you generated with n_samples >= max(k) (run the eval
with N_SAMPLES=16 if you want pass@16).

Estimators (unbiased, Chen et al. 2021 "Codex" style), averaged over prompts:
    pass@k  = mean_prompt [ 1 - C(n-c, k) / C(n, k) ]   # >=1 correct in a k-subset
    worst@k = mean_prompt [     C(c,   k) / C(n, k) ]    # ALL k correct in a k-subset
Prompts with n < k are excluded from that k's average (and counted in n_excl).

Usage:
    python recipe/annealed_sampling/codeRL/recompute_passk.py
    python recipe/annealed_sampling/codeRL/recompute_passk.py \\
        --root ./logs/inference_only_eval --k 1,2,4,8,16 --csv passk.csv --md passk.md

Pure stdlib (math.comb); no third-party deps.
"""

import argparse
import csv as _csv
import glob
import json
import os
from math import comb
from typing import Any, Dict, List, Optional

SUCCESS_THRESHOLD = 0.999


def pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    if k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:  # not enough failures to fill a k-subset -> always >=1 correct
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def worst_at_k(n: int, c: int, k: int) -> Optional[float]:
    if k > n:
        return None
    if c < k:  # cannot fill a k-subset with all-correct
        return 0.0
    return comb(c, k) / comb(n, k)


def _bench_name(summary: Dict[str, Any], path: str) -> str:
    src = (summary.get("eval_parquet") or "").lower()
    if "humaneval" in src:
        return "HumanEval+"
    if "livecodebench" in src or "release_v" in src:
        return "LiveCodeBench"
    if "mbpp" in src:
        return "MBPP+"
    return os.path.basename(os.path.dirname(path)) or "unknown"


def _config_label(summary: Dict[str, Any]) -> str:
    cfg = summary.get("config", {})
    mode = summary.get("mode", "fixed")
    if mode == "fixed":
        return f"T={cfg.get('temperature')}"
    return (f"EAD d={cfg.get('decay_freq')} "
            f"({cfg.get('start_temp')}->{cfg.get('end_temp')})")


def _load_sibling_summary(per_prompt_path: str) -> Dict[str, Any]:
    """per_prompt__<tag>.jsonl -> summary__<tag>.json in the same dir, if present."""
    base = os.path.basename(per_prompt_path)
    if base.startswith("per_prompt__"):
        tag = base[len("per_prompt__"):].rsplit(".jsonl", 1)[0]
        cand = os.path.join(os.path.dirname(per_prompt_path), f"summary__{tag}.json")
        if os.path.exists(cand):
            try:
                with open(cand) as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def process_file(path: str, ks: List[int]) -> Optional[Dict[str, Any]]:
    vectors: List[List[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            succ = row.get("successes")
            if isinstance(succ, list) and succ:
                vectors.append([float(s) for s in succ])
    if not vectors:
        print(f"[warn] no 'successes' vectors in {path}")
        return None

    summary = _load_sibling_summary(path)
    out: Dict[str, Any] = {
        "model": os.path.basename(summary.get("model", "?")),
        "benchmark": _bench_name(summary, path),
        "mode": summary.get("mode", "?"),
        "config": _config_label(summary) if summary else "?",
        "num_prompts": len(vectors),
        "n_max": max(len(v) for v in vectors),
        "path": path,
    }
    for k in ks:
        pvals, wvals, excl = [], [], 0
        for v in vectors:
            n = len(v)
            c = sum(1 for s in v if s >= SUCCESS_THRESHOLD)
            pk = pass_at_k(n, c, k)
            wk = worst_at_k(n, c, k)
            if pk is None or wk is None:
                excl += 1
                continue
            pvals.append(pk)
            wvals.append(wk)
        out[f"pass@{k}"] = (sum(pvals) / len(pvals)) if pvals else None
        out[f"worst@{k}"] = (sum(wvals) / len(wvals)) if wvals else None
        out[f"n_excl@{k}"] = excl
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def _columns(ks: List[int]) -> List[str]:
    cols = ["model", "benchmark", "mode", "config", "num_prompts", "n_max"]
    for k in ks:
        cols += [f"pass@{k}", f"worst@{k}"]
    return cols


def print_table(rows: List[Dict[str, Any]], cols: List[str]) -> None:
    if not rows:
        print("No per_prompt__*.jsonl files with 'successes' found.")
        return
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(r.get(c))))
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def write_csv(rows: List[Dict[str, Any]], cols: List[str], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols + ["path"])
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols + ["path"]})
    print(f"[ok] wrote CSV -> {path}")


def write_md(rows: List[Dict[str, Any]], cols: List[str], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n")
    print(f"[ok] wrote Markdown -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./logs/inference_only_eval",
                    help="Directory tree containing per_prompt__*.jsonl files.")
    ap.add_argument("--k", default="1,2,4,8,16",
                    help="Comma-separated k values (e.g. '1,2,4,8,16').")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--model", default=None, help="Substring filter on model.")
    ap.add_argument("--benchmark", default=None, help="Substring filter on benchmark.")
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    files = sorted(glob.glob(os.path.join(args.root, "**", "per_prompt__*.jsonl"),
                             recursive=True))
    rows = []
    for f in files:
        r = process_file(f, ks)
        if r:
            rows.append(r)

    if args.model:
        rows = [r for r in rows if args.model.lower() in r["model"].lower()]
    if args.benchmark:
        rows = [r for r in rows if args.benchmark.lower() in r["benchmark"].lower()]
    rows.sort(key=lambda r: (r["model"], r["benchmark"], r["mode"], r["config"]))

    cols = _columns(ks)
    print_table(rows, cols)

    # Warn if any requested k exceeds the samples actually generated.
    for r in rows:
        too_big = [k for k in ks if k > r["n_max"]]
        if too_big:
            print(f"[note] {r['model']}/{r['benchmark']}/{r['config']}: "
                  f"k={too_big} exceed n_samples={r['n_max']} -> reported blank; "
                  f"re-run generation with N_SAMPLES>={max(ks)} to fill them.")

    if args.csv:
        write_csv(rows, cols, args.csv)
    if args.md:
        write_md(rows, cols, args.md)


if __name__ == "__main__":
    main()

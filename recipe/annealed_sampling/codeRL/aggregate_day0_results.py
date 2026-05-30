"""
Aggregate Day-0 inference-only eval results into a single comparison table.

Walks the summary JSON files written by ``inference_only_eval.py`` /
``eval_grid_worker.sh`` (one per (model, benchmark, sampling-config) cell) and
collapses them into:

  * a console table (always),
  * a CSV  (--csv path),
  * a Markdown table (--md path), ready to paste into the appendix,
  * a per-(model, benchmark) "EAD best vs. fixed best" delta summary
    (--summary), which is the headline number for the rebuttal.

Default input root matches the layout produced by the eval grid:
    logs/inference_only_eval/<model>/<benchmark>/summary__*.json

Usage:
    python recipe/annealed_sampling/codeRL/aggregate_day0_results.py
    python recipe/annealed_sampling/codeRL/aggregate_day0_results.py \\
        --root ./logs/inference_only_eval --csv day0.csv --md day0.md --summary

No third-party dependencies (no pandas); pure stdlib so it runs anywhere.
"""

import argparse
import csv as _csv
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _pick_metric(summary: Dict[str, Any], prefix: str, k: int) -> Optional[float]:
    """Return the {prefix}@K metric.

    Prefer the exact key for the run's K (``n_samples_per_prompt``); otherwise
    fall back to any ``{prefix}@N`` with N != 1 (so we never mistake the always
    -present ``pass@1`` for ``pass@K``).
    """
    exact = summary.get(f"{prefix}@{k}") if k else None
    if isinstance(exact, (int, float)):
        return float(exact)
    for key, v in summary.items():
        if key.startswith(f"{prefix}@") and key != f"{prefix}@1" \
                and isinstance(v, (int, float)):
            return float(v)
    return None


def _bench_name(summary: Dict[str, Any], path: str) -> str:
    src = (summary.get("eval_parquet") or "").lower()
    if "humanevalplus" in src or "humaneval" in src:
        return "HumanEval+"
    if "livecodebench" in src or "release_v" in src:
        return "LiveCodeBench"
    if "mbpp" in src:
        return "MBPP+"
    # fall back to the parent directory name (eval_grid_worker puts the
    # benchmark in its own subdir)
    return os.path.basename(os.path.dirname(path)) or "unknown"


def _config_label(summary: Dict[str, Any]) -> str:
    cfg = summary.get("config", {})
    mode = summary.get("mode", "fixed")
    if mode == "fixed":
        return f"T={cfg.get('temperature')}"
    # EAD: distinguish the decay_freq ablation rows
    return (f"EAD d={cfg.get('decay_freq')} "
            f"({cfg.get('start_temp')}->{cfg.get('end_temp')})")


def _k_of(summary: Dict[str, Any]) -> int:
    n = summary.get("n_samples_per_prompt")
    return int(n) if n else 0


def collect(root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(root, "**", "summary__*.json")
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(f, "r") as fh:
                s = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skipping {f}: {e}")
            continue
        k = _k_of(s)
        cfg = s.get("config", {})
        rows.append({
            "model": os.path.basename(s.get("model", "?")),
            "benchmark": _bench_name(s, f),
            "mode": s.get("mode", "fixed"),
            "config": _config_label(s),
            "K": k,
            "pass@1": s.get("pass@1"),
            "pass@K": _pick_metric(s, "pass", k),
            "worst@K": _pick_metric(s, "worst", k),
            "num_prompts": s.get("num_prompts"),
            "max_tokens": cfg.get("max_tokens"),
            "thinking": cfg.get("enable_thinking"),
            "path": f,
        })
    return rows


# columns shown in the console / CSV / markdown table
_COLS = ["model", "benchmark", "mode", "config", "K",
         "pass@1", "pass@K", "worst@K", "num_prompts",
         "max_tokens", "thinking"]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No summary files found.")
        return
    widths = {c: len(c) for c in _COLS}
    for r in rows:
        for c in _COLS:
            widths[c] = max(widths[c], len(_fmt(r.get(c))))
    header = "  ".join(c.ljust(widths[c]) for c in _COLS)
    print(header)
    print("  ".join("-" * widths[c] for c in _COLS))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in _COLS))


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=_COLS + ["path"])
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLS + ["path"]})
    print(f"[ok] wrote CSV -> {path}")


def write_md(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("| " + " | ".join(_COLS) + " |\n")
        fh.write("| " + " | ".join("---" for _ in _COLS) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(_fmt(r.get(c)) for c in _COLS) + " |\n")
    print(f"[ok] wrote Markdown -> {path}")


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """For each (model, benchmark): best fixed-T vs. best EAD config, on each metric."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["benchmark"]), []).append(r)

    print("\n=== EAD-best vs fixed-best (Day-0 headline) ===")
    metrics = ["pass@1", "pass@K", "worst@K"]
    for (model, bench), grp in sorted(groups.items()):
        fixed = [r for r in grp if r["mode"] == "fixed"]
        ead = [r for r in grp if r["mode"] == "ead"]
        print(f"\n[{model} / {bench}]  (K={grp[0]['K']}, n_prompts={grp[0].get('num_prompts')})")
        for m in metrics:
            fv = max((r[m] for r in fixed if r.get(m) is not None), default=None)
            ev = max((r[m] for r in ead if r.get(m) is not None), default=None)
            if fv is None and ev is None:
                continue
            delta = (ev - fv) if (fv is not None and ev is not None) else None
            d = f"  (EAD {'+' if (delta or 0) >= 0 else ''}{delta:.4f})" if delta is not None else ""
            print(f"  {m:8s}  fixed_best={_fmt(fv):8s}  ead_best={_fmt(ev):8s}{d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./logs/inference_only_eval",
                    help="Directory tree containing summary__*.json files.")
    ap.add_argument("--csv", default=None, help="Optional CSV output path.")
    ap.add_argument("--md", default=None, help="Optional Markdown output path.")
    ap.add_argument("--summary", action="store_true",
                    help="Also print the EAD-best vs fixed-best delta summary.")
    ap.add_argument("--model", default=None,
                    help="Substring filter on model name (e.g. 'Qwen3-4B').")
    ap.add_argument("--benchmark", default=None,
                    help="Substring filter on benchmark (e.g. 'HumanEval').")
    args = ap.parse_args()

    rows = collect(args.root)
    if args.model:
        rows = [r for r in rows if args.model.lower() in r["model"].lower()]
    if args.benchmark:
        rows = [r for r in rows if args.benchmark.lower() in r["benchmark"].lower()]

    # stable, human-friendly ordering
    rows.sort(key=lambda r: (r["model"], r["benchmark"], r["mode"], r["config"]))

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.md:
        write_md(rows, args.md)
    if args.summary:
        print_summary(rows)


if __name__ == "__main__":
    main()

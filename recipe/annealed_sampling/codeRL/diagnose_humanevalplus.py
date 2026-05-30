"""
Diagnostic for the HumanEval+ reward path. No GPU / vLLM needed.

Checks
------
1. Eval parquet sanity. For every row, parse `ground_truth` (the JSON dict
   with entry_point/tests) and confirm it has the expected fields.

2. Canonical-solution sanity. For every row, build a candidate from the
   canonical solution embedded in the prompt's HF dataset and score it with
   `humanevalplus.compute_score`. This MUST give 1.0 for nearly every
   problem; if it does not, the bug is in our reward module / the dataset's
   `tests` field, not in the model.

3. Replay an existing per_prompt jsonl, if provided, to show
     a) the distribution of failure statuses
     b) a few example raw completions per failure category.

Usage
-----
    # (1) + (2): verify the reward path on the canonical solutions.
    python recipe/annealed_sampling/codeRL/diagnose_humanevalplus.py \\
        --eval_parquet ./data/humanevalplus/test.parquet \\
        --canonical_check_only

    # (1) + (2) + (3): also categorise the failures from a previous run.
    python recipe/annealed_sampling/codeRL/diagnose_humanevalplus.py \\
        --eval_parquet ./data/humanevalplus/test.parquet \\
        --per_prompt_jsonl ./logs/inference_only_eval/.../per_prompt__test__fixed__T1.0.jsonl
"""

import argparse
import json
import os
from collections import Counter

import datasets

# The reward module under test.
from verl.utils.reward_score import humanevalplus as _hep


def _load_canonical_solutions():
    """Fetch HumanEval+ once just to get the canonical_solution field per task_id."""
    ds = datasets.load_dataset("evalplus/humanevalplus", split="test", trust_remote_code=True)
    out = {}
    for ex in ds:
        out[ex["task_id"]] = {
            "prompt": ex["prompt"],
            "canonical_solution": ex.get("canonical_solution", ""),
            "entry_point": ex["entry_point"],
        }
    return out


def check_parquet(parquet_path: str):
    print(f"\n[1/3] Parquet sanity: {parquet_path}", flush=True)
    ds = datasets.Dataset.from_parquet(parquet_path)
    n = len(ds)
    bad = 0
    sample_keys = None
    for i, row in enumerate(ds):
        gt_str = row["reward_model"]["ground_truth"]
        try:
            gt = json.loads(gt_str)
        except Exception as e:
            print(f"  row {i}: ground_truth not valid JSON: {e}")
            bad += 1
            continue
        if sample_keys is None:
            sample_keys = list(gt.keys())
        missing = [k for k in ("entry_point", "tests") if not gt.get(k)]
        if missing:
            print(f"  row {i} ({row['extra_info'].get('task_id')}): missing/empty {missing}")
            bad += 1
    print(f"  rows={n}  bad={bad}  sample keys={sample_keys}")
    if bad == n and n > 0:
        print("  >>> Every row has missing/empty `tests` or `entry_point`.")
        print("  >>> This is almost certainly the reason your run scored 0/N.")
        print("  >>> Fix `prepare_eval_humanevalplus.py` to populate `tests` from")
        print("  >>> the correct HF field (e.g. `test` vs `test_plus`).")
    return bad < n


def check_canonical(parquet_path: str):
    print("\n[2/3] Canonical-solution sanity (no vLLM)", flush=True)
    ds = datasets.Dataset.from_parquet(parquet_path)
    canon = _load_canonical_solutions()
    n_pass = 0
    n_total = 0
    failures = []
    for row in ds:
        task_id = row["extra_info"].get("task_id")
        gt = json.loads(row["reward_model"]["ground_truth"])
        entry_point = gt.get("entry_point")
        meta = canon.get(task_id)
        if meta is None:
            continue
        # Build a "completion" that mimics what a model would output:
        # the original prompt (signature + docstring) concatenated with the
        # canonical solution body, wrapped in a python code block.
        candidate_src = meta["prompt"] + meta["canonical_solution"]
        completion = f"```python\n{candidate_src}\n```"
        score, hep_meta = _hep.compute_score(completion, json.dumps(gt))
        n_total += 1
        if score >= 0.999:
            n_pass += 1
        else:
            status = (hep_meta[0] if hep_meta else {}).get("status") \
                or (hep_meta[0] if hep_meta else {}).get("error")
            failures.append((task_id, status))
    print(f"  canonical Pass@1 = {n_pass}/{n_total} = {n_pass/max(n_total,1):.3f}")
    if n_pass < n_total:
        print("  >>> The canonical solution should score ~1.0 on every problem.")
        print("  >>> Any non-passing case below is a bug in the reward module")
        print("  >>> or in the `tests` field we packaged into ground_truth:")
        for task_id, status in failures[:10]:
            print(f"    - {task_id}: {status}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    return n_pass == n_total


def replay_per_prompt(jsonl_path: str, n_examples: int = 5):
    if not jsonl_path or not os.path.isfile(jsonl_path):
        print(f"\n[3/3] Skipping per-prompt replay (path missing or not provided).")
        return
    print(f"\n[3/3] Replaying {jsonl_path}", flush=True)
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    if not rows:
        print("  per_prompt file is empty.")
        return

    has_statuses = any(r.get("statuses") for r in rows)
    has_completion = any(r.get("first_completion") for r in rows)
    if not has_statuses:
        print("  This per_prompt file was produced by an older version of")
        print("  inference_only_eval.py that did not save `statuses` or")
        print("  `first_completion`. Re-run one config with the patched script.")
        # still print high-level pass rates
        n = len(rows)
        p1 = sum(r["pass@1"] for r in rows) / n
        pk = sum(r["pass@k"] for r in rows) / n
        wk = sum(r["worst@k"] for r in rows) / n
        print(f"  rows={n}  pass@1={p1:.3f}  pass@K={pk:.3f}  worst@K={wk:.3f}")
        return

    status_ctr = Counter()
    for r in rows:
        for s in (r.get("statuses") or []):
            status_ctr[str(s) if s is not None else "<None>"] += 1
    print("  failure-status counts (over all completions):")
    for status, c in status_ctr.most_common():
        print(f"    {c:6d}  {status[:120]}")

    # Show a few examples per category, with the raw completion text.
    print("\n  example completions per category:")
    examples_per_cat = {}
    for r in rows:
        for s in (r.get("statuses") or []):
            key = str(s)
            if key not in examples_per_cat:
                examples_per_cat[key] = (r.get("task_id"), r.get("first_completion", ""))
            if len(examples_per_cat) >= n_examples * 4:
                break
    for status, (task_id, comp) in list(examples_per_cat.items())[:n_examples * 2]:
        head = comp[:600].replace("\n", "\n      ")
        print(f"  --- status={status}  task_id={task_id}")
        print(f"      first 600 chars of completion:\n      {head}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_parquet", required=True)
    ap.add_argument("--per_prompt_jsonl", default=None)
    ap.add_argument("--canonical_check_only", action="store_true")
    ap.add_argument("--n_examples", type=int, default=5)
    args = ap.parse_args()

    ok_p = check_parquet(args.eval_parquet)
    ok_c = check_canonical(args.eval_parquet) if ok_p else False
    if not args.canonical_check_only:
        replay_per_prompt(args.per_prompt_jsonl, args.n_examples)

    print("\nSummary:")
    print(f"  parquet ok          : {ok_p}")
    print(f"  canonical Pass@1=1.0: {ok_c}")
    if ok_p and ok_c:
        print("  >>> Reward path is sound. If your model still scores 0%,")
        print("  >>> the model's completions are mis-formatted; inspect the")
        print("  >>> `first_completion` field of the per_prompt jsonl after")
        print("  >>> re-running ONE config with the patched eval script.")


if __name__ == "__main__":
    main()

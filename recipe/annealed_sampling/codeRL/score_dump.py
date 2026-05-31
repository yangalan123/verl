"""
CPU-only scorer for generate-only dumps produced by inference_only_eval.py
(--generate_only). Reads a gen__<tag>.jsonl (which contains completions +
ground_truth + data_source per prompt) and writes the SAME summary__<tag>.json /
per_prompt__<tag>.jsonl format as the live eval, so aggregate_day0_results.py and
recompute_passk.py work unchanged.

This is the "CPU half" of the GPU/CPU split: the GPU process generates and exits
quickly, then this runs on CPU (overlapping the next GPU job) to do the slow,
sandbox-bound scoring.

We parallelize across (prompt, completion) pairs with a THREAD pool -- not a
process pool -- because the code reward modules themselves spawn child processes
for sandboxed execution, and daemonic Pool workers cannot have children. Threads
release the GIL while waiting on those child processes, so we still get real
concurrency.

Usage:
    python recipe/annealed_sampling/codeRL/score_dump.py \\
        --gen_jsonl ./logs/.../gen__test__ead__neg_1.2_0.3_d40_w8.jsonl \\
        --num_workers 8
"""

import argparse
import json
import os

# CPU-only: hide GPUs so any transitive torch import never grabs a device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from multiprocessing.pool import ThreadPool

from inference_only_eval import score_completion

SUCCESS_THRESHOLD = 0.999


def _derive_paths(gen_jsonl):
    d = os.path.dirname(gen_jsonl)
    base = os.path.basename(gen_jsonl)
    assert base.startswith("gen__") and base.endswith(".jsonl"), \
        f"expected gen__<tag>.jsonl, got {base}"
    tag = base[len("gen__"):-len(".jsonl")]
    meta = os.path.join(d, f"genmeta__{tag}.json")
    summary = os.path.join(d, f"summary__{tag}.json")
    per_prompt = os.path.join(d, f"per_prompt__{tag}.jsonl")
    return tag, meta, summary, per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_jsonl", required=True)
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Thread-pool size for concurrent scoring.")
    ap.add_argument("--keep_completions", action="store_true",
                    help="Also copy the completion texts into per_prompt JSONL.")
    args = ap.parse_args()

    tag, meta_path, summary_path, per_prompt_path = _derive_paths(args.gen_jsonl)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    rows = []
    with open(args.gen_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"[score_dump] no rows in {args.gen_jsonl}; nothing to do.")
        return

    # Flatten to (prompt_idx, completion_idx, data_source, text, gt) work items.
    work = []
    for pi, r in enumerate(rows):
        ds = r["data_source"]
        gt = r["ground_truth"]
        for ci, text in enumerate(r["completions"]):
            work.append((pi, ci, ds, text, gt))

    def _do(item):
        pi, ci, ds, text, gt = item
        score, status = score_completion(ds, text, gt)
        return pi, ci, score, status

    n_workers = max(1, args.num_workers)
    print(f"[score_dump] scoring {len(work)} completions across "
          f"{len(rows)} prompts with {n_workers} threads ...", flush=True)
    with ThreadPool(n_workers) as pool:
        results = pool.map(_do, work)

    # Reassemble per-prompt success/status vectors in original order.
    n_samples = max(len(r["completions"]) for r in rows)
    succ = [[0.0] * len(r["completions"]) for r in rows]
    stat = [[None] * len(r["completions"]) for r in rows]
    for pi, ci, score, status in results:
        succ[pi][ci] = score
        stat[pi][ci] = status

    pass1 = passk = worstk = 0.0
    per_prompt = []
    for pi, r in enumerate(rows):
        successes = succ[pi]
        n_ok = sum(1 for s in successes if s >= SUCCESS_THRESHOLD)
        any_ok = 1.0 if n_ok > 0 else 0.0
        all_ok = 1.0 if n_ok == len(successes) else 0.0
        first_ok = 1.0 if successes and successes[0] >= SUCCESS_THRESHOLD else 0.0
        pass1 += first_ok
        passk += any_ok
        worstk += all_ok
        rec = {
            "task_id": r.get("task_id"),
            "successes": successes,
            "statuses": stat[pi],
            "token_lens": r.get("token_lens"),
            "pass@1": first_ok,
            "pass@k": any_ok,
            "worst@k": all_ok,
        }
        if args.keep_completions:
            rec["completions"] = r["completions"]
        per_prompt.append(rec)

    n = len(rows)
    summary = {
        "model": meta.get("model", "?"),
        "eval_parquet": meta.get("eval_parquet", "?"),
        "mode": meta.get("mode", "?"),
        "n_samples_per_prompt": n_samples,
        "num_prompts": n,
        "pass@1": pass1 / max(n, 1),
        f"pass@{n_samples}": passk / max(n, 1),
        f"worst@{n_samples}": worstk / max(n, 1),
        "config": meta.get("config", {}),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(per_prompt_path, "w") as f:
        for rec in per_prompt:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[score_dump] wrote {summary_path}")
    print(f"[score_dump] wrote {per_prompt_path}")


if __name__ == "__main__":
    main()

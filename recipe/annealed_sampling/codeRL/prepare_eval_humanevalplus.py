"""
Build a HumanEval+ eval parquet that plugs into verl with the local
`humanevalplus` reward (function-style execution, no Docker, no firejail).

Source: `evalplus/humanevalplus` (HuggingFace).
Each row contains:
    task_id           -- e.g. "HumanEval/0"
    prompt            -- function signature + docstring
    canonical_solution
    test              -- python source that defines `def check(candidate): ...`
                         with assertion blocks
    entry_point       -- the function name to call

We pack `entry_point` + `test` + (optional) `prompt` header into a single JSON
ground_truth blob so verl's reward dispatcher can execute the candidate
against the reference test harness.

Usage:
    python recipe/annealed_sampling/codeRL/prepare_eval_humanevalplus.py \\
        --local_dir ./data/humanevalplus
"""

import argparse
import json
import os

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="evalplus/humanevalplus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_dir", default="./data/humanevalplus")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Loading {args.hf_repo} ...", flush=True)
    ds = datasets.load_dataset(args.hf_repo, split=args.split, trust_remote_code=True)
    print(f"Raw rows: {len(ds)}", flush=True)

    instruction = (
        "Complete the function below. Put your final code inside a single "
        "```python ... ``` block. The block should contain the full function "
        "definition (signature + body)."
    )
    system_prompt = "You are an expert Python programmer."

    rows = []
    for i, ex in enumerate(ds):
        prompt = ex["prompt"]
        entry_point = ex["entry_point"]
        tests = ex.get("test", "")
        gt = {
            "entry_point": entry_point,
            "tests": tests,
            "prompt_header": "",
        }
        rows.append({
            "data_source": "humanevalplus",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + "\n\n" + instruction},
            ],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(gt),
            },
            "extra_info": {
                "split": "test",
                "index": i,
                "task_id": ex.get("task_id", str(i)),
                "source": "humanevalplus",
            },
        })

    print(f"Kept {len(rows)} eval problems.", flush=True)
    out_path = os.path.join(args.local_dir, "test.parquet")
    datasets.Dataset.from_list(rows).to_parquet(out_path)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

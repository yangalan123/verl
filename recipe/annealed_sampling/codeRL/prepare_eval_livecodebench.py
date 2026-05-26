"""
Build a LiveCodeBench eval parquet that plugs into verl with the existing
prime_code reward (stdin/stdout style execution, no Docker, no firejail).

Notes
-----
The official `livecodebench/code_generation_lite` HF dataset stores test cases
in `public_test_cases` / `private_test_cases` as JSON-encoded lists of
`{"input": "...", "output": "..."}` dicts. We normalize them to the standard
{"inputs": [...], "outputs": [...]} format expected by prime_code.

Usage:
    python recipe/annealed_sampling/codeRL/prepare_eval_livecodebench.py \\
        --version release_v2 \\
        --local_dir ./data/livecodebench
"""

import argparse
import json
import os
from typing import List, Tuple

import datasets


def _decode_tests(field) -> List[Tuple[str, str]]:
    """Return list of (input, output) pairs from either a JSON string or list."""
    if field is None:
        return []
    if isinstance(field, str):
        try:
            field = json.loads(field)
        except Exception:
            return []
    out = []
    if isinstance(field, list):
        for item in field:
            if not isinstance(item, dict):
                continue
            inp = item.get("input", "")
            outp = item.get("output", "")
            if isinstance(inp, str) and isinstance(outp, str):
                out.append((inp, outp))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="livecodebench/code_generation_lite")
    parser.add_argument("--version", default="release_v2",
                        help="HuggingFace config version, e.g. release_v1 / release_v2 / release_v5")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local_dir", default="./data/livecodebench")
    parser.add_argument("--max_examples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Loading {args.hf_repo} (version={args.version}) ...", flush=True)
    ds = datasets.load_dataset(args.hf_repo, version_tag=args.version,
                               split=args.split, trust_remote_code=True)
    print(f"Raw rows: {len(ds)}", flush=True)

    instruction = (
        "Read the problem carefully. Write a single self-contained Python "
        "program that reads from standard input and writes the required "
        "output to standard output. Put your final program inside a single "
        "```python ... ``` block."
    )
    system_prompt = "You are an expert competitive programmer."

    rows = []
    for i, ex in enumerate(ds):
        if args.max_examples > 0 and i >= args.max_examples:
            break
        prompt = ex.get("question_content") or ex.get("question") or ex.get("problem")
        if not prompt:
            continue
        tests = []
        tests.extend(_decode_tests(ex.get("public_test_cases")))
        tests.extend(_decode_tests(ex.get("private_test_cases")))
        if not tests:
            continue
        inputs, outputs = zip(*tests)
        ground_truth = {"inputs": list(inputs), "outputs": list(outputs)}

        rows.append({
            "data_source": "codecontests",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + "\n\n" + instruction},
            ],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(ground_truth),
            },
            "extra_info": {
                "split": "test",
                "index": i,
                "task_id": ex.get("question_id") or ex.get("task_id") or str(i),
                "source": "livecodebench",
                "version": args.version,
            },
        })

    print(f"Kept {len(rows)} eval problems.", flush=True)
    out_path = os.path.join(args.local_dir, f"{args.version}_test.parquet")
    datasets.Dataset.from_list(rows).to_parquet(out_path)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

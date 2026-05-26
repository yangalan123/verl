"""
Build a small train parquet for code RL from the public PRIME-RL/Eurus-2-RL-Data
(or any compatible TACO / APPS-style dataset that ships ground_truth in the
{"inputs": [...], "outputs": [...]} stdio format).

The output `data_source` is fixed to "codecontests" so that the existing
verl reward dispatcher (verl/utils/reward_score/__init__.py) routes scoring
to `prime_code.compute_score`, which runs each candidate in a local Python
subprocess with a SIGALRM timeout -- no Docker, no firejail required.

Usage:
    python recipe/annealed_sampling/codeRL/prepare_train_eurus2_code.py \\
        --hf_repo "PRIME-RL/Eurus-2-RL-Data" \\
        --hf_split train \\
        --local_dir ./data/eurus2_code \\
        --max_examples 10000
"""

import argparse
import json
import os
import random
from typing import Optional

import datasets


def _to_inout_dict(raw):
    """Normalize a row's ground_truth into a dict {"inputs": [...], "outputs": [...]}.

    The Eurus-2-RL-Data code rows usually already ship test cases in this
    format inside `reward_model.ground_truth` (as a JSON string). If the row
    instead uses a `tests` / `input_output` field we try a couple of
    conventional layouts before giving up.
    """
    if raw is None:
        return None
    if isinstance(raw, dict) and "inputs" in raw and "outputs" in raw:
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "inputs" in obj and "outputs" in obj:
                return obj
        except Exception:
            return None
    return None


def _extract_problem_text(example):
    """Try a few common field names for the prompt text."""
    for key in ("problem", "prompt", "question", "task", "instruction"):
        v = example.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # chat-style prompt -- take last user message
            for msg in reversed(v):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    return msg["content"]
    return None


def _extract_ability(example) -> Optional[str]:
    for key in ("ability", "task", "domain"):
        v = example.get(key)
        if isinstance(v, str):
            return v.lower()
    return None


def _extract_ground_truth(example):
    """Pull the raw ground-truth and normalize it to a dict."""
    if "reward_model" in example and isinstance(example["reward_model"], dict):
        raw = example["reward_model"].get("ground_truth")
        norm = _to_inout_dict(raw)
        if norm is not None:
            return norm
    for key in ("ground_truth", "tests", "input_output"):
        if key in example:
            norm = _to_inout_dict(example[key])
            if norm is not None:
                return norm
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="PRIME-RL/Eurus-2-RL-Data")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_subset", default=None, help="Optional config name")
    parser.add_argument("--local_dir", default="./data/eurus2_code")
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_ability_substr", default="code",
                        help="Only keep rows whose 'ability'/'task' field contains this substring.")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Loading {args.hf_repo} ...", flush=True)
    if args.hf_subset:
        ds = datasets.load_dataset(args.hf_repo, args.hf_subset, split=args.hf_split,
                                   trust_remote_code=True)
    else:
        ds = datasets.load_dataset(args.hf_repo, split=args.hf_split,
                                   trust_remote_code=True)
    print(f"Raw rows: {len(ds)}", flush=True)

    instruction = (
        "Read the problem carefully. Write a single Python program that reads "
        "from standard input and writes the required output to standard output. "
        "Put your final program inside a single ```python ... ``` block."
    )
    system_prompt = "You are an expert competitive programmer."

    rows = []
    n_skipped_no_gt = 0
    n_skipped_no_prompt = 0
    n_skipped_ability = 0
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    for idx in indices:
        if len(rows) >= args.max_examples:
            break
        ex = ds[int(idx)]

        ability = _extract_ability(ex) or ""
        if args.keep_ability_substr and args.keep_ability_substr not in ability:
            n_skipped_ability += 1
            continue

        prompt = _extract_problem_text(ex)
        if prompt is None:
            n_skipped_no_prompt += 1
            continue

        gt = _extract_ground_truth(ex)
        if gt is None:
            n_skipped_no_gt += 1
            continue

        rows.append({
            "data_source": "codecontests",
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
                "split": "train",
                "index": len(rows),
                "source_repo": args.hf_repo,
            },
        })

    print(
        f"Kept {len(rows)} | skipped (ability={n_skipped_ability}, "
        f"no_prompt={n_skipped_no_prompt}, no_gt={n_skipped_no_gt})",
        flush=True,
    )
    if len(rows) == 0:
        raise RuntimeError(
            "No usable rows. Inspect the source dataset's schema and adjust "
            "`_extract_*` helpers above."
        )

    out_path = os.path.join(args.local_dir, "train.parquet")
    datasets.Dataset.from_list(rows).to_parquet(out_path)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

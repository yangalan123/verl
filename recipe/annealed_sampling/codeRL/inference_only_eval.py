"""
Inference-only evaluation of EAD vs. fixed-temperature sampling on code
benchmarks. No RL training, no Docker, no firejail.

Loads the prepared parquet (HumanEval+ or LiveCodeBench format), generates K
samples per prompt with vLLM, scores each sample through the same reward
modules that verl uses during training, and reports pass@1 / pass@K (mean over
prompts) and worst@K (worst-of-K success fraction per prompt).

Two sampling regimes are supported:
  --mode fixed   : standard temperature sampling (--temperature T)
  --mode ead     : Exploratory Annealed Decoding via vLLM LogitsProcessor
                   with the same negexp schedule used in the paper.

Example:
    python recipe/annealed_sampling/codeRL/inference_only_eval.py \\
        --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --eval_parquet ./data/humanevalplus/test.parquet \\
        --mode fixed --temperature 1.0 --n_samples 8

    python recipe/annealed_sampling/codeRL/inference_only_eval.py \\
        --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --eval_parquet ./data/livecodebench/release_v2_test.parquet \\
        --mode ead --start_temp 1.2 --end_temp 0.1 --decay_freq 200 \\
        --n_samples 8
"""

import argparse
import json
import os
import time
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets

from verl.utils.reward_score import default_compute_score


def _build_prompts(rows, tokenizer, enable_thinking=None) -> List[str]:
    """Apply the model's chat template to each row's prompt field.

    `enable_thinking` is a tri-state:
      None  -> use the model's chat-template default (don't pass the kwarg)
      True  -> request thinking traces (Qwen3 dual-mode models)
      False -> suppress thinking traces

    Non-Qwen3 templates do not accept the kwarg; we retry without it.
    """
    out = []
    for row in rows:
        msgs = list(row["prompt"])
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            text = tokenizer.apply_chat_template(msgs, **kwargs)
        except TypeError:
            # Template doesn't support enable_thinking; drop it and retry.
            kwargs.pop("enable_thinking", None)
            try:
                text = tokenizer.apply_chat_template(msgs, **kwargs)
            except Exception:
                text = "\n".join(m.get("content", "") for m in msgs)
        except Exception:
            text = "\n".join(m.get("content", "") for m in msgs)
        out.append(text)
    return out


def score_completion(data_source, text, ground_truth):
    """Score ONE completion. Returns (score_float, status_str_or_None).

    Module-level so the CPU-only scorer (score_dump.py) can reuse the exact same
    scoring path as the live eval. The heavy reward submodules are imported
    lazily so importing this module never pulls in vllm/torch.
    """
    try:
        if data_source in ("humanevalplus", "mbppplus"):
            from verl.utils.reward_score import humanevalplus as _hep
            score_val, meta = _hep.compute_score(text, ground_truth)
            status = (meta[0] if meta else {}).get("status") \
                or (meta[0] if meta else {}).get("error")
            return float(score_val), status
        if data_source in ("codecontests", "apps", "codeforces", "taco"):
            from verl.utils.reward_score import prime_code as _pc
            success, _meta = _pc.compute_score(text, ground_truth, continuous=True)
            score = (1.0 if success is True
                     else float(success) if isinstance(success, (int, float)) else 0.0)
            return score, ("passed" if score >= 0.999 else "failed")
        res = default_compute_score(data_source, text, ground_truth)
        if isinstance(res, dict):
            return float(res.get("score", 0.0)), None
        return float(res), None
    except Exception as e:  # noqa: BLE001
        return 0.0, f"exception: {type(e).__name__}: {e}"


def _completion_stats(completion, max_pos):
    """Extract (n_tokens, mean_neglogp, neglogp_by_pos) from a vLLM output.

    mean_neglogp is the negative average log-likelihood of the sampled tokens --
    an unbiased Monte-Carlo estimate of the average per-token entropy (the
    long-sequence entropy proxy used in "How Alignment Shrinks the Generative
    Horizon", yangalan123/LLMBranchingFactor). Requires logprobs to have been
    requested; otherwise mean_neglogp/neglogp_by_pos may be None.
    """
    token_ids = list(getattr(completion, "token_ids", []) or [])
    n = len(token_ids)
    mean_neglogp = None
    neglogp_by_pos = None
    cum = getattr(completion, "cumulative_logprob", None)
    if cum is not None and n > 0:
        mean_neglogp = -float(cum) / n
    lp = getattr(completion, "logprobs", None)
    if lp:
        seq = []
        for pos, tid in enumerate(token_ids):
            d = lp[pos] if pos < len(lp) else None
            entry = d.get(tid) if d else None
            if entry is not None:
                seq.append(-float(entry.logprob))
        if seq:
            mean_neglogp = sum(seq) / len(seq)
            neglogp_by_pos = seq[: max_pos] if max_pos and max_pos > 0 else seq
    return n, mean_neglogp, neglogp_by_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_parquet", required=True)
    parser.add_argument("--output_dir", default="./logs/inference_only_eval")
    parser.add_argument("--mode", choices=["fixed", "ead"], default="fixed")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable_thinking", choices=["auto", "on", "off"], default="auto",
        help="Qwen3 dual-mode chat template: 'on' requests <think> traces, "
             "'off' suppresses them, 'auto' uses the template default. "
             "Ignored by templates without thinking support.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=-1,
        help="vLLM context window. -1 lets vLLM use the model default; set "
             "explicitly for long-reasoning models to bound KV-cache memory.",
    )
    parser.add_argument(
        "--save_completions", action="store_true",
        help="Persist ALL completion texts (not just the first) to the "
             "per_prompt JSONL so they can be re-scored later under a different "
             "reward/extraction. Off by default (can be large for reasoning "
             "models). pass@k/worst@k for any k<=n_samples are recomputable "
             "from the saved 'successes' vector regardless of this flag.",
    )
    parser.add_argument(
        "--completion_char_cap", type=int, default=0,
        help="With --save_completions, truncate each saved completion to this "
             "many chars (0 = no cap). Use a cap only if disk is tight; note "
             "truncated text cannot be reliably re-scored.",
    )
    parser.add_argument(
        "--logprobs", type=int, default=0,
        help="If >0, request this many vLLM logprobs so we can record per-"
             "completion token length and negative avg log-likelihood (entropy "
             "proxy) for the smoke/analysis phase. 1 is enough.",
    )
    parser.add_argument(
        "--logprob_max_pos", type=int, default=512,
        help="When --logprobs>0, cap the per-position neg-logprob array stored "
             "per completion to this many leading positions (keeps dumps small).",
    )
    parser.add_argument(
        "--generate_only", action="store_true",
        help="Generate and dump completions (+ length/entropy stats) WITHOUT "
             "scoring, so the GPU is freed immediately and a separate CPU "
             "process (score_dump.py) can score the dump. Writes gen__<tag>.jsonl "
             "and genmeta__<tag>.json instead of summary/per_prompt.",
    )
    parser.add_argument(
        "--tag_suffix", default="",
        help="Optional suffix appended to the output tag (e.g. 'smoke') so "
             "different runs of the same config don't overwrite each other.",
    )
    # EAD-specific
    parser.add_argument("--start_temp", type=float, default=1.2)
    parser.add_argument("--end_temp", type=float, default=0.1)
    parser.add_argument("--decay_freq", type=int, default=200)
    parser.add_argument("--warmup_period", type=int, default=10)
    parser.add_argument("--decay_mode", default="negexp")
    parser.add_argument("--decay_freq_cap_large", type=int, default=40000)
    parser.add_argument("--decay_freq_increase_factor", type=int, default=5)
    parser.add_argument("--global_step", type=int, default=0,
                        help="Used only by the EAD schedule (0 for inference-only).")
    # vLLM scaffolding
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_USE_V1", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    ds = datasets.Dataset.from_parquet(args.eval_parquet)
    rows = list(ds)
    if args.max_prompts > 0:
        rows = rows[: args.max_prompts]

    enable_thinking = {"auto": None, "on": True, "off": False}[args.enable_thinking]
    prompts = _build_prompts(rows, tokenizer, enable_thinking=enable_thinking)
    print(
        f"Loaded {len(rows)} prompts from {args.eval_parquet} "
        f"(enable_thinking={enable_thinking}, max_tokens={args.max_tokens})",
        flush=True,
    )

    llm_kwargs = dict(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.mode == "ead":
        # Per-request LogitsProcessor; the vLLM v1 wrapper picks it up from
        # SamplingParams.extra_args (see WrapperAdapterLogitsProcessor).
        from verl.workers.rollout.vllm_rollout.annealed_sampling import (
            WrapperAdapterLogitsProcessor,
        )
        llm_kwargs["logits_processors"] = [WrapperAdapterLogitsProcessor]

    print("Spinning up vLLM ...", flush=True)
    llm = LLM(**llm_kwargs)

    logprobs_arg = args.logprobs if args.logprobs and args.logprobs > 0 else None
    if args.mode == "fixed":
        sampling_params = SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            logprobs=logprobs_arg,
        )
    else:
        sampling_params = SamplingParams(
            n=args.n_samples,
            temperature=1.0,  # base T; the logits proc rescales every step
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            logprobs=logprobs_arg,
            extra_args={
                "exploration_temp": args.start_temp,
                "stability_temp": args.end_temp,
                "decay_freq": args.decay_freq,
                "global_step": args.global_step,
                "decay_mode": args.decay_mode,
                "warmup_period": args.warmup_period,
                "decay_freq_increase_factor": args.decay_freq_increase_factor,
                "decay_freq_cap_large": args.decay_freq_cap_large,
            },
        )

    print(f"Generating with mode={args.mode} ...", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params)
    print(f"Generation done in {time.time() - t0:.1f}s", flush=True)

    out_tag = (
        os.path.basename(args.eval_parquet).replace(".parquet", "")
        + f"__{args.mode}"
        + (f"__T{args.temperature}" if args.mode == "fixed"
           else f"__neg_{args.start_temp}_{args.end_temp}_d{args.decay_freq}_w{args.warmup_period}")
    )
    if args.tag_suffix:
        out_tag += f"__{args.tag_suffix}"

    config = {
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "start_temp": args.start_temp,
        "end_temp": args.end_temp,
        "decay_freq": args.decay_freq,
        "decay_mode": args.decay_mode,
        "decay_freq_cap_large": args.decay_freq_cap_large,
        "decay_freq_increase_factor": args.decay_freq_increase_factor,
        "warmup_period": args.warmup_period,
        "logprobs": args.logprobs,
    }

    # ---- generate-only mode: dump completions + stats, skip GPU-blocking scoring ----
    if args.generate_only:
        gen_path = os.path.join(args.output_dir, f"gen__{out_tag}.jsonl")
        meta_path = os.path.join(args.output_dir, f"genmeta__{out_tag}.json")
        with open(gen_path, "w") as f:
            for row, out in zip(rows, outs):
                texts, lens, neglogps, neglogp_pos = [], [], [], []
                for c in out.outputs:
                    nt, mnl, npos = _completion_stats(c, args.logprob_max_pos)
                    texts.append(c.text)
                    lens.append(nt)
                    neglogps.append(mnl)
                    if npos is not None:
                        neglogp_pos.append(npos)
                rec = {
                    "task_id": row.get("extra_info", {}).get("task_id"),
                    "data_source": row["data_source"],
                    "ground_truth": row["reward_model"]["ground_truth"],
                    "completions": texts,
                    "token_lens": lens,
                    "mean_neglogp": neglogps,
                }
                if neglogp_pos:
                    rec["neglogp_by_pos"] = neglogp_pos
                f.write(json.dumps(rec) + "\n")
        with open(meta_path, "w") as f:
            json.dump({
                "model": args.model_name_or_path,
                "eval_parquet": args.eval_parquet,
                "mode": args.mode,
                "n_samples_per_prompt": args.n_samples,
                "num_prompts": len(rows),
                "config": config,
                "out_tag": out_tag,
            }, f, indent=2)
        print(f"[generate_only] wrote {gen_path}")
        print(f"[generate_only] wrote {meta_path}")
        print("[generate_only] score later with: python "
              "recipe/annealed_sampling/codeRL/score_dump.py "
              f"--gen_jsonl {gen_path}")
        return

    # ---- normal mode: score inline ----
    pass_at_1 = 0.0
    pass_at_k = 0.0
    worst_at_k = 0.0
    per_prompt = []
    for row, out in zip(rows, outs):
        data_source = row["data_source"]
        ground_truth = row["reward_model"]["ground_truth"]
        successes = []
        statuses = []  # one per completion; populated for humanevalplus / prime_code
        lens = []
        for completion in out.outputs:
            score, status_detail = score_completion(data_source, completion.text, ground_truth)
            successes.append(score)
            statuses.append(status_detail)
            lens.append(len(getattr(completion, "token_ids", []) or []))
        n_ok = sum(1 for s in successes if s >= 0.999)
        any_ok = 1.0 if n_ok > 0 else 0.0
        all_ok = 1.0 if n_ok == len(successes) else 0.0
        first_ok = 1.0 if successes and successes[0] >= 0.999 else 0.0
        pass_at_1 += first_ok
        pass_at_k += any_ok
        worst_at_k += all_ok
        # Keep the first completion's raw text so we can inspect format failures
        # without re-running vLLM. Truncate to avoid blowing up the log.
        first_completion = out.outputs[0].text if out.outputs else ""
        if len(first_completion) > 4000:
            first_completion = first_completion[:4000] + " ...[truncated]"
        record = {
            "task_id": row.get("extra_info", {}).get("task_id"),
            "successes": successes,
            "statuses": statuses,
            "token_lens": lens,
            "first_completion": first_completion,
            "pass@1": first_ok,
            "pass@k": any_ok,
            "worst@k": all_ok,
        }
        if args.save_completions:
            texts = [c.text for c in out.outputs]
            if args.completion_char_cap > 0:
                cap = args.completion_char_cap
                texts = [t if len(t) <= cap else t[:cap] + " ...[truncated]"
                         for t in texts]
            record["completions"] = texts
        per_prompt.append(record)

    n = len(rows)
    summary = {
        "model": args.model_name_or_path,
        "eval_parquet": args.eval_parquet,
        "mode": args.mode,
        "n_samples_per_prompt": args.n_samples,
        "num_prompts": n,
        "pass@1": pass_at_1 / max(n, 1),
        f"pass@{args.n_samples}": pass_at_k / max(n, 1),
        f"worst@{args.n_samples}": worst_at_k / max(n, 1),
        "config": config,
    }

    summary_path = os.path.join(args.output_dir, f"summary__{out_tag}.json")
    per_prompt_path = os.path.join(args.output_dir, f"per_prompt__{out_tag}.jsonl")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(per_prompt_path, "w") as f:
        for r in per_prompt:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_prompt_path}")


if __name__ == "__main__":
    main()

"""
EAD parameter-sweep orchestrator (Day-0, inference-only) on 4 GPUs.

Goal: find an EAD (`negexp`) configuration that BEATS the fixed-temperature
baseline on the code benchmarks. The paper's defaults were tuned for long math
responses; code completions are much shorter, so the schedule must be re-sized
from the actual response-length distribution. This script automates that:

  Phase 1  BASELINE + SMOKE (gather data)
     For each (model, benchmark):
       - if the fixed-T=1.0 baseline is missing, generate it (generate-only,
         with logprobs) and score it on CPU;
       - that same dump (token lengths + per-position neg-logprob entropy proxy)
         is reused for analysis. If a baseline already exists but no dump is
         present, run a small smoke generate-only job with logprobs instead.
  Phase 2  ANALYZE (in-process, fast)
       analyze_smoke.recommend_search_space() -> warmup_period + decay_freq
       candidates (sized to median response length) + temp grids.
  Phase 3  SWEEP
       grid = start_temps x end_temps x decay_freqs (decay_mode=negexp,
       warmup_period fixed from analysis). Each config: generate-only on GPU,
       then score on CPU.
  Phase 4  REPORT
       compare every EAD config to the baseline; list the winners.

GPU/CPU split: generation (GPU-bound) and scoring (CPU/sandbox-bound) are
separate processes. The scheduler keeps all GPUs busy generating while scoring
runs concurrently on CPU, so neither resource idles waiting for the other.

Usage:
    # full pipeline on 4 GPUs, default models/benchmarks
    python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3

    # see the plan without launching anything
    python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 --dry_run

    # one model / one benchmark, quick
    python recipe/annealed_sampling/codeRL/ead_sweep.py --gpus 0,1,2,3 \\
        --models Qwen2.5-Coder-1.5B --benchmarks humanevalplus
"""

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_PY = os.path.join(HERE, "inference_only_eval.py")
SCORE_PY = os.path.join(HERE, "score_dump.py")

# import the analysis helper in-process
sys.path.insert(0, HERE)
from analyze_smoke import recommend_search_space  # noqa: E402


# --- Model registry: (id, max_tokens, enable_thinking, tp, max_model_len, gpu_mem) ---
MODELS = [
    {"model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "max_tokens": 2048,
     "thinking": "off", "tp": 1, "max_model_len": -1, "gpu_mem": 0.85},
    {"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "max_tokens": 2048,
     "thinking": "off", "tp": 1, "max_model_len": -1, "gpu_mem": 0.90},
    {"model": "Qwen/Qwen3-4B", "max_tokens": 16384,
     "thinking": "on", "tp": 1, "max_model_len": 20480, "gpu_mem": 0.90},
    {"model": "Qwen/Qwen3-8B", "max_tokens": 16384,
     "thinking": "on", "tp": 1, "max_model_len": 20480, "gpu_mem": 0.90},
    {"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "max_tokens": 4096,
     "thinking": "off", "tp": 2, "max_model_len": 16384, "gpu_mem": 0.90},
]

# benchmark -> (subdir, parquet filename); livecodebench filename is templated
BENCHMARKS = {
    "humanevalplus": ("humanevalplus", "test.parquet"),
    "livecodebench": ("livecodebench", "{lcb}_test.parquet"),
}


def model_tag(model_id: str) -> str:
    return os.path.basename(model_id)


def parquet_basename(parquet_path: str) -> str:
    return os.path.basename(parquet_path).replace(".parquet", "")


def predict_tag(pbase: str, mode: str, *, temperature=1.0, start_temp=1.2,
                end_temp=0.1, decay_freq=200, warmup=10, suffix="") -> str:
    """Mirror inference_only_eval.py's out_tag EXACTLY so we can find the dumps."""
    if mode == "fixed":
        tag = f"{pbase}__fixed__T{float(temperature)}"
    else:
        tag = (f"{pbase}__ead__neg_{float(start_temp)}_{float(end_temp)}"
               f"_d{int(decay_freq)}_w{int(warmup)}")
    if suffix:
        tag += f"__{suffix}"
    return tag


# ----------------------------- job scheduler ------------------------------- #
class Job:
    def __init__(self, name: str, kind: str, cmd: List[str], tp: int = 0,
                 env: Optional[Dict[str, str]] = None,
                 logfile: Optional[str] = None, score_cmd: Optional[List[str]] = None,
                 score_name: Optional[str] = None, score_log: Optional[str] = None):
        self.name = name
        self.kind = kind            # 'gen' (GPU) or 'score' (CPU)
        self.cmd = cmd
        self.tp = tp
        self.env = env or {}
        self.logfile = logfile
        self.score_cmd = score_cmd  # follow-up CPU job to enqueue after a gen
        self.score_name = score_name
        self.score_log = score_log
        self.gpus: List[int] = []
        self.popen: Optional[subprocess.Popen] = None


def run_schedule(gen_jobs: List[Job], gpus: List[int], cpu_slots: int) -> None:
    free = list(gpus)
    gen_pending = list(gen_jobs)
    gen_running: List[Job] = []
    score_pending: List[Job] = []
    score_running: List[Job] = []

    def launch(job: Job):
        env = dict(os.environ)
        env.update(job.env)
        if job.kind == "gen":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in job.gpus)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only scoring
        env["TOKENIZERS_PARALLELISM"] = "false"
        out = open(job.logfile, "w") if job.logfile else None
        job._out = out  # type: ignore[attr-defined]
        job.popen = subprocess.Popen(job.cmd, cwd=os.getcwd(), env=env,
                                     stdout=out, stderr=subprocess.STDOUT)
        where = (f"GPU={env.get('CUDA_VISIBLE_DEVICES')}" if job.kind == "gen"
                 else "CPU")
        print(f"[sched] launch {job.kind} '{job.name}' {where} -> {job.logfile}",
              flush=True)

    while gen_pending or gen_running or score_pending or score_running:
        # start GPU gen jobs that fit
        i = 0
        while i < len(gen_pending):
            job = gen_pending[i]
            if len(free) >= job.tp:
                job.gpus = [free.pop(0) for _ in range(job.tp)]
                launch(job)
                gen_running.append(job)
                gen_pending.pop(i)
            else:
                i += 1
        # start CPU score jobs up to the slot cap
        while score_pending and len(score_running) < cpu_slots:
            job = score_pending.pop(0)
            launch(job)
            score_running.append(job)

        # poll GPU gens
        still: List[Job] = []
        for job in gen_running:
            rc = job.popen.poll()
            if rc is None:
                still.append(job)
                continue
            if getattr(job, "_out", None):
                job._out.close()
            free.extend(job.gpus)
            if rc == 0:
                print(f"[sched] done gen '{job.name}'", flush=True)
                if job.score_cmd:
                    score_pending.append(Job(job.score_name or (job.name + ":score"),
                                             "score", job.score_cmd,
                                             logfile=job.score_log))
            else:
                print(f"[sched] FAILED gen '{job.name}' rc={rc} (see {job.logfile})",
                      flush=True)
        gen_running = still

        # poll CPU scores
        still_s: List[Job] = []
        for job in score_running:
            rc = job.popen.poll()
            if rc is None:
                still_s.append(job)
                continue
            if getattr(job, "_out", None):
                job._out.close()
            tag = "done" if rc == 0 else f"FAILED rc={rc}"
            print(f"[sched] {tag} score '{job.name}' (see {job.logfile})", flush=True)
        score_running = still_s

        time.sleep(2.0)


# ----------------------------- command builders ---------------------------- #
def gen_cmd(model: Dict[str, Any], parquet: str, out_dir: str, *, mode: str,
            n_samples: int, max_prompts: int, logprobs: int, suffix: str,
            temperature=1.0, start_temp=1.2, end_temp=0.1, decay_freq=200,
            warmup=10) -> List[str]:
    cmd = [
        sys.executable, EVAL_PY,
        "--model_name_or_path", model["model"],
        "--eval_parquet", parquet,
        "--output_dir", out_dir,
        "--mode", mode,
        "--n_samples", str(n_samples),
        "--max_prompts", str(max_prompts),
        "--max_tokens", str(model["max_tokens"]),
        "--enable_thinking", model["thinking"],
        "--max_model_len", str(model["max_model_len"]),
        "--tensor_parallel_size", str(model["tp"]),
        "--gpu_memory_utilization", str(model["gpu_mem"]),
        "--generate_only",
        "--logprobs", str(logprobs),
    ]
    if suffix:
        cmd += ["--tag_suffix", suffix]
    if mode == "fixed":
        cmd += ["--temperature", str(temperature)]
    else:
        cmd += ["--start_temp", str(start_temp), "--end_temp", str(end_temp),
                "--decay_freq", str(decay_freq), "--decay_mode", "negexp",
                "--warmup_period", str(warmup),
                "--decay_freq_cap_large", "40000",
                "--decay_freq_increase_factor", "5"]
    return cmd


def score_cmd(gen_path: str, num_workers: int) -> List[str]:
    return [sys.executable, SCORE_PY, "--gen_jsonl", gen_path,
            "--num_workers", str(num_workers)]


# ------------------------------- discovery --------------------------------- #
def resolve_parquet(data_root: str, bench: str, lcb_version: str) -> str:
    sub, fname = BENCHMARKS[bench]
    fname = fname.format(lcb=lcb_version)
    return os.path.join(data_root, sub, fname)


def find_gen_for_analysis(out_dir: str) -> Optional[str]:
    """Prefer the baseline fixed-T dump, else any smoke dump."""
    cands = sorted(glob.glob(os.path.join(out_dir, "gen__*fixed__T1.0*.jsonl")))
    cands += sorted(glob.glob(os.path.join(out_dir, "gen__*smoke*.jsonl")))
    return cands[0] if cands else None


def summary_exists(out_dir: str, tag: str) -> bool:
    p = os.path.join(out_dir, f"summary__{tag}.json")
    return os.path.exists(p) and os.path.getsize(p) > 0


# --------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./inference-eval-sample32",
                    help="Eval output root (per the renamed 32-sample dir).")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--cpu_slots", type=int, default=2,
                    help="Concurrent CPU scoring processes.")
    ap.add_argument("--score_workers", type=int, default=16,
                    help="Threads per scoring process.")
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--smoke_prompts", type=int, default=32)
    ap.add_argument("--smoke_samples", type=int, default=4)
    ap.add_argument("--sweep_max_prompts", type=int, default=-1,
                    help="Subsample prompts during the EAD sweep for speed "
                         "(-1 = full set; full set keeps comparison to the "
                         "full baseline apples-to-apples).")
    ap.add_argument("--models", default=None, help="Substring filter on model id.")
    ap.add_argument("--benchmarks", default="humanevalplus,livecodebench")
    ap.add_argument("--lcb_version", default="release_v2")
    ap.add_argument("--phase", default="all",
                    choices=["all", "data", "analyze", "sweep", "report"])
    ap.add_argument("--metric", default="pass@k",
                    choices=["pass@1", "pass@k", "worst@k"],
                    help="Primary metric used to rank EAD winners in the report.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if outputs already exist.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    benches = [b for b in args.benchmarks.split(",") if b.strip()]
    models = [m for m in MODELS
              if (args.models is None or args.models.lower() in m["model"].lower())]
    if not models:
        print(f"No models match '{args.models}'. Known: "
              + ", ".join(m['model'] for m in MODELS))
        return
    log_root = os.path.join(args.root, "_sweep_logs")
    os.makedirs(log_root, exist_ok=True)

    pairs = [(m, b) for m in models for b in benches]
    print(f"[ead_sweep] models={[model_tag(m['model']) for m in models]} "
          f"benchmarks={benches} gpus={gpus}")

    # ---------------- Phase 1: baseline + smoke (data gathering) ------------
    if args.phase in ("all", "data"):
        jobs: List[Job] = []
        for m, b in pairs:
            tag = model_tag(m["model"])
            parquet = resolve_parquet(args.data_root, b, args.lcb_version)
            out_dir = os.path.join(args.root, tag, b)
            os.makedirs(out_dir, exist_ok=True)
            pbase = parquet_basename(parquet)
            base_tag = predict_tag(pbase, "fixed", temperature=1.0)

            if not args.force and summary_exists(out_dir, base_tag):
                # baseline done; ensure we have a dump to analyze, else smoke it
                if find_gen_for_analysis(out_dir) is None:
                    smoke_suffix = "smoke"
                    smoke_tag = predict_tag(pbase, "fixed", temperature=1.0,
                                            suffix=smoke_suffix)
                    cmd = gen_cmd(m, parquet, out_dir, mode="fixed",
                                  n_samples=args.smoke_samples,
                                  max_prompts=args.smoke_prompts, logprobs=1,
                                  suffix=smoke_suffix, temperature=1.0)
                    jobs.append(Job(f"{tag}/{b}:smoke", "gen", cmd, tp=m["tp"],
                                    logfile=os.path.join(log_root, f"{tag}__{b}__smoke.log")))
                continue

            # baseline missing -> full generate-only WITH logprobs (serves both
            # the baseline score AND the analysis), then score on CPU.
            gen_path = os.path.join(out_dir, f"gen__{base_tag}.jsonl")
            cmd = gen_cmd(m, parquet, out_dir, mode="fixed",
                          n_samples=args.n_samples, max_prompts=-1, logprobs=1,
                          suffix="", temperature=1.0)
            jobs.append(Job(
                f"{tag}/{b}:baseline", "gen", cmd, tp=m["tp"],
                logfile=os.path.join(log_root, f"{tag}__{b}__baseline.log"),
                score_cmd=score_cmd(gen_path, args.score_workers),
                score_name=f"{tag}/{b}:baseline:score",
                score_log=os.path.join(log_root, f"{tag}__{b}__baseline_score.log")))

        if args.dry_run:
            print(f"[dry_run] phase=data would launch {len(jobs)} gen jobs:")
            for j in jobs:
                print("   ", j.name, "TP=", j.tp)
        elif jobs:
            run_schedule(jobs, gpus, args.cpu_slots)
        else:
            print("[ead_sweep] data phase: nothing to do (all present).")

    # ---------------- Phase 2: analyze -> search space ----------------------
    search_spaces: Dict[str, Dict[str, Any]] = {}
    if args.phase in ("all", "analyze", "sweep", "report"):
        for m, b in pairs:
            tag = model_tag(m["model"])
            out_dir = os.path.join(args.root, tag, b)
            ss_path = os.path.join(out_dir, "search_space.json")
            if not args.force and os.path.exists(ss_path):
                with open(ss_path) as f:
                    search_spaces[(tag, b)] = json.load(f)
                continue
            gen = find_gen_for_analysis(out_dir)
            if gen is None:
                print(f"[analyze] no dump for {tag}/{b}; run --phase data first.")
                continue
            space = recommend_search_space(gen)
            os.makedirs(out_dir, exist_ok=True)
            with open(ss_path, "w") as f:
                json.dump(space, f, indent=2)
            search_spaces[(tag, b)] = space
            print(f"[analyze] {tag}/{b}: warmup={space['warmup_period']} "
                  f"decay_freqs={space['decay_freqs']} "
                  f"(median_len={space['length_stats']['p50']:.0f})")

    # ---------------- Phase 3: EAD sweep ------------------------------------
    if args.phase in ("all", "sweep"):
        jobs = []
        for m, b in pairs:
            tag = model_tag(m["model"])
            space = search_spaces.get((tag, b))
            if space is None:
                print(f"[sweep] no search space for {tag}/{b}; skipping.")
                continue
            parquet = resolve_parquet(args.data_root, b, args.lcb_version)
            out_dir = os.path.join(args.root, tag, b)
            pbase = parquet_basename(parquet)
            warmup = space["warmup_period"]
            grid = list(itertools.product(space["start_temps"],
                                          space["end_temps"],
                                          space["decay_freqs"]))
            print(f"[sweep] {tag}/{b}: {len(grid)} EAD configs (warmup={warmup})")
            for st, et, df in grid:
                ead_tag = predict_tag(pbase, "ead", start_temp=st, end_temp=et,
                                      decay_freq=df, warmup=warmup)
                if not args.force and summary_exists(out_dir, ead_tag):
                    continue
                gen_path = os.path.join(out_dir, f"gen__{ead_tag}.jsonl")
                cmd = gen_cmd(m, parquet, out_dir, mode="ead",
                              n_samples=args.n_samples,
                              max_prompts=args.sweep_max_prompts, logprobs=0,
                              suffix="", start_temp=st, end_temp=et,
                              decay_freq=df, warmup=warmup)
                jobs.append(Job(
                    f"{tag}/{b}:ead_{st}_{et}_d{df}", "gen", cmd, tp=m["tp"],
                    logfile=os.path.join(log_root, f"{tag}__{b}__ead_{st}_{et}_d{df}.log"),
                    score_cmd=score_cmd(gen_path, args.score_workers),
                    score_name=f"{tag}/{b}:ead_{st}_{et}_d{df}:score",
                    score_log=os.path.join(log_root, f"{tag}__{b}__ead_{st}_{et}_d{df}_score.log")))

        if args.dry_run:
            print(f"[dry_run] phase=sweep would launch {len(jobs)} gen jobs")
            for j in jobs[:20]:
                print("   ", j.name)
            if len(jobs) > 20:
                print(f"    ... and {len(jobs) - 20} more")
        elif jobs:
            run_schedule(jobs, gpus, args.cpu_slots)
        else:
            print("[ead_sweep] sweep phase: nothing to do (all present).")

    # ---------------- Phase 4: report ---------------------------------------
    if args.phase in ("all", "report"):
        report = {}
        for m, b in pairs:
            tag = model_tag(m["model"])
            out_dir = os.path.join(args.root, tag, b)
            pbase = parquet_basename(resolve_parquet(args.data_root, b, args.lcb_version))
            base = _read_summary(out_dir, predict_tag(pbase, "fixed", temperature=1.0))
            if base is None:
                continue
            metrics_keys = ["pass@1", _kkey(base, "pass"), _kkey(base, "worst")]
            ead_rows = []
            for sp in sorted(glob.glob(os.path.join(out_dir, "summary__*__ead__*.json"))):
                s = _load(sp)
                if s is None:
                    continue
                cfg = s.get("config", {})
                row = {
                    "config": f"st={cfg.get('start_temp')} et={cfg.get('end_temp')} "
                              f"d={cfg.get('decay_freq')} w={cfg.get('warmup_period')}",
                    "pass@1": s.get("pass@1"),
                    "pass@k": _val(s, "pass"),
                    "worst@k": _val(s, "worst"),
                }
                ead_rows.append(row)
            base_row = {"pass@1": base.get("pass@1"),
                        "pass@k": _val(base, "pass"),
                        "worst@k": _val(base, "worst")}
            key = args.metric
            winners = [r for r in ead_rows
                       if r[key] is not None and base_row[key] is not None
                       and r[key] > base_row[key]]
            winners.sort(key=lambda r: r[key], reverse=True)
            report[f"{tag}/{b}"] = {"baseline": base_row, "winners": winners,
                                    "num_ead_configs": len(ead_rows)}
            print(f"\n=== {tag}/{b} === baseline {key}={base_row[key]}")
            if not ead_rows:
                print("  (no EAD configs scored yet)")
            for r in winners[:10]:
                d = r[key] - base_row[key]
                print(f"  WIN {r['config']:45s} {key}={r[key]:.4f} (+{d:.4f})")
            if ead_rows and not winners:
                best = max(ead_rows, key=lambda r: (r[key] if r[key] is not None else -1))
                print(f"  no winner; best EAD {key}={best[key]} ({best['config']})")
        rep_path = os.path.join(args.root, "ead_sweep_report.json")
        with open(rep_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[ead_sweep] wrote report -> {rep_path}")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_summary(out_dir, tag):
    return _load(os.path.join(out_dir, f"summary__{tag}.json"))


def _kkey(summary, prefix):
    for k in summary:
        if k.startswith(f"{prefix}@") and k != f"{prefix}@1":
            return k
    return f"{prefix}@k"


def _val(summary, prefix):
    for k, v in summary.items():
        if k.startswith(f"{prefix}@") and k != f"{prefix}@1" and isinstance(v, (int, float)):
            return float(v)
    return None


if __name__ == "__main__":
    main()

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Minimal HumanEval+ / MBPP+ style reward.

The ground_truth string is a JSON dict::

    {
        "entry_point": "add",
        "tests": "<python source defining `check(candidate)` that asserts>",
        "prompt_header": "<optional snippet to prepend to the candidate>"
    }

We `exec` the candidate function plus the test harness in a forked subprocess
with a hard SIGALRM timeout, so this reward needs **no Docker and no firejail**.
The same multiprocessing+signal pattern is what `prime_code` uses in this repo.
"""

import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import platform
import queue as _queue_mod
import signal
import tempfile


# NFS-safe scratch root. multiprocessing.Manager() (and tempfile.TemporaryDirectory)
# default to tempfile.gettempdir(), which on shared clusters is often NFS-backed.
# On NFS, the Manager socket gets silly-renamed to '.nfs...' on cleanup and the
# finalizer raises `OSError: [Errno 16] Device or resource busy: '.nfsXXXX'` at
# interpreter shutdown. We avoid Manager entirely (using a Queue instead, which
# uses anonymous pipes), and we route TemporaryDirectory at a known-local path.
_DEFAULT_LOCAL_TMP_CANDIDATES = ["/dev/shm", "/tmp", "/var/tmp"]


def _pick_local_tmp() -> str:
    """Pick a non-NFS-backed scratch root for tempfile.TemporaryDirectory.

    Priority: $HUMANEVALPLUS_TMPDIR > $TMPDIR (if it looks local) > /dev/shm > /tmp.
    Falls back to tempfile.gettempdir() if none of the candidates are writable.
    """
    override = os.environ.get("HUMANEVALPLUS_TMPDIR")
    if override and os.path.isdir(override) and os.access(override, os.W_OK):
        return override
    for cand in _DEFAULT_LOCAL_TMP_CANDIDATES:
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return tempfile.gettempdir()


def _unsafe_execute(check_program: str, result_q, timeout: float):
    """Run `check_program` and push the resulting status string onto `result_q`."""
    try:
        with _create_tempdir():
            # tweak the runtime so the candidate can't trivially DoS us
            rmtree = __import__("shutil").rmtree
            rmdir = os.rmdir
            chdir = os.chdir
            try:
                with _swallow_io():
                    with _time_limit(timeout):
                        exec_globals = {}
                        exec(check_program, exec_globals)
                status = "passed"
            except TimeoutException:
                status = "timed out"
            except BaseException as e:  # noqa: BLE001
                status = f"failed: {type(e).__name__}: {e}"
            # restore tampered builtins / fs ops
            __import__("shutil").rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
    except BaseException as e:  # noqa: BLE001
        status = f"harness_error: {type(e).__name__}: {e}"
    try:
        result_q.put(status)
    except Exception:
        # Parent already gave up on us; nothing we can do.
        pass


def _check_correctness(check_program: str, timeout: float):
    """Run the test program in a separate process with a SIGALRM timeout.

    Uses a `multiprocessing.Queue` (anonymous pipes, no on-disk artefacts) so
    this works on NFS-only $TMPDIR clusters.
    """
    ctx = multiprocessing.get_context("fork") if hasattr(os, "fork") \
        else multiprocessing.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_unsafe_execute, args=(check_program, q, timeout))
    p.start()
    # Drain before join to avoid a pipe-buffer deadlock on large payloads.
    try:
        status = q.get(timeout=timeout + 2)
    except _queue_mod.Empty:
        status = "timed out"
    if p.is_alive():
        p.kill()
    p.join()
    # Close so the pipe isn't held open into the next call.
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass
    return status


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutException("timed out")
    if platform.system() != "Windows":
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if platform.system() != "Windows":
            signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


@contextlib.contextmanager
def _create_tempdir():
    """Cd into a scratch dir under a non-NFS root to avoid '.nfs*' cleanup errors."""
    root = _pick_local_tmp()
    with tempfile.TemporaryDirectory(dir=root) as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(cwd)


def _strip_reasoning_trace(completion: str) -> str:
    """Drop a leading <think>...</think> reasoning trace (Qwen3 dual-mode etc.).

    Long-reasoning models often emit scratch code inside the trace; we score the
    final answer, so we keep only what follows the closing tag. An unterminated
    trace (still 'thinking' when truncated) leaves nothing useful, so we return
    the original text in that case.
    """
    close = "</think>"
    if close in completion:
        return completion.split(close, 1)[1]
    return completion


def _extract_python_block(completion: str) -> str:
    """Pull a ```python ... ``` block (or a generic ``` ... ``` block) if present.

    Falls back to the raw completion so that models which forget the fence are
    not silently zeroed out.
    """
    completion = _strip_reasoning_trace(completion)
    if "```python" in completion:
        tail = completion.split("```python", 1)[1]
        return tail.split("```", 1)[0]
    if "```" in completion:
        # generic fenced block: take what's between the first pair of fences
        parts = completion.split("```")
        if len(parts) >= 3:
            return parts[1]
        # only an opening fence: take what follows it
        return parts[1] if len(parts) >= 2 else completion
    return completion


def compute_score(completion: str, ground_truth, continuous: bool = False, timeout: float = 8.0):
    """Return (success_float, [metadata]).

    `ground_truth` may be a dict or a JSON string with keys
    `entry_point`, `tests`, optional `prompt_header`.
    """
    try:
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        if not isinstance(ground_truth, dict):
            return 0.0, [{"error": "ground_truth is not a dict"}]
        entry_point = ground_truth.get("entry_point")
        tests_src = ground_truth.get("tests", "")
        prompt_header = ground_truth.get("prompt_header", "")
        if not entry_point or not tests_src:
            return 0.0, [{"error": "missing entry_point or tests"}]

        candidate = _extract_python_block(completion)
        check_program = (
            prompt_header
            + "\n"
            + candidate
            + "\n"
            + tests_src
            + f"\n\ncheck({entry_point})\n"
        )
        status = _check_correctness(check_program, timeout)
        passed = status == "passed"
        return (1.0 if passed else 0.0), [{"status": status}]
    except Exception as e:  # noqa: BLE001
        return 0.0, [{"error": str(e)}]

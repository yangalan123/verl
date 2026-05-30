# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import queue as _queue_mod
import sys
import traceback
from typing import Optional

from .testing_util import run_test


def _temp_run(sample, generation, debug, result_q, timeout):
    # NOTE: results are returned via a multiprocessing.Queue (anonymous pipes)
    # rather than a multiprocessing.Manager().list(). Manager spawns a helper
    # process that listens on a socket file under tempfile.gettempdir(); on
    # NFS-backed $TMPDIR that socket gets silly-renamed to '.nfsXXXX' on
    # cleanup and raises `OSError: [Errno 16] Device or resource busy` at
    # shutdown. Queue avoids any on-disk artefact.
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result_q.put((res, metadata))
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result_q.put(([-1 for i in range(len(sample["inputs"]))], {}))


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    ctx = multiprocessing.get_context("fork") if hasattr(os, "fork") else multiprocessing.get_context()
    result_q = ctx.Queue()
    p = ctx.Process(target=_temp_run, args=(in_outs, generation, debug, result_q, timeout))
    p.start()
    # Drain the queue *before* join: a large payload can fill the OS pipe
    # buffer, which would block the child's feeder thread and deadlock a join.
    try:
        res, metadata = result_q.get(timeout=timeout + 1)
        result = [res]
        metadata_list = [metadata]
    except _queue_mod.Empty:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = []
        if debug:
            print("global timeout")
    if p.is_alive():
        p.kill()
        # p.terminate()
    p.join()
    # Close so the pipe isn't held open into the next call.
    try:
        result_q.close()
        result_q.join_thread()
    except Exception:
        pass
    return result[0], metadata_list

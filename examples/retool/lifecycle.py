# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native Ray terminal status and cleanup of explicitly owned children."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TextIO

from examples.retool.execution_service import put


def require_job_success(env: Mapping[str, str], receipt: Path, deadline: float) -> None:
    """The submit CLI's exit code does not establish native job success."""
    submission_id = env["RETOOL_RAY_SUBMISSION_ID"]
    # Both SDK construction and status lookup perform HTTP calls without a
    # timeout. A child process makes the entire query killable and reapable.
    query = (
        "import json, sys; from ray.job_submission import JobSubmissionClient; "
        "status = JobSubmissionClient(sys.argv[1]).get_job_status(sys.argv[2]); "
        "print(json.dumps(getattr(status, 'value', status)))"
    )
    try:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("native Ray status query", 0)
        result = subprocess.run(
            [sys.executable, "-c", query, env["RAY_ADDRESS"], submission_id],
            env=dict(env),
            capture_output=True,
            text=True,
            check=True,
            timeout=remaining,
        )
        status = json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        put(receipt, dict(submission_id=submission_id, status="QUERY_FAILED", error=type(error).__name__))
        raise RuntimeError("Native Ray status query failed within the run deadline") from error
    put(receipt, dict(submission_id=submission_id, status=status))
    if status != "SUCCEEDED":
        raise RuntimeError(f"Native Ray job {submission_id} ended with {status}; expected SUCCEEDED")


def require_training_success(root: Path, mode: str, updates: int) -> None:
    """A final checkpoint and CLI exit zero alone do not prove completion."""
    stage = root / f"gpu_{mode}"
    wait = json.loads((stage / "training_actual_wait.json").read_text())
    assert wait.get("actual_parent_wait") is True and wait.get("returncode") == 0, (
        f"{mode.upper()} did not exit successfully"
    )
    job = json.loads((stage / "ray_job_status.json").read_text())
    assert job.get("submission_id") and job.get("status") == "SUCCEEDED", (
        f"{mode.upper()} lacks native Ray success evidence"
    )
    assert (root / f"checkpoint_{mode}.log").read_text().strip() == str(updates)


def close_owned_children(
    children: Sequence[tuple[str, subprocess.Popen, dict, TextIO]], helpers: ModuleType, timeout: float = 90
) -> list[dict]:
    """Attempt every child, escalate only identity-matched groups, reap and
    close."""
    ordered = list(reversed(children))
    errors = {}
    # Send TERM to all children before waiting for any one child.
    for name, proc, ident, _log in ordered:
        try:
            if proc.poll() is None and helpers.identity(proc.pid) == ident:
                os.killpg(ident["pgid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as error:
            errors[name] = repr(error)
    closed = []
    for name, proc, ident, log in ordered:
        record = dict(name=name, pid=proc.pid, identity=ident, actual_parent_wait=False)
        try:
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if helpers.identity(proc.pid) != ident:
                    raise RuntimeError("Child identity changed; refusing to signal its group") from None
                os.killpg(ident["pgid"], signal.SIGKILL)
                record["escalated_to_sigkill"] = True
                code = proc.wait(timeout=15)
            record.update(actual_parent_wait=True, returncode=code)
        except Exception as error:
            record["error"] = repr(error)
        finally:
            log.close()
        if name in errors:
            record["termination_error"] = errors[name]
        closed.append(record)
    return closed

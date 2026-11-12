# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native Ray terminal status and cleanup of explicitly owned children."""

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TextIO

from examples.retool.execution_service import put


def require_job_success(env: Mapping[str, str], receipt: Path) -> None:
    """The submit CLI's exit code does not establish native job success."""
    from ray.job_submission import JobSubmissionClient

    submission_id = env["RETOOL_RAY_SUBMISSION_ID"]
    status = JobSubmissionClient(env["RAY_ADDRESS"]).get_job_status(submission_id)
    status = getattr(status, "value", status)
    put(receipt, dict(submission_id=submission_id, status=status))
    if status != "SUCCEEDED":
        raise RuntimeError(f"Native Ray job {submission_id} ended with {status}; expected SUCCEEDED")


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

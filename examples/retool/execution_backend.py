# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Select and freeze a guarded backend before any student code is accepted."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from examples.retool import landlock_sandbox


def verify_guard(result: dict, policy: str) -> None:
    """Reject missing, wrong-profile, or incomplete execution isolation."""
    guard = result.get("guard")
    if result.get("guard_error") or not isinstance(guard, dict) or not guard:
        raise RuntimeError("Missing execution guard: " + diagnostic(result))
    try:
        cap_eff = int(guard.get("cap_eff", "1"), 16)
    except (ValueError, TypeError):
        cap_eff = -1
    if (
        guard.get("profile") != policy
        or guard.get("uid") != 65534
        or guard.get("gid") != 65534
        or cap_eff != 0
        or guard.get("no_new_privs") != 1
        or guard.get("seccomp_mode") != 2
        or guard.get("process_creation_allowed") is not False
    ):
        raise RuntimeError("Invalid execution guard: " + diagnostic(result))


def diagnostic(result: dict) -> str:
    """Retain useful bounded infrastructure context, not just
    AssertionError."""
    return repr(
        {
            "returncode": result.get("returncode"),
            "wrapper_returncode": result.get("wrapper_returncode"),
            "timed_out": result.get("timed_out"),
            "guard_error": result.get("guard_error"),
            "guard": result.get("guard"),
            "stderr": result.get("stderr", "")[-2048:],
        }
    )


@dataclass(frozen=True)
class Backend:
    name: str
    policy: str
    run_isolated: Callable[..., dict[str, Any]]
    close: Callable[[], None]
    startup: dict[str, Any]


def _build(name: str) -> Backend:
    if name == "landlock":
        return Backend(name, landlock_sandbox.POLICY, landlock_sandbox.run_isolated, lambda: None, {})
    from examples.retool.apptainer_sandbox import POLICY, Runtime

    runtime = Runtime()
    return Backend(name, POLICY, runtime.run_isolated, runtime.close, {})


def select_backend(root: Path, requested: str | None = None) -> Backend:
    """Only trusted bootstrap is probed; payload execution never switches
    backends."""
    requested = requested if requested is not None else os.environ.get("RETOOL_SANDBOX_BACKEND", "auto")
    if requested not in {"auto", "landlock", "apptainer"}:
        raise ValueError("RETOOL_SANDBOX_BACKEND must be auto, landlock, or apptainer")
    failures = {}
    for name in ("landlock", "apptainer") if requested == "auto" else (requested,):
        backend = None
        try:
            backend = _build(name)
            result = backend.run_isolated("", root, timeout=10, cpu_seconds=3, memory_mb=1024, probe_only=True)
            verify_guard(result, backend.policy)
            if result["returncode"] != 0 or result["timed_out"]:
                raise RuntimeError("Guard bootstrap failed: " + diagnostic(result))
            return Backend(
                backend.name,
                backend.policy,
                backend.run_isolated,
                backend.close,
                dict(
                    requested=requested,
                    selected=name,
                    profile=backend.policy,
                    prior_failures=failures,
                    actual_guard=result["guard"],
                    student_code_executed=False,
                ),
            )
        except (RuntimeError, OSError, ValueError) as error:
            if backend is not None:
                backend.close()
            failures[name] = str(error)
    raise RuntimeError("No requested sandbox backend is available: " + repr(failures))

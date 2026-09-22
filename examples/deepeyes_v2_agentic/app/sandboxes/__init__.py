# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pluggable sandbox subsystem.

Public API:

* :class:`BaseSandbox` / :class:`BaseSandboxSession` — backend ABCs.
* :class:`SandboxCapability` — frozenset members a backend can advertise.
* :class:`ExecutionResult` / :class:`ExecutionError` / :class:`FileEntry` —
  normalised return shapes.
* :class:`SandboxExecutor` — singleton concurrency controller; recipes call
  ``acquire_session(...)`` on it.
* :func:`get_sandbox_backend` — factory mirroring ``custom_rm_path`` resolution.
* :func:`register_backend` — third-party hook for registering backends out of
  tree.
"""

from typing import Any, Optional

from .backends import _REGISTRY, register_backend
from .base import (
    RELAX_SANDBOX_DEFAULT_CODE_TIMEOUT_S,
    RELAX_SANDBOX_DEFAULT_REQUEST_TIMEOUT_S,
    BaseSandbox,
    BaseSandboxSession,
    ExecutionError,
    ExecutionResult,
    FileEntry,
    SandboxCapability,
)
from .exceptions import (
    SandboxCapabilityError,
    SandboxCreateFailed,
    SandboxError,
    SandboxTimeout,
    SandboxTransientError,
)
from .executor import SandboxExecutor
from .retry import run_with_transient_retry


__all__ = [
    "BaseSandbox",
    "BaseSandboxSession",
    "ExecutionError",
    "ExecutionResult",
    "FileEntry",
    "RELAX_SANDBOX_DEFAULT_CODE_TIMEOUT_S",
    "RELAX_SANDBOX_DEFAULT_REQUEST_TIMEOUT_S",
    "SandboxCapability",
    "SandboxCapabilityError",
    "SandboxCreateFailed",
    "SandboxError",
    "SandboxExecutor",
    "SandboxTimeout",
    "SandboxTransientError",
    "get_sandbox_backend",
    "register_backend",
    "run_with_transient_retry",
]


def get_sandbox_backend(
    name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
    custom_path: Optional[str] = None,
) -> BaseSandbox:
    """Resolve a backend class and instantiate it with ``config``.

    ``custom_path`` (a dotted import path to a :class:`BaseSandbox` subclass)
    takes precedence over ``name`` and mirrors how ``--custom-rm-path`` wins
    over ``--rm-type`` in :class:`relax.engine.rewards.RewardExecutor`.
    """
    if custom_path:
        from relax.utils.misc import load_function

        cls: Any = load_function(custom_path)
    else:
        if name is None:
            raise ValueError("get_sandbox_backend requires either name or custom_path")
        if name not in _REGISTRY:
            raise ValueError(f"Unknown sandbox backend {name!r}; registered backends: {sorted(_REGISTRY)}")
        cls = _REGISTRY[name]
    return cls(**(config or {}))

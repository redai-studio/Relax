# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 11 straggler profiler: opt-in, non-blocking per-stage timing.

The package is inert unless ``RELAX_STRAGGLER_ENABLE`` is set. When it is off,
:func:`get_straggler_timers` returns ``None`` and the training backend keeps
Megatron's ``config.timers = None``, so the default deployment is bit-for-bit
the upstream one.

When it is on, the returned object replaces ``config.timers`` and records every
Megatron stage boundary (``forward-backward``, ``forward-compute``, the
optimizer and all-reduce phases, ...) without synchronising the device or adding
a collective. Reading those intervals back, comparing ranks and reporting them
is the job of the observer/detector/collector layers, which run off the training
thread.
"""

import os
from typing import Optional

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.runtime import StragglerRuntime


logger = get_logger(__name__)

#: Provenance guard (Task 11 acceptance): every process that imports the
#: profiler -- the Ray driver and every Megatron actor -- records which ``relax``
#: package it resolved. A run whose driver or actor resolves another worktree
#: (e.g. task4-pr) is invalid, and this is the line a manifest records to prove
#: it. Exactly one line per process at import time; no I/O beyond the logger's.
_RELAX_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info("straggler provenance: relax_root=%s module=%s", _RELAX_ROOT, __file__)

_TIMERS: Optional[StragglerTimers] = None
_RUNTIME: Optional[StragglerRuntime] = None
_INITIALIZED = False


def get_straggler_runtime() -> Optional[StragglerRuntime]:
    """Return the process-wide runtime, or ``None`` when the profiler is off.

    The runtime is what a platform integration polls: it carries the identity,
    the observer/collector/sender counters and the pending verdicts. It is
    built on first use, once per process, and never raises.
    """
    global _TIMERS, _RUNTIME, _INITIALIZED  # noqa: PLW0603 - process-wide singleton by design
    if _INITIALIZED:
        return _RUNTIME
    _INITIALIZED = True
    try:
        config = StragglerConfig.from_env()
        if not config.enabled:
            return None
        runtime = StragglerRuntime(config).start()
        _RUNTIME = runtime
        _TIMERS = runtime.timers
        logger.info(
            "straggler profiler enabled: role=%s, log_level=%d, event_pool=%d, window=%.1fs, collector=%s%s",
            runtime.role,
            config.timer_log_level,
            config.event_pool,
            config.window_seconds,
            config.collector_addr or "local",
            f", clamped={config.clamped}" if config.clamped else "",
        )
    except Exception:
        logger.warning("straggler profiler failed to initialise; Megatron timers stay disabled", exc_info=True)
        _TIMERS = None
        _RUNTIME = None
    return _RUNTIME


def get_straggler_timers() -> Optional[StragglerTimers]:
    """Return the process-wide ``config.timers`` replacement, or ``None``.

    ``None`` means the profiler is off (the default) or failed to start;
    callers must then leave Megatron's ``config.timers`` exactly as it was.
    Initialisation happens once per process and never raises.
    """
    get_straggler_runtime()
    return _TIMERS


def is_straggler_profiler_enabled() -> bool:
    """Return whether the profiler is active in this process."""
    return get_straggler_timers() is not None


def reset_straggler_state_for_tests() -> None:
    """Forget the cached singletons so a test can re-read the environment."""
    global _TIMERS, _RUNTIME, _INITIALIZED  # noqa: PLW0603 - process-wide singleton by design
    if _RUNTIME is not None:
        try:
            _RUNTIME.close()
        except Exception:
            pass
    _TIMERS = None
    _RUNTIME = None
    _INITIALIZED = False


__all__ = [
    "StragglerConfig",
    "StragglerRuntime",
    "StragglerTimers",
    "get_straggler_runtime",
    "get_straggler_timers",
    "is_straggler_profiler_enabled",
    "reset_straggler_state_for_tests",
]

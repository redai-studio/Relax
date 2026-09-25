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

from typing import Optional

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.megatron_timer_shim import StragglerTimers


logger = get_logger(__name__)

_TIMERS: Optional[StragglerTimers] = None
_INITIALIZED = False


def get_straggler_timers() -> Optional[StragglerTimers]:
    """Return the process-wide ``config.timers`` replacement, or ``None``.

    ``None`` means the profiler is off (the default) or failed to start;
    callers must then leave Megatron's ``config.timers`` exactly as it was.
    Initialisation happens once per process and never raises.
    """
    global _TIMERS, _INITIALIZED  # noqa: PLW0603 - process-wide singleton by design
    if _INITIALIZED:
        return _TIMERS
    _INITIALIZED = True
    try:
        config = StragglerConfig.from_env()
        if not config.enabled:
            return None
        _TIMERS = StragglerTimers(config)
        logger.info(
            "straggler profiler enabled: log_level=%d, event_pool=%d, window=%.1fs, collector=%s%s",
            config.timer_log_level,
            config.event_pool,
            config.window_seconds,
            config.collector_addr or "local",
            f", clamped={config.clamped}" if config.clamped else "",
        )
    except Exception:
        logger.warning("straggler profiler failed to initialise; Megatron timers stay disabled", exc_info=True)
        _TIMERS = None
    return _TIMERS


def is_straggler_profiler_enabled() -> bool:
    """Return whether the profiler is active in this process."""
    return get_straggler_timers() is not None


def reset_straggler_state_for_tests() -> None:
    """Forget the cached singleton so a test can re-read the environment."""
    global _TIMERS, _INITIALIZED  # noqa: PLW0603 - process-wide singleton by design
    _TIMERS = None
    _INITIALIZED = False


__all__ = [
    "StragglerConfig",
    "StragglerTimers",
    "get_straggler_timers",
    "is_straggler_profiler_enabled",
    "reset_straggler_state_for_tests",
]

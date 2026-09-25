# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A Megatron-compatible ``config.timers`` object that never blocks the host.

Megatron's own :class:`megatron.core.timers.Timer` calls
``torch.cuda.synchronize()`` on every ``start``/``stop`` (and a
``torch.distributed.barrier()`` when ``barrier=True``), which serialises the
step it is supposed to measure. Relax therefore disables timers by setting
``config.timers = None`` in ``relax/backends/megatron/model.py``, which also
means the framework has no per-stage timing at all.

This module supplies the drop-in replacement used by Task 11: the same call
shapes (``timers('forward-compute', log_level=2).start()`` /
``timers('forward-compute').stop()``) and the same log-level filtering, but
``start``/``stop`` only record a host timestamp and (optionally) a CUDA event on
the current stream. Reading the elapsed time happens later, off the training
thread, so the training path performs no GPU-CPU synchronisation and no
collective.

Design rules, in priority order:

1. Never raise into the schedule. Every entry point is wrapped and degrades to a
   no-op; failures are counted, not propagated.
2. Never synchronise. ``torch.cuda.synchronize`` and ``torch.distributed.barrier``
   are never called; the ``barrier`` argument is accepted and counted so the
   evidence can show how many upstream barriers were suppressed.
3. Never add a collective. Cross-rank aggregation is the collector's job and
   travels out of band (see :mod:`relax.utils.straggler.observer`).
"""

import time
from typing import Any, Callable, Dict, Optional, Protocol

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig


logger = get_logger(__name__)

#: Highest ``log_level`` Megatron uses; mirrors ``Timers._max_log_level``.
MAX_LOG_LEVEL = 2


class TimerSink(Protocol):
    """Destination for completed timing intervals.

    The shim hands over *unread* CUDA events: reading them back is the sink's
    job (and must happen off the training thread). ``acquire_interval`` may
    return ``None`` — meaning "host timing only" — when the event pool is
    exhausted or the device has no CUDA support.
    """

    def acquire_interval(self, name: str, log_level: int) -> Any:
        """Record the interval's start marker and return an opaque token."""
        ...

    def complete_interval(
        self,
        token: Any,
        name: str,
        log_level: int,
        host_start: float,
        host_end: float,
        barrier: bool,
    ) -> None:
        """Record the interval's end marker and hand the interval over."""
        ...


class NullTimerSink:
    """Sink that discards intervals; used until a collector is wired in."""

    def acquire_interval(self, name: str, log_level: int) -> None:
        """Return no token: the shim then measures host time only."""
        return None

    def complete_interval(
        self,
        token: Any,
        name: str,
        log_level: int,
        host_start: float,
        host_end: float,
        barrier: bool,
    ) -> None:
        """Discard the completed interval."""
        return None

    def stats(self) -> Dict[str, int]:
        """Return sink-level counters (none for the null sink)."""
        return {}


class _NoopTimer:
    """Mirrors :class:`megatron.core.timers.DummyTimer` for filtered levels.

    ``elapsed``/``active_time`` raise exactly as upstream does: call sites that
    reach them are asking for a level-filtered timer, and silently returning a
    number would hide that bug rather than reproduce Megatron's contract.
    """

    name = "dummy timer"

    def start(self, barrier: bool = False) -> None:
        """Ignore the call."""
        return None

    def stop(self, barrier: bool = False) -> None:
        """Ignore the call."""
        return None

    def reset(self) -> None:
        """Ignore the call."""
        return None

    def set_barrier_group(self, barrier_group: Any) -> None:
        """Ignore the call."""
        return None

    def set_elapsed(self, value: float) -> None:
        """Ignore the call."""
        return None

    def elapsed(self, reset: bool = True, barrier: bool = False) -> float:
        """Raise: a level-filtered timer must not be read."""
        raise RuntimeError(
            "dummy timer should not be used to calculate elapsed time, check if timer's log_level <= self._log_level."
        )

    def active_time(self) -> float:
        """Raise: a level-filtered timer must not be read."""
        raise RuntimeError(
            "active timer should not be used to calculate elapsed time, check if timer's log_level <= self._log_level."
        )


class StragglerTimerHandle:
    """One named Megatron timer: records intervals, reads nothing back."""

    __slots__ = (
        "_name",
        "_log_level",
        "_owner",
        "_started",
        "_host_start",
        "_token",
        "_last_host_duration",
        "_active_host_time",
    )

    def __init__(self, name: str, log_level: int, owner: "StragglerTimers") -> None:
        self._name = name
        self._log_level = log_level
        self._owner = owner
        self._started = False
        self._host_start = 0.0
        self._token: Any = None
        #: Host duration of the most recent completed interval. This is *not*
        #: device time: reading device time requires the event to complete and
        #: is done asynchronously by the sink.
        self._last_host_duration = 0.0
        self._active_host_time = 0.0

    @property
    def name(self) -> str:
        """Name of this timer, matching Megatron's ``Timer.name``."""
        return self._name

    def start(self, barrier: bool = False) -> None:
        """Open an interval without synchronising the device or the ranks."""
        owner = self._owner
        try:
            if self._started:
                owner.note_unbalanced("start")
                return
            if barrier:
                owner.note_ignored_barrier()
            host_start = owner.clock()
            token = owner.acquire(self._name, self._log_level)
            self._host_start = host_start
            self._token = token
            self._started = True
        except Exception:  # pragma: no cover - defensive, must never escape
            owner.note_sink_error()

    def stop(self, barrier: bool = False) -> None:
        """Close an interval and hand it to the sink without reading it
        back."""
        owner = self._owner
        try:
            if not self._started:
                owner.note_unbalanced("stop")
                return
            if barrier:
                owner.note_ignored_barrier()
            host_end = owner.clock()
            host_start = self._host_start
            token = self._token
            self._started = False
            self._host_start = 0.0
            self._token = None
            duration = host_end - host_start
            if duration >= 0.0:
                self._last_host_duration = duration
                self._active_host_time += duration
            owner.complete(token, self._name, self._log_level, host_start, host_end, bool(barrier))
        except Exception:  # pragma: no cover - defensive, must never escape
            owner.note_sink_error()

    def reset(self) -> None:
        """Clear the accumulated elapsed time, as Megatron's ``Timer.reset``.

        A running interval is closed first so its CUDA events return to the
        pool instead of leaking. ``_active_host_time`` deliberately survives,
        which is also upstream behaviour.
        """
        try:
            if self._started:
                self.stop()
            self._last_host_duration = 0.0
        except Exception:  # pragma: no cover - defensive, must never escape
            self._owner.note_sink_error()

    def set_barrier_group(self, barrier_group: Any) -> None:
        """Accept and ignore Megatron's barrier group: no barrier is issued."""
        return None

    def set_elapsed(self, value: float) -> None:
        """Set the last host duration, matching Megatron's ``set_elapsed``."""
        try:
            self._last_host_duration = float(value)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._owner.note_sink_error()

    def elapsed(self, reset: bool = True, barrier: bool = False) -> float:
        """Return the last completed **host** duration without reading the
        device.

        Megatron's implementation stops, reads, optionally resets and restarts
        the timer. None of that is possible without a synchronisation, so this
        diverges deliberately: the running interval is left untouched and the
        most recent completed host duration is returned. Megatron core never
        calls ``elapsed`` on a config timer (all 61 call sites use only
        ``start``/``stop``), so the divergence is unreachable in practice.
        """
        duration = self._last_host_duration
        if reset:
            self._last_host_duration = 0.0
        return duration

    def active_time(self) -> float:
        """Return the cumulative host time this timer has been active."""
        return self._active_host_time


class StragglerTimers:
    """Drop-in stand-in for :class:`megatron.core.timers.Timers`.

    The object is assigned to ``config.timers`` in the training backend, so it
    must satisfy exactly the interface Megatron exercises there: ``timers(name,
    log_level=...)`` returning a handle with ``start``/``stop``.
    """

    def __init__(
        self,
        config: StragglerConfig,
        sink: Optional[TimerSink] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._log_level = max(1, min(MAX_LOG_LEVEL, int(config.timer_log_level)))
        self._sink: TimerSink = sink if sink is not None else NullTimerSink()
        self._clock = clock
        self._timers: Dict[str, StragglerTimerHandle] = {}
        self._log_levels: Dict[str, int] = {}
        self._noop = _NoopTimer()
        self._counters: Dict[str, int] = {
            "intervals": 0,
            "ignored_barriers": 0,
            "unbalanced_start": 0,
            "unbalanced_stop": 0,
            "sink_errors": 0,
            "level_mismatch": 0,
            "call_errors": 0,
        }

    @property
    def log_level(self) -> int:
        """Highest log level this instance captures."""
        return self._log_level

    def __deepcopy__(self, memo: Dict[int, Any]) -> "StragglerTimers":
        """Return ``self``: Megatron deep-copies configs during construction.

        Megatron's attention/MoE modules run ``copy.deepcopy(self.config)``
        while they are being built, and its config logger runs
        ``dataclasses.asdict``. The profiler is process-wide, so sharing the
        instance is both correct and the only way such a copy can succeed: its
        event pool, locks and readout thread are not copyable.
        """
        return self

    def clock(self) -> float:
        """Return the monotonic host clock used for interval boundaries."""
        return self._clock()

    def __call__(self, name: str, log_level: Optional[int] = None) -> Any:
        """Return the handle for ``name``, mirroring ``Timers.__call__``."""
        try:
            if name in self._timers:
                if log_level is not None and int(log_level) != self._log_levels[name]:
                    # Megatron asserts here; the profiler must not be able to
                    # abort a step, so the mismatch is counted and the existing
                    # handle is reused.
                    self._counters["level_mismatch"] += 1
                return self._timers[name]
            resolved = MAX_LOG_LEVEL if log_level is None else int(log_level)
            if resolved > MAX_LOG_LEVEL or resolved > self._log_level:
                return self._noop
            handle = StragglerTimerHandle(name, resolved, self)
            self._timers[name] = handle
            self._log_levels[name] = resolved
            return handle
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["call_errors"] += 1
            return self._noop

    def acquire(self, name: str, log_level: int) -> Any:
        """Ask the sink for an interval token, degrading to host-only
        timing."""
        try:
            return self._sink.acquire_interval(name, log_level)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["sink_errors"] += 1
            return None

    def complete(
        self,
        token: Any,
        name: str,
        log_level: int,
        host_start: float,
        host_end: float,
        barrier: bool,
    ) -> None:
        """Hand a completed interval to the sink, counting the attempt."""
        self._counters["intervals"] += 1
        try:
            self._sink.complete_interval(token, name, log_level, host_start, host_end, barrier)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["sink_errors"] += 1

    def note_ignored_barrier(self) -> None:
        """Count a suppressed ``barrier=True`` request."""
        self._counters["ignored_barriers"] += 1

    def note_unbalanced(self, action: str) -> None:
        """Count an unpaired ``start``/``stop`` instead of asserting."""
        key = f"unbalanced_{action}"
        if key in self._counters:
            self._counters[key] += 1

    def note_sink_error(self) -> None:
        """Count a swallowed failure from the sink or a handle."""
        self._counters["sink_errors"] += 1

    def stats(self) -> Dict[str, Any]:
        """Return shim counters plus the sink's own counters."""
        stats: Dict[str, Any] = dict(self._counters)
        stats["timer_names"] = len(self._timers)
        stats["log_level"] = self._log_level
        try:
            stats["sink"] = dict(self._sink.stats())
        except Exception:  # pragma: no cover - defensive, must never escape
            stats["sink"] = {}
        return stats


__all__ = [
    "MAX_LOG_LEVEL",
    "NullTimerSink",
    "StragglerTimerHandle",
    "StragglerTimers",
    "TimerSink",
]

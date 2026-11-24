# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Non-synchronizing stand-in for a framework's named timers.

Megatron's ``megatron.core.timers.Timer.start/stop`` call
``torch.cuda.synchronize()`` on every bracket, which is why Relax sets
``config.timers = None`` everywhere. ``StragglerTimers`` speaks the same
protocol (``timers(name, log_level).start(barrier) / .stop(barrier)``) but only
records a pair of device timing events plus a CPU timestamp and hands them to a
sink. Nothing here blocks on the device, so DP grad-reduce overlap and PP
communication overlap are untouched. Which names exist and what they measure is
the adapter's business (``megatron_adapter``).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

from relax.utils.straggler.core.stats import FIELD_INDEX


class EventSink(Protocol):
    """Describe the owner a ``_SegmentTimer`` reports to (the recorder)."""

    def record_event(self) -> Any:
        """Return a timing event already recorded on the current stream, or
        ``None`` when recording is not possible right now (e.g. the stream is
        being captured into a CUDA graph)."""

    def push(self, segment_index: int, phase_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        """Hand over one completed bracket for later, lazy readout;
        ``phase_index`` is a PP phase field, a gradient sync's phase slot (see
        ``stats.DP_REDUCTION_PHASE``), -1, or a module bracket's layer number
        (negated for an MTP layer)."""

    def discard_event(self, event: Any) -> None:
        """Return an event whose bracket could not be completed."""

    def on_error(self, where: str, exc: BaseException) -> None:
        """Count an exception caught on the hot path instead of raising it into
        Megatron's schedule."""

    def on_degraded(self, reason: str) -> None:
        """Record a lasting gap in what is measured (see
        ``ProfilerHealth.mark_degraded``)."""


class EventPool:
    """Reuse timing-event objects so the hot path never allocates.

    An empty pool grows by one event per ``acquire`` and never shrinks;
    ``created`` counts every event ever built.
    """

    __slots__ = ("_free", "_factory", "created")

    def __init__(self, size: int, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._free: list[Any] = [factory() for _ in range(size)]
        self.created: int = size

    def acquire(self) -> Any:
        if self._free:
            return self._free.pop()
        self.created += 1
        return self._factory()

    def release(self, *events: Any) -> None:
        self._free.extend(events)

    def __len__(self) -> int:
        return len(self._free)


class _NullTimer:
    """The timer of every name that is not timed; takes any arguments."""

    __slots__ = ()

    def start(self, *args: Any, **kwargs: Any) -> None:
        return None

    def stop(self, *args: Any, **kwargs: Any) -> None:
        return None


class _SegmentTimer:
    """One timer name bound to one straggler segment and to the phase slot of
    its brackets (-1 if it has none).

    ``barrier`` is accepted for protocol compatibility and ignored: a barrier
    here would reintroduce exactly the synchronization we are avoiding. Any
    other argument is ignored too, and reported to ``sink.on_degraded``. A
    second ``start`` without ``stop`` is ignored rather than asserted, and an
    exception from the sink skips the bracket and goes to ``sink.on_error``, so
    the profiler can never take the training loop down.
    """

    __slots__ = ("_segment_index", "_phase_index", "_sink", "_start_event", "_start_cpu")

    def __init__(self, segment_index: int, phase_index: int, sink: EventSink) -> None:
        self._segment_index = segment_index
        self._phase_index = phase_index
        self._sink = sink
        self._start_event: Any = None
        self._start_cpu: float = 0.0

    def start(self, barrier: bool = False, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            self._sink.on_degraded(_unexpected_arguments("a timer's start()", args, kwargs))
        if self._start_event is not None:
            return
        try:
            self._start_event = self._sink.record_event()  # None while capturing -> bracket is skipped
        except Exception as exc:
            self._sink.on_error("timer", exc)
            return
        self._start_cpu = perf_counter()

    def stop(self, barrier: bool = False, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            self._sink.on_degraded(_unexpected_arguments("a timer's stop()", args, kwargs))
        start_event = self._start_event
        if start_event is None:
            return
        self._start_event = None
        try:
            end_event = self._sink.record_event()
            if end_event is None:
                self._sink.discard_event(start_event)
                return
            self._sink.push(
                self._segment_index,
                self._phase_index,
                start_event,
                end_event,
                (perf_counter() - self._start_cpu) * 1e3,
            )
        except Exception as exc:
            self._sink.discard_event(start_event)
            self._sink.on_error("timer", exc)

    @property
    def active(self) -> bool:
        return self._start_event is not None


class StragglerTimers:
    """Stand in for Megatron's ``Timers`` on ``ModelParallelConfig`` /
    ``OptimizerConfig``: timer name -> segment (``name_to_segment``) and
    phase slot (``name_to_phase``); ``ignored`` names are left untimed without
    a warning.

    Only the subset of the ``megatron.core.timers.Timers`` protocol that
    megatron.core itself uses is implemented (``__call__`` returning an object
    with ``start``/``stop``). Whatever falls outside it (an unknown name, an
    argument it does not take, ``log``/``write``) is ignored rather than raised
    into the training loop, and reported to ``sink.on_degraded`` so that it is
    noticed.
    """

    def __init__(
        self,
        sink: EventSink,
        name_to_segment: Mapping[str, str],
        name_to_phase: Mapping[str, int] | None = None,
        ignored: frozenset[str] = frozenset(),
    ) -> None:
        self._sink = sink
        self._name_to_segment = dict(name_to_segment)
        self._name_to_phase = dict(name_to_phase or {})
        self._ignored = ignored
        self._timers: dict[str, _SegmentTimer | _NullTimer] = {}
        self._null = _NullTimer()

    def __call__(
        self, name: str, log_level: int | None = None, *args: Any, **kwargs: Any
    ) -> _SegmentTimer | _NullTimer:
        """Return the timer bound to ``name``; untracked names share a no-op
        timer and ``log_level`` is ignored.

        A name that is neither mapped nor ignored is reported the first time it
        is seen, so stale or misspelled names are not silently swallowed.
        """
        if args or kwargs:
            self._sink.on_degraded(_unexpected_arguments("timers()", args, kwargs))
        timer = self._timers.get(name)
        if timer is None:
            segment = self._name_to_segment.get(name)
            if segment is not None:
                timer = _SegmentTimer(FIELD_INDEX[segment], self._name_to_phase.get(name, -1), self._sink)
            else:
                timer = self._null
                if name not in self._ignored:
                    self._sink.on_degraded(
                        f"unknown timer {name!r}: not mapped to a segment, so its time is not recorded; map it, or "
                        "list it as ignored, in the adapter that builds these timers"
                    )
            self._timers[name] = timer
        return timer

    def names_seen(self) -> frozenset[str]:
        """Every timer name asked for so far, timed or not."""
        return frozenset(self._timers)

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Report the call as a gap: no per-name totals are kept here."""
        self._sink.on_degraded(_unsupported("log"))

    def write(self, *args: Any, **kwargs: Any) -> None:
        """Report the call as a gap: no per-name totals are kept here."""
        self._sink.on_degraded(_unsupported("write"))


def _unexpected_arguments(call: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    passed = [f"{len(args)} positional"] if args else []
    passed += sorted(kwargs)
    return f"{call} got arguments the profiler does not take ({', '.join(passed)}); they are ignored"


def _unsupported(method: str) -> str:
    return f"Timers.{method}() was called, but the profiler keeps no per-name totals, so that caller gets nothing"

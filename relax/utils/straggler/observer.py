# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Off-thread readout of timing intervals produced by the timers shim.

The shim records, per interval, one host timestamp pair and (when CUDA is
available) a pair of CUDA events on the current stream. Reading an event back
requires it to be complete; doing that on the training thread would reintroduce
the very synchronisation the shim removed. So this module owns:

* a fixed pool of CUDA events, acquired at ``start`` and released once read;
* a bounded pending list of finished-but-unread intervals;
* one daemon thread that polls the events, reads the elapsed device time when it
  becomes available, and hands a :class:`TimingEnvelope` to a consumer.

Nothing here is allowed to block or raise on the training thread: pool
exhaustion degrades an interval to host timing only, a full queue drops with a
counted reason, and a failing consumer is counted rather than propagated.

The observer also owns an explicit lifecycle state machine: counted failures
move it ``active`` -> ``degraded`` -> ``disabled``. ``disabled`` is terminal
because a profiler that keeps flapping leaves gaps in the evidence that are
worse than the profiler simply being off. Once disabled, the recording entry
points are cheap counted no-ops: no CUDA event is created and no thread starts.
"""

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity, discover_identity
from relax.utils.straggler.protocol import WORKLOAD_FIELDS


logger = get_logger(__name__)


def _bounded_workload(value: Any) -> Optional[Dict[str, Any]]:
    """Sanitise a decoded workload mapping; never raises, never unbounded.

    Only the declared fields survive, only finite numeric values count, and the
    result is a fresh small dict. A hostile packet therefore cannot smuggle an
    arbitrarily large mapping into a detector sample.
    """
    if not isinstance(value, dict):
        return None
    workload: Dict[str, Any] = {}
    for field in WORKLOAD_FIELDS:
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            continue
        try:
            if not math.isfinite(float(number)):
                continue
        except (TypeError, ValueError):
            continue
        workload[field] = number
    return workload or None


#: Observer states, in escalation order. ``DISABLED`` is terminal: a profiler
#: that flaps between states produces gaps in the evidence that are worse than
#: no evidence at all, so once disabled it never returns to ``ACTIVE``.
STATE_ACTIVE = "active"
STATE_DEGRADED = "degraded"
STATE_DISABLED = "disabled"

#: Fixed key set of ``StragglerObserver._counters``: every key is created once,
#: at construction, so no drop reason or failure kind can grow the dict.
COUNTER_KEYS: Tuple[str, ...] = (
    "intervals",
    "device_intervals",
    "host_only_intervals",
    "dropped_pending_full",
    "dropped_output_full",
    "readout_timeouts",
    "observer_errors",
    "consumer_errors",
    "disabled_skips",
    "name_evictions",
)

#: Cumulative failures (full pending buffer, pool exhaustion, readout timeouts,
#: consumer/observer errors) after which the observer disables itself for good.
DISABLE_AFTER_FAILURES = 64

#: Hard cap on the transition-history deque: history is diagnostic, so a record
#: per failure would make the profiler the unbounded leak it watches for.
STATE_HISTORY_MAX = 64

#: Hard cap on the per-timer-name failure map. Timer names come from Megatron
#: call sites and are not enumerable, so the oldest name is evicted (and the
#: eviction counted) once the budget is reached.
NAME_BUDGET = 128


class EventBackend(Protocol):
    """Minimal device-timing surface the observer needs.

    A fake implementation makes the whole readout path testable on CPU.
    """

    def available(self) -> bool:
        """Return whether device timing can be used at all."""
        ...

    def create_event(self) -> Any:
        """Create one timing-capable device event."""
        ...

    def record(self, event: Any) -> None:
        """Record the event on the current stream (must not synchronise)."""
        ...

    def is_complete(self, event: Any) -> bool:
        """Return whether the event has completed (must not block)."""
        ...

    def elapsed_ms(self, start: Any, end: Any) -> float:
        """Return the device milliseconds between two completed events."""
        ...


class TorchCudaEventBackend:
    """CUDA-event backend; all torch access is lazy and guarded."""

    def available(self) -> bool:
        """Return whether CUDA is usable in this process."""
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def create_event(self) -> Any:
        """Create a timing-enabled CUDA event."""
        import torch

        return torch.cuda.Event(enable_timing=True)

    def record(self, event: Any) -> None:
        """Record on the current stream; never calls ``synchronize``."""
        event.record()

    def is_complete(self, event: Any) -> bool:
        """Query completion without blocking the host."""
        return bool(event.query())

    def elapsed_ms(self, start: Any, end: Any) -> float:
        """Read the device time between two completed events."""
        return float(start.elapsed_time(end))


class IntervalToken:
    """Events and metadata for one in-flight interval.

    ``seq`` is deliberately *not* set at acquire time: it is stamped by the
    observer when the interval completes. Megatron nests timers, so a start-
    order sequence would leave the enclosing interval behind the inner
    intervals it wraps on the wire, and the collector -- which requires a per-
    rank sequence that never decreases -- would drop it as out-of-order
    transport.
    """

    __slots__ = (
        "start_event",
        "end_event",
        "name",
        "log_level",
        "host_start",
        "host_end",
        "barrier",
        "seq",
    )

    def __init__(self, start_event: Any, end_event: Any) -> None:
        self.start_event = start_event
        self.end_event = end_event
        self.name = ""
        self.log_level = 0
        self.host_start = 0.0
        self.host_end = 0.0
        self.barrier = False
        #: Stamped by ``StragglerObserver.complete_interval``; 0 means the
        #: interval has not completed yet and must never be shipped.
        self.seq = 0


class EventPool:
    """Fixed-size CUDA event pool; exhaustion degrades, never blocks."""

    def __init__(self, backend: EventBackend, size: int) -> None:
        self._backend = backend
        self._size = max(2, int(size))
        self._free: List[Any] = []
        self._created = 0
        self.exhausted = 0

    def acquire_pair(self) -> Optional[Tuple[Any, Any]]:
        """Return ``(start_event, end_event)`` or ``None`` when exhausted."""
        while len(self._free) < 2 and self._created < self._size:
            self._free.append(self._backend.create_event())
            self._created += 1
        if len(self._free) < 2:
            self.exhausted += 1
            return None
        return self._free.pop(), self._free.pop()

    def release(self, start_event: Any, end_event: Any) -> None:
        """Return a pair to the pool."""
        if start_event is not None:
            self._free.append(start_event)
        if end_event is not None:
            self._free.append(end_event)

    @property
    def size(self) -> int:
        """Configured capacity in events."""
        return self._size

    def stats(self) -> Dict[str, int]:
        """Return pool occupancy counters."""
        return {"size": self._size, "created": self._created, "free": len(self._free), "exhausted": self.exhausted}


#: Measurement vocabulary for one interval. ``device`` means a CUDA event was
#: read back; ``host_only`` means only host timestamps exist.
MEASUREMENT_DEVICE = "device"
MEASUREMENT_HOST_ONLY = "host_only"
MEASUREMENT_UNKNOWN = "unknown"


def _measurement_kind(raw: Any, device_ms: Any) -> str:
    """Return the protocol's measurement classification for one interval.

    A payload that already carries ``device`` or ``host_only`` keeps it;
    anything else (including ``unknown``) is derived from whether a device
    duration exists, because every envelope has exactly one clock story. Never
    raises, so a malformed payload still yields a usable envelope.
    """
    if raw in (MEASUREMENT_DEVICE, MEASUREMENT_HOST_ONLY):
        return str(raw)
    return MEASUREMENT_HOST_ONLY if device_ms is None else MEASUREMENT_DEVICE


@dataclass(frozen=True)
class TimingEnvelope:
    """One observed interval, ready to ship to the collector."""

    run_id: str
    rank: int
    cohort: str
    label: str
    world_size: int
    name: str
    log_level: int
    seq: int
    host_start: float
    host_end: float
    device_ms: Optional[float]
    barrier: bool
    reason: str
    #: Optional per-interval workload counters (tokens/sequences/microbatches).
    #: Advisory: the detector reports the peer-relative difference next to the
    #: timing gap. It rides the wire so a collector can actually see it; only
    #: the declared finite-numeric fields survive :meth:`from_dict`.
    workload: Optional[Dict[str, Any]] = None
    #: Which clock produced this interval: ``device`` when a CUDA event was
    #: read back, ``host_only`` when only host timestamps exist. Stamped at
    #: build time so a reader never has to guess from ``device_ms``.
    measurement_kind: str = MEASUREMENT_UNKNOWN

    @property
    def host_ms(self) -> float:
        """Host-observed interval length in milliseconds."""
        return (self.host_end - self.host_start) * 1000.0

    def __post_init__(self) -> None:
        """Derive the clock classification when the builder did not set one.

        Every envelope has exactly one clock story: a readable CUDA event
        (``device``) or host timestamps only (``host_only``). Deriving it here
        means no construction path can ship an unclassified interval.
        """
        if self.measurement_kind == MEASUREMENT_UNKNOWN:
            object.__setattr__(
                self,
                "measurement_kind",
                _measurement_kind(self.measurement_kind, self.device_ms),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat JSON-friendly mapping."""
        return {
            "run_id": self.run_id,
            "rank": self.rank,
            "cohort": self.cohort,
            "label": self.label,
            "world_size": self.world_size,
            "name": self.name,
            "log_level": self.log_level,
            "seq": self.seq,
            "host_start": self.host_start,
            "host_end": self.host_end,
            "host_ms": self.host_ms,
            "device_ms": self.device_ms,
            "barrier": self.barrier,
            "reason": self.reason,
            "workload": self.workload,
            "measurement_kind": self.measurement_kind,
        }

    def to_json(self) -> str:
        """Serialise to one JSONL line without a trailing newline."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TimingEnvelope":
        """Rebuild an envelope received from another rank.

        ``host_ms`` is derived, so it is ignored on the way back in; a
        malformed payload still yields a usable envelope because every field
        falls back to a neutral value.
        """
        return cls(
            run_id=str(payload.get("run_id", "")),
            rank=int(payload.get("rank", -1)),
            cohort=str(payload.get("cohort", "")),
            label=str(payload.get("label", "")),
            world_size=int(payload.get("world_size", 1) or 1),
            name=str(payload.get("name", "")),
            log_level=int(payload.get("log_level", 0) or 0),
            seq=int(payload.get("seq", 0) or 0),
            host_start=float(payload.get("host_start", 0.0) or 0.0),
            host_end=float(payload.get("host_end", 0.0) or 0.0),
            device_ms=None if payload.get("device_ms") is None else float(payload["device_ms"]),
            barrier=bool(payload.get("barrier", False)),
            reason=str(payload.get("reason", "")),
            workload=_bounded_workload(payload.get("workload")),
            measurement_kind=_measurement_kind(payload.get("measurement_kind"), payload.get("device_ms")),
        )


class StragglerObserver:
    """Timer sink that reads intervals back without blocking the trainer."""

    def __init__(
        self,
        config: StragglerConfig,
        identity: Optional[RuntimeIdentity] = None,
        backend: Optional[EventBackend] = None,
        consumer: Optional[Callable[[TimingEnvelope], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
        readout_timeout_s: float = 30.0,
        poll_interval_s: float = 0.001,
        disable_after_failures: int = DISABLE_AFTER_FAILURES,
        state_history_max: int = STATE_HISTORY_MAX,
        name_budget: int = NAME_BUDGET,
    ) -> None:
        self._config = config
        self._identity = identity if identity is not None else discover_identity()
        self._backend: EventBackend = backend if backend is not None else TorchCudaEventBackend()
        self._consumer = consumer
        self._clock = clock
        self._readout_timeout_s = max(0.01, float(readout_timeout_s))
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._device_enabled = bool(self._backend.available())
        self._pool: Optional[EventPool] = EventPool(self._backend, config.event_pool) if self._device_enabled else None
        self._pending: Deque[IntervalToken] = deque()
        self._delivered: Deque[TimingEnvelope] = deque()
        self._cv = threading.Condition()
        self._delivered_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._closed = False
        self._seq = 0
        self._disable_after_failures = max(1, int(disable_after_failures))
        self._state_history_max = max(1, int(state_history_max))
        self._name_budget = max(1, int(name_budget))
        self._counters: Dict[str, int] = {key: 0 for key in COUNTER_KEYS}
        self._name_failures: Dict[str, int] = {}
        self._state_lock = threading.Lock()
        self._state = STATE_ACTIVE
        self._disable_reason: Optional[str] = None
        self._state_history: Deque[Dict[str, Any]] = deque()
        self._record_transition(STATE_ACTIVE, "init")

    @property
    def identity(self) -> RuntimeIdentity:
        """Identity used to label every envelope."""
        return self._identity

    @property
    def device_timing_enabled(self) -> bool:
        """Return whether CUDA events back the intervals."""
        return self._device_enabled

    @property
    def state(self) -> str:
        """Current lifecycle state: ``active``, ``degraded`` or
        ``disabled``."""
        return self._state

    @property
    def disable_reason(self) -> Optional[str]:
        """Reason the observer disabled itself, or ``None`` while it runs."""
        return self._disable_reason

    def _failure_total(self) -> int:
        """Return the cumulative count of counted failures."""
        pool_exhausted = self._pool.exhausted if self._pool is not None else 0
        return (
            self._counters["dropped_pending_full"]
            + self._counters["readout_timeouts"]
            + self._counters["consumer_errors"]
            + self._counters["observer_errors"]
            + pool_exhausted
        )

    def _note_name_failure(self, name: str) -> None:
        """Count one failure against a timer name, within a hard budget."""
        try:
            key = name or "unknown"
            failures = self._name_failures
            if key in failures:
                failures[key] += 1
                return
            if len(failures) >= self._name_budget:
                # Bounded: evict the oldest name and count the eviction, so the
                # audit can see that the budget was actually hit.
                failures.pop(next(iter(failures)))
                self._counters["name_evictions"] += 1
            failures[key] = 1
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["observer_errors"] += 1

    def _note_failure(self, reason: str) -> None:
        """Advance the state machine after a counted failure; never raises."""
        try:
            failures = self._failure_total()
            if failures <= 0:
                return
            with self._state_lock:
                if self._state == STATE_DISABLED:
                    return
                if self._state == STATE_ACTIVE:
                    self._transition_locked(STATE_DEGRADED, reason)
                if failures >= self._disable_after_failures:
                    self._transition_locked(STATE_DISABLED, reason)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["observer_errors"] += 1

    def _disable(self, reason: str) -> None:
        """Force the terminal ``disabled`` state; never raises."""
        try:
            with self._state_lock:
                if self._state == STATE_DISABLED:
                    return
                if self._state == STATE_ACTIVE:
                    self._transition_locked(STATE_DEGRADED, reason)
                self._transition_locked(STATE_DISABLED, reason)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["observer_errors"] += 1

    def _record_transition(self, state: str, reason: str) -> None:
        """Record one transition under the state lock; never raises."""
        try:
            with self._state_lock:
                self._transition_locked(state, reason)
        except Exception:  # pragma: no cover - defensive, must never escape
            self._counters["observer_errors"] += 1

    def _transition_locked(self, state: str, reason: str) -> None:
        """Apply and record one transition; caller holds ``_state_lock``."""
        self._state = state
        if state == STATE_DISABLED and self._disable_reason is None:
            self._disable_reason = reason
        self._state_history.append(
            {
                "state": state,
                "reason": reason,
                "time": float(self._clock()),
                "failures": self._failure_total(),
                "counters": dict(self._counters),
            }
        )
        while len(self._state_history) > self._state_history_max:
            self._state_history.popleft()

    def acquire_interval(self, name: str, log_level: int) -> Optional[IntervalToken]:
        """Record an interval start; returns ``None`` for host-only timing.

        In the terminal ``disabled`` state this is a cheap counted no-op: no
        event is created and no thread is started, and nothing can raise.
        """
        if self._state == STATE_DISABLED:
            self._counters["disabled_skips"] += 1
            return None
        if not self._device_enabled or self._pool is None:
            return None
        pair: Optional[Tuple[Any, Any]] = None
        token: Optional[IntervalToken] = None
        try:
            if not self._ensure_thread():
                if self._state == STATE_DISABLED:
                    self._counters["disabled_skips"] += 1
                return None
            pair = self._pool.acquire_pair()
            if pair is None:
                self._counters["host_only_intervals"] += 1
                self._note_name_failure(name)
                self._note_failure("pool_exhausted")
                return None
            token = IntervalToken(pair[0], pair[1])
            token.name = name
            token.log_level = log_level
            self._backend.record(token.start_event)
            return token
        except Exception:
            # A broken CUDA context surfaces here (event creation or recording)
            # and there is no recoverable action: the failure is counted and the
            # state machine escalates towards DISABLED, but it must not escape.
            self._counters["observer_errors"] += 1
            if token is not None:
                self._release(token)
            elif pair is not None:
                self._pool.release(pair[0], pair[1])
            self._note_failure("acquire_error")
            return None

    def _next_seq(self) -> int:
        """Hand every completed interval a distinct sequence number.

        The wire protocol keys idempotency on ``(run_id, topology_epoch, rank,
        sample_seq)``. Host-only intervals -- the whole no-CUDA path, and any
        interval whose event acquisition failed -- used to carry ``seq=0``, so
        every one of them after the first looked like a duplicate of the first
        and the degraded path silently stopped being judged at all. The sequence
        is allocated from :meth:`complete_interval` (never from
        :meth:`acquire_interval`) so it follows *completion* order: a nested
        outer timer completes after the inner timers it wraps, and a start-order
        sequence made the collector reject it as late. Only the training thread
        mutates this counter, so a plain increment is atomic enough and
        allocates nothing.
        """
        self._seq += 1
        return self._seq

    def complete_interval(
        self,
        token: Any,
        name: str,
        log_level: int,
        host_start: float,
        host_end: float,
        barrier: bool,
    ) -> None:
        """Record an interval end and queue it for off-thread readout.

        In the terminal ``disabled`` state this is a cheap counted no-op: a
        token acquired before disabling is released so its events cannot leak,
        and nothing here can raise.
        """
        if self._state == STATE_DISABLED:
            self._counters["disabled_skips"] += 1
            self._release(token)
            return
        try:
            self._counters["intervals"] += 1
            if token is None:
                self._counters["host_only_intervals"] += 1
                self._deliver(
                    self._build_envelope(
                        name=name,
                        log_level=log_level,
                        seq=self._next_seq(),
                        host_start=host_start,
                        host_end=host_end,
                        device_ms=None,
                        barrier=barrier,
                        reason="no_event_pair",
                    )
                )
                return
            token.name = name
            token.log_level = log_level
            token.host_start = host_start
            token.host_end = host_end
            token.barrier = barrier
            # Stamp the sequence now, at completion, so the sequence order is
            # the order intervals finish. Megatron's level-1 timer wraps the
            # level-2 ones, so a start-time sequence gives the outer interval the
            # lowest number but the latest completion, and the collector's
            # never-decreasing-per-rank check dropped every enclosing interval as
            # "late" transport noise. Completion order is also exactly the order
            # in which this training thread appends tokens to ``_pending``.
            token.seq = self._next_seq()
            self._backend.record(token.end_event)
            with self._cv:
                full = len(self._pending) >= self._config.queue_max
                if not full:
                    self._pending.append(token)
                    self._cv.notify_all()
            if full:
                self._counters["dropped_pending_full"] += 1
                self._note_name_failure(name)
                self._release(token)
                self._note_failure("pending_full")
        except Exception:
            self._counters["observer_errors"] += 1
            self._release(token)
            self._note_failure("complete_error")

    def _release(self, token: Any) -> None:
        """Return a token's events to the pool."""
        if token is None or self._pool is None:
            return
        try:
            self._pool.release(token.start_event, token.end_event)
            token.start_event = None
            token.end_event = None
        except Exception:
            self._counters["observer_errors"] += 1
            self._note_failure("release_error")

    def _ensure_thread(self) -> bool:
        """Start the readout thread once; ``False`` means it is unavailable.

        A thread that cannot be started, or that has died, disables the
        observer: nothing is left to drain the pending queue, so recording more
        intervals would only fill it up.
        """
        if self._state == STATE_DISABLED:
            return False
        thread = self._thread
        if thread is not None:
            if thread.is_alive():
                return True
            if not self._stopping and not self._closed:
                self._counters["observer_errors"] += 1
                self._disable("readout_thread_died")
            return False
        with self._cv:
            if self._thread is not None:
                return self._thread.is_alive()
            if self._stopping or self._closed:
                return False
            try:
                thread = threading.Thread(target=self._readout_loop, name="straggler-readout", daemon=True)
                thread.start()
                self._thread = thread
            except Exception:
                self._counters["observer_errors"] += 1
                self._disable("thread_start_failed")
                return False
        return True

    def _readout_loop(self) -> None:
        """Poll pending intervals, disabling the observer if the poll dies."""
        try:
            self._readout_until_stopped()
        except BaseException:
            # Any escape from the poll loop -- including a backend raising a
            # BaseException -- leaves the queue unserviced, so the observer
            # disables itself rather than accumulate unreadable intervals.
            try:
                self._counters["observer_errors"] += 1
                self._disable("readout_thread_died")
            except Exception:  # pragma: no cover - defensive, must never escape
                pass

    def _readout_until_stopped(self) -> None:
        """Body of the readout thread, separated so its death is observable."""
        while True:
            with self._cv:
                if not self._pending:
                    if self._stopping:
                        return
                    self._cv.wait(0.05)
                    continue
                token = self._pending.popleft()
                stopping = self._stopping
            if stopping:
                # Closing: never requeue, so the thread can actually exit.
                self._counters["readout_timeouts"] += 1
                self._deliver(self._envelope_from_token(token, None, "closed"))
                self._release(token)
                continue
            if self._process(token):
                with self._cv:
                    self._pending.append(token)
                    self._cv.notify_all()
                if self._poll_interval_s > 0:
                    time.sleep(self._poll_interval_s)

    def _process(self, token: IntervalToken) -> bool:
        """Read one interval; returns ``True`` when it should be retried."""
        if self._state == STATE_DISABLED:
            self._release(token)
            return False
        try:
            if self._backend.is_complete(token.end_event):
                device_ms = self._backend.elapsed_ms(token.start_event, token.end_event)
                self._counters["device_intervals"] += 1
                self._deliver(self._envelope_from_token(token, device_ms, "device"))
                self._release(token)
                return False
            if self._clock() - token.host_end > self._readout_timeout_s:
                self._counters["readout_timeouts"] += 1
                self._note_name_failure(token.name)
                self._deliver(self._envelope_from_token(token, None, "readout_timeout"))
                self._release(token)
                self._note_failure("readout_timeout")
                return False
            return True
        except Exception:
            self._counters["observer_errors"] += 1
            self._release(token)
            self._note_failure("readout_error")
            return False

    def _envelope_from_token(self, token: IntervalToken, device_ms: Optional[float], reason: str) -> TimingEnvelope:
        """Build an envelope from a completed token."""
        return self._build_envelope(
            name=token.name,
            log_level=token.log_level,
            seq=token.seq,
            host_start=token.host_start,
            host_end=token.host_end,
            device_ms=device_ms,
            barrier=token.barrier,
            reason=reason,
        )

    def _build_envelope(
        self,
        name: str,
        log_level: int,
        seq: int,
        host_start: float,
        host_end: float,
        device_ms: Optional[float],
        barrier: bool,
        reason: str,
    ) -> TimingEnvelope:
        """Build an envelope carrying this process's identity."""
        identity = self._identity
        return TimingEnvelope(
            run_id=identity.run_id,
            rank=identity.rank,
            cohort=identity.cohort,
            label=identity.label,
            world_size=identity.world_size,
            name=name,
            log_level=log_level,
            seq=seq,
            host_start=host_start,
            host_end=host_end,
            device_ms=device_ms,
            barrier=barrier,
            reason=reason,
            measurement_kind=MEASUREMENT_HOST_ONLY if device_ms is None else MEASUREMENT_DEVICE,
            workload=self._workload_from_context(),
        )

    @staticmethod
    def _workload_from_context() -> Optional[Dict[str, Any]]:
        """Read this rank's published local work for the current step.

        Runs on the straggler-readout thread, never the training thread. A
        missing or malformed context yields ``None`` rather than guessing, so a
        vehicle whose data path does not publish still ships valid envelopes.
        """
        try:
            from relax.utils.straggler.context import snapshot

            current = snapshot()
        except Exception:
            return None
        workload = {field: current.get(field) for field in WORKLOAD_FIELDS if isinstance(current.get(field), int)}
        return workload or None

    def _deliver(self, envelope: TimingEnvelope) -> None:
        """Hand an envelope to the consumer, or buffer it locally."""
        if self._consumer is not None:
            try:
                self._consumer(envelope)
            except Exception:
                self._counters["consumer_errors"] += 1
                self._note_name_failure(envelope.name)
                self._note_failure("consumer_error")
            return
        with self._delivered_lock:
            if len(self._delivered) >= self._config.queue_max:
                # No consumer means the local buffer is the only sink, so a full
                # buffer is expected back-pressure, not an observer failure: it
                # is counted but does not move the state machine.
                self._counters["dropped_output_full"] += 1
                return
            self._delivered.append(envelope)

    def drain(self) -> List[TimingEnvelope]:
        """Return and clear locally buffered envelopes."""
        with self._delivered_lock:
            envelopes = list(self._delivered)
            self._delivered.clear()
        return envelopes

    def start(self) -> None:
        """Start the readout thread explicitly (otherwise started on
        demand)."""
        self._ensure_thread()

    def close(self, timeout: float = 2.0) -> None:
        """Stop the readout thread, flushing what it can; idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        # Anything still pending is delivered with host timing only: losing the
        # interval entirely would bias the evidence towards the healthy ranks.
        with self._cv:
            remaining = list(self._pending)
            self._pending.clear()
        for token in remaining:
            try:
                self._counters["readout_timeouts"] += 1
                self._deliver(self._envelope_from_token(token, None, "closed"))
            except Exception:
                self._counters["observer_errors"] += 1
            finally:
                self._release(token)

    def stats(self) -> Dict[str, Any]:
        """Return counters plus lifecycle, pool and identity state."""
        stats: Dict[str, Any] = dict(self._counters)
        stats["device_timing_enabled"] = self._device_enabled
        stats["pending"] = len(self._pending)
        stats["delivered"] = len(self._delivered)
        stats["readout_thread_alive"] = bool(self._thread is not None and self._thread.is_alive())
        stats["pool"] = self._pool.stats() if self._pool is not None else {}
        stats["identity"] = self._identity.label
        stats["state"] = self._state
        stats["disable_reason"] = self._disable_reason
        with self._state_lock:
            stats["state_history"] = [dict(record) for record in self._state_history]
        stats["name_failures"] = dict(self._name_failures)
        return stats


__all__ = [
    "COUNTER_KEYS",
    "DISABLE_AFTER_FAILURES",
    "EventBackend",
    "EventPool",
    "IntervalToken",
    "NAME_BUDGET",
    "STATE_ACTIVE",
    "STATE_DEGRADED",
    "STATE_DISABLED",
    "STATE_HISTORY_MAX",
    "StragglerObserver",
    "TimingEnvelope",
    "TorchCudaEventBackend",
]

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
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity, discover_identity


logger = get_logger(__name__)


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
    """Events and metadata for one in-flight interval."""

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

    def __init__(self, start_event: Any, end_event: Any, seq: int) -> None:
        self.start_event = start_event
        self.end_event = end_event
        self.name = ""
        self.log_level = 0
        self.host_start = 0.0
        self.host_end = 0.0
        self.barrier = False
        self.seq = seq


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

    @property
    def host_ms(self) -> float:
        """Host-observed interval length in milliseconds."""
        return (self.host_end - self.host_start) * 1000.0

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
        }

    def to_json(self) -> str:
        """Serialise to one JSONL line without a trailing newline."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


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
        self._counters: Dict[str, int] = {
            "intervals": 0,
            "device_intervals": 0,
            "host_only_intervals": 0,
            "dropped_pending_full": 0,
            "dropped_output_full": 0,
            "readout_timeouts": 0,
            "observer_errors": 0,
            "consumer_errors": 0,
        }

    @property
    def identity(self) -> RuntimeIdentity:
        """Identity used to label every envelope."""
        return self._identity

    @property
    def device_timing_enabled(self) -> bool:
        """Return whether CUDA events back the intervals."""
        return self._device_enabled

    def acquire_interval(self, name: str, log_level: int) -> Optional[IntervalToken]:
        """Record an interval start; returns ``None`` for host-only timing."""
        if not self._device_enabled or self._pool is None:
            return None
        pair: Optional[Tuple[Any, Any]] = None
        token: Optional[IntervalToken] = None
        try:
            pair = self._pool.acquire_pair()
            if pair is None:
                self._counters["host_only_intervals"] += 1
                return None
            self._seq += 1
            token = IntervalToken(pair[0], pair[1], self._seq)
            token.name = name
            token.log_level = log_level
            self._backend.record(token.start_event)
            self._ensure_thread()
            return token
        except Exception:
            self._counters["observer_errors"] += 1
            if token is not None:
                self._release(token)
            elif pair is not None:
                self._pool.release(pair[0], pair[1])
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
        """Record an interval end and queue it for off-thread readout."""
        try:
            self._counters["intervals"] += 1
            if token is None:
                self._counters["host_only_intervals"] += 1
                self._deliver(
                    self._build_envelope(
                        name=name,
                        log_level=log_level,
                        seq=0,
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
            self._backend.record(token.end_event)
            with self._cv:
                if len(self._pending) >= self._config.queue_max:
                    self._counters["dropped_pending_full"] += 1
                    self._release(token)
                    return
                self._pending.append(token)
                self._cv.notify_all()
        except Exception:
            self._counters["observer_errors"] += 1
            self._release(token)

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

    def _ensure_thread(self) -> None:
        """Start the readout thread once, on first use."""
        if self._thread is not None:
            return
        with self._cv:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._readout_loop, name="straggler-readout", daemon=True)
            self._thread.start()

    def _readout_loop(self) -> None:
        """Poll pending intervals until closed and drained."""
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
        try:
            if self._backend.is_complete(token.end_event):
                device_ms = self._backend.elapsed_ms(token.start_event, token.end_event)
                self._counters["device_intervals"] += 1
                self._deliver(self._envelope_from_token(token, device_ms, "device"))
                self._release(token)
                return False
            if self._clock() - token.host_end > self._readout_timeout_s:
                self._counters["readout_timeouts"] += 1
                self._deliver(self._envelope_from_token(token, None, "readout_timeout"))
                self._release(token)
                return False
            return True
        except Exception:
            self._counters["observer_errors"] += 1
            self._release(token)
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
        )

    def _deliver(self, envelope: TimingEnvelope) -> None:
        """Hand an envelope to the consumer, or buffer it locally."""
        if self._consumer is not None:
            try:
                self._consumer(envelope)
            except Exception:
                self._counters["consumer_errors"] += 1
            return
        with self._delivered_lock:
            if len(self._delivered) >= self._config.queue_max:
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
        """Return counters plus pool and identity state."""
        stats: Dict[str, Any] = dict(self._counters)
        stats["device_timing_enabled"] = self._device_enabled
        stats["pending"] = len(self._pending)
        stats["delivered"] = len(self._delivered)
        stats["readout_thread_alive"] = bool(self._thread is not None and self._thread.is_alive())
        stats["pool"] = self._pool.stats() if self._pool is not None else {}
        stats["identity"] = self._identity.label
        return stats


__all__ = [
    "EventBackend",
    "EventPool",
    "IntervalToken",
    "StragglerObserver",
    "TimingEnvelope",
    "TorchCudaEventBackend",
]

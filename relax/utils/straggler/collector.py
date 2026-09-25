# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collector for straggler envelopes: aggregate, judge, report, persist.

One collector runs per training run (rank 0 by default). Every other rank ships
its envelopes to it over a same-host TCP socket, because cross-rank comparison
needs one process to see all ranks and a collective inside the step is exactly
what this design refuses to add.

The collector owns three jobs:

* feed :class:`~relax.utils.straggler.detector.StragglerDetector` and forward the
  verdicts it produces;
* persist envelopes and verdicts as JSONL when ``RELAX_STRAGGLER_OUTPUT_DIR`` is
  set, and emit a periodic one-line summary so the existing log/metrics pipeline
  carries the diagnosis without a new endpoint;
* expose :meth:`TimingCollector.status` for whatever the platform polls.

Transport is best effort by design: a full queue, a broken connection or a failed
write is counted and dropped, never retried in the training path. Losing
observations is acceptable; delaying a step is not.
"""

import json
import os
import socket
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from relax.utils.logging_utils import get_logger
from relax.utils.straggler import protocol
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import StragglerDetector, Verdict
from relax.utils.straggler.identity import RuntimeIdentity


logger = get_logger(__name__)

#: Bound on concurrently served sender connections.
MAX_CONNECTIONS = 64

#: Read buffer for a single JSONL line.
MAX_LINE_BYTES = 1 << 20

#: Lines buffered before a single write. The consumer runs on the training thread
#: when no device backend is available, so persistence must not open a file per
#: envelope.
WRITE_BATCH = 256

#: Hard cap on buffered-but-unwritten JSONL lines per file. The training thread
#: may never perform file I/O, so its ingest path can only append; the cap keeps
#: that buffer bounded (oldest lines are dropped and counted) until a background
#: thread or an explicit ``report``/``flush``/``close`` drains it.
MAX_PENDING_LINES = 65536

#: Bound and TTL of the collector's idempotency tracker. A reconnect can resend
#: what was already queued, and a slow link can deliver an older sample after a
#: newer one; both are dropped (and counted) before they can bias a window. The
#: bound keeps a flooding peer from growing the collector's memory.
DEDUP_MAX_ENTRIES = protocol.DEDUP_MAX_ENTRIES
DEDUP_TTL_S = protocol.DEDUP_TTL_S


def parse_address(address: str) -> Tuple[str, int]:
    """Split a ``host:port`` string; raises ``ValueError`` when malformed."""
    host, _, port = str(address).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"invalid collector address {address!r}; expected host:port")
    return host, int(port)


class TimingCollector:
    """Aggregates envelopes, judges them, and reports."""

    def __init__(
        self,
        config: StragglerConfig,
        identity: Optional[RuntimeIdentity] = None,
        on_verdict: Optional[Callable[[Verdict], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._identity = identity
        self._detector = StragglerDetector(config)
        self._on_verdict = on_verdict
        self._clock = clock
        self._dedup = protocol.BoundedDedup(max_entries=DEDUP_MAX_ENTRIES, ttl_s=DEDUP_TTL_S, clock=clock)
        self._verdicts: Deque[Verdict] = deque(maxlen=256)
        self._started_at = self._clock()
        self._last_report = self._started_at
        self._envelope_path: Optional[str] = None
        self._verdict_path: Optional[str] = None
        self._writers_ready = False
        self._pending: Dict[str, List[str]] = {}
        # Serialises the pending buffers and the counter updates they make. The
        # lock is never held across the file write itself, so a training-thread
        # caller of ``_append`` can never block on a background thread's I/O.
        self._write_lock = threading.Lock()
        # Serialises every mutation of the detector, the dedup tracker, the
        # collector counters and the retained verdicts. ``ingest`` runs on the
        # observer's readout thread *and* on one thread per accepted connection,
        # so without this the detector's window close could race a concurrent
        # ``_Window.add`` (dropping a window's verdicts with only a log line)
        # and the dedup tracker's expiry walk could raise and silently forget a
        # key. Re-entrant because ``_maybe_report`` re-enters ``status``.
        self._state_lock = threading.RLock()
        #: Thread that constructed the collector (the actor's training thread).
        #: Ingest from this thread must not write files.
        self._training_thread_id = threading.get_ident()
        self._counters: Dict[str, int] = {
            "envelopes": 0,
            "judged_packets": 0,
            "invalid_packets": 0,
            "duplicate_packets": 0,
            "late_packets": 0,
            "ingest_errors": 0,
            "flushed_lines": 0,
            "verdicts": 0,
            "reports": 0,
            "write_errors": 0,
            "verdict_callback_errors": 0,
            "pending_line_drops": 0,
        }
        if config.output_dir:
            self._open_writers(config.output_dir)

    def _open_writers(self, output_dir: str) -> None:
        """Create the JSONL sinks; a failure only disables persistence."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self._envelope_path = os.path.join(output_dir, "straggler_envelopes.jsonl")
            self._verdict_path = os.path.join(output_dir, "straggler_verdicts.jsonl")
            # Touch both files so an empty run is still visible in the evidence.
            for path in (self._envelope_path, self._verdict_path):
                with open(path, "a", encoding="utf-8"):
                    pass
            self._writers_ready = True
        except Exception:
            self._counters["write_errors"] += 1
            self._writers_ready = False
            logger.warning("straggler collector could not open output files", exc_info=True)

    def ingest(self, envelope: Any) -> List[Verdict]:
        """Absorb one envelope, persisting and judging it at most once.

        The gates run in a fixed order so a bad packet can never reach the
        detector: protocol validation first (an invalid packet is counted and
        dropped), then transport idempotency (a resend or an out-of-order
        packet is counted and dropped), and only then the detector. The method
        is the consumer of both the observer thread and the socket reader, so
        it never raises, whatever the input.
        """
        with self._state_lock:
            self._counters["envelopes"] += 1
            try:
                verdicts = self._ingest_checked(envelope)
            except Exception:
                self._counters["ingest_errors"] += 1
                verdicts = []
            try:
                self._maybe_report()
            except Exception:
                self._counters["ingest_errors"] += 1
        return verdicts

    def _ingest_checked(self, envelope: Any) -> List[Verdict]:
        """Run the validation/dedup gates and judge a packet that passes."""
        ok, _reason = protocol.validate(envelope)
        if not ok:
            self._counters["invalid_packets"] += 1
            return []
        decision = self._dedup.check(protocol.dedup_key(envelope))
        if decision == "duplicate":
            self._counters["duplicate_packets"] += 1
            return []
        if decision == "late":
            self._counters["late_packets"] += 1
            return []
        self._counters["judged_packets"] += 1
        self._append(self._envelope_path, getattr(envelope, "to_json", None))
        verdicts = self._detector.observe(envelope)
        self._handle(verdicts)
        return verdicts

    def _handle(self, verdicts: List[Verdict]) -> None:
        """Retain, persist and forward verdicts."""
        for verdict in verdicts:
            self._counters["verdicts"] += 1
            self._verdicts.append(verdict)
            self._append(self._verdict_path, verdict.to_json)
            if self._on_verdict is not None:
                try:
                    self._on_verdict(verdict)
                except Exception:
                    self._counters["verdict_callback_errors"] += 1

    def _append(self, path: Optional[str], serialiser: Any) -> None:
        """Buffer one JSONL line, counting (not raising) a failure.

        Appending is safe from any thread: the pending buffer and its counter
        updates are guarded by a lock that is released before any file is
        touched. A batch is only written when the caller is *not* the training
        thread, so the training path never performs file I/O; lines that pile up
        there are capped by :data:`MAX_PENDING_LINES` and drained by the
        background/close paths.
        """
        if not self._writers_ready or not path or serialiser is None:
            return
        try:
            line = serialiser() if callable(serialiser) else str(serialiser)
        except Exception:
            with self._write_lock:
                self._counters["write_errors"] += 1
            return
        with self._write_lock:
            pending = self._pending.setdefault(path, [])
            pending.append(line)
            overflow = len(pending) - MAX_PENDING_LINES
            if overflow > 0:
                # Oldest-first eviction keeps memory flat on a run whose
                # consumer is the training thread (no device backend).
                del pending[:overflow]
                self._counters["pending_line_drops"] += overflow
            batch_ready = len(pending) >= WRITE_BATCH and not self._on_training_thread()
        if batch_ready:
            self._flush_path(path)

    def _on_training_thread(self) -> bool:
        """Return whether the caller is the thread that built the collector."""
        return threading.get_ident() == self._training_thread_id

    def _flush_path(self, path: str) -> None:
        """Write the buffered lines of one file in a single call.

        The buffer is swapped out under the lock and written outside it, so two
        writers can never serialise the same lines twice and a training-thread
        ``_append`` can never block behind the write.
        """
        with self._write_lock:
            pending = self._pending.get(path)
            if not pending:
                return
            self._pending[path] = []
        self._write_batch(path, pending)

    def _write_batch(self, path: str, lines: List[str]) -> None:
        """Append one already-swapped batch to its file, counting failures."""
        if not lines:
            return
        written = False
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            written = True
        except Exception:
            with self._write_lock:
                self._counters["write_errors"] += 1
                if self._counters["write_errors"] > 100:
                    self._writers_ready = False
        if written:
            # Two concurrent flushes would otherwise lose a ``+=`` here even
            # though every line reached the file.
            with self._write_lock:
                self._counters["flushed_lines"] += len(lines)

    def _flush_writers(self) -> None:
        """Flush every buffered file."""
        for path in list(self._pending):
            self._flush_path(path)

    def _maybe_report(self) -> None:
        """Emit the periodic summary line when the interval has elapsed."""
        now = self._clock()
        if now - self._last_report < self._config.report_interval_seconds:
            return
        self._last_report = now
        self._counters["reports"] += 1
        # Ingest can run on the training thread (the no-device path), so the
        # periodic summary must not touch a file; only the explicit/close path
        # is allowed to flush.
        self.summary()

    def report(self, flush: bool = True) -> Dict[str, Any]:
        """Log and return the current summary.

        Args:
            flush: When ``True`` (the explicit diagnostic and shutdown path)
                buffered JSONL lines are written first. Callers on the training
                thread must use :meth:`summary`, which never writes.
        """
        if flush:
            self._flush_writers()
        status = self.status()
        active = status["active_stragglers"]
        logger.info(
            "straggler[%s]: envelopes=%d judged=%d invalid=%d duplicate=%d late=%d "
            "windows=%d verdicts=%d stragglers=%d active=%s",
            "rank0" if self._identity is None else self._identity.label,
            status["envelopes"],
            status["judged_packets"],
            status["invalid_packets"],
            status["duplicate_packets"],
            status["late_packets"],
            status["windows_closed"],
            status["verdicts"],
            status["stragglers_reported"],
            active if active else "none",
        )
        for entry in active:
            logger.info("straggler active: %s", entry)
        return status

    def summary(self) -> Dict[str, Any]:
        """Return the summary without flushing persistence.

        This is the training-thread-safe accessor: it performs no file I/O, so
        a per-rollout metrics read cannot block the step. The buffered lines
        stay pending until the batch bound, an explicit :meth:`report` or
        :meth:`flush`/close.
        """
        return self.report(flush=False)

    def drain_verdicts(self) -> List[Verdict]:
        """Return and clear verdicts not yet consumed by the caller."""
        with self._state_lock:
            verdicts = list(self._verdicts)
            self._verdicts.clear()
        return verdicts

    def flush(self) -> List[Verdict]:
        """Close open windows, handle the verdicts and persist everything."""
        with self._state_lock:
            verdicts = self._detector.flush()
            self._handle(verdicts)
        # The file write stays outside the state lock: a training-thread ingest
        # must never wait behind background I/O.
        self._flush_writers()
        return verdicts

    def status(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot for logs, TUI or metrics."""
        with self._state_lock:
            detector_stats = self._detector.stats()
            status: Dict[str, Any] = dict(detector_stats)
            status.update(self._counters)
            status["active_stragglers"] = self._detector.active_stragglers()
            status["dedup"] = self._dedup.stats()
            status["uptime_s"] = self._clock() - self._started_at
            with self._write_lock:
                status["pending_lines"] = {path: len(lines) for path, lines in self._pending.items()}
            status["envelope_path"] = self._envelope_path
            status["verdict_path"] = self._verdict_path
            status["identity"] = None if self._identity is None else self._identity.label
        return status


class EnvelopeSender:
    """Ships envelopes to the collector from a background thread."""

    def __init__(
        self,
        address: str,
        queue_max: int = 4096,
        reconnect_interval_s: float = 1.0,
    ) -> None:
        self._address = parse_address(address)
        self._queue: Deque[str] = deque()
        self._queue_max = max(1, int(queue_max))
        self._cv = threading.Condition()
        self._reconnect_interval_s = max(0.1, float(reconnect_interval_s))
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._closed = False
        self._counters: Dict[str, int] = {"queued": 0, "sent": 0, "dropped_queue_full": 0, "send_errors": 0}

    def send(self, envelope: Any) -> None:
        """Queue one envelope as a versioned packet; never blocks the
        caller."""
        try:
            line = self._encode(envelope)
        except Exception:
            self._counters["send_errors"] += 1
            return
        with self._cv:
            if len(self._queue) >= self._queue_max:
                self._counters["dropped_queue_full"] += 1
                return
            self._queue.append(line)
            self._counters["queued"] += 1
            self._ensure_thread()
            self._cv.notify_all()

    @staticmethod
    def _encode(envelope: Any) -> str:
        """Serialise one envelope, stamping the protocol version on the wire.

        The receiver rejects a decoded dict that does not declare its schema,
        so the sender is the one place that must add it: doing it here keeps
        the observer (which owns the envelope dataclass) unaware of the
        transport.
        """
        to_dict = getattr(envelope, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
        else:
            to_json = getattr(envelope, "to_json", None)
            payload = json.loads(to_json() if callable(to_json) else str(envelope))
        if not isinstance(payload, dict):
            raise ValueError("envelope does not serialise to a mapping")
        payload.setdefault("schema_version", protocol.SCHEMA_VERSION)
        payload.setdefault("protocol", protocol.PROTOCOL_NAME)
        return json.dumps(payload, separators=(",", ":"))

    def _ensure_thread(self) -> None:
        """Start the sender thread once."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="straggler-sender", daemon=True)
        self._thread.start()

    def _connect(self) -> Optional[socket.socket]:
        """Return a connected socket, or ``None`` while the collector is
        away."""
        try:
            sock = socket.create_connection(self._address, timeout=self._reconnect_interval_s)
            sock.settimeout(None)
            return sock
        except Exception:
            return None

    def _run(self) -> None:
        """Drain the queue, reconnecting with a bounded delay."""
        while True:
            with self._cv:
                if not self._queue:
                    if self._stopping:
                        return
                    self._cv.wait(0.05)
                    continue
                line = self._queue.popleft()
            if self._socket is None:
                self._socket = self._connect()
                if self._socket is None:
                    self._counters["send_errors"] += 1
                    time.sleep(self._reconnect_interval_s)
                    with self._cv:
                        self._queue.appendleft(line)
                    continue
            try:
                self._socket.sendall((line + "\n").encode("utf-8"))
                self._counters["sent"] += 1
            except Exception:
                self._counters["send_errors"] += 1
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
                with self._cv:
                    self._queue.appendleft(line)

    def close(self, timeout: float = 2.0) -> None:
        """Stop the sender thread; idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def stats(self) -> Dict[str, Any]:
        """Return queue and connection counters."""
        return {
            **self._counters,
            "pending": len(self._queue),
            "queue_max": self._queue_max,
            "connected": self._socket is not None,
            "address": f"{self._address[0]}:{self._address[1]}",
        }


class EnvelopeReceiver:
    """Accepts envelopes from the other ranks and hands them to a callback."""

    def __init__(
        self,
        address: str,
        on_envelope: Callable[[Any], None],
        bind_host: Optional[str] = None,
    ) -> None:
        host, port = parse_address(address)
        self._bind = (bind_host or host, port)
        self._on_envelope = on_envelope
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._connections: List[threading.Thread] = []
        self._closed = False
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "connections": 0,
            "lines": 0,
            "parse_errors": 0,
            "invalid_packets": 0,
            "callback_errors": 0,
            "accept_errors": 0,
        }

    def start(self) -> None:
        """Bind and start accepting; a bind failure is counted, not raised."""
        if self._thread is not None:
            return
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(self._bind)
            server.listen(MAX_CONNECTIONS)
        except Exception:
            self._counters["accept_errors"] += 1
            logger.warning("straggler collector could not bind %s:%d", self._bind[0], self._bind[1], exc_info=True)
            return
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, name="straggler-accept", daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        """Accept connections until closed."""
        while not self._closed and self._server is not None:
            try:
                connection, _ = self._server.accept()
            except Exception:
                if self._closed:
                    return
                self._counters["accept_errors"] += 1
                time.sleep(0.05)
                continue
            with self._lock:
                # Finished reader threads linger in this list; prune them before
                # applying the cap, otherwise the cap counts *lifetime*
                # connections and a long run (or a reconnecting sender) is
                # refused forever once MAX_CONNECTIONS have ever been accepted.
                self._connections = [thread for thread in self._connections if thread.is_alive()]
                if len(self._connections) >= MAX_CONNECTIONS:
                    try:
                        connection.close()
                    except Exception:
                        pass
                    self._counters["accept_errors"] += 1
                    continue
                self._counters["connections"] += 1
                thread = threading.Thread(
                    target=self._read_loop, args=(connection,), name="straggler-recv", daemon=True
                )
                self._connections.append(thread)
                thread.start()

    def _read_loop(self, connection: socket.socket) -> None:
        """Read JSONL lines from one connection until it closes."""
        try:
            with connection.makefile("rb") as stream:
                while not self._closed:
                    line = stream.readline(MAX_LINE_BYTES)
                    if not line:
                        return
                    self._handle_line(line)
        except Exception:
            self._counters["parse_errors"] += 1
        finally:
            try:
                connection.close()
            except Exception:
                pass

    def _handle_line(self, line: bytes) -> None:
        """Decode one line, validate it, and forward it to the callback.

        The receiver is transport, not a judge: it counts protocol violations
        (so a version skew is visible) but still hands the decoded dict on,
        because the collector owns the gate that decides whether a packet is
        judged. Neither a parse failure nor an invalid packet may raise here.
        """
        try:
            payload = json.loads(line.decode("utf-8"))
        except Exception:
            self._counters["parse_errors"] += 1
            return
        self._counters["lines"] += 1
        ok, _reason = protocol.validate(payload)
        if not ok:
            self._counters["invalid_packets"] += 1
        try:
            self._on_envelope(payload)
        except Exception:
            self._counters["callback_errors"] += 1

    @property
    def address(self) -> Tuple[str, int]:
        """Actual bound address (useful when port 0 was requested)."""
        if self._server is not None:
            try:
                host, port = self._server.getsockname()
                return str(host), int(port)
            except Exception:
                pass
        return self._bind

    def close(self, timeout: float = 2.0) -> None:
        """Stop accepting and join the reader threads; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        for thread in (self._thread, *self._connections):
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(0.0, float(timeout)))

    def stats(self) -> Dict[str, Any]:
        """Return connection and parsing counters."""
        return {
            **self._counters,
            "listening": self._server is not None,
            "address": f"{self.address[0]}:{self.address[1]}",
        }


__all__ = [
    "DEDUP_MAX_ENTRIES",
    "DEDUP_TTL_S",
    "EnvelopeReceiver",
    "EnvelopeSender",
    "MAX_PENDING_LINES",
    "TimingCollector",
    "parse_address",
]

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
        self._verdicts: Deque[Verdict] = deque(maxlen=256)
        self._started_at = self._clock()
        self._last_report = self._started_at
        self._envelope_path: Optional[str] = None
        self._verdict_path: Optional[str] = None
        self._writers_ready = False
        self._pending: Dict[str, List[str]] = {}
        self._counters: Dict[str, int] = {
            "envelopes": 0,
            "flushed_lines": 0,
            "verdicts": 0,
            "reports": 0,
            "write_errors": 0,
            "verdict_callback_errors": 0,
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
        """Absorb one envelope, persist it and return any new verdicts."""
        self._counters["envelopes"] += 1
        self._append(self._envelope_path, getattr(envelope, "to_json", None))
        verdicts = self._detector.observe(envelope)
        self._handle(verdicts)
        self._maybe_report()
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
        """Buffer one JSONL line, counting (not raising) a failure."""
        if not self._writers_ready or not path or serialiser is None:
            return
        try:
            line = serialiser() if callable(serialiser) else str(serialiser)
        except Exception:
            self._counters["write_errors"] += 1
            return
        pending = self._pending.setdefault(path, [])
        pending.append(line)
        if len(pending) >= WRITE_BATCH:
            self._flush_path(path)

    def _flush_path(self, path: str) -> None:
        """Write the buffered lines of one file in a single call."""
        pending = self._pending.get(path)
        if not pending:
            return
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(pending) + "\n")
            self._counters["flushed_lines"] += len(pending)
        except Exception:
            self._counters["write_errors"] += 1
            if self._counters["write_errors"] > 100:
                self._writers_ready = False
        finally:
            pending.clear()

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
        self.report()

    def report(self) -> Dict[str, Any]:
        """Log and return the current summary."""
        self._flush_writers()
        status = self.status()
        active = status["active_stragglers"]
        logger.info(
            "straggler[%s]: envelopes=%d windows=%d verdicts=%d stragglers=%d active=%s",
            "rank0" if self._identity is None else self._identity.label,
            status["envelopes"],
            status["windows_closed"],
            status["verdicts"],
            status["stragglers_reported"],
            active if active else "none",
        )
        for entry in active:
            logger.info("straggler active: %s", entry)
        return status

    def drain_verdicts(self) -> List[Verdict]:
        """Return and clear verdicts not yet consumed by the caller."""
        verdicts = list(self._verdicts)
        self._verdicts.clear()
        return verdicts

    def flush(self) -> List[Verdict]:
        """Close open windows, handle the verdicts and persist everything."""
        verdicts = self._detector.flush()
        self._handle(verdicts)
        self._flush_writers()
        return verdicts

    def status(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot for logs, TUI or metrics."""
        detector_stats = self._detector.stats()
        status: Dict[str, Any] = dict(detector_stats)
        status.update(self._counters)
        status["active_stragglers"] = self._detector.active_stragglers()
        status["uptime_s"] = self._clock() - self._started_at
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
        """Queue one envelope; never blocks the caller."""
        try:
            line = envelope.to_json()
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
        """Decode and forward one envelope."""
        try:
            payload = json.loads(line.decode("utf-8"))
        except Exception:
            self._counters["parse_errors"] += 1
            return
        self._counters["lines"] += 1
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
    "EnvelopeReceiver",
    "EnvelopeSender",
    "TimingCollector",
    "parse_address",
]

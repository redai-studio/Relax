# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wires the profiler together inside one training process.

The layout depends only on the rank and on ``RELAX_STRAGGLER_COLLECTOR_ADDR``:

* address unset — every process keeps its own collector, so the observation is
  rank-local (a single-rank run still gets full stage timings, a multi-rank run
  reports ``uncertain`` because it has no peer to compare against);
* address set, rank 0 — this process is the collector: it binds the socket and
  judges every rank's envelopes;
* address set, other ranks — this process ships its envelopes and judges nothing.

Building is fail-open: if any stage cannot start, the runtime reports ``None``
timers and the training backend keeps Megatron's ``config.timers = None``.
"""

import atexit
import json
import os
import re
import sys
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, Optional

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.collector import EnvelopeReceiver, EnvelopeSender, TimingCollector, parse_address
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity, discover_identity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope


logger = get_logger(__name__)

#: Upper bound on the periodic status-writer cadence. The writer runs on its own
#: daemon thread, so the cap only bounds how stale ``runtime_status.json`` can be
#: when the process is killed without a graceful ``close()`` (Ray SIGTERMs the
#: actors at job end, and atexit does not run for a signal death).
STATUS_WRITE_MAX_INTERVAL_S = 10.0


def _sanitise(value: Any) -> str:
    """Turn a run id or identity label into one safe path component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


class StragglerRuntime:
    """Owns the profiler components of one process and their shutdown."""

    def __init__(
        self,
        config: StragglerConfig,
        identity: Optional[RuntimeIdentity] = None,
        register_atexit: bool = True,
    ) -> None:
        self._config = config
        self._identity = identity if identity is not None else discover_identity()
        self._timers: Optional[StragglerTimers] = None
        self._observer: Optional[StragglerObserver] = None
        self._collector: Optional[TimingCollector] = None
        self._sender: Optional[EnvelopeSender] = None
        self._receiver: Optional[EnvelopeReceiver] = None
        self._started = False
        self._closed = False
        self._errors = 0
        self._status_thread: Optional[threading.Thread] = None
        self._status_stop = threading.Event()
        if register_atexit:
            atexit.register(self.close)

    @property
    def config(self) -> StragglerConfig:
        """Configuration this runtime was built from."""
        return self._config

    @property
    def identity(self) -> RuntimeIdentity:
        """Identity of the process owning this runtime."""
        return self._identity

    @property
    def timers(self) -> Optional[StragglerTimers]:
        """The object to assign to ``config.timers``, or ``None``."""
        return self._timers

    @property
    def collector(self) -> Optional[TimingCollector]:
        """The local collector, when this process owns one."""
        return self._collector

    @property
    def observer(self) -> Optional[StragglerObserver]:
        """The observer reading intervals back off the training thread."""
        return self._observer

    @property
    def sender(self) -> Optional[EnvelopeSender]:
        """The transport used when this rank only ships envelopes."""
        return self._sender

    @property
    def receiver(self) -> Optional[EnvelopeReceiver]:
        """The transport used when this rank hosts the collector."""
        return self._receiver

    @property
    def role(self) -> str:
        """``collector``, ``sender`` or ``local``, for the log and the
        evidence."""
        if self._receiver is not None or (self._collector is not None and not self._config.collector_addr):
            return "collector" if self._receiver is not None else "local"
        if self._sender is not None:
            return "sender"
        return "unstarted"

    def start(self) -> "StragglerRuntime":
        """Build the pipeline; returns ``self`` even when it fails."""
        if self._started:
            return self
        self._started = True
        try:
            consumer = self._build_sink()
            observer = StragglerObserver(self._config, identity=self._identity, consumer=consumer)
            self._observer = observer
            self._timers = StragglerTimers(self._config, sink=observer)
            logger.info(
                "straggler profiler started: role=%s identity=%s collector=%s",
                self.role,
                self._identity.label,
                self._config.collector_addr or "local",
            )
        except Exception:
            self._errors += 1
            self._timers = None
            logger.warning("straggler profiler failed to start; Megatron timers stay disabled", exc_info=True)
        if self._timers is not None:
            self._start_status_writer()
        return self

    def _start_status_writer(self) -> None:
        """Persist the counters periodically, off the training thread.

        ``close()`` writes the status files, but Ray tears the actors down with
        SIGTERM at job end, so atexit never runs and no status file was
        produced for a completed arm. This daemon thread bounds that gap: an
        arm that is killed keeps a status file at most one interval stale.
        Fail-open and bounded (one thread, a constant interval, errors
        counted).
        """
        if not self._config.output_dir or self._status_thread is not None:
            return
        interval = max(0.1, min(self._config.report_interval_seconds, STATUS_WRITE_MAX_INTERVAL_S))
        try:
            self._status_thread = threading.Thread(
                target=self._status_writer_loop,
                args=(interval,),
                name="straggler-status-writer",
                daemon=True,
            )
            self._status_thread.start()
        except Exception:
            self._errors += 1
            self._status_thread = None

    def _status_writer_loop(self, interval: float) -> None:
        """Flush the collector and write the status files every ``interval``.

        Flushing here rather than only in ``close()`` is what makes the JSONL
        durable while the run is alive: Ray SIGTERMs the actor at job end, so
        ``close()`` never runs and a completed ON arm kept ``envelopes=132`` in
        the log while ``straggler_envelopes.jsonl`` stayed at zero bytes. This
        is a daemon thread, so the I/O stays off the training thread, and the
        collector's pending buffers are already bounded with counted drops.
        """
        while not self._status_stop.wait(interval):
            try:
                if self._collector is not None:
                    self._collector.flush()
                self._write_status()
            except Exception:
                self._errors += 1

    def _run_dir(self) -> str:
        """Run-scoped output directory shared by every rank of one run.

        The launcher gives each arm its own base directory and the run id (the
        Ray job id) makes the scope unique even if two arms ever shared a base,
        so two runs' JSONL streams can never interleave.
        """
        return os.path.join(self._config.output_dir, f"run_{_sanitise(self._identity.run_id)}")

    def _collector_config(self) -> StragglerConfig:
        """Point the collector's persistence at this run's scoped directory."""
        if not self._config.output_dir:
            return self._config
        return replace(self._config, output_dir=self._run_dir())

    def _build_sink(self) -> Optional[Callable[[TimingEnvelope], None]]:
        """Return the callable that receives each read-back envelope."""
        address = self._config.collector_addr
        if not address:
            self._collector = TimingCollector(self._collector_config(), self._identity)
            return self._collector.ingest
        parse_address(address)  # validate early so a typo is reported, not guessed
        if self._identity.rank == 0:
            self._collector = TimingCollector(self._collector_config(), self._identity)
            self._receiver = EnvelopeReceiver(address, self._ingest_payload)
            self._receiver.start()
            return self._collector.ingest
        self._sender = EnvelopeSender(address, queue_max=self._config.queue_max)
        return self._sender.send

    def _ingest_payload(self, payload: Dict[str, Any]) -> None:
        """Rebuild and ingest an envelope received from another rank."""
        if self._collector is None:
            return
        self._collector.ingest(TimingEnvelope.from_dict(payload))

    def status(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of every component."""
        status: Dict[str, Any] = {
            "role": self.role,
            "identity": self._identity.as_dict(),
            "enabled": self._config.enabled,
            "errors": self._errors,
            "started": self._started,
            "closed": self._closed,
        }
        if self._timers is not None:
            status["timers"] = self._timers.stats()
        if self._observer is not None:
            status["observer"] = self._observer.stats()
        if self._collector is not None:
            status["collector"] = self._collector.status()
        if self._sender is not None:
            status["sender"] = self._sender.stats()
        if self._receiver is not None:
            status["receiver"] = self._receiver.stats()
        return status

    def report(self) -> Dict[str, Any]:
        """Force a collector summary (used by on-demand diagnostics).

        Flushes buffered persistence, so it must not be called on the training
        thread; use :meth:`summary` there instead.
        """
        if self._collector is not None:
            return self._collector.report()
        return self.status()

    def summary(self) -> Dict[str, Any]:
        """Return a non-flushing summary, safe for the training thread.

        The platform metrics path calls this once per rollout to read counters
        and pending verdicts without performing any file I/O.
        """
        if self._collector is not None:
            return self._collector.summary()
        return self.status()

    def drain_verdicts(self) -> list:
        """Return and clear verdicts accumulated by the local collector."""
        if self._collector is None:
            return []
        return self._collector.drain_verdicts()

    def _write_status(self) -> None:
        """Persist the component counters inside this run's scoped directory.

        Every rank writes its own ``runtime_status_<label>.json``; the
        unsuffixed ``collector_status.json`` / ``runtime_status.json`` that
        ``analyze_run.py`` reads are written only by the collector, because all
        ranks share one run directory and four writers racing on one filename
        would leave an arbitrary rank's counters behind. Fail-open: a write
        error is counted, never raised into training. Called from the status
        writer thread and from ``close()``, never from the training path.
        """
        if not self._config.output_dir:
            return
        run_dir = self._run_dir()
        os.makedirs(run_dir, exist_ok=True)
        status = self.status()
        label = _sanitise(self._identity.label)
        self._write_json(run_dir, f"runtime_status_{label}.json", status)
        # Exactly one writer for the unsuffixed names analyze_run.py reads: the
        # reference rank. With a collector address that rank owns the shared
        # collector; in rank-local mode every rank owns one, so gating on rank 0
        # is what stops four writers racing on one filename.
        if self._identity.rank == 0:
            self._write_json(run_dir, "runtime_status.json", status)
            if "collector" in status:
                self._write_json(run_dir, f"collector_status_{label}.json", status["collector"])
                self._write_json(run_dir, "collector_status.json", status["collector"])

    @staticmethod
    def _write_json(output_dir: str, name: str, payload: Any) -> None:
        """Atomic status snapshot: write tmp, flush, then rename into place."""
        final = os.path.join(output_dir, name)
        temp = f"{final}.tmp"
        with open(temp, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
        os.replace(temp, final)

    def close(self, timeout: float = 2.0) -> None:
        """Stop reading, flush the windows and close the transport;
        idempotent."""
        if self._closed:
            return
        self._closed = True
        self._status_stop.set()
        if self._status_thread is not None:
            try:
                self._status_thread.join(timeout=1.0)
            except Exception:
                self._errors += 1
            self._status_thread = None
        try:
            if self._observer is not None:
                self._observer.close(timeout)
        except Exception:
            self._errors += 1
        try:
            if self._collector is not None:
                self._collector.flush()
                # Logging during interpreter shutdown races with a closed
                # stderr; the verdicts are already persisted by then.
                if not sys.is_finalizing():
                    self._collector.report()
        except Exception:
            self._errors += 1
        for component in (self._sender, self._receiver):
            try:
                if component is not None:
                    component.close(timeout)
            except Exception:
                self._errors += 1
        try:
            if not sys.is_finalizing():
                self._write_status()
        except Exception:
            self._errors += 1


__all__ = ["StragglerRuntime"]

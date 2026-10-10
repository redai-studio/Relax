# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-rank event collection, lazy readout and the per-window gather.

Hot path (Megatron's timer call sites): ``record_event`` / ``push`` record
events on the current stream and append to a queue. Once per step ``end_step``
reads completed pairs with ``event.query()`` (never ``synchronize``); every
``report_interval`` steps all ranks gather one small vector and the primary
rank queues the table for the background analysis, whose reports a later
``end_step`` returns.

A "step" in this package is one ``end_step`` call, i.e. one rollout, which may
contain several optimizer steps.
"""

from __future__ import annotations

import gc
from collections import deque
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Callable

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.detector import DetectorConfig, DetectorState, RankMeta, WindowReport, analyze_window
from relax.utils.straggler.health import HEALTH_DISABLE, HealthConfig, ProfilerHealth
from relax.utils.straggler.stats import (
    FIELD_INDEX,
    FORWARD_ONLY_TIMER_SEGMENTS,
    MEGATRON_TIMER_SEGMENTS,
    WindowStats,
)
from relax.utils.straggler.timers import EventPool, StragglerTimers
from relax.utils.straggler.worker import AnalysisWorker


logger = get_logger(__name__)

_FWD_INDEX = FIELD_INDEX["fwd"]
_BWD_INDEX = FIELD_INDEX["bwd"]
_CPU_FWD_INDEX = FIELD_INDEX["cpu_fwd"]
_CPU_BWD_INDEX = FIELD_INDEX["cpu_bwd"]
_NUM_FWD_INDEX = FIELD_INDEX["num_fwd"]
_GC_INDEX = FIELD_INDEX["gc"]
_DROPPED_INDEX = FIELD_INDEX["dropped"]
_ERRORS_INDEX = FIELD_INDEX["errors"]
_HEALTH_INDEX = FIELD_INDEX["health"]

# An analysis takes milliseconds (35 ms at 2048 ranks); this only bounds a stuck worker at the end of a run.
_FINAL_FLUSH_TIMEOUT_S = 10.0

GatherFn = Callable[[list[float]], list[list[float]]]
GatherObjectsFn = Callable[[Any], list[Any]]


@dataclass
class _WindowJob:
    """One closed window on its way to the analysis; ``ready`` carries a report
    that needs no analysis (the switch-off notice) through the same queue so
    reports stay in window order."""

    table: list[list[float]] | None
    meta: list[RankMeta]
    window_start_wall: float
    first_rollout: int
    last_rollout: int
    closed_wall: float = 0.0
    gather_ms: float = 0.0
    ready: WindowReport | None = None


class StragglerCollector:
    """Queue this rank's timing events, fold them lazily, and gather one window
    every ``report_interval`` steps.

    Torch-free: CUDA events, stream-capture detection and the gathers are
    injected (``runtime`` passes the real ones, tests pass fakes).
    """

    def __init__(
        self,
        *,
        rank_meta: RankMeta,
        is_primary: bool,
        event_factory: Callable[[], Any],
        gather: GatherFn,
        gather_objects: GatherObjectsFn,
        is_capturing: Callable[[], bool],
        report_interval: int = 10,
        detector_config: DetectorConfig | None = None,
        pool_size: int = 4096,
        max_pending: int = 16384,
        clock: Callable[[], float] = perf_counter,
        register_gc_callback: bool = True,
        health_config: HealthConfig | None = None,
        background: bool = True,
        on_report: Callable[[WindowReport], None] | None = None,
        max_backlog: int = 4,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        if report_interval < 1:
            raise ValueError(
                f"straggler report_interval must be >= 1, got {report_interval}: a window closes every "
                "report_interval rollouts. Set RELAX_STRAGGLER_REPORT_INTERVAL to a positive integer."
            )
        self.rank_meta = rank_meta
        self.is_primary = is_primary
        self.report_interval = report_interval
        self.detector_config = detector_config or DetectorConfig()
        self.detector_state = DetectorState()
        self._pool = EventPool(pool_size, event_factory)
        self._max_pending = max_pending
        self._pending: deque[tuple[int, Any, Any, float]] = deque()
        self._window = WindowStats()
        self._wall = wall_clock
        self.window_start_wall = wall_clock()
        self._gather = gather
        self._gather_objects = gather_objects
        self._clock = clock
        self._is_capturing = is_capturing
        self._all_meta: list[RankMeta] | None = None
        # True from the first bracket of a step until ``end_step``; GC pauses are
        # only attributed to the step while it is set (not during train_wait,
        # offload / clear_memory, etc.).
        self._step_open = False
        self._gc_start: float | None = None
        self.health = ProfilerHealth(health_config)
        # Plain attribute (not ``health.disabled``) because ``record_event`` checks it on every bracket.
        self._disabled = False
        # Appended by the worker, drained by ``end_step`` (deque append / popleft are atomic).
        self._reports: deque[WindowReport] = deque()
        self._on_report = on_report
        self._max_backlog = max_backlog
        self._window_first_rollout: int | None = None
        self._window_last_rollout = -1
        self._worker = AnalysisWorker(self._handle_job) if background and is_primary else None
        if register_gc_callback:
            gc.callbacks.append(self._on_gc)
        # One timers object per phase so the same Megatron call sites land in
        # different buckets during training vs. forward-only passes.
        self.train_timers = StragglerTimers(self, MEGATRON_TIMER_SEGMENTS)
        self.forward_only_timers = StragglerTimers(self, FORWARD_ONLY_TIMER_SEGMENTS)

    # ---- hot path -----------------------------------------------------------------------------------------------

    def record_event(self) -> Any:
        """Record a pooled event on the current stream and open the step for GC
        attribution.

        Returns ``None`` (the bracket is skipped) once the profiler is
        disabled, and during CUDA graph capture, where ``elapsed_time`` would
        be meaningless on replay.
        """
        if self._disabled or self._is_capturing():
            return None
        self._step_open = True
        event = self._pool.acquire()
        event.record()
        return event

    def discard_event(self, event: Any) -> None:
        """Return the start event of a bracket that could not be closed."""
        self._pool.release(event)

    def on_error(self, where: str, exc: BaseException) -> None:
        """Count an exception caught anywhere in the profiler (see
        ``ProfilerHealth``)."""
        self.health.record_error(where, exc)

    def push(self, segment_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        """Queue one bracket for ``drain``; when the queue is full drop the
        oldest pair and count it in ``dropped``, so a rank that stops draining
        cannot grow without bound."""
        if len(self._pending) >= self._max_pending:
            _, old_start, old_end, _ = self._pending.popleft()
            self._pool.release(old_start)
            self._pool.release(old_end)
            self._window.add_index(_DROPPED_INDEX, 1.0)
        self._pending.append((segment_index, start_event, end_event, cpu_ms))

    def add_tokens(self, tokens: int) -> None:
        """Count the tokens this rank trained on in the current step."""
        self._window.add("tokens", float(tokens))

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            self._gc_start = self._clock() if self._step_open else None
        elif self._gc_start is not None:
            self._window.add_index(_GC_INDEX, (self._clock() - self._gc_start) * 1e3)
            self._gc_start = None

    # ---- cold path ----------------------------------------------------------------------------------------------

    @property
    def pending_events(self) -> int:
        return len(self._pending)

    @property
    def window(self) -> WindowStats:
        return self._window

    def drain(self) -> int:
        """Fold completed event pairs from the front of the queue into the
        window; stop at the first pair that is still in flight.

        Pairs are queued in stream order, so an incomplete pair implies every
        later pair is incomplete too. Each pair is popped only after it has
        been folded, so an exception leaves the queue consistent and no event
        is released twice. Returns the number of pairs folded.
        """
        started = self._clock()
        folded = 0
        pending = self._pending
        window = self._window
        while pending:
            segment_index, start_event, end_event, cpu_ms = pending[0]
            if not (end_event.query() and start_event.query()):
                break
            window.add_index(segment_index, start_event.elapsed_time(end_event))
            if segment_index == _FWD_INDEX:
                window.add_index(_CPU_FWD_INDEX, cpu_ms)
                window.add_index(_NUM_FWD_INDEX, 1.0)
            elif segment_index == _BWD_INDEX:
                window.add_index(_CPU_BWD_INDEX, cpu_ms)
            pending.popleft()
            self._pool.release(start_event)
            self._pool.release(end_event)
            folded += 1
        window.add("overhead", (self._clock() - started) * 1e3)
        return folded

    @property
    def disabled(self) -> bool:
        return self._disabled

    def end_step(self, rollout_id: int = -1, final: bool = False) -> list[WindowReport]:
        """Close one training step; every ``report_interval`` steps gather and
        analyze.

        All ranks must call this at the same logical step (it contains a
        collective). Never raises: profiler failures are counted by
        ``self.health``. Returns the reports that are ready on the primary rank
        (empty elsewhere and between windows). On the ``final`` step of a run
        it waits (bounded) for the pending analyses, since no later call would
        pick them up.
        """
        if self._disabled:
            return self._take_reports()
        try:
            self.drain()
        except Exception as exc:
            self.on_error("drain", exc)
        self._step_open = False
        self._window.add("num_steps", 1.0)
        if self._window_first_rollout is None:
            self._window_first_rollout = rollout_id
        self._window_last_rollout = rollout_id
        if self._window.get("num_steps") >= self.report_interval:
            self._close_window()
        if final and not self.flush(timeout=_FINAL_FLUSH_TIMEOUT_S):
            self.on_error("analyze", TimeoutError(f"analysis still running {_FINAL_FLUSH_TIMEOUT_S} s after the run"))
        return self._take_reports()

    def _close_window(self) -> None:
        closed_wall = self._wall()
        errors, code = self.health.close_window(self._window.get("dropped"))
        if code >= HEALTH_DISABLE:
            logger.warning(
                "[straggler] rank %d asks to switch the profiler off: %s",
                self.rank_meta.rank,
                self.health.request_reason(),
            )
        self._window.set("unread", float(len(self._pending)))
        self._window.set("errors", errors)
        self._window.set("health", code)
        started = self._clock()
        try:
            if self._all_meta is None:
                self._all_meta = list(self._gather_objects(self.rank_meta))
            table = self._gather(self._window.as_list())
        except Exception as exc:
            self.on_error("gather", exc)
            self._disable(
                f"the window gather failed ({type(exc).__name__}: {exc}); only this rank can tell, so it stops "
                "joining straggler gathers"
            )
            return
        gathered = self._clock()

        requesting = [self._all_meta[i].rank for i, row in enumerate(table) if row[_HEALTH_INDEX] >= HEALTH_DISABLE]
        if requesting:
            note = (
                f"rank {requesting[0]} asked to switch the profiler off"
                + (f" (with {len(requesting) - 1} more)" if len(requesting) > 1 else "")
                + "; the reason is in that rank's log"
            )
            if self.is_primary:
                notice = WindowReport(
                    metrics={
                        "straggler/health/state": float(HEALTH_DISABLE),
                        "straggler/health/requested_by": float(requesting[0]),
                        "straggler/health/errors": float(sum(row[_ERRORS_INDEX] for row in table)),
                    },
                    alerts=[],
                    rows=[],
                    note=note,
                )
                self._submit(self._job(None, closed_wall=closed_wall, ready=notice))
            self._disable(note)
            return

        if self.is_primary:
            self._submit(self._job(table, closed_wall=closed_wall, gather_ms=(gathered - started) * 1e3))
        self._window.reset()
        self.window_start_wall = self._wall()
        self._window_first_rollout = None

    def _job(self, table: list[list[float]] | None, **kwargs: Any) -> _WindowJob:
        first = self._window_first_rollout
        return _WindowJob(
            table=table,
            meta=list(self._all_meta or []),
            window_start_wall=self.window_start_wall,
            first_rollout=self._window_last_rollout if first is None else first,
            last_rollout=self._window_last_rollout,
            **kwargs,
        )

    def _submit(self, job: _WindowJob) -> None:
        if self._worker is None:
            self._handle_job(job)
            return
        if self._worker.backlog >= self._max_backlog:
            self.on_error(
                "analyze",
                RuntimeError(
                    f"{self._worker.backlog} windows are still waiting for analysis; the window ending at rollout "
                    f"{job.last_rollout} is dropped"
                ),
            )
            return
        self._worker.submit(job)

    def _handle_job(self, job: _WindowJob) -> None:
        """Analyze one window (worker thread, or inline without a worker) and
        queue its report for ``end_step``; never raises.

        Only this method touches ``detector_state``, and jobs run one at a time
        in window order.
        """
        report = job.ready
        if report is None:
            started = self._clock()
            try:
                report = analyze_window(job.table, job.meta, self.detector_config, self.detector_state)
            except Exception as exc:
                self.on_error("analyze", exc)
                return
            report.metrics["straggler/gather_ms"] = job.gather_ms
            report.metrics["straggler/analyze_ms"] = (self._clock() - started) * 1e3
            report.window_start_wall = job.window_start_wall
        report.first_rollout = job.first_rollout
        report.last_rollout = job.last_rollout
        report.closed_wall = job.closed_wall
        report.analyzed_wall = self._wall()
        if self._on_report is not None:
            try:
                self._on_report(report)
            except Exception as exc:
                self.on_error("report", exc)
        self._reports.append(report)

    def _take_reports(self) -> list[WindowReport]:
        reports: list[WindowReport] = []
        while self._reports:
            reports.append(self._reports.popleft())
        return reports

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for queued analyses; True when none is left."""
        return self._worker.flush(timeout) if self._worker is not None else True

    def _disable(self, reason: str) -> None:
        """Switch the profiler off on this rank for the rest of the run."""
        if self._disabled:
            return
        self._disabled = True
        self.health.disable(reason)
        while self._pending:
            _, start_event, end_event, _ = self._pending.popleft()
            self._pool.release(start_event)
            self._pool.release(end_event)
        self.close()
        logger.warning(
            "[straggler] profiler disabled on rank %d: %s. Training continues; config.timers now records nothing.",
            self.rank_meta.rank,
            reason,
        )

    def close(self) -> None:
        """Unregister the GC callback and stop the worker once its queue is
        empty; safe to call more than once."""
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if self._worker is not None:
            self._worker.stop()

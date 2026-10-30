# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The window close: the per-window gather and keeping every rank on the same
collectives.

This rank's brackets are recorded and read into its window statistics by
``recorder.EventRecorder``. Every ``report_interval`` steps ``end_step``
closes the window: all ranks gather one small vector, and the primary rank
queues the table for the ``analyze`` passed in
(``analysis.report.WindowAnalyzer`` in production) on a background thread,
whose reports a later ``end_step`` returns.

With tracing on, the first step of every window after the first is sampled
(see ``recorder``), and the window close gathers every rank's sample to the
primary rank: once tracing is on, in every window, with ``None`` from a rank
without one.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Callable

from relax.utils.straggler.core.health import HEALTH_DISABLE, HealthConfig, ProfilerHealth
from relax.utils.straggler.core.recorder import CpuEventsFn, EventRecorder
from relax.utils.straggler.core.stats import (
    FIELD_INDEX,
    LEVELS,
    RankMeta,
    RankTrace,
    WindowStats,
    disabled_row,
)
from relax.utils.straggler.core.worker import AnalysisWorker


logger = logging.getLogger(__name__)

_HEALTH_INDEX = FIELD_INDEX["health"]
_CONTROL_INDEX = FIELD_INDEX["control"]

# An analysis takes milliseconds (35 ms at 2048 ranks); this only bounds a stuck worker at the end of a run.
_FINAL_FLUSH_TIMEOUT_S = 10.0

GatherFn = Callable[[list[float]], list[list[float]]]
GatherObjectsFn = Callable[[Any], list[Any]]
# (object, destination rank) -> every rank's object on the destination, None elsewhere.
GatherToFn = Callable[[Any, int], list[Any] | None]


@dataclass
class ClosedWindow:
    """One closed window of the primary rank on its way to ``analyze``: every
    rank's row and metadata, the sampled step's traces (``None`` below the
    detail level) and the level it ran at.

    A paused window (level 0, without a table: its report needs none) and the
    switch-off notice (``switch_off`` says why) take the same queue, so reports
    stay in window order.
    """

    table: list[list[float]]
    meta: list[RankMeta]
    window_start_wall: float
    first_rollout: int
    last_rollout: int
    closed_wall: float = 0.0
    gather_ms: float = 0.0
    traces: list[RankTrace | None] | None = None
    level: int = 1
    switch_off: str = ""
    # The first rank whose row asked to switch off.
    requested_by: int = -1

    @property
    def paused(self) -> bool:
        return not self.level and not self.switch_off


# (closed window, guard) -> its report, which ``end_step`` hands back; ``guard(where, call, *args, default=None)``
# runs ``call`` and counts its exception under ``where``.
AnalyzeFn = Callable[[ClosedWindow, Callable[..., Any]], Any]


class StragglerCollector:
    """Record this rank's brackets (``recorder``) and gather one window every
    ``report_interval`` steps.

    Torch-free: device events, stream-capture detection, the gathers and what
    becomes of a closed window (``analyze``) are injected (``relax_integration``
    passes the real ones, tests pass fakes); the timers that feed the recorder
    are built by the adapter around it (``megatron_adapter.phase_timers``).
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
        analyze: AnalyzeFn,
        report_interval: int = 10,
        pool_size: int = 4096,
        max_pending: int = 16384,
        clock: Callable[[], float] = perf_counter,
        register_gc_callback: bool = True,
        health_config: HealthConfig | None = None,
        background: bool = True,
        max_backlog: int = 4,
        wall_clock: Callable[[], float] = time,
        trace: bool = False,
        synchronize: Callable[[], None] = lambda: None,
        gather_to: GatherToFn | None = None,
        primary_rank: int | None = None,
        cpu_events: CpuEventsFn | None = None,
        max_trace_brackets: int = 4096,
        max_module_brackets: int = 8192,
        level: int = 1,
        level_request: Callable[[], int | None] = lambda: None,
        interleaved: bool = False,
    ) -> None:
        if report_interval < 1:
            raise ValueError(
                f"straggler report_interval must be >= 1, got {report_interval}: a window closes every "
                "report_interval rollouts. Set RELAX_STRAGGLER_REPORT_INTERVAL to a positive integer."
            )
        if not 0 <= level < len(LEVELS):
            raise ValueError(
                f"straggler level must be one of 0..{len(LEVELS) - 1} ({', '.join(LEVELS)}), got {level}. "
                "Set RELAX_STRAGGLER_LEVEL accordingly."
            )
        self.rank_meta = rank_meta
        self.is_primary = is_primary
        self.report_interval = report_interval
        self.health = ProfilerHealth(health_config)
        self.recorder = EventRecorder(
            event_factory=event_factory,
            is_capturing=is_capturing,
            health=self.health,
            pool_size=pool_size,
            max_pending=max_pending,
            clock=clock,
            wall_clock=wall_clock,
            synchronize=synchronize,
            cpu_events=cpu_events if is_primary else None,
            max_trace_brackets=max_trace_brackets,
            max_module_brackets=max_module_brackets,
            interleaved=interleaved,
            register_gc_callback=register_gc_callback,
        )
        self._analyze = analyze
        self._trace_enabled = trace
        self._disabled = False
        self.level = self._usable(level)
        self._steer_recorder()
        # Primary rank: the level the operator asks for, ``None`` without a request.
        self._level_request = level_request
        self._gather_to = gather_to
        # Where samples go, the same on every rank (agreed on at install); ``None``: no samples.
        self._primary_rank = primary_rank
        self._sample_gather = trace and gather_to is not None and primary_rank is not None
        self._wall = wall_clock
        self.window_start_wall = wall_clock()
        self._gather = gather
        self._gather_objects = gather_objects
        self._clock = clock
        self._all_meta: list[RankMeta] | None = None
        self._meta_gathered = False
        # Reports from ``analyze``: appended by the worker, drained by ``end_step`` (deque append / popleft are atomic).
        self._reports: deque[Any] = deque()
        self._max_backlog = max_backlog
        # A paused window waits in the analysis queue: set on the training thread, cleared by the worker once its
        # report is out. A race can only add one more paused window or skip one.
        self._paused_queued = False
        self._window_first_rollout: int | None = None
        self._window_last_rollout = -1
        self._worker = AnalysisWorker(self._handle_job) if background and is_primary else None

    @property
    def window(self) -> WindowStats:
        return self.recorder.window

    @property
    def disabled(self) -> bool:
        return self._disabled

    def on_error(self, where: str, exc: BaseException) -> None:
        """Count an exception caught anywhere in the profiler (see
        ``ProfilerHealth``)."""
        self.health.record_error(where, exc)

    def on_degraded(self, reason: str) -> None:
        """Record a lasting gap in what this rank measures (see
        ``ProfilerHealth.mark_degraded``)."""
        self.health.mark_degraded(reason)

    def end_step(self, rollout_id: int = -1, final: bool = False) -> list[Any]:
        """Close one training step; every ``report_interval`` steps gather and
        analyze.

        All ranks must call this at the same logical step (it contains a
        collective), including a switched-off one: it stops recording time but
        still joins every gather with a row carrying its ``health`` code. Never
        raises: profiler failures are counted by ``self.health``. Returns the
        reports that are ready on the primary rank (empty elsewhere and between
        windows). On the ``final`` step of a run it waits (bounded) for the
        pending analyses, since no later call would pick them up.
        """
        self.recorder.end_step()
        if self._window_first_rollout is None:
            self._window_first_rollout = rollout_id
        self._window_last_rollout = rollout_id
        if self.window.get("num_steps") >= self.report_interval:
            self._close_window()
        if final and not self.flush(timeout=_FINAL_FLUSH_TIMEOUT_S):
            self.on_error("analyze", TimeoutError(f"analysis still running {_FINAL_FLUSH_TIMEOUT_S} s after the run"))
        return self._take_reports()

    def _close_window(self) -> None:
        closed_wall = self._wall()
        errors, code = self.health.close_window(self.window.get("dropped"))
        code = self._health_code(code)
        self.window.set("unread", float(self.recorder.unread))
        self.window.set("errors", errors)
        self.window.set("health", code)
        self.window.set("control", self._control_code())
        level = self.level
        try:
            started = self._clock()
            table = self._gather_window(code)
            gathered = self._clock()
            # Before anything reads the table, so a rank whose table is missing joins it like its peers.
            traces = self._gather_traces() if self._sample_gather else None
            if table is None:
                return
            if self._handle_switch_off(table, closed_wall):
                return
            # A switched-off primary has already stopped its worker.
            if self.is_primary and not self._disabled:
                self._submit(
                    self._job(
                        table if level else [],
                        closed_wall=closed_wall,
                        gather_ms=(gathered - started) * 1e3,
                        traces=traces if level >= 2 else None,
                        level=level,
                    )
                )
            requested = int(max(row[_CONTROL_INDEX] for row in table)) - 1
            if requested >= 0:
                self._set_level(requested)
        finally:
            self._reset_window()

    def _control_code(self) -> float:
        """This window's ``control`` field: the level the operator asks for,
        plus one, on the primary rank; 0 everywhere else or without a
        request."""
        if not self.is_primary:
            return 0.0
        requested = self.health.guarded("control", self._level_request)
        return 0.0 if requested is None else float(requested + 1)

    def _usable(self, level: int) -> int:
        """Detail and modules need somewhere to write the trace; without it,
        coarse."""
        return 1 if level >= 2 and not self._trace_enabled else level

    def _set_level(self, level: int) -> None:
        """Switch every rank to ``level`` from the next window on."""
        level = self._usable(level)
        if level == self.level or not 0 <= level < len(LEVELS):
            return
        if self.is_primary:
            logger.info(
                "[straggler] level %s -> %s from rollout %d on",
                LEVELS[self.level],
                LEVELS[level],
                self._window_last_rollout + 1,
            )
        self.level = level
        self._steer_recorder()

    def _steer_recorder(self) -> None:
        """Record at every level but paused, and nothing once switched off;
        time module brackets at the modules level."""
        self.recorder.recording = self.level > 0 and not self._disabled
        self.recorder.modules_on = self.level == 3

    def status(self) -> dict[str, Any]:
        """What the primary rank reports to the control page."""
        return {"level": self.level, "rollout": self._window_last_rollout, "health": self.health.state}

    def _gather_traces(self) -> list[RankTrace | None] | None:
        """Hand this window's sample to the primary rank.

        Every rank joins in every window, with ``None`` when it has no complete
        sample (paused, below the detail level, switched off), so a rank that
        failed to read the table, and missed a level switch or switch-off
        because of it, still issues the same gathers as its peers.

        Returns the gathered list on the primary rank, ``None`` elsewhere or on
        failure.
        """
        sample = self.recorder.take_sample()
        return self.health.guarded("trace", self._gather_to, None if self._disabled else sample, self._primary_rank)

    def _health_code(self, code: float) -> float:
        """Return this window's ``health`` code, logging a new local switch-off
        request once.

        A rank that is already off keeps asking in every window, so a peer that
        has not seen the request yet still does.
        """
        if self._disabled:
            return HEALTH_DISABLE
        if code >= HEALTH_DISABLE:
            logger.warning(
                "[straggler] rank %d asks to switch the profiler off: %s",
                self.rank_meta.rank,
                self.health.request_reason(),
            )
        return code

    def _gather_window(self, code: float) -> list[list[float]] | None:
        """Gather this window's row on every rank, or ``None`` if it failed.

        Every rank issues the same collectives whatever failed locally, or the
        gathers of different ranks would pair up wrongly: the metadata gather
        runs once, in the first window, and every window gathers one row. A
        rank left without the metadata cannot retry it (that object gather
        would be its alone), so its row asks every rank to switch off at this
        window. A failed row gather is counted and the next window tries again;
        the dedicated short-timeout group bounds a group that is really broken,
        and once health gives up the rank also stops recording, so a failed
        group cannot keep costing timer time for the rest of the run. A table
        with a NaN row (a rank that could not build its row) is skipped on
        every rank alike.
        """
        if not self._meta_gathered:
            self._meta_gathered = True
            try:
                self._all_meta = list(self._gather_objects(self.rank_meta))
            except Exception as exc:
                self.on_error("gather", exc)
                logger.warning(
                    "[straggler] rank %d asks to switch the profiler off: the rank metadata gather failed (%s: %s)",
                    self.rank_meta.rank,
                    type(exc).__name__,
                    exc,
                )
        if self._all_meta is None:
            code = HEALTH_DISABLE
            self.window.set("health", code)
        try:
            table = self._gather(disabled_row() if self._disabled else self.window.as_list())
        except Exception as exc:
            self.on_error("gather", exc)
            if code >= HEALTH_DISABLE and not self._disabled:
                self._disable(f"the window gather kept failing ({type(exc).__name__}: {exc})")
            return None
        # The rank that sent it raised and counts the failure itself.
        return None if any(math.isnan(row[_HEALTH_INDEX]) for row in table) else table

    def _handle_switch_off(self, table: list[list[float]], closed_wall: float) -> bool:
        """Switch off when a gathered row asks for it; ``True`` when it did.

        On the primary rank the notice joins the analysis queue, so it stays in
        window order with the reports around it.
        """
        # Rows are in gather-group rank order, which is the global rank.
        requesting = [i for i, row in enumerate(table) if row[_HEALTH_INDEX] >= HEALTH_DISABLE]
        if not requesting or self._disabled:
            return False
        note = (
            f"rank {requesting[0]} asked to switch the profiler off"
            + (f" (with {len(requesting) - 1} more)" if len(requesting) > 1 else "")
            + "; the reason is in that rank's log"
        )
        if self.is_primary:
            self._submit(self._job(table, closed_wall=closed_wall, switch_off=note, requested_by=requesting[0]))
        self._disable(note)
        return True

    def _reset_window(self) -> None:
        """Start the next window; every close path has to reach this, or the
        window would close (and gather) again on every following step."""
        self.recorder.reset_window(sample_next=self._sample_gather and self.level >= 2 and not self._disabled)
        self.window_start_wall = self._wall()
        self._window_first_rollout = None

    def _job(self, table: list[list[float]], **kwargs: Any) -> ClosedWindow:
        first = self._window_first_rollout
        return ClosedWindow(
            table=table,
            meta=list(self._all_meta or []),
            window_start_wall=self.window_start_wall,
            first_rollout=self._window_last_rollout if first is None else first,
            last_rollout=self._window_last_rollout,
            **kwargs,
        )

    def _submit(self, job: ClosedWindow) -> None:
        if self._worker is None:
            self._handle_job(job)
            return
        # A paused window costs nothing to report and may arrive while the
        # analysis is behind, which is when an operator pauses: it does not
        # count against the backlog, and one waiting in the queue stands for
        # the paused windows after it. The switch-off notice (health/state=2)
        # has to reach end_step even when the queue is full.
        if job.paused:
            if self._paused_queued:
                return
            self._paused_queued = True
        elif not job.switch_off and self._worker.backlog >= self._max_backlog:
            self.on_error(
                "analyze",
                RuntimeError(
                    f"{self._worker.backlog} windows are still waiting for analysis; the window ending at rollout "
                    f"{job.last_rollout} is dropped"
                ),
            )
            return
        self._worker.submit(job)

    def _handle_job(self, job: ClosedWindow) -> None:
        """Hand one window to ``analyze`` (worker thread, or inline without a
        worker) and queue its report for ``end_step``; never raises.

        Windows arrive one at a time, in window order.
        """
        report = self.health.guarded("analyze", self._analyze, job, self.health.guarded)
        if report is not None:
            self._reports.append(report)
        if job.paused:
            self._paused_queued = False

    def _take_reports(self) -> list[Any]:
        reports: list[Any] = []
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
        self.recorder.stop()
        self.close()
        logger.warning(
            "[straggler] profiler disabled on rank %d: %s. Training continues; config.timers now records nothing.",
            self.rank_meta.rank,
            reason,
        )

    def close(self) -> None:
        """Unregister the GC callback and stop the worker once its queue is
        empty; safe to call more than once."""
        self.recorder.close()
        if self._worker is not None:
            self._worker.stop()

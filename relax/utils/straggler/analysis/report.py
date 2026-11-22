# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Turn the primary rank's closed windows into reports, and each report into
scalars, timeline bars and log lines.

* ``WindowAnalyzer`` -> the collector's ``analyze``: the detector's verdict,
  the sampled-step trace file and the log lines, on the analysis thread
* the scalars in ``REPORTED_METRICS`` -> returned for the step's metrics call
* per-rank segment bars -> one timeline row per rank
* alerts, recoveries, uncertain ranks and the rest of the window's numbers -> logger, from the analysis thread

Where they go (which metrics call, which timeline, which step number) is the
integration's business (``relax_integration``).
"""

from __future__ import annotations

import logging
from time import perf_counter, time
from typing import Any, Callable, NamedTuple

from relax.utils.straggler.analysis.detector import DetectorConfig, DetectorState, WindowReport, analyze_window
from relax.utils.straggler.analysis.trace import TIMELINE_PID_BASE
from relax.utils.straggler.core.collector import ClosedWindow
from relax.utils.straggler.core.health import HEALTH_DISABLE
from relax.utils.straggler.core.stats import FIELD_INDEX, GPU_SEGMENTS, RankMeta, RankTrace


logger = logging.getLogger(__name__)

_TABLE_COLUMNS: tuple[str, ...] = (
    "rank",
    "tag",
    "host",
    "device",
    "self_ms",
    "wait_ms",
    "late_ms",
    "tokens",
    "ms_per_ktok",
    "gc_ms",
    "steady_ms",
    "mb_fwd_max",
)

# The scalars that go out with the step's metrics call. Every key adds training-thread time to that
# step's flush, so the analysis thread logs the rest of a window's numbers instead.
REPORTED_METRICS: frozenset[str] = frozenset(
    {
        "straggler/flagged/count",
        "straggler/flagged/rank",
        "straggler/flagged/reason",
        "straggler/uncertain/count",
        "straggler/self/median_ms",
        "straggler/self/max_ms",
        "straggler/self/max_rank",
        "straggler/late/max_ms",
        "straggler/health/state",
        "straggler/health/errors",
        "straggler/health/requested_by",
        "straggler/dropped_events",
        "straggler/unread_events",
        "straggler/trace/lost",
        "straggler/control/level",
        "straggler/self_overhead_ms",
        "straggler/window/first_rollout",
        "straggler/window/last_rollout",
        "straggler/latency/analyzed_ms",
    }
)

_ERRORS_INDEX = FIELD_INDEX["errors"]

# (gathered traces, rank metadata, first rollout of the window) -> trace scalars.
TraceSink = Callable[[list[RankTrace | None], list[RankMeta], int], dict[str, float]]


class WindowAnalyzer:
    """The collector's ``analyze`` on the primary rank: turn a closed window
    into its report (the detector's verdict, the paused note or the switch-off
    notice), add the sampled-step trace's scalars and hand the report to
    ``on_report``.

    The collector calls it one window at a time in window order, so ``state``
    needs no lock. ``guard`` is the collector's (see
    ``core.collector.AnalyzeFn``), which also counts a failing analysis.
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        trace_sink: TraceSink | None = None,
        on_report: Callable[[WindowReport], None] | None = None,
        clock: Callable[[], float] = perf_counter,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        self.config = config or DetectorConfig()
        self.state = DetectorState()
        self._trace_sink = trace_sink
        self._on_report = on_report
        self._clock = clock
        self._wall = wall_clock

    def __call__(self, window: ClosedWindow, guard: Callable[..., Any]) -> WindowReport:
        if window.switch_off:
            report = WindowReport(
                metrics={
                    "straggler/health/state": float(HEALTH_DISABLE),
                    "straggler/health/requested_by": float(window.requested_by),
                    "straggler/health/errors": float(sum(row[_ERRORS_INDEX] for row in window.table)),
                },
                alerts=[],
                rows=[],
                note=window.switch_off,
            )
        elif window.level == 0:
            report = WindowReport(metrics={"straggler/control/level": 0.0}, alerts=[], rows=[], note="paused")
        else:
            started = self._clock()
            report = analyze_window(window.table, window.meta, self.config, self.state)
            report.metrics["straggler/gather_ms"] = window.gather_ms
            report.metrics["straggler/analyze_ms"] = (self._clock() - started) * 1e3
            report.metrics["straggler/control/level"] = float(window.level)
            report.window_start_wall = window.window_start_wall
            if window.traces is not None and self._trace_sink is not None:
                report.metrics.update(
                    guard("trace", self._trace_sink, window.traces, window.meta, window.first_rollout, default={})
                )
        report.first_rollout = window.first_rollout
        report.last_rollout = window.last_rollout
        report.closed_wall = window.closed_wall
        report.analyzed_wall = self._wall()
        if self._on_report is not None:
            guard("report", self._on_report, report)
        return report


class TimelineBar(NamedTuple):
    name: str
    start: float
    end: float
    pid: int
    tid: int


def timeline_bars(report: WindowReport) -> list[TimelineBar]:
    """Lay each rank's per-step segment averages out as consecutive bars, in
    wall-clock seconds.

    The bars start at the window's wall-clock start so they line up with the
    primary rank's own CPU timeline of the same window.
    """
    bars: list[TimelineBar] = []
    for row in report.rows:
        cursor = report.window_start_wall
        pid = TIMELINE_PID_BASE + int(row["rank"])
        for tid, name in enumerate(GPU_SEGMENTS):
            ms = float(row[name])
            if ms <= 0.0:
                continue
            bars.append(TimelineBar(f"straggler/{name} {row['tag']}", cursor, cursor + ms / 1e3, pid, tid))
            cursor += ms / 1e3
    return bars


def _format_table(report: WindowReport) -> str:
    header = " | ".join(f"{column:>11}" for column in _TABLE_COLUMNS) + " | reason | uncertain"
    lines = [header]
    for row in report.rows:
        cells = []
        for column in _TABLE_COLUMNS:
            value = row[column]
            cells.append(f"{value:>11.1f}" if isinstance(value, float) else f"{str(value):>11}")
        lines.append(" | ".join(cells) + f" | {row['reason']} | {row.get('uncertain', '')}")
    return "\n".join(lines)


def switched_off(report: WindowReport) -> bool:
    return report.metrics.get("straggler/health/state", 0.0) >= HEALTH_DISABLE


def reported_scalars(report: WindowReport) -> dict[str, float]:
    """The window's ``REPORTED_METRICS``, with the rollouts it covers (it is
    usually picked up a step after it closed, hence ``straggler/window/*``) and
    how long after its close the analysis ended."""
    metrics = report.metrics
    metrics["straggler/window/first_rollout"] = float(report.first_rollout)
    metrics["straggler/window/last_rollout"] = float(report.last_rollout)
    if report.closed_wall > 0.0:
        metrics["straggler/latency/analyzed_ms"] = (report.analyzed_wall - report.closed_wall) * 1e3
    return {key: value for key, value in metrics.items() if key in REPORTED_METRICS}


def log_window(report: WindowReport, step: int) -> None:
    """Log one window's alerts, recoveries, uncertain ranks, summary and the
    numbers left out of ``REPORTED_METRICS`` at ``step``, the window's own.

    Only touches the logger, so it is safe on the analysis thread, which runs
    it as soon as the analysis ends.
    """
    metrics = report.metrics
    if switched_off(report):
        logger.warning("[straggler] step=%d profiler switched off on every rank: %s", step, report.note)
        return
    if report.note == "paused":
        logger.info(
            "[straggler] step=%d paused: rollouts %d-%d not analyzed", step, report.first_rollout, report.last_rollout
        )
        return

    for alert in report.alerts:
        if alert.new:
            logger.warning("[straggler] step=%d %s", step, alert.message)
    for recovery in report.recovered:
        logger.warning("[straggler] step=%d %s", step, recovery.message)
    for unsure in report.uncertain:
        if unsure.new:
            logger.warning("[straggler] step=%d %s", step, unsure.message)
    if report.alerts:
        logger.info("[straggler] step=%d per-rank window (ms/step):\n%s", step, _format_table(report))
    else:
        logger.info(
            "[straggler] step=%d no straggler (uncertain %d); self median %.1f ms max %.1f ms (rank %d, +%.0f%%), "
            "latest to grad-sync rank %d by %.1f ms (peers idle %.1f ms), pp_stage_imbalance %.2f, "
            "overhead %.2f ms/step",
            step,
            len(report.uncertain),
            metrics.get("straggler/self/median_ms", 0.0),
            metrics.get("straggler/self/max_ms", 0.0),
            int(metrics.get("straggler/self/max_rank", -1)),
            100.0 * metrics.get("straggler/self/spread", 0.0),
            int(metrics.get("straggler/late/max_rank", -1)),
            metrics.get("straggler/late/max_ms", 0.0),
            metrics.get("straggler/late/peer_idle_ms", 0.0),
            metrics.get("straggler/pp_stage_imbalance", 1.0),
            metrics.get("straggler/self_overhead_ms", 0.0),
        )
    others = sorted(key for key in metrics if key not in REPORTED_METRICS)
    if others:
        logger.info(
            "[straggler] step=%d other metrics: %s",
            step,
            ", ".join(f"{key.removeprefix('straggler/')} {round(metrics[key], 3):g}" for key in others),
        )

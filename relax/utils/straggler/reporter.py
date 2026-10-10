# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Turn one analyzed straggler window into scalars, timeline events and log
lines.

* scalars -> returned for the step's ``log_perf_data``
* per-rank segment bars -> ``Timer().records``, one Perfetto row per rank
* alerts, recoveries, uncertain ranks -> logger, from the analysis thread
"""

from __future__ import annotations

from time import time
from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.straggler.detector import WindowReport
from relax.utils.straggler.health import HEALTH_DISABLE
from relax.utils.straggler.stats import GPU_SEGMENTS
from relax.utils.timer import TimelineEvent, Timer


logger = get_logger(__name__)

# Above any real pid (kernel.pid_max <= 2**22) so the straggler rows never merge
# with a live process's track in Perfetto.
TIMELINE_PID_BASE = 1 << 30
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
)


def _timeline_events(report: WindowReport) -> list[TimelineEvent]:
    """Lay each rank's per-step segment averages out as consecutive bars.

    The bars start at the window's wall-clock start so they line up with the
    primary rank's CPU-side ``actor_train`` event in the same trace file.
    """
    events: list[TimelineEvent] = []
    for row in report.rows:
        cursor = report.window_start_wall
        pid = TIMELINE_PID_BASE + int(row["rank"])
        for tid, name in enumerate(GPU_SEGMENTS):
            ms = float(row[name])
            if ms <= 0.0:
                continue
            events.append(
                TimelineEvent(
                    name=f"straggler/{name} {row['tag']}",
                    start_ts=cursor,
                    end_ts=cursor + ms / 1e3,
                    pid=pid,
                    tid=tid,
                )
            )
            cursor += ms / 1e3
    return events


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


def _switched_off(report: WindowReport) -> bool:
    return report.metrics.get("straggler/health/state", 0.0) >= HEALTH_DISABLE


def _ms_after_close(report: WindowReport, wall: float) -> float:
    return (wall - report.closed_wall) * 1e3


def report_straggler_window(args: Any, report: WindowReport, now: float | None = None) -> dict[str, float]:
    """Emit timeline events for one window and return its scalars for the
    step's ``log_perf_data`` (primary rank, training thread).

    That step can be later than the window, hence ``straggler/window/*``.
    """
    report.emitted_wall = time() if now is None else now
    metrics = report.metrics
    metrics["straggler/window/first_rollout"] = float(report.first_rollout)
    metrics["straggler/window/last_rollout"] = float(report.last_rollout)
    if report.closed_wall > 0.0:
        metrics["straggler/latency/analyzed_ms"] = _ms_after_close(report, report.analyzed_wall)
        metrics["straggler/latency/emitted_ms"] = _ms_after_close(report, report.emitted_wall)
    if getattr(args, "timeline_dump_dir", None) and not _switched_off(report):
        # Flushed, and stamped with the step, by the ``log_perf_data`` that follows.
        Timer().records.extend(_timeline_events(report))
    return metrics


def log_straggler_delivery(args: Any, report: WindowReport, rollout_id: int, now: float | None = None) -> float:
    """Log how long one window took to reach the tracking backend and return
    close-to-delivered in ms.

    Call it once the ``log_perf_data`` that carried the report's scalars has
    returned: with ``--use-metrics-service`` the metrics service has then
    acknowledged them; with TensorBoard / WandB the writer has accepted them.
    """
    delivered_ms = _ms_after_close(report, time() if now is None else now)
    logger.info(
        "[straggler] step=%d window rollouts %d-%d: closed -> analyzed +%.1f ms (alerts logged), emitted +%.1f ms, "
        "delivered +%.1f ms",
        compute_rollout_step(args, rollout_id),
        report.first_rollout,
        report.last_rollout,
        _ms_after_close(report, report.analyzed_wall),
        _ms_after_close(report, report.emitted_wall),
        delivered_ms,
    )
    return delivered_ms


def log_straggler_window(args: Any, report: WindowReport) -> None:
    """Log one window's alerts, recoveries, uncertain ranks and summary at the
    window's own step.

    Only touches the logger, so it is safe on the analysis thread, which runs
    it as soon as the analysis ends.
    """
    step = compute_rollout_step(args, report.last_rollout)
    metrics = report.metrics
    if _switched_off(report):
        logger.warning("[straggler] step=%d profiler switched off on every rank: %s", step, report.note)
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

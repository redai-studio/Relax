# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for how a window report reaches metrics, timeline and logs."""

from argparse import Namespace

import pytest

from relax.utils import tracking_utils
from relax.utils.straggler import reporter
from relax.utils.straggler.detector import DetectorConfig, DetectorState, WindowReport, analyze_window
from relax.utils.timer import Timer
from tests.utils.straggler_helpers import meta as _meta
from tests.utils.straggler_helpers import row as _row


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    calls = []
    monkeypatch.setattr(tracking_utils, "log", lambda args, metrics, step_key: calls.append((metrics, step_key)))
    return calls


@pytest.fixture(autouse=True)
def records():
    Timer().records.clear()
    yield Timer().records
    Timer().records.clear()


@pytest.fixture
def captured(monkeypatch):
    warnings, infos = [], []
    monkeypatch.setattr(reporter.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(reporter.logger, "info", lambda msg, *a: infos.append(msg % a))
    return warnings, infos


def _args(timeline_dump_dir=None):
    return Namespace(wandb_always_use_train_step=False, timeline_dump_dir=timeline_dump_dir)


def _report(flag_rank=None, first_rollout=10, last_rollout=19):
    table = [_row(fwd=100.0, bwd=200.0, tokens=1000.0) for _ in range(4)]
    if flag_rank is not None:
        table[flag_rank] = _row(fwd=150.0, bwd=300.0, tokens=1000.0)
    report = analyze_window(table, _meta(4), DetectorConfig(persist_windows=1), DetectorState())
    report.window_start_wall = 1000.0
    report.first_rollout, report.last_rollout = first_rollout, last_rollout
    return report


def test_report_returns_scalars_instead_of_logging_them(logged, captured):
    # With --use-metrics-service each tracking_utils.log is a synchronous HTTP request, so the
    # scalars go back to the actor and ride along with log_perf_data for the same step.
    metrics = reporter.report_straggler_window(_args(), report=_report(flag_rank=2))
    assert logged == []
    assert captured == ([], []), "logging happens on the analysis worker, not at pickup"
    assert metrics["straggler/flagged/count"] == 1
    assert metrics["straggler/fwd/median_ms"] == 100.0
    assert "rollout/step" not in metrics


def test_scalars_carry_the_rollout_range_of_their_window():
    # The report is usually picked up one rollout after its window closed.
    metrics = reporter.report_straggler_window(_args(), report=_report(first_rollout=10, last_rollout=19))
    assert metrics["straggler/window/first_rollout"] == 10.0
    assert metrics["straggler/window/last_rollout"] == 19.0


def test_scalars_include_close_to_analysis_and_close_to_emit_latency():
    report = _report()
    report.closed_wall, report.analyzed_wall = 100.0, 100.004
    metrics = reporter.report_straggler_window(_args(), report=report, now=100.9)
    assert metrics["straggler/latency/analyzed_ms"] == pytest.approx(4.0)
    assert metrics["straggler/latency/emitted_ms"] == pytest.approx(900.0)


def test_delivery_line_reports_every_stage_and_returns_close_to_delivered(captured):
    _, infos = captured
    report = _report(first_rollout=10, last_rollout=19)
    report.closed_wall, report.analyzed_wall = 100.0, 100.004
    reporter.report_straggler_window(_args(), report=report, now=100.9)
    delivered_ms = reporter.log_straggler_delivery(_args(), report, rollout_id=20, now=101.2)
    assert delivered_ms == pytest.approx(1200.0)
    (line,) = infos
    for piece in ("rollouts 10-19", "analyzed +4.0 ms", "emitted +900.0 ms", "delivered +1200.0 ms", "step=20"):
        assert piece in line, (piece, line)


def test_timeline_events_one_row_per_rank_when_enabled(records):
    reporter.report_straggler_window(_args(timeline_dump_dir="/tmp/tl"), report=_report())
    events = list(records)
    # 4 ranks x 2 non-zero segments (fwd, bwd)
    assert len(events) == 8
    pids = {event.pid for event in events}
    assert pids == {reporter.TIMELINE_PID_BASE + r for r in range(4)}
    rank0 = sorted((event for event in events if event.pid == reporter.TIMELINE_PID_BASE), key=lambda e: e.start_ts)
    assert rank0[0].name.startswith("straggler/fwd") and rank0[0].start_ts == 1000.0
    assert rank0[0].end_ts == 1000.1 and rank0[1].start_ts == 1000.1
    # Rows must never collide with a real process's track.
    assert reporter.TIMELINE_PID_BASE > 2**22  # Linux upper bound for kernel.pid_max


def test_no_timeline_events_when_disabled(records):
    reporter.report_straggler_window(_args(), report=_report())
    assert records == []


def test_alert_is_logged_as_warning_with_table_at_the_window_step(captured):
    warnings, infos = captured
    reporter.log_straggler_window(_args(), _report(flag_rank=2, last_rollout=5))
    assert len(warnings) == 1 and "rank 2" in warnings[0] and "slow_device" in warnings[0]
    assert "step=5" in warnings[0]
    assert len(infos) == 1 and "per-rank window" in infos[0] and "slow_device" in infos[0]


def test_switch_off_report_logs_why_instead_of_a_healthy_summary(captured, records):
    warnings, infos = captured
    report = WindowReport(
        metrics={"straggler/health/state": 2.0, "straggler/health/requested_by": 3.0},
        alerts=[],
        rows=[],
        note="rank 3 asked to switch the profiler off",
    )
    reporter.log_straggler_window(_args(), report)
    assert len(warnings) == 1 and "rank 3 asked" in warnings[0]
    assert infos == [], "a switched-off profiler must not print a 'no straggler' summary"
    metrics = reporter.report_straggler_window(_args(timeline_dump_dir="/tmp/tl"), report=report)
    assert metrics["straggler/health/state"] == 2.0 and records == []


def test_new_uncertain_ranks_are_warned_once_and_counted_in_the_summary(captured):
    warnings, infos = captured
    state = DetectorState()
    table = [_row(fwd=100.0, tokens=1000.0), _row(fwd=500.0, tokens=1000.0)]
    for _ in range(2):
        reporter.log_straggler_window(_args(), analyze_window(table, _meta(2, pp_size=2), DetectorConfig(), state))
    assert len(warnings) == 2, "one WARNING per rank when it becomes uncertain, not every window"
    assert all("uncertain" in line and "no other rank" in line for line in warnings)
    assert len(infos) == 2 and all("uncertain 2" in line for line in infos)


def test_recovery_is_logged_as_warning(captured):
    warnings, _ = captured
    config, state = DetectorConfig(persist_windows=1, recover_windows=1), DetectorState()
    slow = [_row(fwd=100.0, bwd=200.0, tokens=1000.0) for _ in range(4)]
    slow[1] = _row(fwd=150.0, bwd=300.0, tokens=1000.0)
    analyze_window(slow, _meta(4), config, state)
    clean = [_row(fwd=100.0, bwd=200.0, tokens=1000.0) for _ in range(4)]
    reporter.log_straggler_window(_args(), analyze_window(clean, _meta(4), config, state))
    assert len(warnings) == 1 and "recovered from slow_device" in warnings[0]

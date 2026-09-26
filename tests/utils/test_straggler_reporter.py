# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for how a window report reaches metrics, timeline and logs."""

from argparse import Namespace

import pytest

from relax.utils import tracking_utils
from relax.utils.straggler import reporter
from relax.utils.straggler.detector import DetectorConfig, DetectorState, analyze_window
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


def _args(timeline_dump_dir=None):
    return Namespace(wandb_always_use_train_step=False, timeline_dump_dir=timeline_dump_dir)


def _report(flag_rank=None):
    table = [_row(fwd=100.0, bwd=200.0, tokens=1000.0) for _ in range(4)]
    if flag_rank is not None:
        table[flag_rank] = _row(fwd=150.0, bwd=300.0, tokens=1000.0)
    report = analyze_window(table, _meta(4), DetectorConfig(persist_windows=1), DetectorState())
    report.window_start_wall = 1000.0
    return report


def test_report_returns_scalars_instead_of_logging_them(logged):
    # With --use-metrics-service each tracking_utils.log is a synchronous HTTP request, so the
    # scalars go back to the actor and ride along with log_perf_data for the same step.
    metrics = reporter.report_straggler_window(_args(), rollout_id=7, report=_report())
    assert logged == []
    assert metrics["straggler/flagged/count"] == 0
    assert metrics["straggler/fwd/median_ms"] == 100.0
    assert "rollout/step" not in metrics


def test_timeline_events_one_row_per_rank_when_enabled(records):
    reporter.report_straggler_window(_args(timeline_dump_dir="/tmp/tl"), rollout_id=3, report=_report())
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
    reporter.report_straggler_window(_args(), rollout_id=3, report=_report())
    assert records == []


def test_alert_is_logged_as_warning_with_table(monkeypatch):
    warnings, infos = [], []
    monkeypatch.setattr(reporter.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(reporter.logger, "info", lambda msg, *a: infos.append(msg % a))
    reporter.report_straggler_window(_args(), rollout_id=5, report=_report(flag_rank=2))
    assert len(warnings) == 1 and "rank 2" in warnings[0] and "slow_device" in warnings[0]
    assert len(infos) == 1 and "per-rank window" in infos[0] and "slow_device" in infos[0]

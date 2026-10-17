# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for running the window analysis off the training thread."""

import threading

from relax.utils.straggler import collector as collector_module
from relax.utils.straggler.health import HealthConfig
from relax.utils.straggler.stats import FIELD_INDEX
from tests.utils.straggler_helpers import bracket, make_collector


def _collector(**kwargs):
    return make_collector(background=True, **kwargs)


def _step(collector, rollout_id):
    bracket(collector)
    collector.add_tokens(1000)
    return collector.end_step(rollout_id)


def _gated_analysis(monkeypatch):
    """Make the analysis wait until the test opens the gate."""
    gate, started, real = threading.Event(), threading.Event(), collector_module.analyze_window

    def gated(*args):
        started.set()
        assert gate.wait(10), "test never opened the gate"
        return real(*args)

    monkeypatch.setattr(collector_module, "analyze_window", gated)
    return gate, started


def test_end_step_returns_while_the_analysis_is_still_running(monkeypatch):
    gate, started = _gated_analysis(monkeypatch)
    collector = _collector()
    try:
        assert _step(collector, rollout_id=0) == []  # returns although the analysis is blocked
        assert started.wait(5)
        gate.set()
        assert collector.flush(timeout=5)
        (report,) = _step(collector, rollout_id=1)
        assert (report.first_rollout, report.last_rollout) == (0, 0)
    finally:
        gate.set()
        collector.close()


def test_on_report_runs_on_the_worker_thread_as_soon_as_the_analysis_ends():
    seen = []
    collector = _collector(on_report=lambda report: seen.append(threading.current_thread().name))
    try:
        _step(collector, rollout_id=0)
        assert collector.flush(timeout=5)
        assert seen and seen[0] != threading.current_thread().name
        assert seen[0].startswith("straggler")
    finally:
        collector.close()


def test_windows_are_analyzed_in_order_by_one_worker(monkeypatch):
    gate, _ = _gated_analysis(monkeypatch)

    def gather(values):
        slow = list(values)
        slow[FIELD_INDEX["fwd"]] *= 3.0
        return [list(values), slow]

    collector = _collector(gather=gather, persist=2)
    try:
        _step(collector, rollout_id=0)
        _step(collector, rollout_id=1)  # queued behind the first window
        gate.set()
        assert collector.flush(timeout=5)
        first, second = _step(collector, rollout_id=2)
        assert (first.last_rollout, second.last_rollout) == (0, 1)
        assert first.alerts == [] and [a.rank for a in second.alerts] == [1], "one detector state, in window order"
    finally:
        gate.set()
        collector.close()


def test_final_step_waits_for_its_own_window_so_the_last_report_is_not_lost():
    collector = _collector()
    try:
        bracket(collector)
        (report,) = collector.end_step(rollout_id=59, final=True)
        assert report.last_rollout == 59
    finally:
        collector.close()


def test_a_failing_analysis_is_counted_and_training_carries_on(monkeypatch):
    def boom(*args):
        raise ValueError("bad table")

    monkeypatch.setattr(collector_module, "analyze_window", boom)
    collector = _collector()
    try:
        _step(collector, rollout_id=0)
        assert collector.flush(timeout=5)
        assert collector.health.errors_total == 1
        assert _step(collector, rollout_id=1) == []
    finally:
        collector.close()


def test_a_failing_on_report_is_counted_and_the_report_still_arrives():
    def boom(report):
        raise RuntimeError("logger broke")

    collector = _collector(on_report=boom)
    try:
        _step(collector, rollout_id=0)
        assert collector.flush(timeout=5)
        assert collector.health.errors_total == 1
        assert len(_step(collector, rollout_id=1)) == 1
    finally:
        collector.close()


def test_backlog_drops_ordinary_windows_and_counts_errors(monkeypatch):
    """Once ``max_backlog`` windows wait (the in-flight one included), new
    windows are dropped and counted as errors."""
    gate, started = _gated_analysis(monkeypatch)
    collector = _collector(max_backlog=2, health_config=HealthConfig(max_errors=100))
    try:
        _step(collector, rollout_id=0)  # in flight, blocked on the gate
        assert started.wait(5)
        _step(collector, rollout_id=1)  # queued: backlog is now full
        for rollout_id in (2, 3, 4):
            _step(collector, rollout_id=rollout_id)  # dropped
        assert collector.health.errors_total == 3
        assert collector._worker.backlog == 2
    finally:
        gate.set()
        collector.close()


def test_switch_off_notice_is_not_dropped_when_backlog_is_full(monkeypatch):
    """The switch-off notice (health/state=2) bypasses a full backlog."""
    gate, started = _gated_analysis(monkeypatch)
    collector = _collector(max_backlog=2, health_config=HealthConfig(max_errors=3))
    try:
        _step(collector, rollout_id=0)  # in flight
        assert started.wait(5)
        _step(collector, rollout_id=1)  # backlog full
        for rollout_id in (2, 3, 4):
            _step(collector, rollout_id=rollout_id)  # three drops reach max_errors
        _step(collector, rollout_id=5)  # this close disables and queues the notice
        assert collector.disabled
        gate.set()
        assert collector.flush(timeout=5)
        states = [r.metrics.get("straggler/health/state") for r in _step(collector, rollout_id=6)]
        assert 2.0 in states, f"expected health/state=2 in {states}"
    finally:
        gate.set()
        collector.close()

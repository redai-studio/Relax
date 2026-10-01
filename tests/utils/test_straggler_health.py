# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the profiler's own health state machine and how the collector
switches itself off without ever raising into training."""

import gc

import pytest

from relax.utils.straggler.health import (
    HEALTH_ACTIVE,
    HEALTH_DEGRADED,
    HEALTH_DISABLE,
    HealthConfig,
    ProfilerHealth,
)
from relax.utils.straggler.stats import FIELD_INDEX
from tests.utils.straggler_helpers import bracket, make_collector


def test_health_goes_degraded_on_drops_and_back_to_active_on_a_clean_window():
    health = ProfilerHealth(HealthConfig(max_errors=3, max_degraded_windows=3))
    assert health.close_window(dropped=0.0) == (0.0, float(HEALTH_ACTIVE))
    assert health.close_window(dropped=5.0) == (0.0, float(HEALTH_DEGRADED))
    assert health.state == "degraded"
    assert health.close_window(dropped=0.0) == (0.0, float(HEALTH_ACTIVE))
    assert health.state == "active"


def test_health_requests_disable_after_consecutive_degraded_windows():
    health = ProfilerHealth(HealthConfig(max_errors=100, max_degraded_windows=3))
    codes = [health.close_window(dropped=1.0)[1] for _ in range(3)]
    assert codes == [HEALTH_DEGRADED, HEALTH_DEGRADED, HEALTH_DISABLE]
    assert "3 consecutive" in health.request_reason()


def test_health_requests_disable_after_max_errors_even_across_windows():
    health = ProfilerHealth(HealthConfig(max_errors=3, max_degraded_windows=100))
    health.record_error("drain", RuntimeError("a"))
    assert health.close_window(dropped=0.0) == (1.0, float(HEALTH_DEGRADED))
    health.record_error("drain", RuntimeError("b"))
    health.record_error("analyze", ValueError("c"))
    errors, code = health.close_window(dropped=0.0)
    assert errors == 2.0 and code == HEALTH_DISABLE
    assert "3 errors" in health.request_reason()


def test_disabled_is_terminal():
    health = ProfilerHealth()
    health.disable("because")
    assert health.disabled and health.state == "disabled" and health.disabled_reason == "because"
    health.close_window(dropped=0.0)
    assert health.state == "disabled"


def _collector(world=2, gather=None, **kwargs):
    """``make_collector`` that also records what this rank sends."""
    calls = []

    def counting_gather(values):
        calls.append(list(values))
        return gather(values) if gather is not None else [list(values) for _ in range(world)]

    collector = make_collector(world, gather=counting_gather, **kwargs)
    collector.gather_calls = calls
    return collector


def test_drain_error_never_reaches_training_and_is_counted_in_the_window():
    collector = _collector()
    bracket(collector)
    _, start, _, _ = collector._pending[0]

    def boom(other):
        raise RuntimeError("device error")

    start.elapsed_time = boom
    collector.end_step(rollout_id=0)  # must not raise
    sent = collector.gather_calls[-1]
    assert sent[FIELD_INDEX["errors"]] == 1.0
    assert sent[FIELD_INDEX["health"]] == HEALTH_DEGRADED
    assert sent[FIELD_INDEX["unread"]] == 1.0, "the pair that failed to read is still queued"


def test_timer_errors_skip_the_bracket_and_are_counted():
    collector = _collector()

    def broken_factory():
        raise RuntimeError("cudaErrorLaunchFailure")

    collector._pool._free.clear()
    collector._pool._factory = broken_factory
    bracket(collector)  # must not raise
    assert collector.pending_events == 0
    assert collector.health.errors_total == 1


def test_one_rank_asking_to_disable_turns_the_profiler_off_on_every_rank():
    def gather(values):
        other = list(values)
        other[FIELD_INDEX["health"]] = float(HEALTH_DISABLE)
        return [list(values), other]

    collector = _collector(gather=gather, register_gc=True)
    before = len(gc.callbacks)
    bracket(collector)
    reports = collector.end_step(rollout_id=4)
    assert collector.disabled
    assert len(reports) == 1 and reports[0].metrics["straggler/health/state"] == HEALTH_DISABLE
    assert reports[0].metrics["straggler/health/requested_by"] == 1.0
    assert "rank 1" in reports[0].note
    # Off means off: no more collectives, no events, no GC hook, pool whole again.
    assert len(gc.callbacks) == before - 1
    bracket(collector)
    assert collector.pending_events == 0 and len(collector._pool) == 4
    assert collector.end_step(rollout_id=5) == []
    assert len(collector.gather_calls) == 1


def test_non_primary_rank_disables_too_but_returns_no_report():
    def gather(values):
        other = list(values)
        other[FIELD_INDEX["health"]] = float(HEALTH_DISABLE)
        return [other, list(values)]

    collector = _collector(gather=gather, is_primary=False)
    assert collector.end_step(rollout_id=0) == []
    assert collector.disabled


def test_gather_failure_disables_locally_without_raising():
    def gather(values):
        raise RuntimeError("gloo timeout")

    collector = _collector(gather=gather)
    bracket(collector)
    assert collector.end_step(rollout_id=0) == []
    assert collector.disabled
    assert "gather" in collector.health.disabled_reason


def test_repeated_drain_errors_escalate_to_disable_across_windows():
    collector = _collector(health_config=HealthConfig(max_errors=2, max_degraded_windows=100))

    def boom(other):
        raise RuntimeError("device error")

    bracket(collector)
    collector._pending[0][1].elapsed_time = boom
    collector.end_step(rollout_id=0)  # 1 error -> degraded
    assert not collector.disabled
    collector.end_step(rollout_id=1)  # same pair fails again -> 2 errors -> disable requested and applied
    assert collector.disabled


@pytest.mark.parametrize("dropped", [0, 3])
def test_health_scalars_ride_along_with_every_report(dropped):
    collector = _collector(max_pending=1)
    for _ in range(dropped + 1):
        bracket(collector)
    (report,) = collector.end_step(rollout_id=0)
    assert report.metrics["straggler/health/state"] == (HEALTH_DEGRADED if dropped else HEALTH_ACTIVE)
    assert report.metrics["straggler/health/errors"] == 0.0

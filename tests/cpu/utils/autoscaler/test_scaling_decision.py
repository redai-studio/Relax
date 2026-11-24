# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the autoscaler scaling decision engine."""

from relax.utils.autoscaler.config import AutoscalerConfig
from relax.utils.autoscaler.metrics_collector import AggregatedMetrics
from relax.utils.autoscaler.scaling_decision import (
    SCALE_IN_TERMINAL_STATUSES,
    SCALE_OUT_TERMINAL_STATUSES,
    ScalingAction,
    ScalingDecisionEngine,
    is_scale_request_terminal,
)


def _config(**overrides) -> AutoscalerConfig:
    """Build a config without debounce for single-frame tests."""
    cfg = AutoscalerConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.scale_out_policy.condition_duration_secs = 0.0
    cfg.scale_in_policy.condition_duration_secs = 0.0
    cfg.validate()
    return cfg


def _engine():
    return ScalingDecisionEngine(_config())


def _evaluate(engine, agg, current_engines, pending=None):
    return engine.evaluate(
        aggregated_metrics=agg,
        current_engines=current_engines,
        last_scale_time=None,  # no cooldown
        last_scale_action=None,
        pending_requests=pending or [],
    )


def _idle_metrics(coverage=1.0):
    """Non-empty snapshot where every value says 'no load' (true idle)."""
    return AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        total_running_reqs=0,
        avg_token_usage=0.0,
        total_throughput=100.0,
        throughput_variance=0.0,
        is_empty=False,
        coverage=coverage,
    )


def _busy_metrics(coverage=1.0):
    """Non-empty snapshot with high token usage -> scale-out condition met."""
    return AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        avg_token_usage=0.95,
        throughput_variance=0.0,
        is_empty=False,
        coverage=coverage,
    )


def test_scaling_decision_empty_snapshot_freezes_scale_in():
    """is_empty=True must NOT be read as low load -> no scale-in (fail-
    safe)."""
    engine = _engine()
    agg = AggregatedMetrics(is_empty=True, coverage=0.0)
    decision = _evaluate(engine, agg, current_engines=8)
    assert decision.action == ScalingAction.NONE
    assert "empty" in decision.reason.lower()


def test_scaling_decision_empty_snapshot_freezes_scale_out():
    """is_empty=True freezes scale-out too (both directions frozen)."""
    engine = _engine()
    # Even though current < max, an empty snapshot yields no data to act on.
    agg = AggregatedMetrics(is_empty=True, coverage=0.0)
    decision = _evaluate(engine, agg, current_engines=2)
    assert decision.action == ScalingAction.NONE


def test_scaling_decision_idle_load_allows_scale_in():
    engine = _engine()
    agg = _idle_metrics(coverage=1.0)
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.SCALE_IN
    assert decision.delta >= 1


def test_scaling_decision_partial_coverage_blocks_scale_in():
    """coverage=0.5 < min_coverage_scale_in(1.0) -> scale-in frozen."""
    engine = _engine()
    agg = _idle_metrics(coverage=0.5)
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.NONE


def test_scaling_decision_partial_coverage_allows_scale_out():
    """coverage=0.5 >= min_coverage_scale_out(0.5) -> scale-out allowed."""
    engine = _engine()
    agg = _busy_metrics(coverage=0.5)
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.SCALE_OUT
    assert decision.delta >= 1


def test_scaling_decision_coverage_below_scale_out_threshold_blocks_scale_out():
    """coverage below min_coverage_scale_out freezes scale-out even under
    load."""
    engine = _engine()
    agg = _busy_metrics(coverage=0.4)
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.NONE


def test_scaling_decision_full_coverage_allows_scale_out():
    engine = _engine()
    agg = _busy_metrics(coverage=1.0)
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.SCALE_OUT


def test_scaling_decision_partial_pending_does_not_block():
    """a PARTIAL scale-out request is terminal and must not block new decisions
    via the active_pending gate."""
    engine = _engine()
    agg = _busy_metrics(coverage=1.0)
    pending = [{"action": "scale_out", "status": "PARTIAL", "delta": 2}]
    decision = _evaluate(engine, agg, current_engines=4, pending=pending)
    assert decision.action == ScalingAction.SCALE_OUT


def test_scaling_decision_non_terminal_pending_blocks():
    """A genuinely non-terminal scale-out (CREATING) still blocks decisions."""
    engine = _engine()
    agg = _busy_metrics(coverage=1.0)
    pending = [{"action": "scale_out", "status": "CREATING", "delta": 2}]
    decision = _evaluate(engine, agg, current_engines=4, pending=pending)
    assert decision.action == ScalingAction.NONE
    assert "pending" in decision.reason.lower()


def test_is_scale_request_terminal_scale_out():
    for status in ("ACTIVE", "PARTIAL", "FAILED", "CANCELLED"):
        assert is_scale_request_terminal("scale_out", status) is True
    for status in ("PENDING", "CREATING", "CONNECTING", "READY", "WEIGHT_SYNCING"):
        assert is_scale_request_terminal("scale_out", status) is False


def test_is_scale_request_terminal_scale_in():
    for status in ("COMPLETED", "FAILED"):
        assert is_scale_request_terminal("scale_in", status) is True
    for status in ("PENDING", "DRAINING", "REMOVING"):
        assert is_scale_request_terminal("scale_in", status) is False
    # scale_in never reaches ACTIVE or CANCELLED (no such ScaleInStatus) -> not terminal
    assert is_scale_request_terminal("scale_in", "ACTIVE") is False
    assert is_scale_request_terminal("scale_in", "CANCELLED") is False


def test_scale_in_terminal_statuses_match_real_enum():
    """SCALE_IN_TERMINAL_STATUSES must only contain real ScaleInStatus terminal
    values.

    ScaleInStatus has no CANCELLED, so classifying "CANCELLED" as a scale-in
    terminal is a dead string that never matches a real request.
    """
    assert SCALE_IN_TERMINAL_STATUSES == frozenset({"COMPLETED", "FAILED"})
    assert is_scale_request_terminal("scale_in", "CANCELLED") is False


def test_is_scale_request_terminal_defaults_to_scale_out():
    """Unknown/missing action is treated as scale_out (matches service
    default)."""
    assert is_scale_request_terminal(None, "PARTIAL") is True
    assert is_scale_request_terminal("scale_out", None) is False


def test_terminal_sets_classify_partial_consistently():
    assert "PARTIAL" in SCALE_OUT_TERMINAL_STATUSES
    assert "PARTIAL" not in SCALE_IN_TERMINAL_STATUSES
    assert "ACTIVE" in SCALE_OUT_TERMINAL_STATUSES
    assert "COMPLETED" in SCALE_IN_TERMINAL_STATUSES


def test_scaling_decision_queue_latency_overflow_triggers_scale_out():
    engine = _engine()
    # max_queue_time_p95 (5.0) is BELOW the default 5.0 threshold via '>', so only
    # the explicit overflow flag can trigger the condition here.
    agg = AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        avg_token_usage=0.1,
        throughput_variance=0.0,
        is_empty=False,
        coverage=1.0,
        max_queue_time_p95=5.0,
        max_queue_time_p95_overflow=True,
    )
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.SCALE_OUT
    assert "queue_latency_high" in decision.triggered_conditions


def test_scaling_decision_ttft_overflow_triggers_scale_out():
    """Same for TTFT p95 overflow."""
    engine = _engine()
    agg = AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        avg_token_usage=0.1,
        throughput_variance=0.0,
        is_empty=False,
        coverage=1.0,
        max_ttft_p95=1.0,
        max_ttft_p95_overflow=True,
    )
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.SCALE_OUT
    assert "ttft_high" in decision.triggered_conditions


def test_scaling_decision_no_overflow_no_latency_trigger():
    """Sanity: without overflow and below thresholds, latency does not trigger.

    token_usage is set in the neutral band (0.3 < 0.5 < 0.85) so neither the
    scale-in nor the token scale-out condition fires -- isolating the latency
    conditions under test.
    """
    engine = _engine()
    agg = AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        avg_token_usage=0.5,
        throughput_variance=0.5,
        is_empty=False,
        coverage=1.0,
        max_queue_time_p95=1.0,
        max_ttft_p95=1.0,
        max_queue_time_p95_overflow=False,
        max_ttft_p95_overflow=False,
    )
    decision = _evaluate(engine, agg, current_engines=4)
    assert decision.action == ScalingAction.NONE


#
# These tests inject a fake MONOTONIC clock via the ScalingDecisionEngine(clock=)
# keyword. They never patch the global ``time`` module: cooldown keeps its own
# wall clock (disabled here via last_scale_time=None) and only the duration
# tracker consumes the injected clock (-- the two are kept separate).


class _FakeClock:
    """Deterministic injectable monotonic clock."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _duration_config(out_secs=30.0, in_secs=120.0, **overrides) -> AutoscalerConfig:
    """Config with explicit (NON-zero) condition durations for debounce
    tests."""
    cfg = AutoscalerConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.scale_out_policy.condition_duration_secs = out_secs
    cfg.scale_in_policy.condition_duration_secs = in_secs
    cfg.validate()
    return cfg


# ---- false -> true first valid frame must not count prior gap ----


def test_i1_false_then_true_first_frame_does_not_count_prior_gap():
    """idle@t0 -> busy@t30 keeps elapsed=0 (no trigger); the prior unsatisfied
    interval must NOT be credited to the streak."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    # t0: seed a false (unsatisfied) state via an idle frame.
    _evaluate(engine, _idle_metrics(), current_engines=4)

    # t30: FIRST satisfied frame -> elapsed=0 -> below 30s -> no scale-out.
    clock.advance(30.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)
    assert d.action == ScalingAction.NONE


def test_i1_true_then_true_accumulates_interval():
    """busy@t30 -> busy@t60 accumulates exactly 30s and triggers."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _idle_metrics(), current_engines=4)  # seed false @ t0
    clock.advance(30.0)
    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.NONE  # first true, 0s
    clock.advance(30.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)  # now 30s
    assert d.action == ScalingAction.SCALE_OUT
    assert "token_usage_high" in d.triggered_conditions


def test_i1_transient_spike_below_duration_does_not_trigger():
    """A short busy spike (< duration) never triggers."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # first true, elapsed 0
    clock.advance(10.0)
    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.NONE  # 10s < 30s


def test_i1_sustained_above_duration_triggers():
    """Continuous busy for >= duration triggers scale-out."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 0
    clock.advance(15.0)
    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 15
    clock.advance(15.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 30
    assert d.action == ScalingAction.SCALE_OUT


def test_i1_drop_below_resets_streak():
    """A break in the streak (unsatisfied frame) resets elapsed to zero."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 0
    clock.advance(20.0)
    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 20
    clock.advance(5.0)
    _evaluate(engine, _idle_metrics(), current_engines=4)  # break -> reset
    clock.advance(30.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)  # first true again, elapsed 0
    assert d.action == ScalingAction.NONE


# ---- continuity gap / clock jump must reset, not fill in time ----


def test_i1_event_loop_gap_resets_and_first_frame_not_triggered():
    """dt > max_gap (2x evaluation_interval) => reset; the recovery frame is a
    new first frame and must not trigger even though wall time passed."""
    clock = _FakeClock(0.0)
    # evaluation_interval default 30 -> max_gap 60.
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 0
    clock.advance(1000.0)  # loop stalled way beyond max_gap
    d = _evaluate(engine, _busy_metrics(), current_engines=4)
    assert d.action == ScalingAction.NONE  # reset, treated as new first frame


def test_i1_negative_clock_jump_resets():
    """a backward clock jump (dt < 0) resets the streak."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 0
    clock.advance(25.0)
    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 25
    clock.t = 5.0  # clock jumped backwards
    d = _evaluate(engine, _busy_metrics(), current_engines=4)
    assert d.action == ScalingAction.NONE  # dt<0 -> reset


def test_i1_recovery_after_gap_accumulates_normally():
    """After a gap reset, normal-interval frames accumulate again."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)
    clock.advance(1000.0)  # gap -> reset
    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.NONE  # new first frame
    clock.advance(30.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)
    assert d.action == ScalingAction.SCALE_OUT


# ---- duration == 0 => immediate trigger on first valid frame ----


def test_i1_duration_zero_triggers_on_first_frame_scale_out():
    """condition_duration_secs=0 triggers on the first valid satisfied frame
    (no debounce)."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=0.0), clock=clock)
    d = _evaluate(engine, _busy_metrics(), current_engines=4)
    assert d.action == ScalingAction.SCALE_OUT


def test_i1_duration_zero_triggers_on_first_frame_scale_in():
    """(scale-in side): duration=0 -> first idle frame scales in."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(in_secs=0.0), clock=clock)
    d = _evaluate(engine, _idle_metrics(), current_engines=4)
    assert d.action == ScalingAction.SCALE_IN


# ---- scale-in: ALL conditions must be sustained together ----


def test_i1_scale_in_all_conditions_sustained_triggers():
    """scale-in requires ALL conditions sustained for >= duration."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(in_secs=60.0), clock=clock)

    _evaluate(engine, _idle_metrics(), current_engines=4)  # elapsed 0
    clock.advance(30.0)
    assert _evaluate(engine, _idle_metrics(), current_engines=4).action == ScalingAction.NONE  # 30 < 60
    clock.advance(30.0)
    d = _evaluate(engine, _idle_metrics(), current_engines=4)  # 60
    assert d.action == ScalingAction.SCALE_IN
    assert engine.is_scale_in_sustained() is True


def test_i1_scale_in_stale_frame_resets_streak():
    """stale reset: an is_empty frame (collection failed) mid-streak resets
    scale-in; the recovered frame must not immediately scale in."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(in_secs=30.0), clock=clock)

    _evaluate(engine, _idle_metrics(), current_engines=4)  # elapsed 0
    clock.advance(30.0)
    # empty snapshot -> both directions reset (and evaluate returns NONE/frozen)
    empty = AggregatedMetrics(is_empty=True, coverage=0.0)
    assert _evaluate(engine, empty, current_engines=4).action == ScalingAction.NONE
    clock.advance(30.0)
    d = _evaluate(engine, _idle_metrics(), current_engines=4)  # first true again
    assert d.action == ScalingAction.NONE
    assert engine.is_scale_in_sustained() is False


def test_i1_scale_in_partial_coverage_resets_streak():
    """coverage < min_coverage_scale_in resets the scale-in streak (frame
    invalid)."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(in_secs=30.0), clock=clock)

    _evaluate(engine, _idle_metrics(coverage=1.0), current_engines=4)  # elapsed 0
    clock.advance(30.0)
    # partial coverage -> invalid frame for scale-in -> reset (evaluate also gates)
    _evaluate(engine, _idle_metrics(coverage=0.5), current_engines=4)
    clock.advance(30.0)
    d = _evaluate(engine, _idle_metrics(coverage=1.0), current_engines=4)  # first true again
    assert d.action == ScalingAction.NONE


def test_reset_condition_trackers_clears_scale_in_streak():
    """an unobservable cycle (engine discovery yielded nothing) must be able to
    reset the debounce streaks explicitly, so accumulated satisfied time does
    not carry through and fire immediately once engines reappear."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(in_secs=30.0), clock=clock)

    _evaluate(engine, _idle_metrics(), current_engines=4)  # seed first true, elapsed 0
    clock.advance(20.0)
    _evaluate(engine, _idle_metrics(), current_engines=4)  # elapsed 20 (< 30, not sustained)

    # Discovery failed this cycle -> reset trackers explicitly (no metrics seen).
    engine.reset_condition_trackers()

    clock.advance(30.0)
    d = _evaluate(engine, _idle_metrics(), current_engines=4)  # first true AGAIN -> elapsed 0
    assert d.action == ScalingAction.NONE
    assert engine.is_scale_in_sustained() is False


# ---- tracker keeps advancing while pending gate is closed ----


def test_i1_streak_advances_across_pending_cycles():
    """The tracker is updated BEFORE the pending gate, so a streak keeps
    advancing while a pending request blocks decisions; once it clears the
    already-sustained condition triggers immediately."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)
    pending = [{"action": "scale_out", "status": "CREATING", "delta": 2}]

    _evaluate(engine, _busy_metrics(), current_engines=4, pending=pending)  # elapsed 0, blocked
    clock.advance(30.0)
    blocked = _evaluate(engine, _busy_metrics(), current_engines=4, pending=pending)  # elapsed 30, still blocked
    assert blocked.action == ScalingAction.NONE
    assert "pending" in blocked.reason.lower()
    clock.advance(1.0)
    d = _evaluate(engine, _busy_metrics(), current_engines=4, pending=[])  # pending cleared
    assert d.action == ScalingAction.SCALE_OUT


# ---- production defaults (30s / 120s) really gate (end-to-end) ----


def test_i1_production_default_scale_out_gates_at_30s():
    """End-to-end with PRODUCTION defaults: scale-out needs a full 30s
    streak."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(AutoscalerConfig(), clock=clock)  # defaults 30/120

    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.NONE  # first frame
    clock.advance(29.0)
    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.NONE  # 29 < 30
    clock.advance(1.0)
    assert _evaluate(engine, _busy_metrics(), current_engines=4).action == ScalingAction.SCALE_OUT  # 30


def test_i1_production_default_scale_in_gates_at_120s():
    """End-to-end with PRODUCTION defaults: scale-in needs a full 120s
    streak."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(AutoscalerConfig(), clock=clock)  # defaults 30/120

    _evaluate(engine, _idle_metrics(), current_engines=4)  # first frame, 0
    clock.advance(60.0)
    assert _evaluate(engine, _idle_metrics(), current_engines=4).action == ScalingAction.NONE  # 60 < 120
    clock.advance(60.0)
    assert _evaluate(engine, _idle_metrics(), current_engines=4).action == ScalingAction.SCALE_IN  # 120


def test_i1_condition_observation_snapshot_does_not_advance_tracker():
    """condition_observation() is a pure GET: repeated calls (with the clock
    advancing) must not change elapsed -- only evaluate() advances the
    tracker."""
    clock = _FakeClock(0.0)
    engine = ScalingDecisionEngine(_duration_config(out_secs=30.0), clock=clock)

    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 0
    clock.advance(15.0)
    _evaluate(engine, _busy_metrics(), current_engines=4)  # elapsed 15
    obs1 = engine.condition_observation()
    cond1 = obs1["scale_out"]["conditions"]["token_usage_high"]
    assert cond1["elapsed_secs"] == 15.0
    assert cond1["satisfied"] is True
    assert cond1["sustained"] is False  # 15 < 30
    assert obs1["scale_out"]["condition_duration_secs"] == 30.0

    # Advance the clock and read again WITHOUT evaluate -> elapsed unchanged.
    clock.advance(100.0)
    obs2 = engine.condition_observation()
    assert obs2["scale_out"]["conditions"]["token_usage_high"]["elapsed_secs"] == 15.0


def test_i1_config_rejects_negative_scale_out_duration():
    import pytest

    with pytest.raises(ValueError, match="condition_duration_secs must be >= 0"):
        _duration_config(out_secs=-1.0)


def test_i1_config_rejects_negative_scale_in_duration():
    import pytest

    with pytest.raises(ValueError, match="condition_duration_secs must be >= 0"):
        _duration_config(in_secs=-5.0)

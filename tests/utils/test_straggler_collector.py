# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-rank collector: lazy readout, windows, gather, GC
hook, role gating."""

import gc

import pytest

from relax.utils.straggler import collector as collector_module
from relax.utils.straggler.collector import StragglerCollector, install_straggler_collector
from relax.utils.straggler.detector import DetectorConfig, RankMeta
from relax.utils.straggler.stats import FIELD_INDEX
from tests.utils.straggler_helpers import FakeEvent, meta


class _FakeCapture:
    """Toggle standing in for ``torch.cuda.is_current_stream_capturing``."""

    def __init__(self):
        self.active = False

    def __call__(self):
        return self.active


def _collector(world, interval=2, is_primary=True, register_gc=False, gather=None, is_capturing=None, **kwargs):
    gather_calls, meta_calls = [], []

    def counting_gather(values):
        gather_calls.append(list(values))
        return gather(values) if gather is not None else [values] * world

    def counting_gather_objects(obj):
        meta_calls.append(obj)
        return meta(world)

    collector = StragglerCollector(
        rank_meta=RankMeta(rank=0, dp=0, tp=0, pp=0),
        is_primary=is_primary,
        report_interval=interval,
        detector_config=DetectorConfig(persist_windows=1),
        event_factory=FakeEvent,
        pool_size=4,
        gather=counting_gather,
        gather_objects=counting_gather_objects,
        register_gc_callback=register_gc,
        is_capturing=is_capturing or _FakeCapture(),
        **kwargs,
    )
    collector.gather_calls = gather_calls
    collector.meta_calls = meta_calls
    return collector


def _one_bracket(collector, name="forward-compute"):
    collector.train_timers(name, log_level=2).start()
    collector.train_timers(name).stop()


def test_drain_folds_completed_events_in_order_and_recycles_them():
    collector = _collector(world=1)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    # Hold the second pair back: its end event is "still running".
    _, _, pending_end, _ = collector._pending[1]
    pending_end.done = False
    assert collector.pending_events == 2
    assert collector.drain() == 1
    assert collector.pending_events == 1
    assert collector.window.get("fwd") == 1.0
    assert collector.window.get("num_fwd") == 1.0
    assert collector.window.get("bwd") == 0.0
    pending_end.done = True
    assert collector.drain() == 1
    assert collector.window.get("bwd") == 1.0
    assert collector.window.get("cpu_bwd") >= 0.0
    assert collector.window.get("overhead") >= 0.0
    # The pool got all four events back.
    assert len(collector._pool) == 4


def test_drain_stops_at_first_incomplete_pair_and_requires_both_events():
    collector = _collector(world=1)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    # Pairs complete in stream order: an incomplete *start* of the first pair
    # blocks the (complete) second pair as well, so nothing is folded.
    _, first_start, _, _ = collector._pending[0]
    first_start.done = False
    assert collector.drain() == 0
    assert collector.pending_events == 2
    first_start.done = True
    assert collector.drain() == 2


def test_drain_leaves_queue_consistent_when_readout_raises():
    collector = _collector(world=1)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    _, start, _, _ = collector._pending[1]

    def boom(other):
        raise RuntimeError("device error")

    start.elapsed_time = boom
    with pytest.raises(RuntimeError):
        collector.drain()
    # First pair folded once and released; the failing pair is still queued and
    # its events were not handed back to the pool.
    assert collector.window.get("fwd") == 1.0
    assert collector.pending_events == 1
    assert len(collector._pool) == 2
    with pytest.raises(RuntimeError):
        collector.drain()
    assert collector.window.get("fwd") == 1.0  # not double counted


def test_pending_queue_is_bounded_and_counts_drops():
    collector = _collector(world=1, max_pending=2)
    for _ in range(4):
        _one_bracket(collector)
    assert collector.pending_events == 2
    assert collector.window.get("dropped") == 2.0
    # Dropped pairs go back to the pool, so the pool stops growing once the queue is full.
    created = collector._pool.created
    assert created <= 2 * (2 + 1)
    for _ in range(20):
        _one_bracket(collector)
    assert collector._pool.created == created
    assert collector.window.get("dropped") == 22.0


def test_no_events_are_recorded_while_the_stream_is_being_captured():
    capture = _FakeCapture()
    collector = _collector(world=1, is_capturing=capture)
    capture.active = True
    _one_bracket(collector)
    assert collector.pending_events == 0 and len(collector._pool) == 4
    # Capture begins between start and stop: the started event is recycled.
    capture.active = False
    collector.train_timers("forward-compute").start()
    capture.active = True
    collector.train_timers("forward-compute").stop()
    assert collector.pending_events == 0 and len(collector._pool) == 4
    capture.active = False
    _one_bracket(collector)
    assert collector.pending_events == 1


def test_end_step_reports_every_interval_and_resets_window():
    collector = _collector(world=2, interval=3)
    reports = []
    for _ in range(6):
        _one_bracket(collector)
        collector.add_tokens(1000)
        reports.append(collector.end_step())
    assert [report is not None for report in reports] == [False, False, True, False, False, True]
    report = reports[2]
    assert report.metrics["straggler/fwd/median_ms"] == pytest.approx(1.0)  # per step
    assert report.metrics["straggler/tokens/median"] == pytest.approx(1000.0)
    assert report.metrics["straggler/flagged/count"] == 0
    assert "straggler/gather_ms" in report.metrics
    assert report.window_start_wall > 0.0
    assert collector.window.get("fwd") == 0.0 and collector.window.get("num_steps") == 0.0
    assert len(collector.gather_calls) == 2
    assert len(collector.meta_calls) == 1, "rank metadata is static and must be gathered only once"


def test_gather_and_analysis_are_timed_separately(monkeypatch):
    now = [0.0]
    real_analyze = collector_module.analyze_window

    def slow_gather(values):
        now[0] += 0.005
        return [values, values]

    def slow_analyze(*args):
        now[0] += 0.007
        return real_analyze(*args)

    monkeypatch.setattr(collector_module, "analyze_window", slow_analyze)
    collector = _collector(world=2, interval=1, gather=slow_gather, clock=lambda: now[0])
    _one_bracket(collector)
    report = collector.end_step()
    assert report.metrics["straggler/gather_ms"] == pytest.approx(5.0), "gather_ms must stop before the analysis"
    assert report.metrics["straggler/analyze_ms"] == pytest.approx(7.0)


def test_non_primary_rank_participates_but_returns_no_report():
    collector = _collector(world=2, interval=1, is_primary=False)
    _one_bracket(collector)
    assert collector.end_step() is None
    assert len(collector.gather_calls) == 1
    assert collector.gather_calls[0][FIELD_INDEX["fwd"]] == 1.0
    assert collector.window.get("fwd") == 0.0


def test_gather_table_from_other_ranks_drives_detection():
    # Rank 1 reports 3x compute for the same tokens -> flagged as slow_device.
    def gather(values):
        slow = list(values)
        for name in ("fwd", "bwd", "optim"):
            slow[FIELD_INDEX[name]] *= 3.0
        return [values, slow]

    collector = _collector(world=2, interval=1, gather=gather)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    collector.add_tokens(500)
    report = collector.end_step()
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.alerts and report.alerts[0].reason == "slow_device"


def test_gc_is_attributed_only_while_a_step_is_open():
    collector = _collector(world=1, register_gc=True)
    before = len(gc.callbacks)
    try:
        gc.collect()  # before any bracket: e.g. clear_memory() during offload
        assert collector.window.get("gc") == 0.0
        _one_bracket(collector)
        gc.collect()
        assert collector.window.get("gc") > 0.0
        collector.end_step()
        collector.end_step()  # closes the window
        gc.collect()  # between steps (train_wait)
        assert collector.window.get("gc") == 0.0
    finally:
        collector.close()
    assert len(gc.callbacks) == before - 1
    collector.close()  # idempotent


def test_forward_only_timers_land_in_lp_buckets():
    collector = _collector(world=1)
    collector.forward_only_timers("forward-compute", log_level=2).start()
    collector.forward_only_timers("forward-compute").stop()
    collector.drain()
    assert collector.window.get("lp_fwd") == 1.0
    assert collector.window.get("fwd") == 0.0


def test_report_interval_must_be_positive():
    with pytest.raises(ValueError):
        _collector(world=1, interval=0)


def test_install_is_a_no_op_when_disabled_or_for_non_training_roles(monkeypatch):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "0")
    assert install_straggler_collector("actor") is None
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    # These roles never call end_step, so they must not get a collector
    # (checked before any Megatron parallel state is touched).
    for role in ("critic", "reference", "actor_fwd"):
        assert install_straggler_collector(role) is None

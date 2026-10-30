# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-rank collector: lazy readout, windows, gather, GC
hook."""

import gc
import threading
from dataclasses import replace

import pytest

from relax.utils.straggler.analysis import report as report_module
from relax.utils.straggler.core.health import HEALTH_DISABLE
from relax.utils.straggler.core.modules import module_segment
from relax.utils.straggler.core.stats import FIELD_INDEX, GPU_SEGMENTS, PP_PHASE_FIELDS, RECORD_WIDTH
from relax.utils.straggler.megatron_adapter import GRAD_SYNC_TIMERS, MODULE_NAMES
from tests.utils.straggler_helpers import FakeEvent, bracket, make_collector, meta, timers


_ATTN_FWD = module_segment(MODULE_NAMES, "attn", False)


class _FakeCapture:
    """Toggle standing in for ``torch.cuda.is_current_stream_capturing``."""

    def __init__(self):
        self.active = False

    def __call__(self):
        return self.active


def _collector(world, interval=2, gather=None, **kwargs):
    """``make_collector`` that also records every gather call."""
    gather_calls, meta_calls = [], []

    def counting_gather(values):
        gather_calls.append(list(values))
        return gather(values) if gather is not None else [values] * world

    def counting_gather_objects(obj):
        meta_calls.append(obj)
        return meta(world)

    collector = make_collector(
        world, interval, gather=counting_gather, gather_objects=counting_gather_objects, **kwargs
    )
    collector.gather_calls = gather_calls
    collector.meta_calls = meta_calls
    return collector


def test_drain_folds_completed_events_in_order_and_recycles_them():
    collector = _collector(world=1)
    bracket(collector)
    bracket(collector, "backward-compute")
    # Hold the second pair back: its end event is "still running".
    _, _, _, pending_end, _, _, _ = collector.recorder._pending[1]
    pending_end.done = False
    assert collector.recorder.pending_events == 2
    assert collector.recorder.drain() == 1
    assert collector.recorder.pending_events == 1
    assert collector.window.get("fwd") == 1.0
    assert collector.window.get("num_fwd") == 1.0
    assert collector.window.get("bwd") == 0.0
    pending_end.done = True
    assert collector.recorder.drain() == 1
    assert collector.window.get("bwd") == 1.0
    assert collector.window.get("cpu_bwd") >= 0.0
    assert collector.window.get("overhead") >= 0.0
    # The pool got all four events back.
    assert len(collector.recorder._pool) == 4


def test_drain_stops_at_first_incomplete_pair_and_requires_both_events():
    collector = _collector(world=1)
    bracket(collector)
    bracket(collector, "backward-compute")
    # Pairs complete in stream order: an incomplete *start* of the first pair
    # blocks the (complete) second pair as well, so nothing is folded.
    _, _, first_start, _, _, _, _ = collector.recorder._pending[0]
    first_start.done = False
    assert collector.recorder.drain() == 0
    assert collector.recorder.pending_events == 2
    first_start.done = True
    assert collector.recorder.drain() == 2


def test_drain_leaves_queue_consistent_when_readout_raises():
    collector = _collector(world=1)
    bracket(collector)
    bracket(collector, "backward-compute")
    _, _, start, _, _, _, _ = collector.recorder._pending[1]

    def boom(other):
        raise RuntimeError("device error")

    start.elapsed_time = boom
    with pytest.raises(RuntimeError):
        collector.recorder.drain()
    # First pair folded once and released; the failing pair is still queued and
    # its events were not handed back to the pool.
    assert collector.window.get("fwd") == 1.0
    assert collector.recorder.pending_events == 1
    assert len(collector.recorder._pool) == 2
    with pytest.raises(RuntimeError):
        collector.recorder.drain()
    assert collector.window.get("fwd") == 1.0  # not double counted


def test_pending_queue_is_bounded_and_counts_drops():
    collector = _collector(world=1, max_pending=2)
    for _ in range(4):
        bracket(collector)
    assert collector.recorder.pending_events == 2
    assert collector.window.get("dropped") == 2.0
    # Dropped pairs go back to the pool, so the pool stops growing once the queue is full.
    created = collector.recorder._pool.created
    assert created <= 2 * (2 + 1)
    for _ in range(20):
        bracket(collector)
    assert collector.recorder._pool.created == created
    assert collector.window.get("dropped") == 22.0


def test_no_events_are_recorded_while_the_stream_is_being_captured():
    capture = _FakeCapture()
    collector = _collector(world=1, is_capturing=capture)
    capture.active = True
    bracket(collector)
    assert collector.recorder.pending_events == 0 and len(collector.recorder._pool) == 4
    # Capture begins between start and stop: the started event is recycled.
    capture.active = False
    timers(collector)("forward-compute").start()
    capture.active = True
    timers(collector)("forward-compute").stop()
    assert collector.recorder.pending_events == 0 and len(collector.recorder._pool) == 4
    capture.active = False
    bracket(collector)
    assert collector.recorder.pending_events == 1


def test_end_step_reports_every_interval_and_resets_window():
    collector = _collector(world=2, interval=3)
    reports = []
    for _ in range(6):
        collector.recorder.note_tokens(1000)
        bracket(collector)
        reports.append(collector.end_step())
    assert [len(ready) for ready in reports] == [0, 0, 1, 0, 0, 1]
    report = reports[2][0]
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
    real_analyze = report_module.analyze_window

    def slow_gather(values):
        now[0] += 0.005
        return [values, values]

    def slow_analyze(*args):
        now[0] += 0.007
        return real_analyze(*args)

    monkeypatch.setattr(report_module, "analyze_window", slow_analyze)
    collector = _collector(world=2, interval=1, gather=slow_gather, clock=lambda: now[0])
    bracket(collector)
    (report,) = collector.end_step()
    assert report.metrics["straggler/gather_ms"] == pytest.approx(5.0), "gather_ms must stop before the analysis"
    assert report.metrics["straggler/analyze_ms"] == pytest.approx(7.0)


def test_report_carries_window_close_and_analysis_wall_times():
    now = [0.0]

    def wall():
        now[0] += 1.0
        return now[0]

    collector = _collector(world=2, interval=1, wall_clock=wall)  # window start = 1
    bracket(collector)
    (report,) = collector.end_step()
    assert (report.window_start_wall, report.closed_wall, report.analyzed_wall) == (1.0, 2.0, 3.0)
    assert collector.window_start_wall == 4.0


def test_non_primary_rank_participates_but_returns_no_report():
    collector = _collector(world=2, interval=1, is_primary=False)
    bracket(collector)
    assert collector.end_step() == []
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
    collector.recorder.note_tokens(500)
    bracket(collector)
    bracket(collector, "backward-compute")
    (report,) = collector.end_step()
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.alerts and report.alerts[0].reason == "slow_device"


def test_gc_is_attributed_only_while_a_step_is_open():
    collector = _collector(world=1, register_gc=True)
    before = len(gc.callbacks)
    try:
        gc.collect()  # before any bracket: e.g. clear_memory() during offload
        assert collector.window.get("gc") == 0.0
        bracket(collector)
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
    timers(collector, "forward_only")("forward-compute", log_level=2).start()
    timers(collector, "forward_only")("forward-compute").stop()
    collector.recorder.drain()
    assert collector.window.get("lp_fwd") == 1.0
    assert collector.window.get("fwd") == 0.0


def test_the_window_counts_a_micro_batch_once_whatever_its_model_chunks():
    collector = _collector(world=1)
    collector.recorder.model_chunks = 2
    for tokens in (300, 300, 500, 500):  # each model chunk's forward of micro-batches 0 and 1
        collector.recorder.note_tokens(tokens)
        bracket(collector)
    collector.recorder.drain()
    assert collector.window.get("tokens") == 800.0
    assert collector.window.get("mb_tokens_max") == 500.0


def _timed(collector, name, ms, tokens=None):
    """One bracket lasting ``ms`` fake milliseconds; ``tokens`` is reported
    while it is open, as the training forward_step does."""
    timer = timers(collector)(name)
    timer.start()
    if tokens is not None:
        collector.recorder.note_tokens(tokens)
    for _ in range(ms - 1):
        FakeEvent().record()
    timer.stop()


def test_unnumbered_brackets_carry_no_micro_batch_number_but_keep_their_tokens():
    # Training under combined 1F1B: the schedule table names no bracket.
    collector, calls = _tracing_collector(interleaved=True)
    for rollout_id in range(2):
        collector.end_step(rollout_id=rollout_id)
    collector.recorder.note_schedule(lambda: (), forward_only=False)
    _timed(collector, "forward-compute", 3, tokens=300)
    _timed(collector, "forward-compute", 7, tokens=900)
    _timed(collector, "backward-compute", 5)
    collector.end_step(rollout_id=2)
    window = collector.window
    assert [window.get(name) for name in ("mb_fwd_max", "mb_fwd_argmax", "mb_fwd_argmax_tokens")] == [7.0, -1.0, 900.0]
    collector.end_step(rollout_id=3)
    sample, _ = calls["gathered"][-1]
    assert [(int(r[0]), int(r[2]), int(r[3])) for r in _rows(sample)] == [
        (FIELD_INDEX["fwd"], -1, 300),
        (FIELD_INDEX["fwd"], -1, 900),
        (FIELD_INDEX["bwd"], -1, -1),
    ]


# Megatron's interleaved schedule table for 5 micro-batches, 2 model chunks and groups of 3 (its own example).
_TABLE_5 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (3, 1), (4, 1)]


def test_under_interleaved_pp_brackets_take_the_micro_batch_and_model_chunk_of_the_schedule_table():
    collector, calls = _tracing_collector(interleaved=True)
    for rollout_id in range(2):
        collector.end_step(rollout_id=rollout_id)
    collector.recorder.note_schedule(lambda: _TABLE_5, forward_only=False)
    for name in ("forward-compute", "backward-compute"):
        for _ in _TABLE_5:
            bracket(collector, name)
    # The next call of the step continues the micro-batch numbers; a bracket beyond its table has none,
    # and neither has a log-prob forward, which a training call's table does not cover.
    collector.recorder.note_schedule(lambda: [(0, 0), (0, 1)], forward_only=False)
    for _ in range(3):
        bracket(collector)
    bracket(collector, phase="forward_only")
    collector.end_step(rollout_id=2)
    collector.end_step(rollout_id=3)
    sample, _ = calls["gathered"][-1]
    fwd, bwd = FIELD_INDEX["fwd"], FIELD_INDEX["bwd"]
    # Backward steps follow the table with the model chunks reversed.
    assert [(int(r[0]), int(r[2]), int(r[1])) for r in _rows(sample)] == [
        *((fwd, micro_batch, chunk) for micro_batch, chunk in _TABLE_5),
        *((bwd, micro_batch, 1 - chunk) for micro_batch, chunk in _TABLE_5),
        (fwd, 5, 0),
        (fwd, 5, 1),
        (fwd, -1, -1),
        (FIELD_INDEX["lp_fwd"], -1, -1),
    ]


def test_gradient_syncs_count_in_dp_grad_sync_only():
    collector = _collector(world=1)
    for name in GRAD_SYNC_TIMERS:
        bracket(collector, name)
    collector.recorder.drain()
    assert [collector.window.get(name) for name in GPU_SEGMENTS] == [
        float(len(GRAD_SYNC_TIMERS)) if name == "dp_grad_sync" else 0.0 for name in GPU_SEGMENTS
    ]


def test_training_pp_waits_are_also_split_by_1f1b_phase():
    collector = _collector(world=1)
    for name in ("forward-recv", "backward-send-forward-recv", "forward-send-backward-recv", "backward-recv"):
        bracket(collector, name)
    bracket(collector, "forward-send-forward-recv")  # interleaved schedule: total only
    bracket(collector, "forward-recv", phase="forward_only")
    collector.recorder.drain()
    assert collector.window.get("pp_recv") == 5.0
    assert [collector.window.get(name) for name in PP_PHASE_FIELDS] == [1.0, 1.0, 1.0, 1.0]
    assert collector.window.get("lp_pp_recv") == 1.0


def test_micro_batch_extremes_follow_the_slowest_forward_and_its_tokens():
    collector = _collector(world=1, interval=10)
    _timed(collector, "forward-compute", 3, tokens=300)
    _timed(collector, "forward-compute", 7, tokens=900)
    _timed(collector, "forward-compute", 2)
    _timed(collector, "backward-compute", 5)
    _timed(collector, "backward-compute", 4)
    collector.end_step(rollout_id=0)
    window = collector.window
    assert (window.get("mb_fwd_max"), window.get("mb_fwd_min")) == (7.0, 2.0)
    assert (window.get("mb_fwd_argmax"), window.get("mb_fwd_argmax_tokens")) == (1.0, 900.0)
    assert (window.get("mb_bwd_max"), window.get("mb_bwd_min")) == (5.0, 4.0)
    assert (window.get("mb_tokens_max"), window.get("mb_tokens_min")) == (900.0, 300.0)

    # Numbering restarts every step; a forward without a reported count has -1 tokens.
    _timed(collector, "forward-compute", 1)
    _timed(collector, "forward-compute", 9)
    collector.end_step(rollout_id=1)
    assert (window.get("mb_fwd_argmax"), window.get("mb_fwd_argmax_tokens")) == (1.0, -1.0)
    assert window.get("mb_fwd_min") == 1.0


def test_the_window_keeps_the_slowest_single_step_of_each_peak_segment():
    collector = _collector(world=1, interval=3)
    for step in ((4,), (3, 5), (6,)):
        for ms in step:
            _timed(collector, "forward-compute", ms)
        collector.end_step()
    row = collector.gather_calls[-1]
    assert (row[FIELD_INDEX["fwd"]], row[FIELD_INDEX["fwd_step_max"]]) == (18.0, 8.0)
    assert collector.window.get("fwd_step_max") == 0.0


def _tracing_collector(world=2, interval=2, trace=True, primary=True, level=2, **kwargs):
    """A tracing collector for rank 0 (the primary unless ``primary`` is False)
    that records its synchronizes, trace gathers and sink calls."""
    calls = {"sync": 0, "gathered": [], "sunk": []}

    def synchronize():
        calls["sync"] += 1

    def gather_to(sample, dst):
        calls["gathered"].append((sample, dst))
        return [sample] * world

    def sink(traces, metas, first_rollout):
        calls["sunk"].append(first_rollout)
        return {"straggler/trace/ranks": float(sum(t is not None for t in traces))}

    metas = [replace(m, primary=primary and m.rank == 0) for m in meta(world)]
    collector = make_collector(
        world,
        interval,
        gather_objects=lambda obj: metas,
        trace=trace,
        synchronize=synchronize,
        gather_to=gather_to,
        primary_rank=0 if primary else None,
        trace_sink=sink,
        level=level,
        **kwargs,
    )
    return collector, calls


def _recording_gather():
    """A window gather that keeps this rank's rows and returns two copies."""
    rows = []

    def gather(values):
        rows.append(list(values))
        return [list(values)] * 2

    return rows, gather


def _rows(sample):
    values = list(sample.records)
    return [values[i : i + RECORD_WIDTH] for i in range(0, len(values), RECORD_WIDTH)]


def test_tracing_samples_the_first_step_of_every_window_after_the_first():
    collector, calls = _tracing_collector()
    reports = []
    for rollout_id in range(4):
        bracket(collector)
        bracket(collector, "backward-compute")
        reports += collector.end_step(rollout_id=rollout_id)
    assert calls["sync"] == 1
    (first, first_dst), (second, second_dst) = calls["gathered"]
    assert (first, first_dst, second_dst) == (None, 0, 0)
    # Only rollout 2; FakeEvent stamps the anchor, then each start / stop one apart.
    assert [(int(row[0]), int(row[2]), row[4], row[5]) for row in _rows(second)] == [
        (FIELD_INDEX["fwd"], 0, 1.0, 2.0),
        (FIELD_INDEX["bwd"], 0, 3.0, 4.0),
    ]
    assert calls["sunk"] == [0, 2]
    assert reports[-1].metrics["straggler/trace/ranks"] == 2.0


def test_a_sample_is_capped_and_carries_the_cpu_events_of_its_step():
    collector, calls = _tracing_collector(
        max_trace_brackets=2, cpu_events=lambda start, end: [{"name": "get_data_batch"}]
    )
    for rollout_id in range(2):
        collector.end_step(rollout_id=rollout_id)
    for _ in range(3):
        bracket(collector)
    collector.end_step(rollout_id=2)
    collector.end_step(rollout_id=3)
    sample, _ = calls["gathered"][-1]
    assert (len(_rows(sample)), sample.lost) == (2, 1)
    assert sample.cpu_events == [{"name": "get_data_batch"}]


def test_a_bracket_an_error_skips_in_the_sampled_step_counts_as_lost_and_one_after_that_step_does_not():
    failing = []

    def is_capturing():
        if failing:
            raise RuntimeError("record failed")
        return False

    collector, calls = _tracing_collector(is_capturing=is_capturing)

    def failing_bracket():
        timer = timers(collector)("forward-compute")
        timer.start()
        failing.append(True)
        timer.stop()
        failing.clear()

    for rollout_id in range(2):
        collector.end_step(rollout_id=rollout_id)
    bracket(collector)
    failing_bracket()
    # The sampled step's bracket is still in flight when the next step's bracket fails.
    _, _, _, end, _, _, _ = collector.recorder._pending[0]
    end.done = False
    collector.end_step(rollout_id=2)
    failing_bracket()
    end.done = True
    collector.end_step(rollout_id=3)
    sample, _ = calls["gathered"][-1]
    assert (len(_rows(sample)), sample.lost, collector.health.errors_total) == (1, 1, 2)


def test_with_one_rollout_windows_every_sample_carries_the_cpu_events_of_its_own_rollout():
    read = []

    def cpu_events(start, end):
        read.append(start)
        return [{"name": f"cpu of sampled step {len(read)}"}]

    collector, calls = _tracing_collector(interval=1, cpu_events=cpu_events)
    for rollout_id in range(4):
        bracket(collector)
        collector.end_step(rollout_id=rollout_id)
    samples = [sample for sample, _ in calls["gathered"]]
    assert samples[0] is None
    assert [sample.cpu_events for sample in samples[1:]] == [[{"name": f"cpu of sampled step {n}"}] for n in (1, 2, 3)]
    assert read == [sample.wall for sample in samples[1:]]


def test_a_sample_still_in_flight_at_the_close_is_dropped():
    collector, calls = _tracing_collector(interval=1)
    collector.end_step(rollout_id=0)
    bracket(collector)
    _, _, _, end, _, _, _ = collector.recorder._pending[0]
    end.done = False
    collector.end_step(rollout_id=1)
    assert calls["gathered"][-1][0] is None
    # The anchor is back in the pool; the unread pair still holds its two events.
    assert len(collector.recorder._pool) == 2


@pytest.mark.parametrize(
    ("trace", "primary", "level", "gathers"), [(False, True, 2, 0), (True, False, 2, 0), (True, True, 1, 2)]
)
def test_nothing_is_sampled_without_tracing_a_primary_rank_or_the_detail_level(trace, primary, level, gathers):
    collector, calls = _tracing_collector(trace=trace, primary=primary, level=level)
    for rollout_id in range(4):
        bracket(collector)
        collector.end_step(rollout_id=rollout_id)
    # Below the detail level a tracing run still joins the sample gather of every window, with nothing to send.
    assert calls == {"sync": 0, "gathered": [(None, 0)] * gathers, "sunk": []}


def test_a_rank_that_loses_its_table_still_joins_the_sample_gather_and_takes_the_level_a_window_later():
    readouts = iter([True, False, True])

    def gather(values):
        rows = [list(values), list(values)]
        if len(calls["gathered"]) >= 1:
            # From the second window on, the primary's row asks for coarse.
            rows[0][FIELD_INDEX["control"]] = 2.0
        if not next(readouts):
            raise MemoryError("injected while reading the gathered rows")
        return rows

    collector, calls = _tracing_collector(interval=1, gather=gather)
    levels = []
    for rollout_id in range(3):
        bracket(collector)
        collector.end_step(rollout_id=rollout_id)
        levels.append(collector.level)
    assert len(calls["gathered"]) == 3, "one sample gather per window, the failed one included"
    assert levels == [2, 2, 1]
    assert collector.health.errors_total == 1


def test_a_window_with_a_nan_row_is_skipped_like_a_failed_gather():
    def gather(values):
        return [list(values), [float("nan")] * len(values)]

    collector, calls = _tracing_collector(interval=1, gather=gather)
    for rollout_id in range(2):
        bracket(collector)
        assert collector.end_step(rollout_id=rollout_id) == []
    assert len(calls["gathered"]) == 2 and calls["sunk"] == []
    assert not collector.disabled and collector.health.errors_total == 0


def test_a_paused_rank_records_nothing_and_its_windows_are_not_analyzed():
    analyzer = report_module.WindowAnalyzer()
    collector = _collector(world=1, interval=1, level=0, analyzer=analyzer)
    bracket(collector)
    assert collector.recorder.pending_events == 0
    (report,) = collector.end_step(rollout_id=0)
    assert (report.note, report.metrics) == ("paused", {"straggler/control/level": 0.0})
    assert analyzer.state.consecutive == {}


def test_the_requested_level_travels_in_the_primary_row_and_applies_from_the_next_window():
    collector = _collector(world=2, interval=1, level_request=lambda: 0)
    bracket(collector)
    (report,) = collector.end_step(rollout_id=0)
    assert report.metrics["straggler/control/level"] == 1.0
    assert collector.gather_calls[-1][FIELD_INDEX["control"]] == 1.0
    assert collector.status() == {"level": 0, "rollout": 0, "health": "active"}
    bracket(collector)
    assert collector.recorder.pending_events == 0


class _ModuleTimers:
    def __init__(self):
        self.calls = []

    def attach(self):
        self.calls.append("attach")

    def detach(self):
        self.calls.append("detach")


def _module_bracket(collector, layer):
    start = collector.recorder.record_event()
    end = collector.recorder.record_event()
    if start is not None:
        collector.recorder.push(_ATTN_FWD, layer, start, end, 0.0)


def test_the_modules_level_times_only_the_sampled_step_and_keeps_module_brackets_out_of_the_window():
    gathered, gather = _recording_gather()
    collector, calls = _tracing_collector(level=3, gather=gather, max_trace_brackets=1, max_module_brackets=1)
    timers = collector.recorder.module_timers = _ModuleTimers()
    after_step = []
    for rollout_id in range(4):
        bracket(collector)
        _module_bracket(collector, 1)
        _module_bracket(collector, 2)
        collector.end_step(rollout_id=rollout_id)
        after_step.append(list(timers.calls))
    assert after_step == [[], [], ["attach", "detach"], ["attach", "detach"]]
    sample, _ = calls["gathered"][-1]
    # The fwd bracket and the first module bracket, each under its own cap; the second module bracket is lost.
    rows = _rows(sample)
    assert [int(r[0]) for r in rows] == [FIELD_INDEX["fwd"], _ATTN_FWD] and rows[1][1] == 1.0
    assert sample.lost == 1
    assert [row[FIELD_INDEX["fwd"]] for row in gathered] == [2.0, 2.0]
    assert collector.health.errors_total == 0
    # Switching off in the middle of a sampled step detaches the module timers too.
    bracket(collector)
    collector._disable("test")
    assert timers.calls[-2:] == ["attach", "detach"]


def test_a_module_bracket_finding_the_queue_full_is_dropped_instead_of_a_window_bracket():
    collector, calls = _tracing_collector(level=3, interval=1, max_pending=1)
    collector.end_step(rollout_id=0)
    bracket(collector)
    _module_bracket(collector, 1)
    assert (collector.recorder.pending_events, collector.window.get("dropped")) == (1, 0.0)
    collector.end_step(rollout_id=1)
    sample, _ = calls["gathered"][-1]
    assert [int(r[0]) for r in _rows(sample)] == [FIELD_INDEX["fwd"]] and sample.lost == 1
    assert collector.health.errors_total == 0


def test_module_brackets_take_at_most_half_the_queue_and_make_room_without_counting_as_dropped():
    collector, _ = _tracing_collector(level=3, interval=1, max_pending=4)
    collector.end_step(rollout_id=0)
    for layer in (1, 2, 3):
        _module_bracket(collector, layer)
    assert collector.recorder.pending_events == 2, "the third module bracket found half the queue taken"
    for _ in range(4):
        bracket(collector)
    # Two window brackets filled the queue; the next two pushed out the module brackets.
    assert (collector.recorder.pending_events, collector.window.get("dropped")) == (4, 0.0)
    bracket(collector)
    assert collector.window.get("dropped") == 1.0, "only a window bracket counts once the queue is all its own"
    collector.recorder.drain()
    _module_bracket(collector, 4)
    assert collector.recorder.pending_events == 1


def test_a_window_bracket_pushes_out_a_module_bracket_queued_after_older_window_brackets():
    rows, gather = _recording_gather()
    collector, calls = _tracing_collector(level=3, interval=1, max_pending=4, gather=gather)
    collector.end_step(rollout_id=0)
    bracket(collector, "forward-recv")
    _module_bracket(collector, 1)
    _module_bracket(collector, 2)
    bracket(collector, "forward-compute")
    bracket(collector, "backward-compute")
    collector.end_step(rollout_id=1)
    assert (rows[-1][FIELD_INDEX["dropped"]], rows[-1][FIELD_INDEX["unread"]]) == (0.0, 0.0)
    sample, _ = calls["gathered"][-1]
    records = [(int(r[0]), int(r[1])) for r in _rows(sample)]
    assert [segment for segment, _ in records] == [
        FIELD_INDEX["pp_recv"],
        FIELD_INDEX["fwd"],
        FIELD_INDEX["bwd"],
        _ATTN_FWD,
    ]
    assert records[-1][1] == 2 and sample.lost == 1


def test_module_brackets_in_flight_at_the_close_do_not_count_as_unread():
    rows, gather = _recording_gather()
    collector, calls = _tracing_collector(level=3, interval=1, gather=gather)
    collector.end_step(rollout_id=0)
    bracket(collector)
    start = collector.recorder.record_event()
    end = collector.recorder.record_event()
    end.done = False
    collector.recorder.push(_ATTN_FWD, 1, start, end, 0.0)
    collector.end_step(rollout_id=1)
    assert rows[-1][FIELD_INDEX["unread"]] == 0.0
    # The sample is dropped, its module bracket with it.
    assert calls["gathered"][-1][0] is None and collector.recorder.pending_events == 0


@pytest.mark.parametrize(
    ("requested", "trace", "expected"), [(2, True, 2), (2, False, 1), (3, True, 3), (3, False, 1)]
)
def test_every_rank_follows_the_level_in_the_primary_row_detail_needing_a_trace_dir(requested, trace, expected):
    def gather(values):
        primary_row = list(values)
        primary_row[FIELD_INDEX["control"]] = requested + 1.0
        return [primary_row, list(values)]

    collector = _collector(world=2, interval=1, is_primary=False, gather=gather, trace=trace)
    collector.end_step(rollout_id=0)
    assert collector.level == expected


def test_report_interval_must_be_positive():
    with pytest.raises(ValueError):
        _collector(world=1, interval=0)


def test_gc_on_another_thread_does_not_touch_the_training_threads_pause():
    """``gc.callbacks`` fire on the thread that triggered the collection; one
    on a worker thread must neither add time nor clobber a training-thread
    pause."""
    clock = [0.0]
    collector = make_collector(world=1, clock=lambda: clock[0])
    bracket(collector)  # the first bracket fixes the training thread
    clock[0] = 1.0
    collector.recorder.on_gc("start", {})

    def worker_gc():
        clock[0] = 2.0
        collector.recorder.on_gc("start", {})
        clock[0] = 3.0
        collector.recorder.on_gc("stop", {})

    worker = threading.Thread(target=worker_gc)
    worker.start()
    worker.join()
    clock[0] = 6.0
    collector.recorder.on_gc("stop", {})
    # Checked before end_step, which resets the window at interval=1.
    assert collector.window.get("gc") == pytest.approx(5000.0)


def test_collector_constructed_on_another_thread_uses_bracket_thread():
    """When the collector is built on a background thread but the training loop
    runs on main, the main thread should be recognised as the training
    thread."""
    clock = [0.0]
    collector_box = [None]

    def builder():
        collector_box[0] = make_collector(world=1, clock=lambda: clock[0])

    t = threading.Thread(target=builder)
    t.start()
    t.join()
    collector = collector_box[0]

    # First bracket on the main thread establishes the training thread
    bracket(collector)
    clock[0] = 1.0
    collector.recorder.on_gc("start", {})
    clock[0] = 3.0
    collector.recorder.on_gc("stop", {})
    # Check before end_step, which would reset the window (interval=1)
    assert collector.window.get("gc") == pytest.approx(2000.0)
    collector.close()


def test_a_disabled_rank_keeps_joining_the_gather_with_a_health_row():
    # The other ranks must keep receiving this rank's row, otherwise they block
    # in a gather it never joins.
    collector = _collector(world=2, interval=1)
    try:
        bracket(collector)
        collector._disable("test: switched off by the test")
        assert collector.end_step() == []
        assert len(collector.gather_calls) == 1
        row = collector.gather_calls[0]
        assert row[FIELD_INDEX["health"]] == float(HEALTH_DISABLE)
        assert [v for k, v in enumerate(row) if k != FIELD_INDEX["health"]] == [0.0] * (len(row) - 1)
        assert collector.window.get("fwd") == 0.0
        assert collector.window.get("num_steps") == 0.0
    finally:
        collector.close()


def test_a_failed_gather_counts_an_error_and_keeps_the_rank_participating():
    calls = []

    def gather(values):
        calls.append(list(values))
        if len(calls) == 1:
            raise RuntimeError("rank 1 went away")
        other = list(values)
        other[FIELD_INDEX["health"]] = float(HEALTH_DISABLE)
        return [list(values), other]

    collector = _collector(world=2, interval=1, gather=gather)
    try:
        # A single failed gather is a degradation, not a switch-off: the next
        # window still gathers and can learn about its peers.
        assert collector.end_step() == []
        assert collector.health.errors_total == 1
        assert not collector.disabled
        assert collector.window.get("num_steps") == 0.0, "the failure path must still reset the window"

        (second,) = collector.end_step()
        assert len(calls) == 2
        assert calls[0][FIELD_INDEX["health"]] == 0.0
        assert second.metrics["straggler/health/state"] == float(HEALTH_DISABLE)
        assert collector.disabled
    finally:
        collector.close()


def test_a_failed_metadata_gather_is_not_retried_and_the_row_asks_to_switch_off():
    # The peers that did get the metadata are gathering rows from now on, so a
    # retried object gather would be a collective only this rank issues.
    meta_calls, rows = [], []

    def gather_objects(obj):
        meta_calls.append(obj)
        raise MemoryError("decoding the gathered metadata failed")

    def gather(values):
        rows.append(list(values))
        return [list(values), list(values)]

    collector = make_collector(world=2, interval=1, gather=gather, gather_objects=gather_objects)
    try:
        bracket(collector)
        (notice,) = collector.end_step(rollout_id=0)
        assert len(rows) == 1 and rows[0][FIELD_INDEX["health"]] == float(HEALTH_DISABLE)
        assert collector.disabled
        assert notice.metrics["straggler/health/requested_by"] == 0.0
        assert collector.end_step(rollout_id=1) == []
        assert len(meta_calls) == 1 and len(rows) == 2
    finally:
        collector.close()


def test_a_persistently_failing_gather_switches_the_profiler_off_through_health():
    # Repeated failures reach HealthConfig.max_errors; the rank then switches
    # itself off *and* keeps sending its health row, so peers that can still
    # gather also stop at the same window.
    calls = []

    def gather(values):
        calls.append(list(values))
        raise RuntimeError("gloo group is gone")

    collector = _collector(world=2, interval=1, gather=gather)
    try:
        for step in range(3):
            assert collector.end_step(rollout_id=step) == []
        assert collector.health.errors_total == 3
        assert not collector.disabled, "the switch-off is decided when the window closes"

        assert collector.end_step(rollout_id=3) == []
        assert collector.disabled and "gather" in collector.health.disabled_reason
        assert len(calls) == 4

        collector.end_step(rollout_id=4)
        assert calls[-1][FIELD_INDEX["health"]] == float(HEALTH_DISABLE)
    finally:
        collector.close()

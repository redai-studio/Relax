# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the ``config.timers`` replacement, the Megatron timer names it
is given and the event pool."""

import pytest

from relax.utils.straggler.core.stats import FIELD_INDEX, PP_PHASE_FIELDS
from relax.utils.straggler.core.timers import EventPool, StragglerTimers, _NullTimer, _SegmentTimer
from relax.utils.straggler.megatron_adapter import (
    GRAD_SYNC_TIMERS,
    IGNORED_TIMER_NAMES,
    MEGATRON_TIMER_SEGMENTS,
    PP_PHASE_TIMERS,
    TIMER_PHASES,
    phase_timers,
)
from tests.utils.straggler_helpers import FakeEvent


class _FakeSink:
    def __init__(self, capturing=False):
        self.pool = EventPool(4, FakeEvent)
        self.pushed = []
        self.discarded = []
        self.errors = []
        self.degraded = []
        self.capturing = capturing
        self.fail = False

    def record_event(self):
        if self.fail:
            raise RuntimeError("cudaErrorLaunchFailure")
        if self.capturing:
            return None
        event = self.pool.acquire()
        event.record()
        return event

    def on_error(self, where, exc):
        self.errors.append((where, exc))

    def on_degraded(self, reason):
        self.degraded.append(reason)

    def discard_event(self, event):
        self.discarded.append(event)
        self.pool.release(event)

    def push(self, segment_index, phase_index, start_event, end_event, cpu_ms):
        self.pushed.append((segment_index, phase_index, start_event, end_event, cpu_ms))


def test_event_pool_reuses_released_events_and_grows_when_empty():
    pool = EventPool(2, FakeEvent)
    first, second = pool.acquire(), pool.acquire()
    assert len(pool) == 0 and pool.created == 2
    third = pool.acquire()
    assert pool.created == 3
    pool.release(first)
    assert pool.acquire() is first
    assert third is not second


def test_straggler_timers_maps_known_names_and_ignores_unknown():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    assert isinstance(timers("forward-compute", log_level=2), _SegmentTimer)
    assert isinstance(timers("forward-backward", log_level=1), _NullTimer)
    assert timers("forward-compute") is timers("forward-compute", log_level=2)


def test_segment_timer_start_stop_pushes_pair_with_segment_index():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timers("backward-compute").start(barrier=True)
    timers("backward-compute").stop(barrier=True)
    assert len(sink.pushed) == 1
    segment_index, phase_index, start, end, cpu_ms = sink.pushed[0]
    assert segment_index == FIELD_INDEX["bwd"]
    assert phase_index == -1
    assert start.elapsed_time(end) == 1.0
    assert cpu_ms >= 0.0


def test_training_pp_recv_names_carry_their_1f1b_phase_and_gradient_syncs_their_timer():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS, TIMER_PHASES)
    names = [*PP_PHASE_TIMERS, "forward-send-forward-recv", *GRAD_SYNC_TIMERS]
    for name in names:
        timers(name).start()
        timers(name).stop()
    pushed = [(segment_index, phase_index) for segment_index, phase_index, *_ in sink.pushed]
    assert pushed == [
        *[(FIELD_INDEX["pp_recv"], FIELD_INDEX[PP_PHASE_TIMERS[name]]) for name in PP_PHASE_TIMERS],
        (FIELD_INDEX["pp_recv"], -1),
        *[(FIELD_INDEX["dp_grad_sync"], position) for position in range(len(GRAD_SYNC_TIMERS))],
    ]


def test_phase_timers_are_training_pp_recv_names_covering_every_phase_once():
    assert {MEGATRON_TIMER_SEGMENTS[name] for name in PP_PHASE_TIMERS} == {"pp_recv"}
    assert sorted(PP_PHASE_TIMERS.values()) == sorted(PP_PHASE_FIELDS)


def test_segment_timer_tolerates_double_start_and_stop_without_start():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timer = timers("optimizer-inner-step")
    timer.stop()
    assert sink.pushed == []
    timer.start()
    timer.start()
    assert timer.active
    timer.stop()
    assert len(sink.pushed) == 1
    assert not timer.active


def test_segment_timer_skips_bracket_when_sink_cannot_record():
    # Sink returns None (stream under CUDA-graph capture): nothing is pushed and
    # no event leaks, whether capture starts before ``start`` or between the two.
    sink = _FakeSink(capturing=True)
    timer = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)("forward-compute")
    timer.start()
    assert not timer.active
    timer.stop()
    assert sink.pushed == [] and sink.discarded == []

    sink.capturing = False
    timer.start()
    sink.capturing = True
    timer.stop()
    assert sink.pushed == []
    assert len(sink.discarded) == 1 and len(sink.pool) == 4
    assert not timer.active


def test_sink_errors_skip_the_bracket_and_are_reported_instead_of_raised():
    sink = _FakeSink()
    timer = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)("forward-compute")
    sink.fail = True
    timer.start()  # must not raise into Megatron's schedule
    assert not timer.active and [where for where, _ in sink.errors] == ["timer"]

    sink.fail = False
    timer.start()
    sink.fail = True
    timer.stop()
    assert sink.pushed == [] and len(sink.errors) == 2 and not timer.active
    assert len(sink.pool) == 4, "the start event of a bracket that failed to close goes back to the pool"


def test_timers_log_and_write_do_nothing_and_are_reported():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    assert timers.log(["forward-compute"]) is None
    assert timers.write(["forward-compute"], None, 0) is None
    assert [reason.split(",")[0] for reason in sink.degraded] == [
        "Timers.log() was called",
        "Timers.write() was called",
    ]


def test_known_timer_name_is_not_reported():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timers("forward-compute")
    timers("forward-compute")
    assert sink.degraded == []


@pytest.mark.parametrize("phase", ["train", "forward_only"])
def test_ignored_timer_name_is_not_reported(phase):
    sink = _FakeSink()
    timers = phase_timers(sink)[phase]
    for name in IGNORED_TIMER_NAMES:
        assert isinstance(timers(name), _NullTimer)
        timers(name)
    assert sink.degraded == []


def test_unknown_timer_name_is_reported_exactly_once():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timers("totally-new-timer")
    timers("totally-new-timer")
    timers("totally-new-timer")
    assert len(sink.degraded) == 1
    assert "totally-new-timer" in sink.degraded[0]
    assert timers.names_seen() == {"totally-new-timer"}


def test_arguments_the_timers_do_not_take_are_ignored_and_reported_while_the_bracket_still_counts():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timer = timers("forward-compute", 2, "extra", stream=None)
    timer.start(True, "now")
    timer.stop(barrier=False, sync=True)
    assert len(sink.pushed) == 1
    assert sink.degraded == [
        "timers() got arguments the profiler does not take (1 positional, stream); they are ignored",
        "a timer's start() got arguments the profiler does not take (1 positional); they are ignored",
        "a timer's stop() got arguments the profiler does not take (sync); they are ignored",
    ]


def test_an_untimed_name_accepts_any_arguments():
    sink = _FakeSink()
    timer = phase_timers(sink)["train"]("forward-backward")
    timer.start(True, "now", stream=None)
    timer.stop(sync=True)
    assert sink.pushed == [] and sink.degraded == []

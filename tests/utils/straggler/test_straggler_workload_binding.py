# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: an envelope must carry the workload of the step it measured.

The observer used to read the workload context when it *built* the envelope --
on the readout thread, after the interval had already closed. Whenever the
readback lagged behind the training loop, the envelope married step N's timing
to step N+1's workload, which is exactly the evidence the comparability gate
consumes. The workload is now captured on the training thread at interval
completion and carried on the token, so the readout thread only reads a value
that was already fixed when the interval closed.
"""

import time
from typing import Any, Callable, List

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.context import (
    publish_step_workload,
    reset_training_context_for_tests,
    set_training_context,
)
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope


STAGE_NAME = "forward-backward"


class _FakeEvent:
    __slots__ = ("completed",)

    def __init__(self) -> None:
        self.completed = False


class _HoldFirstBackend:
    def __init__(self) -> None:
        self._first: Any = None
        self.released = False

    def available(self) -> bool:
        return True

    def create_event(self) -> _FakeEvent:
        return _FakeEvent()

    def record(self, event: _FakeEvent) -> None:
        event.completed = True

    def is_complete(self, event: _FakeEvent) -> bool:
        if self._first is None:
            self._first = event
        if event is self._first and not self.released:
            return False
        return event.completed

    def elapsed_ms(self, start: _FakeEvent, end: _FakeEvent) -> float:
        return 1.5


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        run_id="workload-binding",
        rank=0,
        world_size=4,
        tensor_parallel_rank=0,
        pipeline_parallel_rank=0,
        virtual_pipeline_parallel_rank=0,
        context_parallel_rank=0,
        expert_parallel_rank=0,
        expert_tensor_parallel_rank=0,
        data_parallel_rank=0,
        data_parallel_world_size=4,
        model_chunk_index=0,
    )


def _config() -> StragglerConfig:
    return StragglerConfig(
        enabled=True,
        timer_log_level=2,
        event_pool=64,
        queue_max=64,
        warmup_windows=0,
        work_tolerance=0.05,
        min_stage_ms=5.0,
        window_seconds=0.05,
        persist_windows=1,
        min_cohort_size=2,
        report_interval_seconds=3600.0,
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _drive_delayed_readback() -> List[TimingEnvelope]:
    delivered: List[TimingEnvelope] = []
    backend = _HoldFirstBackend()
    clock = _ManualClock()
    observer = StragglerObserver(
        _config(),
        identity=_identity(),
        backend=backend,
        consumer=delivered.append,
        poll_interval_s=0.0,
        clock=clock,
    )
    timers = StragglerTimers(_config(), sink=observer, clock=clock)
    try:
        publish_step_workload(7, [(100, 1, 1), (200, 2, 2)])
        set_training_context(7, 0)
        timer = timers(STAGE_NAME, log_level=1)
        clock.now = 0.10
        timer.start()
        clock.now = 0.12
        timer.stop()
        # The training loop advances to the next step's workload while the
        # interval's event is still unread.
        set_training_context(7, 1)
        time.sleep(0.05)
        backend.released = True
        _wait_until(lambda: len(delivered) >= 1, timeout=5.0)
    finally:
        observer.close()
    return delivered


def test_envelope_carries_the_workload_of_the_interval_s_step() -> None:
    reset_training_context_for_tests()
    try:
        delivered = _drive_delayed_readback()
        assert delivered, "the readout thread delivered nothing"
        workload = delivered[0].workload
        assert workload is not None, "the envelope carries no workload at all"
        assert workload.get("tokens") == 100, f"envelope married the interval to a later step's workload: {workload}"
    finally:
        reset_training_context_for_tests()


def test_binding_survives_across_distinct_published_steps() -> None:
    """Two intervals of two different steps must not swap their workloads."""
    reset_training_context_for_tests()
    try:
        publish_step_workload(9, [(111, 1, 1), (222, 2, 2)])
        set_training_context(9, 0)
        backend = _HoldFirstBackend()
        clock = _ManualClock()
        delivered: List[TimingEnvelope] = []
        observer = StragglerObserver(
            _config(),
            identity=_identity(),
            backend=backend,
            consumer=delivered.append,
            poll_interval_s=0.0,
            clock=clock,
        )
        timers = StragglerTimers(_config(), sink=observer, clock=clock)
        first = timers(STAGE_NAME, log_level=1)
        clock.now = 0.10
        first.start()
        clock.now = 0.12
        first.stop()
        set_training_context(9, 1)
        second = timers(STAGE_NAME, log_level=1)
        clock.now = 0.20
        second.start()
        clock.now = 0.22
        second.stop()
        backend.released = True
        _wait_until(lambda: len(delivered) >= 2, timeout=5.0)
        observer.close()
        assert len(delivered) == 2
        assert [envelope.workload.get("tokens") for envelope in delivered] == [111, 222]
    finally:
        reset_training_context_for_tests()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: a host-only interval must not overtake a pending device interval.

``complete_interval`` used to deliver a host-only interval (``token is None``:
event pool exhausted, or no CUDA at all) inline on the training thread,
bypassing the ``_pending`` queue that the readout thread drains in sequence
order. With a device interval still waiting for its event readback at the
head of the queue, a host-only interval completing later was *shipped* first,
so the collector saw a sequence regression and discarded the device envelope
as ``late`` -- the same failure shape the tail re-queue caused before
``cac4cb6``, reached by the degradation path instead.

Sequence: device interval A completes (seq=1, event held unread), then the
pool is still held by A so interval B degrades to host-only and completes
(seq=2). Wire order must be [1, 2]. Against the inline delivery it is [2, 1].
"""

import time
from typing import Any, Callable, List

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import (
    MEASUREMENT_DEVICE,
    MEASUREMENT_HOST_ONLY,
    StragglerObserver,
    TimingEnvelope,
)


OUTER_NAME = "forward-backward"
INNER_NAME = "forward-compute"


class _FakeEvent:
    """Device event whose completion this test controls."""

    __slots__ = ("completed",)

    def __init__(self) -> None:
        self.completed = False


class _HoldFirstBackend:
    """Backend that keeps the first event the readout asks about incomplete."""

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
        run_id="mixed-degradation-order",
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
    """One event pair in the pool, so the second concurrent interval
    degrades."""
    return StragglerConfig(
        enabled=True,
        timer_log_level=2,
        event_pool=2,
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


def _drive_mixed_degradation() -> List[TimingEnvelope]:
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
        # A acquires the pool's only event pair and completes first (seq=1);
        # the readout thread holds it because its end event never completes.
        outer = timers(OUTER_NAME, log_level=1)
        clock.now = 0.10
        outer.start()
        clock.now = 0.12
        outer.stop()
        # The pair is still held by the unread token, so B degrades to
        # host-only and completes second (seq=2).
        inner = timers(INNER_NAME, log_level=2)
        clock.now = 0.13
        inner.start()
        clock.now = 0.15
        inner.stop()
        # Give any inline delivery ample time to happen before the release.
        time.sleep(0.05)
        backend.released = True
        _wait_until(lambda: len(delivered) >= 2, timeout=5.0)
    finally:
        observer.close()
    return delivered


def test_host_only_interval_does_not_overtake_a_pending_device_interval() -> None:
    delivered = _drive_mixed_degradation()
    seqs = [envelope.seq for envelope in delivered]
    assert len(delivered) == 2, f"expected both intervals, got {seqs}"
    assert seqs == [1, 2], f"delivery order broke sequence order: {seqs}"


def test_mixed_degradation_preserves_measurement_kinds() -> None:
    delivered = _drive_mixed_degradation()
    assert len(delivered) == 2
    first, second = delivered
    assert first.name == OUTER_NAME
    assert first.measurement_kind == MEASUREMENT_DEVICE
    assert second.name == INNER_NAME
    assert second.measurement_kind == MEASUREMENT_HOST_ONLY
    assert second.reason == "no_event_pair"


def test_no_envelope_is_dropped_in_mixed_degradation() -> None:
    delivered = _drive_mixed_degradation()
    assert sorted(envelope.seq for envelope in delivered) == [1, 2]

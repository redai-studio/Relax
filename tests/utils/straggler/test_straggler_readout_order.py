# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression: a retried interval must keep its place in the delivery order.

The collector requires a per-rank sequence that never decreases and discards a
packet whose sequence is behind the newest it has seen as ``late``. The observer
stamps ``seq`` at interval *completion*, so ``_pending`` is filled in sequence
order -- but the readout loop used to re-queue a not-yet-readable interval with
``self._pending.append(token)``, i.e. at the **tail**. The deferred interval was
therefore shipped after later, higher-sequence intervals and the collector
dropped it.

That is not hypothetical. On a real 4-GPU run at ``ce9b637`` the collector
reported ``late=631`` of ``2112`` envelopes with ``invalid=0 duplicate=0``, and
the enclosing ``forward-backward`` interval -- completed in slot 7 of an 11-slot
step, so the longest and the one most likely to be deferred -- was accepted only
30, 30, 32 and 41 times out of 48 across the four ranks, while the short
intervals lost none.

These tests hold one interval's event incomplete so the retry path is taken, and
assert that delivery stays in sequence order. The first test fails against the
tail re-queue.
"""

import time
from typing import Any, Callable, List

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope


FIXED_DEVICE_MS = 1.5
OUTER_NAME = "forward-backward"
INNER_NAME = "forward-compute"
STEPS = 3


class _FakeEvent:
    """Device event whose completion this test controls."""

    __slots__ = ("completed",)

    def __init__(self) -> None:
        self.completed = False


class _HoldFirstBackend:
    """Backend that keeps the very first event incomplete until released.

    The first event the readout thread asks about belongs to the first interval
    to complete, which carries the lowest sequence number of the run. Holding
    it exercises the retry path with the worst possible ordering consequence:
    under a tail re-queue every later, higher-sequence interval overtakes it.
    """

    def __init__(self) -> None:
        self._first: Any = None
        self.released = False

    def available(self) -> bool:
        """Return whether device timing can be used at all."""
        return True

    def create_event(self) -> _FakeEvent:
        """Create one fake timing event."""
        return _FakeEvent()

    def record(self, event: _FakeEvent) -> None:
        """Mark the event complete without touching a device."""
        event.completed = True

    def is_complete(self, event: _FakeEvent) -> bool:
        """Report completion, holding the first event the readout asks
        about."""
        if self._first is None:
            self._first = event
        if event is self._first and not self.released:
            return False
        return event.completed

    def elapsed_ms(self, start: _FakeEvent, end: _FakeEvent) -> float:
        """Return the fixed fake device duration."""
        return FIXED_DEVICE_MS


class _ManualClock:
    """Controllable host clock, so the schedule needs no real waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return the manually advanced time."""
        return self.now


def _identity(rank: int = 0) -> RuntimeIdentity:
    """One data-parallel rank of a single comparable cohort."""
    return RuntimeIdentity(
        run_id="dedup-inflight-order",
        rank=rank,
        world_size=4,
        tensor_parallel_rank=0,
        pipeline_parallel_rank=0,
        virtual_pipeline_parallel_rank=0,
        context_parallel_rank=0,
        expert_parallel_rank=0,
        expert_tensor_parallel_rank=0,
        data_parallel_rank=rank,
        data_parallel_world_size=4,
        model_chunk_index=0,
    )


def _config() -> StragglerConfig:
    """Config with room for every interval to stay a device interval."""
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
    """Poll ``predicate`` until it holds or ``timeout`` elapses."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _run_steps(timers: StragglerTimers, clock: _ManualClock, steps: int) -> None:
    """Run ``steps`` nested pairs; the outer interval always finishes last."""
    for step in range(steps):
        base = step * 0.06
        clock.now = base
        outer = timers(OUTER_NAME, log_level=1)
        outer.start()
        clock.now = base + 0.001
        inner = timers(INNER_NAME, log_level=2)
        inner.start()
        clock.now = base + 0.019
        inner.stop()
        clock.now = base + 0.040
        outer.stop()


def _drive_held_run() -> List[TimingEnvelope]:
    """Run three steps while the first event is held, then release it."""
    delivered: List[TimingEnvelope] = []
    backend = _HoldFirstBackend()
    clock = _ManualClock()
    # The observer must share the manual clock: with a real clock, a host_end
    # taken from the manual clock looks ancient and ``_process`` would skip the
    # retry path entirely by reporting ``readout_timeout``.
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
        _run_steps(timers, clock, STEPS)
        # The first outer interval is now stuck at the head of the queue while
        # every later interval is ready: give the readout thread long enough to
        # take the retry path, then release it and let the queue drain.
        time.sleep(0.05)
        backend.released = True
        _wait_until(lambda: len(delivered) >= 2 * STEPS, timeout=5.0)
    finally:
        observer.close()
    return delivered


def test_retried_interval_is_delivered_in_sequence_order() -> None:
    """Delivery order must follow ``seq`` even when an interval is deferred."""
    delivered = _drive_held_run()
    seqs = [envelope.seq for envelope in delivered]
    assert seqs, "the readout thread delivered nothing"
    assert seqs == sorted(seqs), f"delivery order broke sequence order: {seqs}"


def test_the_deferred_interval_is_still_the_first_delivered() -> None:
    """The held interval has the lowest sequence, so it must come out first.

    Sequence numbers are stamped at completion, and in a nested step the inner
    timer completes before the outer one, so the lowest sequence belongs to
    ``forward-compute`` -- the same ordering the real run showed, where
    ``forward-compute`` occupied slot 1 and ``forward-backward`` slot 7.
    """
    delivered = _drive_held_run()
    assert delivered, "the readout thread delivered nothing"
    assert delivered[0].name == INNER_NAME
    assert delivered[0].seq == min(envelope.seq for envelope in delivered)


def test_no_envelope_is_lost_while_an_interval_is_held() -> None:
    """Ordering must not be bought by dropping or duplicating intervals."""
    delivered = _drive_held_run()
    assert len(delivered) == 2 * STEPS
    assert sorted(envelope.seq for envelope in delivered) == list(range(1, 2 * STEPS + 1))

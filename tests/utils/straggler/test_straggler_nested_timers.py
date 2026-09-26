# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression: nested timers must not look like out-of-order transport.

Megatron nests its timers: the level-1 ``forward-backward`` timer wraps the
level-2 ``forward-compute``/``backward-compute`` timers, so the outer interval
starts first and *completes last*. The observer used to allocate the wire
sequence number at ``acquire`` (start) time while the collector requires a
per-rank sequence that never decreases, so every enclosing interval arrived with
a number behind the intervals it wrapped and was silently dropped as ``late``.
The whole-phase interval was therefore never judged on any GPU run, and a
straggler whose extra time sat in the phase wrapper was invisible.

These tests drive the real shim, the real observer and the real collector
together on CPU (the CUDA-event backend is a fake, so no hardware is needed).
They are written to fail against the start-time sequencing:

* ``test_nested_outer_interval_gets_a_later_sequence_than_the_inner_one`` opens
  an outer timer, wraps an inner one, and asserts the sequence follows
  completion order;
* ``test_stall_inside_the_outer_interval_is_judged_and_reported`` runs four ranks
  where one stalls only inside ``forward-backward`` and asserts the outer
  interval is not counted late, is judged, and is reported as a straggler.
"""

import time
from typing import Any, Callable, List, Tuple

import pytest

from relax.utils.straggler.collector import TimingCollector
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import REASON_HOST_ONLY_STALL, VERDICT_STRAGGLER
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope


#: Deterministic nested-timer schedule, in seconds. Six steps are spread across
#: six 50 ms windows so the detector closes windows without any real waiting.
WINDOW_SECONDS = 0.05
STEP_SECONDS = 0.06
STEPS = 6
INNER_MS = 19.0
OUTER_MS = 40.0
STALL_MS = 200.0
STRAGGLER_RANK = 3
WORLD_SIZE = 4
FIXED_DEVICE_MS = 1.5

OUTER_NAME = "forward-backward"
INNER_NAME = "forward-compute"


class _FakeEvent:
    """Device event whose completion the fake backend controls."""

    __slots__ = ("completed", "device_ms")

    def __init__(self, device_ms: float) -> None:
        self.completed = False
        self.device_ms = device_ms


class _FakeEventBackend:
    """CPU stand-in for CUDA events; records complete immediately."""

    def __init__(self, device_ms: float = FIXED_DEVICE_MS) -> None:
        self._device_ms = device_ms

    def available(self) -> bool:
        """Return whether device timing can be used at all."""
        return True

    def create_event(self) -> _FakeEvent:
        """Create one fake timing event."""
        return _FakeEvent(self._device_ms)

    def record(self, event: _FakeEvent) -> None:
        """Mark the event complete without touching a device."""
        event.completed = True

    def is_complete(self, event: _FakeEvent) -> bool:
        """Report completion without blocking."""
        return event.completed

    def elapsed_ms(self, start: _FakeEvent, end: _FakeEvent) -> float:
        """Return the fixed fake device duration."""
        return self._device_ms


class _ManualClock:
    """Controllable host clock, so the nested schedule needs no sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return the manually advanced time."""
        return self.now


def _identity(rank: int) -> RuntimeIdentity:
    """One data-parallel rank of a single comparable cohort."""
    return RuntimeIdentity(
        run_id="nested-timers",
        rank=rank,
        world_size=WORLD_SIZE,
        tensor_parallel_rank=0,
        pipeline_parallel_rank=0,
        virtual_pipeline_parallel_rank=0,
        context_parallel_rank=0,
        expert_parallel_rank=0,
        expert_tensor_parallel_rank=0,
        data_parallel_rank=rank,
        data_parallel_world_size=WORLD_SIZE,
        model_chunk_index=0,
    )


def _config() -> StragglerConfig:
    """Profiler config tuned so a verdict lands within six deterministic
    steps."""
    return StragglerConfig(
        enabled=True,
        timer_log_level=2,
        # Room for every interval of the run to stay a device interval: an
        # exhausted pool degrades an interval to host timing, which is delivered
        # synchronously on the training thread and would inject an unrelated
        # ordering artefact into this test.
        event_pool=64,
        queue_max=64,
        warmup_windows=0,
        work_tolerance=0.05,
        min_stage_ms=5.0,
        window_seconds=WINDOW_SECONDS,
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


def _open_nested_step(timers: StragglerTimers, clock: _ManualClock, base: float, outer_ms: float) -> None:
    """Run one nested pair: outer wraps an inner, outer finishes last."""
    clock.now = base
    outer = timers(OUTER_NAME, log_level=1)
    outer.start()

    clock.now = base + 0.001
    inner = timers(INNER_NAME, log_level=2)
    inner.start()

    clock.now = base + INNER_MS / 1000.0
    inner.stop()

    clock.now = base + outer_ms / 1000.0
    outer.stop()


def _make_rank(
    config: StragglerConfig,
    rank: int,
    consumer: Callable[[TimingEnvelope], Any],
) -> Tuple[_ManualClock, StragglerTimers, StragglerObserver]:
    """Build a real shim + observer pair for one rank over a fake backend."""
    clock = _ManualClock()
    observer = StragglerObserver(
        config,
        identity=_identity(rank),
        backend=_FakeEventBackend(),
        consumer=consumer,
        poll_interval_s=0.0,
    )
    timers = StragglerTimers(config, sink=observer, clock=clock)
    return clock, timers, observer


def test_nested_outer_interval_gets_a_later_sequence_than_the_inner_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sequence must follow completion order, not start order.

    The outer timer starts first but stops last. With start-time sequencing the
    delivered sequence decreases (inner=2, outer=1); with completion-time
    sequencing it increases (inner=1, outer=2).
    """
    from relax.utils.straggler import megatron_timer_shim as shim_module

    monkeypatch.setattr(shim_module, "_debug_settings", (0.0, -1, ""))

    config = _config()
    seen: List[TimingEnvelope] = []
    clock, timers, observer = _make_rank(config, rank=0, consumer=seen.append)
    try:
        _open_nested_step(timers, clock, base=0.0, outer_ms=OUTER_MS)
        assert _wait_until(lambda: len(seen) == 2)
    finally:
        observer.close()

    assert [envelope.name for envelope in seen] == [INNER_NAME, OUTER_NAME]
    sequences = [envelope.seq for envelope in seen]
    assert all(sequence > 0 for sequence in sequences)
    assert sequences == sorted(sequences), f"sequence went backwards across a nested pair: {sequences}"
    assert seen[0].seq < seen[1].seq


def test_stall_inside_the_outer_interval_is_judged_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank slow only inside the enclosing timer must be reported.

    Four real ranks run the same nested schedule; rank 3 spends an extra 200 ms
    inside ``forward-backward`` only. The outer interval must not be counted
    ``late``, must reach the detector, and (because the extra time is on the
    host, with the device stream identical) must raise a ``host_only_stall``
    straggler verdict.
    """
    from relax.utils.straggler import megatron_timer_shim as shim_module

    monkeypatch.setattr(shim_module, "_debug_settings", (0.0, -1, ""))

    config = _config()
    verdicts: List[Any] = []
    collector = TimingCollector(config, identity=_identity(0), on_verdict=verdicts.append)
    ranks = [_make_rank(config, rank, consumer=collector.ingest) for rank in range(WORLD_SIZE)]
    total_envelopes = WORLD_SIZE * STEPS * 2

    try:
        for step in range(STEPS):
            base = step * STEP_SECONDS
            for rank, (clock, timers, _observer) in enumerate(ranks):
                outer_ms = OUTER_MS + (STALL_MS if rank == STRAGGLER_RANK else 0.0)
                _open_nested_step(timers, clock, base=base, outer_ms=outer_ms)
            # Let the readout threads drain before the next step, so the event
            # pool never exhausts and every packet travels the device path.
            expected = (step + 1) * WORLD_SIZE * 2
            assert _wait_until(lambda: collector.status()["envelopes"] == expected), collector.status()

        assert _wait_until(lambda: collector.status()["envelopes"] == total_envelopes), collector.status()

        for _clock, _timers, observer in ranks:
            observer.close()
        assert _wait_until(lambda: collector.status()["envelopes"] == total_envelopes)

        collector.flush()
    finally:
        for _clock, _timers, observer in ranks:
            observer.close()

    status = collector.status()
    # The enclosing interval was not misread as out-of-order transport, and every
    # packet -- inner and outer -- actually reached the detector.
    assert status["late_packets"] == 0, status
    assert status["dedup"]["late"] == 0
    assert status["duplicate_packets"] == 0
    assert status["judged_packets"] == total_envelopes, status

    outer_names = {envelope.name for envelope in verdicts}
    assert OUTER_NAME in outer_names, f"no verdict judged the enclosing interval: {verdicts}"

    stragglers = [
        verdict
        for verdict in verdicts
        if verdict.kind == VERDICT_STRAGGLER and verdict.name == OUTER_NAME and verdict.rank == STRAGGLER_RANK
    ]
    assert stragglers, f"stall inside {OUTER_NAME} produced no straggler verdict: {verdicts}"
    # The extra 200 ms is host-side only (the fake device stream is identical on
    # every rank), so the attribution must say so.
    assert stragglers[0].reason == REASON_HOST_ONLY_STALL, stragglers[0].reason

    # The inner interval is equal on every rank, so it must never be accused.
    inner_stragglers = [
        verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER and verdict.name == INNER_NAME
    ]
    assert inner_stragglers == []

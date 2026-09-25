# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the off-thread timing readout path.

Every test runs on CPU: the CUDA-event backend is replaced by a fake, which is
the only way to assert the readout, drop and shutdown behaviour
deterministically.
"""

import json
import time
from typing import Any, List, Optional

import pytest

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity, discover_identity
from relax.utils.straggler.observer import EventPool, StragglerObserver, TimingEnvelope


class FakeEvent:
    """Device event whose completion the test controls."""

    __slots__ = ("completed", "device_ms")

    def __init__(self, device_ms: float = 1.5) -> None:
        self.completed = False
        self.device_ms = device_ms


class FakeEventBackend:
    """CPU stand-in for CUDA events."""

    def __init__(self, available: bool = True, auto_complete: bool = True, device_ms: float = 1.5) -> None:
        self._available = available
        self.auto_complete = auto_complete
        self.device_ms = device_ms
        self.created = 0
        self.recorded = 0

    def available(self) -> bool:
        return self._available

    def create_event(self) -> FakeEvent:
        self.created += 1
        return FakeEvent(self.device_ms)

    def record(self, event: FakeEvent) -> None:
        self.recorded += 1
        if self.auto_complete:
            event.completed = True

    def is_complete(self, event: FakeEvent) -> bool:
        return event.completed

    def elapsed_ms(self, start: FakeEvent, end: FakeEvent) -> float:
        return self.device_ms


IDENTITY = RuntimeIdentity(run_id="test-run", rank=3, world_size=8, tensor_parallel_rank=0, data_parallel_rank=3)


def wait_until(predicate, timeout: float = 3.0, interval: float = 0.005) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def make_observer(
    backend: Optional[FakeEventBackend] = None,
    queue_max: int = 64,
    consumer: Optional[Any] = None,
    readout_timeout_s: float = 30.0,
    event_pool: int = 8,
) -> StragglerObserver:
    config = StragglerConfig(enabled=True, queue_max=queue_max, event_pool=event_pool)
    return StragglerObserver(
        config,
        identity=IDENTITY,
        backend=backend if backend is not None else FakeEventBackend(),
        consumer=consumer,
        readout_timeout_s=readout_timeout_s,
        poll_interval_s=0.001,
    )


def test_observer_emits_device_envelope() -> None:
    backend = FakeEventBackend(device_ms=2.5)
    seen: List[TimingEnvelope] = []
    observer = make_observer(backend, consumer=seen.append)

    token = observer.acquire_interval("forward-compute", 2)
    assert token is not None
    observer.complete_interval(token, "forward-compute", 2, 100.0, 100.5, False)

    assert wait_until(lambda: len(seen) == 1)
    assert seen[0].device_ms == pytest.approx(2.5)
    assert seen[0].reason == "device"
    assert observer.stats()["device_intervals"] == 1
    assert backend.recorded == 2  # one start, one stop, and no extra read


def test_drain_returns_and_clears_buffered_envelopes() -> None:
    observer = make_observer(FakeEventBackend())

    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)
    assert wait_until(lambda: observer.stats()["delivered"] == 1)

    assert len(observer.drain()) == 1
    assert observer.drain() == []


def test_observer_envelope_fields() -> None:
    seen: List[TimingEnvelope] = []
    observer = make_observer(FakeEventBackend(device_ms=2.5), consumer=seen.append)

    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 100.0, 100.5, True)

    assert wait_until(lambda: len(seen) == 1)
    envelope = seen[0]
    assert envelope.name == "forward-compute"
    assert envelope.log_level == 2
    assert envelope.rank == 3
    assert envelope.run_id == "test-run"
    assert envelope.host_ms == pytest.approx(500.0)
    assert envelope.device_ms == pytest.approx(2.5)
    assert envelope.barrier is True
    assert envelope.reason == "device"
    assert envelope.to_dict()["host_ms"] == pytest.approx(500.0)


def test_observer_without_cuda_still_reports_host_timing() -> None:
    seen: List[TimingEnvelope] = []
    observer = make_observer(FakeEventBackend(available=False), consumer=seen.append)

    token = observer.acquire_interval("forward-compute", 2)
    assert token is None
    observer.complete_interval(None, "forward-compute", 2, 1.0, 1.25, False)

    assert observer.device_timing_enabled is False
    assert len(seen) == 1
    assert seen[0].device_ms is None
    assert seen[0].reason == "no_event_pair"
    assert seen[0].host_ms == pytest.approx(250.0)
    assert observer.stats()["host_only_intervals"] == 1
    assert observer.stats()["readout_thread_alive"] is False


def test_event_pool_exhaustion_degrades_to_host_only() -> None:
    observer = make_observer(FakeEventBackend(), event_pool=2)

    first = observer.acquire_interval("forward-backward", 1)
    second = observer.acquire_interval("forward-compute", 2)

    assert first is not None
    assert second is None  # only one pair exists
    assert observer.stats()["host_only_intervals"] == 1


def test_events_return_to_the_pool_after_readout() -> None:
    observer = make_observer(FakeEventBackend())
    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)

    assert wait_until(lambda: observer.stats()["pool"]["free"] == 2)
    assert observer.stats()["device_intervals"] == 1


def test_full_pending_queue_drops_with_a_counted_reason() -> None:
    backend = FakeEventBackend(auto_complete=False)
    observer = make_observer(backend, queue_max=2, readout_timeout_s=30.0, event_pool=256)
    tokens = [observer.acquire_interval("forward-compute", 2) for _ in range(60)]
    tokens = [token for token in tokens if token is not None]

    for token in tokens:
        observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)

    assert observer.stats()["dropped_pending_full"] > 0


def test_incomplete_interval_times_out_to_host_only() -> None:
    seen: List[TimingEnvelope] = []
    observer = make_observer(FakeEventBackend(auto_complete=False), consumer=seen.append, readout_timeout_s=0.01)

    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 0.0, 0.001, False)

    assert wait_until(lambda: len(seen) == 1)
    assert seen[0].reason == "readout_timeout"
    assert seen[0].device_ms is None
    assert observer.stats()["readout_timeouts"] == 1


def test_close_flushes_pending_intervals() -> None:
    seen: List[TimingEnvelope] = []
    observer = make_observer(FakeEventBackend(auto_complete=False), consumer=seen.append)

    tokens = [observer.acquire_interval("forward-compute", 2) for _ in range(4)]
    for token in tokens:
        observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)

    observer.close()

    assert wait_until(lambda: len(seen) == 4)
    assert {envelope.reason for envelope in seen} == {"closed"}
    assert observer.stats()["pending"] == 0
    assert observer.stats()["readout_thread_alive"] is False


def test_close_is_idempotent() -> None:
    observer = make_observer(FakeEventBackend())
    observer.acquire_interval("forward-compute", 2)

    observer.close()
    observer.close()

    assert observer.stats()["readout_thread_alive"] is False


def test_failing_consumer_is_counted_not_raised() -> None:
    def explode(envelope: TimingEnvelope) -> None:
        raise RuntimeError("consumer exploded")

    observer = make_observer(FakeEventBackend(), consumer=explode)
    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)

    assert wait_until(lambda: observer.stats()["consumer_errors"] == 1)


def test_local_buffer_is_bounded() -> None:
    # Without device timing no readout thread starts, so the local buffer is
    # filled deterministically through the public API.
    observer = make_observer(FakeEventBackend(available=False), queue_max=2)

    for _ in range(10):
        observer.complete_interval(None, "forward-compute", 2, 0.0, 0.1, False)

    assert observer.stats()["delivered"] == 2
    assert observer.stats()["dropped_output_full"] == 8


def test_stats_expose_pool_and_identity() -> None:
    observer = make_observer(FakeEventBackend())

    stats = observer.stats()

    assert stats["identity"] == IDENTITY.label
    assert stats["device_timing_enabled"] is True
    assert stats["pool"]["size"] == 8
    assert stats["pool"]["created"] == 0


def test_envelope_json_is_flat_and_round_trips() -> None:
    envelope = TimingEnvelope(
        run_id="r",
        rank=1,
        cohort="0:0:0:0:0",
        label="rank1/tp0",
        world_size=4,
        name="forward-compute",
        log_level=2,
        seq=7,
        host_start=10.0,
        host_end=10.25,
        device_ms=12.5,
        barrier=False,
        reason="device",
    )

    payload = json.loads(envelope.to_json())

    assert payload["host_ms"] == pytest.approx(250.0)
    assert payload["device_ms"] == pytest.approx(12.5)
    assert payload["seq"] == 7
    assert "\n" not in envelope.to_json()


class TestEventPool:
    def test_pool_reuses_released_events(self) -> None:
        backend = FakeEventBackend()
        pool = EventPool(backend, 4)

        first = pool.acquire_pair()
        pool.release(*first)
        second = pool.acquire_pair()

        assert backend.created == 2
        assert {second[0], second[1]} == {first[0], first[1]}

    def test_pool_reports_exhaustion(self) -> None:
        pool = EventPool(FakeEventBackend(), 2)

        assert pool.acquire_pair() is not None
        assert pool.acquire_pair() is None
        assert pool.stats()["exhausted"] == 1


class TestIdentity:
    def test_cohort_ignores_data_parallel_position(self) -> None:
        left = RuntimeIdentity(run_id="r", rank=0, world_size=8, tensor_parallel_rank=1, data_parallel_rank=0)
        right = RuntimeIdentity(run_id="r", rank=4, world_size=8, tensor_parallel_rank=1, data_parallel_rank=4)
        other = RuntimeIdentity(run_id="r", rank=5, world_size=8, tensor_parallel_rank=2, data_parallel_rank=4)

        assert left.cohort == right.cohort
        assert left.cohort != other.cohort

    def test_discover_identity_is_safe_without_distributed(self) -> None:
        identity = discover_identity()

        assert identity.rank >= 0
        assert identity.world_size >= 1
        assert isinstance(identity.cohort, str)
        assert isinstance(identity.run_id, str) and identity.run_id

    def test_identity_dict_is_json_friendly(self) -> None:
        payload = IDENTITY.as_dict()

        assert json.loads(json.dumps(payload))["topology"]["dp"] == 3
        assert payload["cohort"] == IDENTITY.cohort

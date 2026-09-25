# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the process wiring: who collects, who ships, who judges."""

import json
import socket
import time
from typing import Any, List

import pytest

import relax.utils.straggler as straggler
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.runtime import StragglerRuntime


def free_port() -> int:
    """Reserve and release a loopback port for a same-host test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def make_config(collector_addr: str | None = None, **overrides: Any) -> StragglerConfig:
    settings: dict = {
        "enabled": True,
        "window_seconds": 1.0,
        "warmup_windows": 0,
        "persist_windows": 1,
        "report_interval_seconds": 3600.0,
        "collector_addr": collector_addr,
    }
    settings.update(overrides)
    return StragglerConfig(**settings)


def identity(rank: int, world_size: int = 4) -> RuntimeIdentity:
    return RuntimeIdentity(run_id="run-1", rank=rank, world_size=world_size, data_parallel_rank=rank)


def run_intervals(runtime: StragglerRuntime, count: int = 3, host_sleep: float = 0.001) -> None:
    """Exercise the shim through the runtime's timers."""
    timers = runtime.timers
    assert timers is not None
    for _ in range(count):
        handle = timers("forward-compute", log_level=2)
        handle.start()
        time.sleep(host_sleep)
        handle.stop()


def wait_until(predicate: Any, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_runtime_is_unstarted_without_a_config() -> None:
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)

    assert runtime.timers is None
    assert runtime.role == "unstarted"
    assert runtime.status()["started"] is False


def test_local_runtime_owns_a_collector_and_timers() -> None:
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)
    try:
        runtime.start()

        assert runtime.role == "local"
        assert runtime.timers is not None
        assert runtime.collector is not None
        status = runtime.status()
        assert status["collector"]["envelopes"] == 0
        assert status["identity"]["rank"] == 0
    finally:
        runtime.close()


def test_local_runtime_reports_after_intervals() -> None:
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)
    try:
        runtime.start()
        run_intervals(runtime)
        assert wait_until(lambda: runtime.collector is not None and runtime.collector.status()["envelopes"] == 3)
        assert runtime.timers is not None
        assert runtime.timers.stats()["intervals"] == 3
    finally:
        runtime.close()


def test_rank_zero_hosts_the_collector() -> None:
    port = free_port()
    runtime = StragglerRuntime(make_config(f"127.0.0.1:{port}"), identity=identity(0), register_atexit=False)
    try:
        runtime.start()

        assert runtime.role == "collector"
        assert runtime.status()["receiver"]["listening"] is True
    finally:
        runtime.close()


def test_other_ranks_ship_instead_of_judging() -> None:
    port = free_port()
    runtime = StragglerRuntime(make_config(f"127.0.0.1:{port}"), identity=identity(2), register_atexit=False)
    try:
        runtime.start()

        assert runtime.role == "sender"
        assert runtime.collector is None
        assert runtime.status()["sender"]["queue_max"] == runtime.config.queue_max
    finally:
        runtime.close()


def test_sender_envelopes_reach_the_rank_zero_collector() -> None:
    port = free_port()
    collector_runtime = StragglerRuntime(make_config(f"127.0.0.1:{port}"), identity=identity(0), register_atexit=False)
    sender_runtime = StragglerRuntime(make_config(f"127.0.0.1:{port}"), identity=identity(1), register_atexit=False)
    collector_runtime.start()
    if not collector_runtime.status().get("receiver", {}).get("listening", False):
        collector_runtime.close()
        sender_runtime.close()
        pytest.skip("loopback TCP is unavailable in this environment")
    try:
        sender_runtime.start()
        run_intervals(sender_runtime)

        assert wait_until(
            lambda: collector_runtime.collector is not None and collector_runtime.collector.status()["envelopes"] >= 3
        ), collector_runtime.status()
        received = collector_runtime.collector.status()["envelopes"]
        assert received == 3
        assert collector_runtime.collector.status()["identity"] == identity(0).label
    finally:
        sender_runtime.close()
        collector_runtime.close()


def test_malformed_collector_address_degrades_to_no_timers() -> None:
    runtime = StragglerRuntime(make_config("not-an-address"), identity=identity(1), register_atexit=False)
    try:
        runtime.start()

        assert runtime.timers is None
        assert runtime.status()["errors"] >= 1
    finally:
        runtime.close()


def test_close_is_idempotent_and_reports_closed() -> None:
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)
    runtime.start()

    runtime.close()
    runtime.close()

    assert runtime.status()["closed"] is True


def test_status_is_json_serialisable() -> None:
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)
    try:
        runtime.start()
        run_intervals(runtime, count=1)

        payload = json.loads(json.dumps(runtime.status()))

        assert payload["role"] == "local"
        assert payload["observer"]["intervals"] == 1
    finally:
        runtime.close()


def test_env_gated_factory_builds_a_local_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    straggler.reset_straggler_state_for_tests()

    runtime = straggler.get_straggler_runtime()

    assert runtime is not None
    assert runtime.role == "local"
    assert straggler.get_straggler_timers() is runtime.timers
    straggler.reset_straggler_state_for_tests()


def test_env_gated_factory_is_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    straggler.reset_straggler_state_for_tests()

    assert straggler.get_straggler_runtime() is straggler.get_straggler_runtime()
    straggler.reset_straggler_state_for_tests()


def test_reset_closes_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    straggler.reset_straggler_state_for_tests()
    runtime = straggler.get_straggler_runtime()
    assert runtime is not None

    straggler.reset_straggler_state_for_tests()

    assert runtime.status()["closed"] is True
    rebuilt = straggler.get_straggler_runtime()
    assert rebuilt is not None and rebuilt is not runtime  # the env still enables it
    straggler.reset_straggler_state_for_tests()


def test_collector_runtime_judges_shipments_from_two_ranks() -> None:
    port = free_port()
    address = f"127.0.0.1:{port}"
    collector_runtime = StragglerRuntime(
        make_config(address, work_tolerance=0.05, persist_windows=1, warmup_windows=0),
        identity=identity(0),
        register_atexit=False,
    )
    fast_runtime = StragglerRuntime(make_config(address), identity=identity(1), register_atexit=False)
    slow_runtime = StragglerRuntime(make_config(address), identity=identity(2), register_atexit=False)
    collector_runtime.start()
    if not collector_runtime.status().get("receiver", {}).get("listening", False):
        for runtime in (collector_runtime, fast_runtime, slow_runtime):
            runtime.close()
        pytest.skip("loopback TCP is unavailable in this environment")
    try:
        fast_runtime.start()
        slow_runtime.start()
        for _ in range(4):
            run_intervals(fast_runtime, count=1, host_sleep=0.001)
            run_intervals(slow_runtime, count=1, host_sleep=0.05)

        assert wait_until(
            lambda: collector_runtime.collector is not None and collector_runtime.collector.status()["envelopes"] >= 8
        ), collector_runtime.status()
        collector_runtime.close()
        stragglers: List[Any] = [
            verdict for verdict in collector_runtime.drain_verdicts() if verdict.kind == "straggler"
        ]
        assert stragglers, collector_runtime.status()
        assert stragglers[0].rank == 2
    finally:
        for runtime in (collector_runtime, fast_runtime, slow_runtime):
            runtime.close()

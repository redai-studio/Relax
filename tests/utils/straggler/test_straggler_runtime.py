# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the process wiring: who collects, who ships, who judges."""

import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any, List

import pytest

import relax.utils.straggler as straggler
from relax.utils.straggler.collector import TimingCollector
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


def run_dir(tmp_path: Any) -> Any:
    """The run-scoped directory shared by every rank of one run."""
    return tmp_path / "run_run-1"


def test_close_persists_the_runtime_and_collector_counters(tmp_path: Any) -> None:
    """The acceptance protocol reads these counters; close must write them."""
    runtime = StragglerRuntime(make_config(output_dir=str(tmp_path)), identity=identity(0), register_atexit=False)
    runtime.start()
    run_intervals(runtime, count=2)
    runtime.close()

    collector_status = json.loads((run_dir(tmp_path) / "collector_status.json").read_text())
    runtime_status = json.loads((run_dir(tmp_path) / "runtime_status.json").read_text())

    assert collector_status["envelopes"] == 2
    assert runtime_status["collector"]["envelopes"] == 2
    assert runtime_status["observer"]["intervals"] == 2
    assert runtime_status["timers"]["intervals"] == 2
    assert runtime_status["collector"]["caps"]["MAX_VERDICTS"] == 512


def test_status_files_are_written_without_a_graceful_close(tmp_path: Any) -> None:
    """Ray SIGTERMs the actor at job end, so ``close``/atexit never runs.

    A real ON arm therefore produced JSONL streams but no status file. The
    periodic writer must leave one behind without any explicit close.
    """
    runtime = StragglerRuntime(
        make_config(output_dir=str(tmp_path), report_interval_seconds=0.1),
        identity=identity(0),
        register_atexit=False,
    )
    runtime.start()
    run_intervals(runtime, count=1)

    deadline = time.time() + 5.0
    while time.time() < deadline and not (run_dir(tmp_path) / "collector_status.json").exists():
        time.sleep(0.05)

    assert (run_dir(tmp_path) / "collector_status.json").exists()
    assert (run_dir(tmp_path) / "runtime_status.json").exists()
    runtime.close()


def test_periodic_writer_flushes_the_jsonl_without_a_close(tmp_path: Any) -> None:
    """The buffered envelopes must reach disk while the run is alive.

    A real ON arm held ``envelopes=132`` in the log and left a zero-byte
    ``straggler_envelopes.jsonl``, because only ``close()`` flushed and Ray
    SIGTERMs the actor. The writer thread must flush periodically instead.
    """
    runtime = StragglerRuntime(
        make_config(output_dir=str(tmp_path), report_interval_seconds=0.1),
        identity=identity(0),
        register_atexit=False,
    )
    runtime.start()
    run_intervals(runtime, count=2)

    envelopes = run_dir(tmp_path) / "straggler_envelopes.jsonl"
    deadline = time.time() + 5.0
    while time.time() < deadline and (not envelopes.exists() or envelopes.stat().st_size == 0):
        time.sleep(0.05)

    lines = [json.loads(line) for line in envelopes.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2
    runtime.close()


def test_output_paths_are_run_scoped_and_per_rank(tmp_path: Any) -> None:
    """Four ranks share one output dir, so no two may write the same file."""
    runtimes = [
        StragglerRuntime(make_config(output_dir=str(tmp_path)), identity=identity(rank), register_atexit=False)
        for rank in (0, 1, 2, 3)
    ]
    for runtime in runtimes:
        runtime.start()
    for runtime in runtimes:
        runtime.close()

    status_files = sorted(path.name for path in run_dir(tmp_path).glob("runtime_status_*.json"))
    assert len(status_files) == 4
    assert len(set(status_files)) == 4
    # The unsuffixed names analyze_run.py reads exist once and belong to the
    # only rank that owns a collector.
    collector = json.loads((run_dir(tmp_path) / "collector_status.json").read_text())
    assert collector["identity"].startswith("rank0/")
    # Every rank scoped its own counters under the same run directory.
    assert len(list(tmp_path.glob("run_run-1/runtime_status_rank*.json"))) == 4


def test_status_writer_never_touches_a_file_on_the_training_thread(tmp_path: Any, monkeypatch: Any) -> None:
    """The flush/write work must stay on the daemon writer thread."""
    main_ident = threading.get_ident()
    writers: List[int] = []
    flushers: List[int] = []
    original_write = StragglerRuntime._write_status
    original_flush = TimingCollector.flush

    def spy_write(self: Any) -> None:
        writers.append(threading.get_ident())
        original_write(self)

    def spy_flush(self: Any) -> Any:
        flushers.append(threading.get_ident())
        return original_flush(self)

    monkeypatch.setattr(StragglerRuntime, "_write_status", spy_write)
    monkeypatch.setattr(TimingCollector, "flush", spy_flush)

    runtime = StragglerRuntime(
        make_config(output_dir=str(tmp_path), report_interval_seconds=0.1),
        identity=identity(0),
        register_atexit=False,
    )
    runtime.start()
    run_intervals(runtime, count=1)
    # The training-thread accessors must not persist anything themselves.
    runtime.summary()
    runtime.status()

    deadline = time.time() + 5.0
    while time.time() < deadline and not writers:
        time.sleep(0.05)

    assert writers, "the periodic writer never ran"
    assert flushers, "the periodic writer never flushed"
    assert main_ident not in writers, "status was written on the training thread"
    assert main_ident not in flushers, "the collector was flushed on the training thread"
    runtime.close()


def test_a_sigterm_loses_at_most_one_flush_interval(tmp_path: Any) -> None:
    """Ray kills the actor with SIGTERM; the run's evidence must survive.

    This is the deliberate termination experiment: the child delivers six
    intervals, waits for at least one periodic flush, then is SIGTERMed without
    any ``close()``. The whole run's JSONL used to be lost; now at most the last
    flush interval may be missing.
    """
    child = textwrap.dedent(
        """
        import sys, time
        from relax.utils.straggler.config import StragglerConfig
        from relax.utils.straggler.identity import RuntimeIdentity
        from relax.utils.straggler.runtime import StragglerRuntime

        cfg = StragglerConfig(
            enabled=True,
            output_dir=sys.argv[1],
            window_seconds=1.0,
            warmup_windows=0,
            persist_windows=1,
            report_interval_seconds=0.1,
        )
        ident = RuntimeIdentity(run_id="sigterm-run", rank=0, world_size=4, data_parallel_rank=0)
        runtime = StragglerRuntime(cfg, identity=ident, register_atexit=False).start()
        for _ in range(6):
            handle = runtime.timers("forward-compute", log_level=2)
            handle.start()
            time.sleep(0.02)
            handle.stop()
        time.sleep(0.5)  # leave time for at least one periodic flush
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert process.stdout is not None
        assert "ready" in process.stdout.readline()
    finally:
        process.terminate()  # SIGTERM, exactly like Ray's actor teardown
        process.wait(timeout=15)

    run = tmp_path / "run_sigterm-run"
    envelopes = run / "straggler_envelopes.jsonl"
    lines = [json.loads(line) for line in envelopes.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2, "a SIGTERM lost the whole run's evidence"
    assert (run / "collector_status.json").exists()


def test_close_persists_the_sender_side_counters(tmp_path: Any) -> None:
    """The shipping (non-zero) rank is the arm whose counters §6 reports."""
    port = free_port()
    collector_runtime = StragglerRuntime(make_config(f"127.0.0.1:{port}"), identity=identity(0), register_atexit=False)
    sender_runtime = StragglerRuntime(
        make_config(f"127.0.0.1:{port}", output_dir=str(tmp_path)),
        identity=identity(1),
        register_atexit=False,
    )
    collector_runtime.start()
    if not collector_runtime.status().get("receiver", {}).get("listening", False):
        collector_runtime.close()
        sender_runtime.close()
        pytest.skip("loopback TCP is unavailable in this environment")
    try:
        sender_runtime.start()
        run_intervals(sender_runtime, count=2)
    finally:
        sender_runtime.close()
        collector_runtime.close()

    runtime_status = json.loads(next(run_dir(tmp_path).glob("runtime_status_rank1*.json")).read_text())
    assert runtime_status["role"] == "sender"
    assert "sender" in runtime_status
    assert runtime_status["observer"]["intervals"] == 2
    assert "collector" not in runtime_status


def test_close_without_an_output_dir_is_a_clean_noop(tmp_path: Any) -> None:
    """No output_dir must not make close fail while persisting counters."""
    runtime = StragglerRuntime(make_config(), identity=identity(0), register_atexit=False)
    runtime.start()
    runtime.close()

    assert runtime.status()["closed"] is True
    assert runtime.status()["errors"] == 0
    assert list(tmp_path.iterdir()) == []


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

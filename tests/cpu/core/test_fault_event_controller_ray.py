# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real isolated CPU Ray/Serve and native thread acceptance, without GPU
models."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
import ray
import requests
from ray import serve

from relax.utils.async_utils import shutdown_async_loop
from relax.utils.fault_events.query_server import FaultQueryServer
from relax.utils.fault_events.runtime import FaultEventRuntime
from tests.cpu.core.fault_event_support import CPUReplica, config, cpu_controller_class


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def cpu_registry(monkeypatch):
    registry = ModuleType("relax.core.registry")
    registry.resolve_algorithm = lambda config: dict.fromkeys(config.cpu_roles, CPUReplica)
    monkeypatch.setitem(sys.modules, "relax.core.registry", registry)


class LocalRay:
    """Reconnect to one head node, as a production driver does after a global
    restart.

    Starting a fresh local cluster from the health checker thread stalls Serve
    router creation on Linux, which production never does.
    """

    def __init__(self, address: str) -> None:
        self.address = address

    def __getattr__(self, name):
        return getattr(ray, name)

    def init(self, **kwargs):
        return ray.init(
            address=self.address,
            log_to_driver=False,
            runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}},
        )


class LocalServe:
    def __getattr__(self, name):
        return getattr(serve, name)

    def start(self, **kwargs):
        return serve.start(proxy_location="Disabled")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def cluster():
    port = _free_port()
    head = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ray.scripts.scripts",
            "start",
            "--head",
            "--block",
            f"--port={port}",
            "--node-ip-address=127.0.0.1",
            "--num-cpus=4",
            "--include-dashboard=false",
            "--disable-usage-stats",
            f"--temp-dir={tempfile.mkdtemp(prefix='r12-', dir='/tmp')}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    local = LocalRay(f"127.0.0.1:{port}")
    try:
        wait_until(lambda: head.poll() is None and _listening(port), timeout=60)
        local.init()
        LocalServe().start()
        yield local
    finally:
        shutdown_async_loop()
        if ray.is_initialized():
            serve.shutdown()
            ray.shutdown()
        # The head runs in its own session, so stopping it never touches other clusters.
        os.killpg(head.pid, signal.SIGTERM)
        try:
            head.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(head.pid, signal.SIGKILL)
            head.wait(10)


def wait_until(predicate, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("CPU Ray integration condition timed out")


def start_training(controller):
    errors = []

    def train():
        try:
            controller.training_loop()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=train, daemon=True)
    thread.start()
    wait_until(lambda: bool(controller._pending_task_refs) or errors)
    assert not errors, errors
    return thread, errors


def trigger_fault(controller, roles):
    checker = controller._health_manager._checker
    checker._stop_event.clear()
    reports = [controller._health_manager.status.report_error.remote(role, "CPU injected fault") for role in roles]
    ray.get(reports, timeout=5)
    asyncio.run(checker._check_loop())  # global restart sets its original stop event


def poll_receipts(controller, runtime, outcome):
    checker = controller._health_manager._checker
    checker._stop_event.clear()

    async def poll():
        task = asyncio.create_task(checker._check_loop())
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if any(e["event_type"] == outcome for e in runtime.query(limit=100)["events"]):
                    return
                await asyncio.sleep(0.01)
            pytest.fail("completion evidence did not reach the driver")
        finally:
            checker._stop_event.set()
            await asyncio.wait_for(task, 5)

    asyncio.run(poll())


def finish(controller, thread):
    for service in controller.serve_dict.values():
        service.handle.finish.remote().result(timeout_s=5)
    thread.join(10)
    assert not thread.is_alive()


def _run_isolated_ray_case(case: str) -> None:
    """Own Ray only in the child, preserving any cluster used by the suite."""
    was_initialized = ray.is_initialized()
    parent_job_id = ray.get_runtime_context().get_job_id() if was_initialized else None
    child_env = os.environ.copy()
    child_env.pop("RAY_ADDRESS", None)
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).resolve()}::{case}",
                "-o",
                "python_functions=check_*",
                "-q",
                "--disable-warnings",
            ],
            cwd=ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Ray case {case} timed out\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}", pytrace=False)
    output = completed.stdout + completed.stderr
    assert ray.is_initialized() == was_initialized
    if was_initialized:
        assert ray.get_runtime_context().get_job_id() == parent_job_id
        marker = {"ray_parent_probe": "still-owned"}
        assert ray.get(ray.put(marker)) == marker
    if completed.returncode != 0 or "1 passed" not in completed.stdout or " skipped" in completed.stdout:
        # pytest.fail keeps the child's report intact; assertion messages are truncated.
        pytest.fail(f"Ray case {case} failed in its subprocess:\n{output}", pytrace=False)


def test_two_real_ray_rebuilds_use_real_worker_completions_and_keep_http():
    _run_isolated_ray_case("check_two_real_ray_rebuilds_use_real_worker_completions_and_keep_http")


@pytest.mark.parametrize("enabled", [False, True])
def test_native_health_and_training_threads_complete_actual_global_restart(enabled):
    _run_isolated_ray_case(f"check_native_health_and_training_threads_complete_actual_global_restart[{enabled}]")


def check_two_real_ray_rebuilds_use_real_worker_completions_and_keep_http(cluster):
    runtime = FaultEventRuntime()
    server = FaultQueryServer(runtime)
    controller = cpu_controller_class(cluster, LocalServe())(
        config(cpu_roles=["actor", "actor_fwd", "reference"]), {}, runtime
    )
    thread, errors = start_training(controller)
    run_id, url = runtime.run_id, server.url
    reads, read_errors = [], []
    stop_reader = threading.Event()

    def read():
        while not stop_reader.wait(0.02):
            try:
                payload = requests.get(url + "/events", timeout=1).json()
                reads.append(payload["run_id"])
            except Exception as exc:
                read_errors.append(type(exc).__name__)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        controller.config.hold_weights = True
        trigger_fault(controller, ["actor", "actor_fwd"])
        wait_until(lambda: bool(ray.get(controller._cpu_gate.entered.remote())))
        pending = controller._fault_recovery
        assert pending.expected == 63 and pending.outcome is None
        envelope = ray.get(controller._health_manager.status.get_unhealthy_services.remote(include_faults=True))
        assert envelope["roles"] == [] and envelope["receipts"]["observed"] == 0
        ray.get(controller._cpu_gate.release.remote())
        wait_until(lambda: pending.handoff and bool(controller._pending_task_refs))
        poll_receipts(controller, runtime, "recovery_succeeded")
        wait_until(
            lambda: (
                len([e for e in runtime.query(limit=100)["events"] if e["event_type"] == "recovery_succeeded"]) == 2
            )
        )
        first = [e for e in runtime.query(limit=100)["events"] if e["event_type"] == "recovery_succeeded"]
        assert {e["role"] for e in first} == {"actor", "actor_fwd"}
        assert len({e["recovery_id"] for e in first}) == 1
        assert all(
            e["source_generation"] == 0 and e["expected_evidence"] == e["observed_evidence"] == 63 for e in first
        )

        controller.config.hold_weights = False
        controller.config.fail_weights = True
        trigger_fault(controller, ["actor"])
        poll_receipts(controller, runtime, "recovery_failed")
        all_events = runtime.query(limit=100)["events"]
        second = [e for e in all_events if e["event_type"] == "recovery_failed"]
        assert len(second) == 1 and second[0]["failed_evidence"] & 56
        assert second[0]["source_generation"] == 1 and second[0]["fault_id"] != first[0]["fault_id"]
        assert runtime.generation == 2 and runtime.run_id == run_id and server.url == url
        assert len({e["event_id"] for e in all_events}) == len(all_events)
        finish(controller, thread)
        # main awaits actual weight responses: the original worker exception
        # now reaches training, independently of the failed completion receipt.
        assert len(errors) == 1 and "injected CPU weight operation failure" in str(errors[0])
        assert reads and set(reads) == {run_id} and not read_errors
    finally:
        stop_reader.set()
        reader.join(3)
        runtime.close()
        server.close()
        runtime._consumer.join(3)
        server._thread.join(3)


@pytest.mark.parametrize("enabled", [False, True])
def check_native_health_and_training_threads_complete_actual_global_restart(cluster, enabled):
    from relax.utils.async_utils import get_async_loop

    runtime = FaultEventRuntime() if enabled else None
    controller = cpu_controller_class(cluster, LocalServe(), native=True)(
        config(cpu_roles=["actor", "actor_fwd"]), {}, runtime
    )
    thread, errors = start_training(controller)
    try:
        checker = controller._health_manager._checker
        assert checker._thread.ident != get_async_loop()._thread.ident
        ray.get(controller._health_manager.status.report_error.remote("actor", "native restart"), timeout=5)
        wait_until(
            lambda: (
                controller._global_restart_count == 1
                and not controller._restarting
                and bool(controller._pending_task_refs)
            )
        )
        if runtime is not None:
            wait_until(lambda: any(e["event_type"] == "recovery_succeeded" for e in runtime.query()["events"]))
            assert runtime.generation == 1
        finish(controller, thread)
        assert not errors
    finally:
        controller.stop_health_check()
        if runtime is not None:
            runtime.close()
            runtime._consumer.join(3)

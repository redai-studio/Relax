# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Actual CPU Ray source-actor to driver to SDK/OTLP delivery.

The helper actor owns the production HealthStatus implementation and an actor-
local SDK tracer. Only the actor envelope and loopback OTLP receiver are test
scaffolding; fault identity, snapshots, Runtime, Store, SDK and HTTP export all
execute their production paths. No models, GPUs or cluster-wide trace
propagation are assumed.
"""

import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import ray
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from relax.utils.fault_events.runtime import start_fault_events
from tests.utils.test_fault_event_otel_exporter import collector as _collector_fixture


collector = _collector_fixture
ROOT = Path(__file__).resolve().parents[2]


@ray.remote(num_cpus=0.1)
class TracedHealthSource:
    """Scheduling wrapper around production HealthStatus, without model
    work."""

    def __init__(self, generation):
        from relax.utils.health_system import HealthStatus

        implementation = HealthStatus.__ray_metadata__.modified_class
        self.status = implementation(fault_generation=generation)
        self.provider = TracerProvider(resource=Resource({"service.name": "test-source"}), shutdown_on_exit=False)
        self.tracer = self.provider.get_tracer("test-actor-origin")

    def report(self, fatal=False):
        with self.tracer.start_as_current_span("actor-upgrade" if fatal else "actor-discovery") as span:
            origin = span.get_span_context()
            self.status.report_error("actor", "PRIVATE_RAW_ERROR", fatal=fatal)
        return (
            self.status.get_unhealthy_services(include_faults=True),
            {"trace_id": origin.trace_id, "span_id": origin.span_id, "trace_flags": int(origin.trace_flags)},
        )

    def snapshot(self):
        return self.status.get_unhealthy_services(include_faults=True)

    def close(self):
        self.provider.shutdown()


def _wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "CPU Ray/OTLP integration did not reach the expected state"


def _drained(runtime):
    with runtime._lock:
        return not runtime._queue and not runtime._inflight


def check_local_ray_otel(collector, monkeypatch):
    """Run the real source/driver check in the isolated child process."""

    if ray.is_initialized():
        pytest.fail("This test must own an isolated local Ray cluster; an existing cluster was left unchanged")
    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", collector.endpoint + "/v1/logs")
    runtime, actor = None, None
    provider = TracerProvider(resource=Resource({"service.name": "test-poller"}), shutdown_on_exit=False)
    tracer = provider.get_tracer("different-driver-poller")
    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=tempfile.mkdtemp(prefix="r12-otel-", dir="/tmp"),
            log_to_driver=False,
            runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}},
        )
        runtime = start_fault_events(enabled=True, health_enabled=True, port=0)
        assert runtime is not None and runtime._log_exporter is not None
        generation, _ = runtime.bind_source()
        actor = TracedHealthSource.remote(generation)
        first, discovery = ray.get(actor.report.remote(), timeout=25)
        with tracer.start_as_current_span("poll-discovery") as poller:
            poller_context = poller.get_span_context()
            runtime.observe_snapshot(first)
        upgraded, escalation = ray.get(actor.report.remote(True), timeout=10)
        with tracer.start_as_current_span("poll-upgrade"):
            runtime.observe_snapshot(upgraded)
        _wait_for(lambda: _drained(runtime) and runtime.diagnostics()["otel_exported"] == 2)
        source_fault = upgraded["faults"][0]
        expected = {source_fault["event_id"]: discovery, source_fault["fatal_event_id"]: escalation}
        exported = {}
        while len(exported) < 2:
            request, path, _, payload = collector.received.get(timeout=5)
            assert path == "/v1/logs"
            assert b"PRIVATE_RAW_ERROR" not in payload
            for resource in request.resource_logs:
                for scope in resource.scope_logs:
                    for record in scope.log_records:
                        attrs = {attribute.key: attribute.value for attribute in record.attributes}
                        event_id = attrs["relax.fault.event_id"].string_value
                        assert event_id not in exported
                        origin = expected[event_id]
                        assert record.trace_id == origin["trace_id"].to_bytes(16, "big")
                        assert record.span_id == origin["span_id"].to_bytes(8, "big")
                        assert record.trace_id != poller_context.trace_id.to_bytes(16, "big")
                        assert record.flags & 255 == origin["trace_flags"]
                        assert attrs["relax.fault.run_id"].string_value == runtime.run_id
                        assert attrs["relax.fault.fault_id"].string_value == source_fault["fault_id"]
                        assert record.body.string_value in {
                            "relax.fault.fault_detected",
                            "relax.fault.fatal_escalated",
                        }
                        assert record.event_name == record.body.string_value
                        exported[event_id] = record
        assert len({item["trace_id"] for item in expected.values()}) == 2
        for _ in range(4):
            with tracer.start_as_current_span("repeated-poller"):
                runtime.observe_snapshot(ray.get(actor.snapshot.remote(), timeout=5))
        _wait_for(lambda: _drained(runtime))
        status = runtime.diagnostics()
        assert status["otel_admitted"] == status["otel_exported"] == 2
        assert status["known_lost_events"] == status["otel_context_errors"] == 0
        events = runtime.query()["events"]
        assert [event["sequence"] for event in events] == [1, 2]
        assert {event["event_id"] for event in events} == set(expected)
        with pytest.raises(queue.Empty):
            collector.received.get_nowait()
    finally:
        try:
            if actor is not None and ray.is_initialized():
                ray.get(actor.close.remote(), timeout=5)
        finally:
            try:
                if runtime is not None:
                    runtime.close()
                    threads = [runtime._consumer]
                    if runtime._log_exporter is not None:
                        threads.append(runtime._log_exporter._worker)
                    if runtime.query_server is not None and runtime.query_server._thread is not None:
                        threads.append(runtime.query_server._thread)
                    for thread in threads:
                        thread.join(3)
                    assert all(not thread.is_alive() for thread in threads)
            finally:
                provider.shutdown()
                if ray.is_initialized():
                    ray.shutdown()


def test_real_ray_source_context_reaches_driver_runtime_and_otlp_once():
    """Preserve Ray clusters initialized by other modules during collection."""

    was_initialized = ray.is_initialized()
    child_env = os.environ.copy()
    child_env.pop("RAY_ADDRESS", None)
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).resolve()),
            "-o",
            "python_functions=check_local_ray_otel",
            "-q",
            "--disable-warnings",
        ],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "1 passed" in completed.stdout and " skipped" not in completed.stdout, output
    if was_initialized:
        marker = {"ray_parent_probe": "still-owned"}
        assert ray.get(ray.put(marker)) == marker

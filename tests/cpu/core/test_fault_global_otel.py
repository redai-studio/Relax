# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Production Controller startup/global evidence exported through the SDK.

Existing CPU seams replace model/RPC work and the completed Controller rebuild
marker. The production training_loop, recovery-plan/handoff logic, Runtime,
HealthStatus receipt slots, Store and official OTLP HTTP export still run.
"""

import os
import queue
import time

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from relax.utils.fault_events.runtime import start_fault_events
from tests.cpu.core.test_fault_global_recovery_cpu import Roles, prepare_handoff, startup_controller
from tests.cpu.utils.test_fault_event_otel_exporter import collector as _collector_fixture
from tests.cpu.utils.test_fault_health_flow import HealthStatusImpl


collector = _collector_fixture


def _drained(runtime):
    with runtime._lock:
        return not runtime._queue and not runtime._inflight


def _wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate(), "global recovery records did not reach the local OTLP receiver"


@pytest.mark.parametrize("outcome", ["success", "operation_failed", "step_failed"])
def test_controller_global_records_reuse_committed_otel_pipeline(collector, monkeypatch, outcome):
    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", collector.endpoint + "/v1/logs")
    provider = TracerProvider(resource=Resource({"service.name": "test-global-origin"}), shutdown_on_exit=False)
    tracer = provider.get_tracer("global-producer")
    runtime = start_fault_events(enabled=True, health_enabled=True)
    assert runtime is not None and runtime._log_exporter is not None
    sink = runtime._log_exporter
    runtime.bind_source()
    controller = startup_controller(
        runtime, (Roles.actor, Roles.actor_fwd), [], failure="step" if outcome == "step_failed" else None
    )
    try:
        with tracer.start_as_current_span("dispatch") as dispatch:
            dispatch_origin = dispatch.get_span_context()
            recovery = prepare_handoff(controller, runtime)
        with tracer.start_as_current_span("startup") as startup:
            startup_origin = startup.get_span_context()
            if outcome == "step_failed":
                with pytest.raises(RuntimeError, match="injected step failure"):
                    controller.training_loop()
            else:
                controller.training_loop()
        assert runtime._log_exporter is sink and runtime.generation == 1
        if outcome != "step_failed":
            token = runtime.token(recovery)
            status = HealthStatusImpl(fault_generation=1, fault_seed=token)
            status.record_recovery_receipt(token, "actor_send", "succeeded" if outcome == "success" else "failed")
            status.record_recovery_receipt(token, "actor_fwd_recv", "succeeded")
            with tracer.start_as_current_span("merge-completions") as completion:
                completion_origin = completion.get_span_context()
                runtime.observe_snapshot(status.get_unhealthy_services(include_faults=True))
        else:
            completion_origin = startup_origin
        expected_events = 3 if outcome == "step_failed" else 4
        _wait_for(lambda: _drained(runtime) and runtime.diagnostics()["otel_exported"] == expected_events)
        local = runtime.query()["events"]
        assert local[0]["event_type"] == "fault_detected"
        global_events = [event for event in local if event["recovery_id"] == recovery.recovery_id]
        terminal = "recovery_succeeded" if outcome == "success" else "recovery_failed"
        assert [event["event_type"] for event in global_events] == (
            ["handling_started", terminal]
            if outcome == "step_failed"
            else ["handling_started", "recovery_handoff", terminal]
        )
        assert global_events[-1]["expected_evidence"] == 31
        assert (
            global_events[-1]["reason_code"]
            == {
                "success": None,
                "operation_failed": "operation_failed",
                "step_failed": "step_restore_failed",
            }[outcome]
        )
        by_identity = {event["event_id"]: event for event in local}
        received = {}
        while len(received) < expected_events:
            request, path, _, payload = collector.received.get(timeout=5)
            assert path == "/v1/logs" and b"injected step failure" not in payload
            for resource in request.resource_logs:
                for scope in resource.scope_logs:
                    for record in scope.log_records:
                        attrs = {attribute.key: attribute.value for attribute in record.attributes}
                        event_id = attrs["relax.fault.event_id"].string_value
                        event = by_identity[event_id]
                        assert event_id not in received
                        assert record.event_name == record.body.string_value == "relax.fault." + event["event_type"]
                        for field in ("run_id", "fault_id", "recovery_id", "recovery_mode"):
                            if event[field] is not None:
                                assert attrs["relax.fault." + field].string_value == event[field]
                        for field in ("source_generation", "sequence", "recorded_at_ms"):
                            assert attrs["relax.fault." + field].string_value == str(event[field])
                        for field in ("expected_evidence", "observed_evidence", "failed_evidence"):
                            assert attrs["relax.fault." + field].int_value == event[field]
                        if event["event_type"] == "fault_detected":
                            assert record.trace_id == record.span_id == b""
                        else:
                            origin = (
                                dispatch_origin
                                if event["event_type"] == "handling_started"
                                else startup_origin
                                if event["event_type"] == "recovery_handoff"
                                else completion_origin
                            )
                            assert record.trace_id == origin.trace_id.to_bytes(16, "big")
                            assert record.span_id == origin.span_id.to_bytes(8, "big")
                        received[event_id] = record
        if outcome != "step_failed":
            with tracer.start_as_current_span("later-poller"):
                runtime.observe_snapshot(status.get_unhealthy_services(include_faults=True))
            _wait_for(lambda: _drained(runtime))
        assert runtime.diagnostics()["otel_admitted"] == expected_events
        assert runtime.diagnostics()["known_lost_events"] == 0
        assert runtime._log_exporter is sink
        with pytest.raises(queue.Empty):
            collector.received.get_nowait()
    finally:
        runtime.close()
        threads = [runtime._consumer, sink._worker, runtime.query_server._thread]
        for thread in threads:
            thread.join(3)
        assert all(not thread.is_alive() for thread in threads)
        provider.shutdown()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Commit/export integration, with real producer contexts and OTLP delivery."""

import threading
import time
from collections import deque

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from relax.utils.fault_events.context import FaultTraceContext
from relax.utils.fault_events.models import EventDraft, RecoveryEvidence
from relax.utils.fault_events.runtime import FaultEventRuntime, start_fault_events
from relax.utils.fault_events.source import FaultSource
from tests.utils.test_fault_event_otel_exporter import collector as _collector_fixture


collector = _collector_fixture


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate()


def draft(runtime, identity=1):
    return EventDraft(runtime.run_id, "f" * 32, f"{identity:032x}", "actor", 0, "fault_detected", "reported_error")


class MemorySink:
    def __init__(self, failure=False):
        self.records = []
        self.failure = failure

    def offer(self, record, context):
        if self.failure:
            raise RuntimeError("PRIVATE_EXPORT_FAILURE")
        self.records.append((record, context))
        return True

    def diagnostics(self):
        return {
            "otel_enabled": True,
            "otel_available": True,
            "otel_queue_capacity": 512,
            "otel_queued": 0,
            "otel_inflight": 0,
            "otel_admitted": len(self.records),
            "otel_rejections": 0,
            "otel_exported": len(self.records),
            "otel_export_failures": 0,
            "otel_shutdown_dropped": 0,
            "otel_context_errors": 0,
            "otel_closed": False,
        }

    def close(self, timeout=0.05):
        pass


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os

    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)


@pytest.fixture
def make_runtime():
    created = []

    def make(**kwargs):
        runtime = FaultEventRuntime(**kwargs)
        created.append(runtime)
        return runtime

    yield make
    for runtime in created:
        runtime.close()
        runtime._consumer.join(3)
        assert not runtime._consumer.is_alive()


@pytest.fixture
def tracer():
    provider = TracerProvider(resource=Resource({"service.name": "test"}), shutdown_on_exit=False)
    yield provider.get_tracer("fault-origin")
    provider.shutdown()


def drained(runtime):
    with runtime._lock:
        return not runtime._queue and not runtime._inflight


def test_runtime_to_real_otlp_keeps_first_origin_and_deduplicates(collector, monkeypatch, tracer):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", collector.endpoint + "/v1/logs")
    runtime = start_fault_events(enabled=True, health_enabled=True)
    entered, release = threading.Event(), threading.Event()
    completed = threading.Event()
    original_commit, original_submit = runtime.store._append_committed, runtime._submit
    submissions = []

    def commit(value):
        entered.set()
        assert release.wait(3)
        return original_commit(value)

    def submit(value, context):
        with tracer.start_as_current_span("unrelated-consumer"):
            original_submit(value, context)
        submissions.append(value.event_id)
        if len(submissions) == 2:
            completed.set()

    monkeypatch.setattr(runtime.store, "_append_committed", commit)
    monkeypatch.setattr(runtime, "_submit", submit)
    try:
        item = draft(runtime)
        with tracer.start_as_current_span("producer") as span:
            origin = span.get_span_context()
            assert runtime.record(item)
        assert entered.wait(3)
        with tracer.start_as_current_span("duplicate-producer"):
            assert runtime.record(item)
        release.set()
        request, path, _, _ = collector.received.get(timeout=3)
        record = request.resource_logs[0].scope_logs[0].log_records[0]
        assert path == "/v1/logs"
        assert record.trace_id == origin.trace_id.to_bytes(16, "big")
        assert record.span_id == origin.span_id.to_bytes(8, "big")
        assert record.event_name == "relax.fault.fault_detected"
        attributes = {attribute.key: attribute.value for attribute in record.attributes}
        assert attributes["relax.fault.sequence"].string_value == "1"
        assert attributes["relax.fault.event_id"].string_value == item.event_id
        wait_for(lambda: drained(runtime))
        assert runtime.record(item)  # Retained duplicate also produces no new export.
        assert completed.wait(3)
        assert runtime._log_exporter.diagnostics()["otel_admitted"] == 1
        assert collector.received.empty()
        assert runtime.store.status()["event_count"] == 1
        assert runtime.diagnostics()["known_lost_events"] == 0
    finally:
        release.set()
        runtime.close()
        runtime._consumer.join(3)
        runtime._log_exporter._worker.join(3)
        runtime.query_server._thread.join(3)
        assert not runtime._consumer.is_alive()
        assert not runtime._log_exporter._worker.is_alive()
        assert not runtime.query_server._thread.is_alive()


def test_exact_commit_export_survives_another_writer_evicting_it(make_runtime, monkeypatch):
    sink = MemorySink()
    runtime = make_runtime(capacity=1, log_exporter=sink)
    original = runtime.store._append_committed
    committed = []
    item, other = draft(runtime), draft(runtime, 2)

    def append(value):
        outcome = original(value)
        committed.append(outcome[1])
        original(other)
        return outcome

    monkeypatch.setattr(runtime.store, "_append_committed", append)
    assert runtime.record(item)
    wait_for(lambda: drained(runtime))
    exported, _ = sink.records[0]
    assert exported == committed[0].draft.record(committed[0].sequence, committed[0].timestamp)
    assert exported["event_id"] == item.event_id and exported["sequence"] == 1
    assert runtime.store.query()["events"][0]["event_id"] == other.event_id
    assert runtime.store.status()["latest_sequence"] == 2


def test_export_offer_exception_never_invalidates_a_definite_local_commit(make_runtime):
    runtime = make_runtime(log_exporter=MemorySink(failure=True))
    assert runtime.record(draft(runtime))
    wait_for(lambda: drained(runtime))
    assert runtime.record(draft(runtime, 2))
    wait_for(lambda: drained(runtime))
    assert [event["sequence"] for event in runtime.query()["events"]] == [1, 2]
    status = runtime.diagnostics()
    assert status["otel_observer_errors"] == 2
    assert status["known_lost_events"] == 0 and not status["coverage_unknown"]
    assert not runtime._store_unavailable.is_set()
    assert "PRIVATE_EXPORT_FAILURE" not in str(status)


def test_queue_admission_failure_releases_pending_identity(make_runtime):
    class UnavailableQueue(deque):
        def append(self, value):
            raise MemoryError("queue allocation unavailable")

    runtime = make_runtime(log_exporter=MemorySink())
    runtime._queue = UnavailableQueue()
    item = draft(runtime)
    assert runtime.record(item) is False
    assert not runtime._pending and not runtime._queue
    runtime._queue = deque()
    assert runtime.record(item)
    wait_for(lambda: drained(runtime))
    assert runtime.query()["events"][0]["event_id"] == item.event_id
    assert runtime.diagnostics()["admission_rejections"] == 1
    assert runtime.diagnostics()["known_lost_events"] == 0


def test_queued_duplicate_preserves_first_context_and_fifo(make_runtime, monkeypatch, tracer):
    sink = MemorySink()
    runtime = make_runtime(queue_capacity=1, log_exporter=sink)
    entered, release = threading.Event(), threading.Event()
    first, queued = draft(runtime), draft(runtime, 2)
    original = runtime.store._append_committed

    def append(value):
        if value.event_id == first.event_id:
            entered.set()
            assert release.wait(3)
        return original(value)

    monkeypatch.setattr(runtime.store, "_append_committed", append)
    try:
        assert runtime.record(first)
        assert entered.wait(3)
        with tracer.start_as_current_span("queued-origin") as span:
            origin = span.get_span_context()
            assert runtime.record(queued)
        with tracer.start_as_current_span("duplicate-origin"):
            assert runtime.record(queued)
        assert runtime.diagnostics()["queued"] == runtime.diagnostics()["inflight"] == 1
    finally:
        release.set()
    wait_for(lambda: drained(runtime))
    assert [record["event_id"] for record, _ in sink.records] == [first.event_id, queued.event_id]
    assert sink.records[1][1] == FaultTraceContext(origin.trace_id, origin.span_id, int(origin.trace_flags))
    assert not runtime._pending


def test_recovery_flags_are_normalized_only_at_runtime_admission(make_runtime):
    class OtherInt(int):
        pass

    sink = MemorySink()
    runtime = make_runtime(log_exporter=sink)
    generation, _ = runtime.bind_source()
    source = FaultSource(generation)
    source.error("advantages", "reported_error")
    runtime.observe_snapshot(source.snapshot(["advantages"]))
    for invalid in (True, OtherInt(7)):
        assert runtime.start_recovery("advantages", "local", invalid) is None
    recovery = runtime.start_recovery("advantages", "local", RecoveryEvidence.LOCAL)
    assert recovery is not None and type(recovery.expected) is int
    for invalid in (True, OtherInt(1)):
        runtime.evidence(recovery, observed=invalid)
        runtime.evidence(recovery, failed=invalid)
        assert recovery.observed == recovery.failed == 0
    runtime.evidence(recovery, observed=RecoveryEvidence.DEPLOYED | RecoveryEvidence.TASK_SUBMITTED)
    runtime.evidence(recovery, failed=RecoveryEvidence.STEP_RESTORED)
    wait_for(lambda: drained(runtime))
    stored = runtime.query()["events"][-1]
    assert stored["event_type"] == "recovery_failed"
    assert [stored[name] for name in ("expected_evidence", "observed_evidence", "failed_evidence")] == [7, 5, 2]
    for record in (stored, *(item for item, _ in sink.records)):
        assert all(type(record[name]) is int for name in ("expected_evidence", "observed_evidence", "failed_evidence"))


@pytest.mark.parametrize("endpoint", [None, "not-a-collector"])
def test_export_off_or_invalid_configuration_preserves_local_queries(monkeypatch, endpoint):
    if endpoint is not None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", endpoint)
    runtime = start_fault_events(enabled=True, health_enabled=True)
    try:
        assert runtime._log_exporter is None
        assert runtime.record(draft(runtime))
        wait_for(lambda: drained(runtime))
        assert runtime.query()["events"][0]["sequence"] == 1
        assert runtime.diagnostics()["otel_startup_errors"] == int(endpoint is not None)
        assert runtime.diagnostics()["known_lost_events"] == 0
    finally:
        runtime.close()
        runtime._consumer.join(3)
        runtime.query_server._thread.join(3)


def test_invalid_otel_configuration_warning_names_setting_without_value(monkeypatch):
    from relax.utils.fault_events import runtime as runtime_module

    warnings = []
    monkeypatch.setattr(runtime_module, "_warn", warnings.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://127.0.0.1:4318/v1/logs")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=Bearer PRIVATE;TOKEN")
    runtime = start_fault_events(enabled=True, health_enabled=True)
    try:
        assert runtime._log_exporter is None
        assert warnings == [
            "Fault event log export disabled: INVALID_OTEL_CONFIG: headers must use the SDK name=value grammar"
        ]
    finally:
        runtime.close()
        runtime._consumer.join(3)
        runtime.query_server._thread.join(3)


def test_source_sidecar_uses_discovery_and_upgrade_origins_not_health_poller(make_runtime, tracer):
    sink = MemorySink()
    runtime = make_runtime(log_exporter=sink)
    generation, _ = runtime.bind_source()
    source = FaultSource(generation)
    with tracer.start_as_current_span("discovery") as discovery:
        source.error("actor", "reported_error")
        discovery_context = discovery.get_span_context()
    with tracer.start_as_current_span("poller"):
        runtime.observe_snapshot(source.snapshot(["actor"]))
    wait_for(lambda: drained(runtime))
    with tracer.start_as_current_span("fatal") as fatal:
        source.error("actor", "fatal_error", fatal=True)
        fatal_context = fatal.get_span_context()
    with tracer.start_as_current_span("poller-again"):
        runtime.observe_snapshot(source.snapshot(["actor"]))
    wait_for(lambda: drained(runtime))
    assert [item[0]["event_type"] for item in sink.records] == ["fault_detected", "fatal_escalated"]
    assert sink.records[0][1] == FaultTraceContext(
        discovery_context.trace_id, discovery_context.span_id, int(discovery_context.trace_flags)
    )
    assert sink.records[1][1] == FaultTraceContext(
        fatal_context.trace_id, fatal_context.span_id, int(fatal_context.trace_flags)
    )
    assert not runtime._pending


def test_malformed_sidecar_degrades_to_untraced_export_without_local_loss(make_runtime):
    sink = MemorySink()
    runtime = make_runtime(log_exporter=sink)
    generation, _ = runtime.bind_source()
    source = FaultSource(generation)
    source.error("actor", "reported_error")
    snapshot = source.snapshot(["actor"])
    event_id = snapshot["faults"][0]["event_id"]
    snapshot["trace_contexts"] = {event_id: {"trace_id": "PRIVATE_TRACE_DATA"}}
    runtime.observe_snapshot(snapshot)
    wait_for(lambda: drained(runtime))
    assert len(sink.records) == 1 and sink.records[0][1] is None
    assert runtime.diagnostics()["otel_context_errors"] == 1
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert not runtime.diagnostics()["coverage_unknown"]
    assert "PRIVATE_TRACE_DATA" not in str(runtime.query())


def test_deferred_local_completion_keeps_completion_producer_context(make_runtime, tracer):
    sink = MemorySink()
    runtime = make_runtime(log_exporter=sink)
    generation, _ = runtime.bind_source()
    source = FaultSource(generation)
    source.error("advantages", "reported_error")
    runtime.observe_snapshot(source.snapshot(["advantages"]))
    recovery = runtime.start_recovery("advantages", "local", 7)
    runtime.evidence(recovery, observed=7)
    with runtime._lock:
        with tracer.start_as_current_span("completion") as span:
            origin = span.get_span_context()
            runtime.complete_local(recovery)
    with tracer.start_as_current_span("later-poller"):
        runtime._call(None)
    wait_for(lambda: drained(runtime))
    expected = FaultTraceContext(origin.trace_id, origin.span_id, int(origin.trace_flags))
    terminal = [(record, context) for record, context in sink.records if record["event_type"] == "recovery_succeeded"]
    assert len(terminal) == 1 and terminal[0][1] == expected
    assert runtime._admission_context is None

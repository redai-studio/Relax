# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Producer context copies and real health flow with an explicit local RPC."""

import asyncio
import builtins
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from relax.utils import health_system
from relax.utils.fault_events.context import (
    FaultTraceContext,
    capture_trace_context,
    parse_trace_context,
    parse_trace_contexts,
)
from relax.utils.fault_events.models import ROLES, FaultContext
from relax.utils.fault_events.source import FaultSource
from tests.utils.test_fault_health_flow import HealthStatusImpl, LocalStatusRPC


def test_trace_context_round_trips_with_fixed_width_ids():
    context = FaultTraceContext(1, 2, 1)
    payload = context.to_payload()
    assert payload == {"trace_id": "0" * 31 + "1", "span_id": "0" * 15 + "2", "trace_flags": 1}
    assert parse_trace_context(payload) == context
    assert parse_trace_contexts({"a" * 32: payload}) == {"a" * 32: context}


def test_capture_copies_actual_optional_otel_api_context():
    from relax.utils.fault_events.context import capture_trace_context

    span = NonRecordingSpan(SpanContext(trace_id=17, span_id=19, is_remote=False, trace_flags=TraceFlags(1)))
    with trace.use_span(span, end_on_exit=False):
        captured = capture_trace_context()
    assert captured == FaultTraceContext(17, 19, 1)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"trace_id": "0" * 32, "span_id": "1" * 16, "trace_flags": 0},
        {"trace_id": "g" * 32, "span_id": "1" * 16, "trace_flags": 0},
        {"trace_id": "1" * 32, "span_id": "1" * 16, "trace_flags": True},
    ],
)
def test_invalid_contexts_use_fixed_error_and_never_raise_from_capture(payload):
    with pytest.raises(ValueError, match="^INVALID_TRACE_CONTEXT: "):
        parse_trace_context(payload)
    with pytest.raises(ValueError, match="^INVALID_TRACE_CONTEXT: "):
        parse_trace_contexts({"a" * 32: payload})


def test_capture_without_optional_sdk_is_untraced(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "opentelemetry":
            raise ModuleNotFoundError("optional telemetry unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert capture_trace_context() is None


def test_fault_source_keeps_business_payload_strict_and_sidecar_optional(monkeypatch):
    context = FaultTraceContext(3, 4, 1)
    monkeypatch.setattr("relax.utils.fault_events.source.capture_trace_context", lambda: context)
    source = FaultSource(0)
    source.error("actor", "reported_error")
    fault = source.get("actor")
    snapshot = source.snapshot(["actor"])
    assert set(fault) == {"role", "fault_id", "source_generation", "event_id", "reason", "fatal", "fatal_event_id"}
    assert set(snapshot) == {"roles", "source_generation", "faults", "observation_error", "trace_contexts"}
    assert snapshot["trace_contexts"][fault["event_id"]] == context.to_payload()

    source.error("actor", "reported_error", fatal=True)
    upgraded = source.get("actor")
    sidecars = source.snapshot(["actor"])["trace_contexts"]
    assert set(sidecars) == {fault["event_id"], upgraded["fatal_event_id"]}
    source.clear("actor")
    assert source.snapshot([]).keys() == {"roles", "source_generation", "faults", "observation_error"}


def test_malformed_optional_context_does_not_change_health_business_payload():
    source = FaultSource(0)
    source.error("actor", "reported_error")
    before = source.get("actor")
    assert source.trace_contexts("actor") == {}
    source._trace_contexts["actor"] = {before["event_id"]: object()}
    assert "trace_contexts" not in source.snapshot(["actor"])
    assert source.get("actor") == before
    assert not source.observation_error


@pytest.mark.parametrize(
    "values", [(0, 1, 0), (1 << 128, 1, 0), (True, 1, 0), (1, 0, 0), (1, 1 << 64, 0), (1, 2, 256)]
)
def test_context_constructor_rejects_invalid_widths_and_is_immutable(values):
    with pytest.raises(ValueError, match="^INVALID_TRACE_CONTEXT: "):
        FaultTraceContext(*values)
    context = FaultTraceContext(1, 2, 1)
    with pytest.raises(FrozenInstanceError):
        context.trace_id = 3


def test_sidecar_map_is_bounded_and_optional():
    payload = FaultTraceContext(1, 2).to_payload()
    assert parse_trace_context(None) is None and parse_trace_contexts(None) == {}
    contexts = {f"{index:032x}": payload for index in range(18)}
    assert len(parse_trace_contexts(contexts)) == 18
    for invalid in ({**contexts, "f" * 32: payload}, {"short": payload}, {"a" * 32: None}):
        with pytest.raises(ValueError, match="^INVALID_TRACE_CONTEXT: "):
            parse_trace_contexts(invalid)


def test_d_u_trace_identity_is_frozen_across_repeated_observations(monkeypatch):
    discovery, upgrade = FaultTraceContext(1, 2), FaultTraceContext(3, 4, 1)
    capture = Mock(side_effect=[discovery, upgrade])
    monkeypatch.setattr("relax.utils.fault_events.source.capture_trace_context", capture)
    source = FaultSource(0)
    source.error("actor", "reported_error")
    first = source.get("actor")
    source.error("actor", "heartbeat_timeout")
    assert capture.call_count == 1
    source.error("actor", "reported_error", fatal=True)
    current = source.get("actor")
    source.error("actor", "reported_error", fatal=True)
    assert capture.call_count == 2
    FaultContext.from_dict(current)
    sidecars = source.trace_contexts("actor")
    assert sidecars == {first["event_id"]: discovery.to_payload(), current["fatal_event_id"]: upgrade.to_payload()}
    sidecars[first["event_id"]]["trace_flags"] = 255
    assert source.trace_contexts("actor")[first["event_id"]] == discovery.to_payload()


def test_source_trace_state_stays_bounded_and_clear_releases_it(monkeypatch):
    monkeypatch.setattr("relax.utils.fault_events.source.capture_trace_context", lambda: FaultTraceContext(1, 2))
    source = FaultSource(0)
    for _ in range(20):
        for role in ROLES:
            source.error(role, "reported_error")
            source.error(role, "reported_error", fatal=True)
        assert len(source.snapshot(list(ROLES))["trace_contexts"]) == 18
        assert sum(map(len, source._trace_contexts.values())) == 18
        for role in ROLES:
            source.clear(role)
        assert source._trace_contexts == {}


def test_broken_optional_capture_preserves_faults_and_health(monkeypatch):
    monkeypatch.setattr(
        "relax.utils.fault_events.source.capture_trace_context", Mock(side_effect=RuntimeError("PRIVATE"))
    )
    status = HealthStatusImpl(fault_generation=0)
    status.report_error("actor", "PRIVATE")
    source = status._fault_source
    assert source.get("actor") is not None and not source.observation_error
    monkeypatch.setattr(source, "trace_contexts", lambda role: {"a" * 32: {"baggage": "PRIVATE"}})
    details = status.get_service_health("actor", include_fault=True)
    assert details["fault"] is not None and "trace_contexts" not in details
    assert not details["observation_error"] and status._fault_source is source
    assert not status.get_unhealthy_services(include_faults=True)["observation_error"]


def test_health_context_gate_keeps_legacy_fault_and_default_return_shape(monkeypatch):
    context = FaultTraceContext(1, 2)
    monkeypatch.setattr("relax.utils.fault_events.source.capture_trace_context", lambda: context)
    status = HealthStatusImpl(fault_generation=0)
    legacy = status.mark_unhealthy("actor", "PRIVATE", return_fault=True)
    FaultContext.from_dict(legacy)
    envelope = status.mark_unhealthy("actor", return_fault=True, include_trace_context=True)
    assert envelope == {"fault": legacy, "trace_contexts": {legacy["event_id"]: context.to_payload()}}
    assert status.get_unhealthy_services(include_faults=True)["trace_contexts"] == envelope["trace_contexts"]
    assert status.mark_unhealthy("actor", include_trace_context=True) is None
    status.mark_healthy("actor")
    assert "trace_contexts" not in status.get_unhealthy_services(include_faults=True)


@pytest.mark.parametrize("stale", [False, True])
def test_real_health_checker_forwards_source_context_in_existing_rpc(monkeypatch, stale):
    context = FaultTraceContext(1, 2, 1)
    monkeypatch.setattr("relax.utils.fault_events.source.capture_trace_context", lambda: context)
    monkeypatch.setattr(health_system.ray, "get", lambda result: result)
    status = HealthStatusImpl(fault_generation=0)
    if stale:
        status.mark_healthy("actor")
        status.set_task_status("actor", True)
        status.state["actor"].last_heartbeat = 0
    else:
        status.report_error("actor", "PRIVATE")
    rpc, observations = LocalStatusRPC(status), []
    observer = SimpleNamespace(
        observe_snapshot=lambda payload: observations.append(("snapshot", payload)),
        observe_fault=lambda fault, role, contexts=None: observations.append(("fault", fault, role, contexts)),
        note_unknown=lambda: observations.append(("unknown",)),
    )

    def recover(role):
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=observer)
    asyncio.run(checker._check_loop())
    fault = next(row for row in observations if row[0] == "fault")
    assert fault[2] == "actor" and fault[3] == {fault[1]["event_id"]: context.to_payload()}
    assert not any(row[0] == "unknown" for row in observations)
    names = [name for name, _, _ in rpc.calls]
    assert names == (
        ["get_unhealthy_services", "get_stale_services", "mark_unhealthy"]
        if stale
        else ["get_unhealthy_services", "get_service_health"]
    )

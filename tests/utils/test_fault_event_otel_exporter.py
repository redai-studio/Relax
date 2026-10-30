# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real SDK/OTLP protobuf delivery with explicit collector and failure
seams."""

import gzip
import logging
import queue
import socket
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry import _logs, trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest, ExportLogsServiceResponse
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from relax.utils.fault_events import telemetry
from relax.utils.fault_events.context import FaultTraceContext
from relax.utils.fault_events.models import MAX_UINT, EventDraft


@pytest.fixture(autouse=True)
def clean_otel_environment(monkeypatch):
    for name in tuple(telemetry.os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)


def record(*, sequence=1, timestamp=1234, event_type="fault_detected"):
    return EventDraft(
        "1" * 32,
        "2" * 32,
        "3" * 32,
        "actor",
        0,
        event_type,
        reason_code="reported_error" if event_type == "fault_detected" else None,
    ).record(sequence, timestamp)


def wait_for(function, expected, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = function()
        if value == expected:
            return
        time.sleep(0.001)
    assert function() == expected


@pytest.fixture
def collector():
    received = queue.Queue()
    settings = SimpleNamespace(status=200, reason="OK", partial_rejected=0)

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            elif self.headers.get("Content-Encoding") == "deflate":
                data = zlib.decompress(data)
            request = ExportLogsServiceRequest()
            request.ParseFromString(data)
            received.put((request, self.path, dict(self.headers), data))
            response = ExportLogsServiceResponse()
            if settings.partial_rejected:
                response.partial_success.rejected_log_records = settings.partial_rejected
                response.partial_success.error_message = "private collector rejection"
            body = response.SerializeToString()
            self.send_response(settings.status, settings.reason)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Receiver)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    yield SimpleNamespace(endpoint=f"http://127.0.0.1:{server.server_port}", received=received, settings=settings)
    server.shutdown()
    server.server_close()
    worker.join(2)
    assert not worker.is_alive()


def close_sink(sink):
    sink.close()
    sink._worker.join(3)
    assert not sink._worker.is_alive()


@pytest.mark.parametrize("with_context", [False, True])
def test_real_sdk_http_protobuf_preserves_origin_and_whitelisted_fields(collector, monkeypatch, with_context):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", collector.endpoint + "/direct/logs")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=Bearer%20private%3Bheader")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "private.resource=must-not-be-exported")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "must-not-be-exported")
    original_logs, original_traces = _logs.get_logger_provider(), trace.get_tracer_provider()
    handlers = tuple(logging.getLogger().handlers)
    provider = TracerProvider(resource=Resource({"service.name": "test-origin"}), shutdown_on_exit=False)
    context = None
    with provider.get_tracer("origin").start_as_current_span("original") as span:
        origin = span.get_span_context()
        assert origin.is_valid
        if with_context:
            context = FaultTraceContext(origin.trace_id, origin.span_id, int(origin.trace_flags))
    sink = telemetry.start_fault_log_exporter(enabled=True)
    try:
        item = record()
        with provider.get_tracer("unrelated").start_as_current_span("must-not-be-used"):
            assert sink.offer(item, context)
        request, path, headers, payload = collector.received.get(timeout=3)
        resource_logs = request.resource_logs[0]
        scope = resource_logs.scope_logs[0]
        received = scope.log_records[0]
        assert path == "/direct/logs"
        assert (
            next(value for key, value in headers.items() if key.lower() == "content-type") == "application/x-protobuf"
        )
        assert (
            next(value for key, value in headers.items() if key.lower() == "authorization") == "Bearer private;header"
        )
        assert {a.key: a.value.string_value for a in resource_logs.resource.attributes} == {"service.name": "relax"}
        assert scope.scope.name == "relax.fault_events"
        assert received.event_name == received.body.string_value == "relax.fault.fault_detected"
        assert received.time_unix_nano == 1_234_000_000
        attrs = {a.key: a.value for a in received.attributes}
        assert set(attrs) == {"relax.fault." + key for key, value in item.items() if value is not None}
        for key, value in item.items():
            if value is None:
                continue
            actual = attrs["relax.fault." + key]
            if key in {"source_generation", "sequence", "recorded_at_ms"}:
                assert actual.string_value == str(value)
            elif type(value) is int:
                assert actual.int_value == value
            else:
                assert actual.string_value == value
        assert received.trace_id == (origin.trace_id.to_bytes(16, "big") if with_context else b"")
        assert received.span_id == (origin.span_id.to_bytes(8, "big") if with_context else b"")
        assert received.flags & 255 == (int(origin.trace_flags) if with_context else 0)
        assert b"private-header" not in payload and b"must-not-be-exported" not in payload
        wait_for(lambda: sink.diagnostics()["otel_exported"], 1)
        assert _logs.get_logger_provider() is original_logs and trace.get_tracer_provider() is original_traces
        assert tuple(logging.getLogger().handlers) == handlers
    finally:
        close_sink(sink)
        provider.shutdown()


def test_uint64_extrema_remain_lossless_without_timestamp_overflow(collector):
    sink = telemetry.FaultLogExporter(collector.endpoint + "/v1/logs")
    try:
        draft = EventDraft(
            "1" * 32,
            "2" * 32,
            "3" * 32,
            "actor",
            MAX_UINT,
            "recovery_failed",
            "step_restore_failed",
            "4" * 32,
            "global",
            63,
            1,
            2,
        )
        item = draft.record(MAX_UINT, MAX_UINT)
        assert sink.offer(item, FaultTraceContext((1 << 128) - 1, MAX_UINT, 255))
        request, *_ = collector.received.get(timeout=3)
        received = request.resource_logs[0].scope_logs[0].log_records[0]
        attrs = {a.key: a.value for a in received.attributes}
        assert set(attrs) == {"relax.fault." + key for key in item}
        for key in ("source_generation", "sequence", "recorded_at_ms"):
            assert attrs["relax.fault." + key].string_value == str(MAX_UINT)
        assert received.time_unix_nano == 0
        assert received.trace_id == b"\xff" * 16 and received.span_id == b"\xff" * 8
        assert attrs["relax.fault.expected_evidence"].int_value == 63
        assert attrs["relax.fault.failed_evidence"].int_value == 2
        assert received.event_name == received.body.string_value == "relax.fault.recovery_failed"
    finally:
        close_sink(sink)


def test_owned_sdk_limits_preserve_all_fields_under_restrictive_ambient_limits(collector, monkeypatch):
    for name in ("OTEL_ATTRIBUTE_COUNT_LIMIT", "OTEL_LOGRECORD_ATTRIBUTE_COUNT_LIMIT"):
        monkeypatch.setenv(name, "1")
    for name in ("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "OTEL_LOGRECORD_ATTRIBUTE_VALUE_LENGTH_LIMIT"):
        monkeypatch.setenv(name, "4")
    sink = telemetry.FaultLogExporter(collector.endpoint + "/v1/logs")
    try:
        item = EventDraft(
            "1" * 32,
            "2" * 32,
            "3" * 32,
            "actor",
            MAX_UINT,
            "recovery_failed",
            "step_restore_failed",
            "4" * 32,
            "global",
            63,
            1,
            2,
        ).record(MAX_UINT, 1_800_000_000_000)
        assert sink.offer(item, FaultTraceContext((1 << 128) - 1, MAX_UINT, 255))
        request, *_ = collector.received.get(timeout=3)
        received = request.resource_logs[0].scope_logs[0].log_records[0]
        attributes = {attribute.key: attribute.value for attribute in received.attributes}
        assert len(attributes) == 17
        for key, value in item.items():
            actual = attributes["relax.fault." + key]
            if key in {"source_generation", "sequence", "recorded_at_ms"}:
                assert actual.string_value == str(value)
            elif type(value) is int:
                assert actual.int_value == value
            else:
                assert actual.string_value == value
        assert received.event_name == received.body.string_value == "relax.fault.recovery_failed"
        assert received.time_unix_nano == 1_800_000_000_000_000_000
        assert received.trace_id == b"\xff" * 16 and received.span_id == b"\xff" * 8
        wait_for(lambda: sink.diagnostics()["otel_exported"], 1)
    finally:
        close_sink(sink)


@pytest.mark.parametrize(
    ("event_type", "reason", "severity"),
    [
        ("fault_detected", "reported_error", SeverityNumber.WARN),
        ("fatal_escalated", "fatal_error", SeverityNumber.ERROR),
        ("handling_started", None, SeverityNumber.INFO),
        ("recovery_handoff", None, SeverityNumber.INFO),
        ("recovery_succeeded", None, SeverityNumber.INFO),
        ("recovery_failed", "restart_exception", SeverityNumber.ERROR),
        ("recovery_interrupted", "superseded", SeverityNumber.WARN),
        ("handling_skipped", "role_unregistered", SeverityNumber.WARN),
        ("termination_requested", "restart_limit", SeverityNumber.ERROR),
    ],
)
def test_all_event_types_have_fixed_native_severity(event_type, reason, severity):
    recovery = "4" * 32 if event_type not in {"fault_detected", "fatal_escalated"} else None
    mode = (
        "global"
        if event_type
        in {
            "handling_started",
            "recovery_handoff",
            "recovery_succeeded",
            "recovery_failed",
            "recovery_interrupted",
            "termination_requested",
        }
        else None
    )
    expected = 0 if event_type == "handling_skipped" else 31 if mode == "global" else 7 if recovery else 0
    observed = expected if event_type == "recovery_succeeded" else 0
    draft = EventDraft(
        "1" * 32,
        "2" * 32,
        "3" * 32,
        "actor",
        0,
        event_type,
        reason,
        recovery,
        mode,
        expected,
        observed,
        0,
    )
    draft.validate()
    item = draft.record(1, 1)
    log_record = telemetry._sdk_record(item, None)
    assert log_record.severity_number == severity
    assert log_record.event_name == f"relax.fault.{event_type}"
    assert log_record.body == f"relax.fault.{event_type}"


def test_generic_endpoint_appends_signal_path_and_signal_endpoint_wins(collector, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint + "/collector/")
    sink = telemetry.start_fault_log_exporter()
    try:
        assert sink.offer(record())
        _, path, *_ = collector.received.get(timeout=3)
        assert path == "/collector/v1/logs"
    finally:
        close_sink(sink)


@pytest.mark.parametrize("compression", ["none", "gzip", "deflate"])
def test_standard_compression_uses_real_sdk_protobuf_wire(collector, monkeypatch, compression):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", collector.endpoint + "/v1/logs")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_COMPRESSION", compression)
    # A valid signal-specific value overrides an invalid generic setting;
    # explicit enum construction prevents the SDK from parsing either again.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_COMPRESSION", "PRIVATE_UNUSED_GENERIC")
    sink = telemetry.start_fault_log_exporter()
    try:
        assert sink.offer(record())
        request, _, headers, _ = collector.received.get(timeout=3)
        received = request.resource_logs[0].scope_logs[0].log_records[0]
        assert received.event_name == "relax.fault.fault_detected"
        encoding = next((value for key, value in headers.items() if key.lower() == "content-encoding"), None)
        assert encoding == (None if compression == "none" else compression)
        wait_for(lambda: sink.diagnostics()["otel_exported"], 1)
    finally:
        close_sink(sink)


@pytest.mark.parametrize(
    "setting",
    [
        "OTEL_EXPORTER_OTLP_LOGS_COMPRESSION",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
    ],
)
def test_invalid_sdk_environment_does_not_construct_exporter_or_log_raw_value(monkeypatch, caplog, setting):
    from opentelemetry.exporter.otlp.proto.http import _log_exporter

    constructor = Mock(side_effect=AssertionError("invalid settings must not reach SDK construction"))
    monkeypatch.setattr(_log_exporter, "OTLPLogExporter", constructor)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://127.0.0.1:4318/v1/logs")
    monkeypatch.setenv(setting, "PRIVATE_CONFIG_VALUE")
    with pytest.raises(ValueError, match="^INVALID_OTEL_CONFIG: "):
        telemetry.start_fault_log_exporter()
    with pytest.raises(ValueError, match="^INVALID_OTEL_CONFIG: "):
        telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs")
    constructor.assert_not_called()
    assert "PRIVATE_CONFIG_VALUE" not in caplog.text


@pytest.mark.parametrize(
    "name,value",
    [
        ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER", "PRIVATE_MISSING_CREDENTIAL_PROVIDER"),
        ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER", "PRIVATE_MISSING_CREDENTIAL_PROVIDER"),
        ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER", "   "),
    ],
)
def test_sdk_credential_provider_is_rejected_before_synchronous_load(monkeypatch, caplog, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="^INVALID_OTEL_CONFIG: "):
        telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs")
    assert not caplog.text


def test_no_endpoint_disabled_feature_and_sdk_disabled_create_no_sink(monkeypatch):
    constructor = Mock(side_effect=AssertionError("must stay disabled"))
    monkeypatch.setattr(telemetry, "FaultLogExporter", constructor)
    assert telemetry.start_fault_log_exporter() is None
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://127.0.0.1:4318/v1/logs")
    assert telemetry.start_fault_log_exporter(enabled=False) is None
    monkeypatch.setenv("OTEL_SDK_DISABLED", "TrUe")
    assert telemetry.start_fault_log_exporter() is None
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "name,value",
    [
        ("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "file:///private-path"),
        ("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://user:private@localhost/v1/logs"),
        ("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://localhost:bad/v1/logs"),
        ("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "grpc"),
        ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "nan"),
        ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "inf"),
        ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "0"),
        ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "31"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "private-malformed-header"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "X-Test=private%0D%0Aheader"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=Bearer private;token"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", 'Authorization=Bearer private"token'),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=Bearer private\\token"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=\nPRIVATE_TOKEN"),
        ("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "Authorization=\x0bPRIVATE_TOKEN"),
    ],
)
def test_invalid_configuration_never_reaches_sdk_or_logs_raw_values(monkeypatch, caplog, name, value):
    from opentelemetry.exporter.otlp.proto.http import _log_exporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://127.0.0.1:4318/v1/logs")
    monkeypatch.setenv(name, value)
    constructor = Mock(side_effect=AssertionError("invalid configuration must not reach SDK construction"))
    monkeypatch.setattr(_log_exporter, "OTLPLogExporter", constructor)
    with pytest.raises(telemetry.OTelConfigError, match="^INVALID_OTEL_CONFIG: ") as raised:
        telemetry.start_fault_log_exporter()
    constructor.assert_not_called()
    assert "private" not in str(raised.value).lower()
    assert value not in caplog.text


@pytest.mark.parametrize("raises", [False, True])
def test_sdk_failure_is_counted_without_exception_content(raises, caplog):
    delegate = Mock()
    delegate.export.return_value = LogRecordExportResult.FAILURE
    if raises:
        delegate.export.side_effect = RuntimeError("PRIVATE_EXCEPTION_CONTENT")
    sink = telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs", exporter_factory=lambda **_: delegate)
    try:
        assert sink.offer(record())
        wait_for(lambda: sink.diagnostics()["otel_export_failures"], 1)
        assert sink.diagnostics()["otel_exported"] == 0
        assert "PRIVATE_EXCEPTION_CONTENT" not in caplog.text
    finally:
        close_sink(sink)
    delegate.shutdown.assert_called_once()


def test_worker_start_failure_does_not_wait_for_blocked_provider_shutdown(monkeypatch):
    delegate = Mock()
    release = threading.Event()

    def blocked_shutdown():
        assert release.wait(3)

    delegate.shutdown.side_effect = blocked_shutdown
    original_start = threading.Thread.start
    calls = 0

    def fail_first_start(thread):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("worker start failed")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_start)
    before = time.monotonic()
    with pytest.raises(RuntimeError, match="^OTEL_STARTUP_UNAVAILABLE$"):
        telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs", exporter_factory=lambda **_: delegate)
    assert time.monotonic() - before < 0.2
    release.set()
    delegate.shutdown.assert_called_once()


def test_incompatible_sdk_client_fails_closed_with_offdriver_cleanup(monkeypatch, caplog):
    from opentelemetry.exporter.otlp.proto.http import _log_exporter

    release, started, finished = threading.Event(), threading.Event(), threading.Event()

    class IncompatibleExporter:
        # A future SDK without the inspected per-instance client policy must
        # never silently start a pipeline with unfiltered network diagnostics.
        def __init__(self, **_kwargs):
            pass

        def shutdown(self):
            started.set()
            release.wait(3)
            finished.set()

    monkeypatch.setattr(_log_exporter, "OTLPLogExporter", IncompatibleExporter)
    before = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="^OTEL_STARTUP_UNAVAILABLE$"):
            telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs")
        assert time.monotonic() - before < 0.2
        assert started.wait(1) and not finished.is_set()
        assert "AttributeError" not in caplog.text
    finally:
        release.set()
        assert finished.wait(1)


def test_thread_exhaustion_preserves_fixed_startup_error(monkeypatch):
    delegate = Mock()
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("PRIVATE_THREAD_ERROR")))
    with pytest.raises(RuntimeError, match="^OTEL_STARTUP_UNAVAILABLE$"):
        telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs", exporter_factory=lambda **_: delegate)
    delegate.shutdown.assert_not_called()


def test_full_queue_one_inflight_and_close_remain_bounded():
    started, release = threading.Event(), threading.Event()
    delegate = Mock()

    def export(_batch):
        started.set()
        assert release.wait(3)
        return LogRecordExportResult.SUCCESS

    delegate.export.side_effect = export
    sink = telemetry.FaultLogExporter(
        "http://127.0.0.1:4318/v1/logs", queue_capacity=2, exporter_factory=lambda **_: delegate
    )
    try:
        assert sink.offer(record()) and started.wait(2)
        assert sink.offer(record(sequence=2)) and sink.offer(record(sequence=3))
        assert not sink.offer(record(sequence=4))
        assert sink.diagnostics()["otel_queued"] == 2 and sink.diagnostics()["otel_inflight"] == 1
        before = time.monotonic()
        sink.close(timeout=0.01)
        assert time.monotonic() - before < 0.2
        assert sink._worker.is_alive()
        assert sink.diagnostics()["otel_shutdown_dropped"] == 2
        assert not sink.offer(record(sequence=5))
        assert sink.diagnostics()["otel_rejections"] == 2
        assert delegate.export.call_count == 1
    finally:
        release.set()
        close_sink(sink)
    assert sink.diagnostics()["otel_exported"] == 1
    delegate.shutdown.assert_called_once()


def test_real_http_error_does_not_echo_receiver_reason_or_body(collector, caplog):
    collector.settings.status, collector.settings.reason = 400, "PRIVATE_RECEIVER_REASON"
    collector.settings.partial_rejected = 1
    sink = telemetry.FaultLogExporter(collector.endpoint + "/v1/logs", timeout=0.1)
    try:
        assert sink.offer(record())
        collector.received.get(timeout=3)
        wait_for(lambda: sink.diagnostics()["otel_export_failures"], 1)
        assert "PRIVATE_RECEIVER_REASON" not in caplog.text
        assert "private collector rejection" not in caplog.text
    finally:
        close_sink(sink)


def test_unreachable_receiver_does_not_block_admission_or_echo_network_error(caplog):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{reservation.getsockname()[1]}/v1/logs"
    sink = telemetry.FaultLogExporter(endpoint, timeout=0.05)
    try:
        before = time.monotonic()
        assert sink.offer(record())
        assert time.monotonic() - before < 0.2
        wait_for(lambda: sink.diagnostics()["otel_export_failures"], 1)
        assert sink.diagnostics()["otel_exported"] == 0
        assert endpoint not in caplog.text
    finally:
        close_sink(sink)


def test_sdk_transport_success_does_not_claim_per_record_collector_acceptance(collector):
    collector.settings.partial_rejected = 1
    sink = telemetry.FaultLogExporter(collector.endpoint + "/v1/logs")
    try:
        assert sink.offer(record())
        collector.received.get(timeout=3)
        # SDK 1.45 does not parse partial_success. This counter deliberately
        # means only SDK transport success, not durable or per-record receipt.
        wait_for(lambda: sink.diagnostics()["otel_exported"], 1)
        assert sink.diagnostics()["otel_export_failures"] == 0
    finally:
        close_sink(sink)


def test_unknown_record_fields_and_invalid_context_are_rejected_without_export():
    delegate = Mock()
    sink = telemetry.FaultLogExporter("http://127.0.0.1:4318/v1/logs", exporter_factory=lambda **_: delegate)
    try:
        assert not sink.offer({**record(), "exception": "PRIVATE_TRAINING_DATA"})
        assert not sink.offer(record(), SimpleNamespace(trace_id=1, span_id=1, trace_flags=1))
        assert sink.diagnostics()["otel_rejections"] == 2
        assert sink.diagnostics()["otel_context_errors"] == 1
        delegate.export.assert_not_called()
    finally:
        close_sink(sink)

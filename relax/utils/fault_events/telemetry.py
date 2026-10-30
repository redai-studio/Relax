# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded SDK log delivery; transport outcomes are not local-history loss."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from .context import FaultTraceContext
from .models import MAX_UINT, EventDraft


_QUEUE_CAPACITY = 512
_DEFAULT_TIMEOUT = 3.0
_ADMISSION_TIMEOUT = 0.005
_MAX_CLOSE_TIMEOUT = 0.05
_DRAFT_FIELDS = (
    "run_id",
    "fault_id",
    "event_id",
    "role",
    "source_generation",
    "event_type",
    "reason_code",
    "recovery_id",
    "recovery_mode",
    "expected_evidence",
    "observed_evidence",
    "failed_evidence",
)
_RECORD_FIELDS = frozenset(_DRAFT_FIELDS) | {
    "schema_version",
    "sequence",
    "recorded_at_ms",
    "result_scope",
    "availability",
}
_WIDE_FIELDS = frozenset(("source_generation", "sequence", "recorded_at_ms"))
_ERROR_EVENTS = frozenset(("fatal_escalated", "recovery_failed", "termination_requested"))
_WARN_EVENTS = frozenset(("fault_detected", "recovery_interrupted", "handling_skipped"))
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
# Mirrors SDK 1.45's liberal environment-header grammar without invoking its
# warning-producing parser on untrusted raw configuration.
_SDK_HEADER = re.compile(
    r"[ \t]*[!#$%&'*+.^_`|~0-9A-Za-z-]+[ \t]*=[ \t]*"
    r"[\x20\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]*[ \t]*"
)
_CREDENTIAL_PROVIDER_ENV = (
    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER",
)
_COUNTERS = ("admitted", "rejections", "exported", "export_failures", "shutdown_dropped", "context_errors")


class OTelConfigError(ValueError):
    """Rejected exporter configuration.

    Messages are fixed text naming the setting and constraint; they never
    include the configured value, so they are safe to log.
    """


def _timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise OTelConfigError("INVALID_OTEL_CONFIG: timeout must be a number of seconds")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise OTelConfigError("INVALID_OTEL_CONFIG: timeout must be a number of seconds") from None
    if not math.isfinite(result) or not 0 < result <= 30:
        raise OTelConfigError("INVALID_OTEL_CONFIG: timeout must be finite and in (0, 30] seconds")
    return result


def _endpoint(value: object) -> str:
    if type(value) is not str or len(value) > 2048 or any(ord(char) <= 32 for char in value):
        raise OTelConfigError(
            "INVALID_OTEL_CONFIG: endpoint must be at most 2048 characters without whitespace or controls"
        )
    try:
        parts = urlsplit(value)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or (parts.port is not None and not 1 <= parts.port <= 65535)
        ):
            raise ValueError
    except (ValueError, UnicodeError):
        raise OTelConfigError(
            "INVALID_OTEL_CONFIG: endpoint must be an http(s) URL with a host and no credentials, query or fragment"
        ) from None
    return value


def _headers() -> dict[str, str]:
    # Validate first: SDK warnings about malformed headers echo their value.
    raw = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS") or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    if len(raw) > 8192:
        raise OTelConfigError("INVALID_OTEL_CONFIG: headers must be at most 8192 characters")
    result = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, separator, value = item.partition("=")
        # SDK 1.45 parses the raw environment a second time even when an
        # explicit ``headers`` mapping is supplied. Mirror its liberal grammar
        # before construction; encoded forms remain valid and are decoded
        # below, while raw controls and credential-bearing malformed values
        # never reach the SDK warning logger.
        if not _SDK_HEADER.fullmatch(item):
            raise OTelConfigError("INVALID_OTEL_CONFIG: headers must use the SDK name=value grammar")
        name, value = unquote(name).strip(), unquote(value).strip()
        if (
            not separator
            or not _HEADER_NAME.fullmatch(name)
            or len(name) > 128
            or len(value) > 2048
            or any(ord(char) < 32 or ord(char) > 126 for char in value)
            or len(result) >= 32
        ):
            raise OTelConfigError(
                "INVALID_OTEL_CONFIG: headers allow at most 32 printable entries with bounded name and value lengths"
            )
        result[name] = value
    return result


def _compression() -> str:
    value = (
        (
            os.environ.get("OTEL_EXPORTER_OTLP_LOGS_COMPRESSION")
            or os.environ.get("OTEL_EXPORTER_OTLP_COMPRESSION")
            or "none"
        )
        .strip()
        .lower()
    )
    if value not in ("none", "gzip", "deflate"):
        raise OTelConfigError("INVALID_OTEL_CONFIG: compression must be none, gzip or deflate")
    return value


def _validate_sdk_environment() -> None:
    # SDK logger/processor constructors warn with the original value for an
    # invalid internal-metrics boolean, even with our NoOpMeterProvider.
    value = os.environ.get("OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED")
    if value is not None and value.strip().lower() not in ("true", "false"):
        raise OTelConfigError("INVALID_OTEL_CONFIG: OTEL_PYTHON_SDK_INTERNAL_METRICS_ENABLED must be true or false")
    # Credential-provider entry points execute synchronously inside the
    # official HTTP exporter constructor. This fault-only interface keeps
    # startup bounded and never invokes an optional provider selected by the
    # ambient training environment.
    if any(os.environ.get(name) for name in _CREDENTIAL_PROVIDER_ENV):
        raise OTelConfigError("INVALID_OTEL_CONFIG: HTTP credential providers are not supported by the fault exporter")


def _validated_record(record: object) -> dict[str, object]:
    if type(record) is not dict or record.keys() != _RECORD_FIELDS or any(type(key) is not str for key in record):
        raise ValueError("INVALID_EVENT")
    draft = EventDraft.from_dict({key: record[key] for key in _DRAFT_FIELDS})
    expected = draft.record(record["sequence"], record["recorded_at_ms"])
    if type(record["schema_version"]) is not int or record != expected:
        raise ValueError("INVALID_EVENT")
    return expected


def _best_effort_provider_shutdown(provider: Any) -> None:
    try:
        provider.shutdown()
    except Exception:
        pass


def _cleanup_after_startup_failure(provider: Any) -> None:
    try:
        cleanup = threading.Thread(target=_best_effort_provider_shutdown, args=(provider,), daemon=True)
        cleanup.start()
        cleanup.join(0.01)
    except Exception:
        # Thread exhaustion must not replace the fixed startup error or cause
        # a synchronous third-party shutdown on the training driver.
        pass


class _QuietTransportLog:
    """Discard this owned client's SDK diagnostics, without global changes."""

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass

    warning = error


def _sdk_pipeline(
    owner: FaultLogExporter, endpoint: str, timeout: float, exporter_factory: Callable | None
) -> tuple[Any, Any]:
    from opentelemetry.exporter.otlp.proto.http import Compression
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.metrics import NoOpMeterProvider
    from opentelemetry.sdk._logs import LoggerProvider, LogRecordLimits
    from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult, SimpleLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    meters = NoOpMeterProvider()
    if exporter_factory is None:
        compression = {
            "none": Compression.NoCompression,
            "gzip": Compression.Gzip,
            "deflate": Compression.Deflate,
        }[_compression()]
        delegate = OTLPLogExporter(
            endpoint=endpoint,
            timeout=timeout,
            headers=_headers(),
            compression=compression,
            max_request_size=128 * 1024,
            meter_provider=meters,
        )
        # SDK 1.45 has no public transport-diagnostic policy argument. Only this
        # exporter instance is changed: neither a global logger nor a handler.
        # The dependency range is pinned to the inspected SDK minor version.
        # If its internal contract differs, fail closed instead of exporting
        # with potentially unsanitized third-party diagnostics.
        try:
            delegate._client._logger = _QuietTransportLog()
        except Exception:
            _cleanup_after_startup_failure(delegate)
            raise RuntimeError("OTEL_STARTUP_UNAVAILABLE") from None
    else:
        delegate = exporter_factory(endpoint=endpoint, timeout=timeout)

    class CountingExporter(LogRecordExporter):
        def export(self, batch: Sequence[Any]) -> Any:
            try:
                result = delegate.export(batch)
            except Exception:
                result = LogRecordExportResult.FAILURE
            owner._increment("exported" if result is LogRecordExportResult.SUCCESS else "export_failures", len(batch))
            return result

        def shutdown(self) -> None:
            try:
                delegate.shutdown()
            except Exception:
                pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    try:
        provider = LoggerProvider(
            resource=Resource({"service.name": "relax"}),
            shutdown_on_exit=False,
            meter_provider=meters,
            log_record_limits=LogRecordLimits(
                max_attributes=17,
                max_attribute_length=64,
                max_log_record_attributes=17,
                max_log_record_attribute_length=64,
            ),
        )
        provider.add_log_record_processor(SimpleLogRecordProcessor(CountingExporter(), meter_provider=meters))
        return provider, provider.get_logger("relax.fault_events")
    except Exception:
        _cleanup_after_startup_failure(delegate)
        raise RuntimeError("OTEL_STARTUP_UNAVAILABLE") from None


def _sdk_record(record: dict[str, object], context: FaultTraceContext | None) -> Any:
    from opentelemetry import trace
    from opentelemetry._logs import LogRecord, SeverityNumber
    from opentelemetry.context import Context

    # Nonempty, identity-only Context prevents falling back to sender ambient
    # context, even when there was no valid originating span.
    span = trace.INVALID_SPAN
    if context is not None:
        if type(context) is not FaultTraceContext:
            raise ValueError("INVALID_TRACE_CONTEXT")
        span = trace.NonRecordingSpan(
            trace.SpanContext(context.trace_id, context.span_id, False, trace.TraceFlags(context.trace_flags))
        )
    clean_context = trace.set_span_in_context(span, Context())
    event_type = record["event_type"]
    event_name = f"relax.fault.{event_type}"
    severity = (
        SeverityNumber.ERROR
        if event_type in _ERROR_EVENTS
        else (SeverityNumber.WARN if event_type in _WARN_EVENTS else SeverityNumber.INFO)
    )
    attributes = {
        f"relax.fault.{key}": str(value) if key in _WIDE_FIELDS else value
        for key, value in record.items()
        if value is not None
    }
    recorded_at_ms = record["recorded_at_ms"]
    timestamp = recorded_at_ms * 1_000_000 if recorded_at_ms <= MAX_UINT // 1_000_000 else 0
    return LogRecord(
        timestamp=timestamp,
        context=clean_context,
        severity_number=severity,
        severity_text=severity.name,
        body=event_name,
        event_name=event_name,
        attributes=attributes,
    )


class FaultLogExporter:
    """At most 512 queued records, one in flight, and one daemon worker."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        queue_capacity: int = _QUEUE_CAPACITY,
        exporter_factory: Callable | None = None,
    ) -> None:
        endpoint, timeout = _endpoint(endpoint), _timeout(timeout)
        _headers()
        _compression()
        _validate_sdk_environment()
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= _QUEUE_CAPACITY:
            raise OTelConfigError("INVALID_OTEL_CONFIG: queue_capacity must be an int between 1 and 512")
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._queue: deque[Any] = deque()
        self._queue_capacity = queue_capacity
        self._counters = dict.fromkeys(_COUNTERS, 0)
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._available = threading.Event()
        self._inflight = 0
        try:
            self._provider, self._logger = _sdk_pipeline(self, endpoint, timeout, exporter_factory)
            self._worker = threading.Thread(target=self._run, name="relax-fault-otel", daemon=True)
            self._available.set()
            self._worker.start()
        except Exception:
            self._available.clear()
            self._closed.set()
            self._wake.set()
            provider = getattr(self, "_provider", None)
            if provider is not None:
                # Exporter shutdown is third-party code and may block. Startup
                # failure must not inherit that unbounded wait; one daemon cleanup
                # thread is allowed only on this exceptional path.
                _cleanup_after_startup_failure(provider)
            raise RuntimeError("OTEL_STARTUP_UNAVAILABLE") from None

    def _increment(self, name: str, count: int = 1) -> None:
        with self._stats_lock:
            self._counters[name] = min(MAX_UINT, self._counters[name] + count)

    def offer(self, record: dict[str, object], context: FaultTraceContext | None = None) -> bool:
        """Admit a copied, whitelisted record without network or SDK export."""
        try:
            record = _validated_record(record)
            log_record = _sdk_record(record, context)
        except Exception:
            if context is not None and type(context) is not FaultTraceContext:
                self.note_context_error()
            self._increment("rejections")
            return False
        if not self._lock.acquire(timeout=_ADMISSION_TIMEOUT):
            self._increment("rejections")
            return False
        try:
            if self._closed.is_set() or not self._available.is_set() or len(self._queue) >= self._queue_capacity:
                self._increment("rejections")
                return False
            self._queue.append(log_record)
            self._increment("admitted")
            self._wake.set()
            return True
        finally:
            self._lock.release()

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._closed.is_set():
                        self._increment("shutdown_dropped", len(self._queue))
                        self._queue.clear()
                        break
                    record = self._queue.popleft() if self._queue else None
                    self._inflight = int(record is not None)
                if record is None:
                    self._wake.wait()
                    self._wake.clear()
                    continue
                try:
                    self._logger.emit(record)
                except Exception:
                    self._increment("export_failures")
                finally:
                    with self._lock:
                        self._inflight = 0
        finally:
            self._available.clear()
            try:
                self._provider.shutdown()
            except Exception:
                pass

    def note_context_error(self) -> None:
        self._increment("context_errors")

    def diagnostics(self) -> dict[str, int | bool]:
        with self._lock:
            queued, inflight = len(self._queue), self._inflight
        with self._stats_lock:
            counters = {f"otel_{name}": value for name, value in self._counters.items()}
        return {
            "otel_enabled": True,
            "otel_available": self._available.is_set() and not self._closed.is_set(),
            "otel_queue_capacity": self._queue_capacity,
            "otel_queued": queued,
            "otel_inflight": inflight,
            "otel_closed": self._closed.is_set(),
            **counters,
        }

    def close(self, timeout: float = _MAX_CLOSE_TIMEOUT) -> None:
        """Stop admission immediately; never wait for an export to finish."""
        try:
            budget = min(float(timeout), _MAX_CLOSE_TIMEOUT) if type(timeout) in (int, float) else 0
            if not math.isfinite(budget):
                budget = 0
        except (ValueError, OverflowError):
            budget = 0
        deadline = time.monotonic() + max(0, budget)
        self._closed.set()
        self._wake.set()
        if self._lock.acquire(timeout=min(_ADMISSION_TIMEOUT, max(0, deadline - time.monotonic()))):
            try:
                self._increment("shutdown_dropped", len(self._queue))
                self._queue.clear()
            finally:
                self._lock.release()
        if threading.current_thread() is not self._worker:
            self._worker.join(max(0, deadline - time.monotonic()))


def start_fault_log_exporter(*, enabled: bool = True) -> FaultLogExporter | None:
    """Use explicit standard OTLP endpoints; never enable a default
    endpoint."""
    if not enabled or os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return None
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if not endpoint:
        generic = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not generic:
            return None
        parts = urlsplit(_endpoint(generic))
        endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/v1/logs", "", ""))
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL") or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    if protocol is not None and protocol.strip().lower() != "http/protobuf":
        raise OTelConfigError("INVALID_OTEL_CONFIG: protocol must be http/protobuf")
    raw_timeout = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT") or os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT")
    try:
        timeout = _timeout(float(raw_timeout)) if raw_timeout is not None else _DEFAULT_TIMEOUT
    except (TypeError, ValueError, OverflowError):
        raise OTelConfigError("INVALID_OTEL_CONFIG: timeout must be a number of seconds") from None
    return FaultLogExporter(endpoint, timeout=timeout)

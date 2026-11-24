# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded, optional span-context copies kept outside business event fields."""

from dataclasses import dataclass


MAX_TRACE_CONTEXTS = 18
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class FaultTraceContext:
    """Primitive OTel identity without retaining spans, baggage or SDK
    state."""

    trace_id: int
    span_id: int
    trace_flags: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.trace_id) is not int
            or not 0 < self.trace_id < (1 << 128)
            or type(self.span_id) is not int
            or not 0 < self.span_id < (1 << 64)
            or type(self.trace_flags) is not int
            or not 0 <= self.trace_flags < 256
        ):
            raise ValueError("INVALID_TRACE_CONTEXT: trace_id, span_id and trace_flags must be in OTel ranges")

    def to_payload(self) -> dict[str, object]:
        """Create one canonical, fixed-width wire copy."""
        return {
            "trace_id": f"{self.trace_id:032x}",
            "span_id": f"{self.span_id:016x}",
            "trace_flags": self.trace_flags,
        }


def parse_trace_context(payload: object) -> FaultTraceContext | None:
    """Return an omitted context or reject malformed metadata with a fixed
    error."""
    if payload is None:
        return None
    try:
        if (
            type(payload) is not dict
            or len(payload) != 3
            or any(type(key) is not str for key in payload)
            or payload.keys() != {"trace_id", "span_id", "trace_flags"}
        ):
            raise ValueError
        trace_id, span_id, flags = payload["trace_id"], payload["span_id"], payload["trace_flags"]
        if (
            type(trace_id) is not str
            or len(trace_id) != 32
            or not all(char in _HEX_DIGITS for char in trace_id)
            or type(span_id) is not str
            or len(span_id) != 16
            or not all(char in _HEX_DIGITS for char in span_id)
        ):
            raise ValueError
        return FaultTraceContext(int(trace_id, 16), int(span_id, 16), flags)
    except Exception:
        raise ValueError("INVALID_TRACE_CONTEXT: payload needs hex trace_id/span_id and int trace_flags") from None


def parse_trace_contexts(payload: object) -> dict[str, FaultTraceContext]:
    """Validate a sidecar containing at most D/U identities for nine roles."""
    if payload is None:
        return {}
    try:
        if type(payload) is not dict or len(payload) > MAX_TRACE_CONTEXTS:
            raise ValueError
        contexts = {}
        for event_id, value in payload.items():
            if type(event_id) is not str or len(event_id) != 32 or not all(char in _HEX_DIGITS for char in event_id):
                raise ValueError
            context = parse_trace_context(value)
            if context is None:
                raise ValueError
            contexts[event_id] = context
        return contexts
    except Exception:
        raise ValueError("INVALID_TRACE_CONTEXT: sidecar maps at most 18 hex event IDs to contexts") from None


def capture_trace_context() -> FaultTraceContext | None:
    """Read the producer's optional OTel API context without initializing an
    SDK."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context is None or context.is_valid is not True:
            return None
        values = (context.trace_id, context.span_id, context.trace_flags)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            return None
        return FaultTraceContext(*(int(value) for value in values))
    except Exception:
        return None

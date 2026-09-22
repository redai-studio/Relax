# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from relax.utils.types import Sample


TRACE_KEY = "agentic_trace"
EVENTS_KEY = "events"


def agentic_trace_events(metadata: dict[str, Any]) -> dict[str, Any]:
    trace = metadata.get(TRACE_KEY)
    if trace is None:
        trace = {}
        metadata[TRACE_KEY] = trace
    if not isinstance(trace, dict):
        raise TypeError(f"{TRACE_KEY} must be a dict, got {type(trace)}")
    events = trace.get(EVENTS_KEY)
    if events is None:
        events = {}
        trace[EVENTS_KEY] = events
    if not isinstance(events, dict):
        raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} must be a dict, got {type(events)}")
    return events


def mark_agentic_event(events: dict[str, Any], key: str, timestamp: float | None = None) -> None:
    events[key] = time.time() if timestamp is None else timestamp


def mark_agentic_event_once(events: dict[str, Any], key: str, timestamp: float) -> None:
    events.setdefault(key, timestamp)


def mark_sample_agentic_event(sample: Sample, key: str, timestamp: float | None = None) -> None:
    mark_agentic_event(agentic_trace_events(sample.metadata), key, timestamp)


def mark_sample_agentic_event_once(sample: Sample, key: str, timestamp: float) -> None:
    mark_agentic_event_once(agentic_trace_events(sample.metadata), key, timestamp)


def merge_agentic_trace(base_trace: dict[str, Any] | None, overlay_trace: dict[str, Any] | None) -> dict[str, Any]:
    if base_trace is not None and not isinstance(base_trace, dict):
        raise TypeError(f"{TRACE_KEY} must be a dict, got {type(base_trace)}")
    if overlay_trace is not None and not isinstance(overlay_trace, dict):
        raise TypeError(f"{TRACE_KEY} overlay must be a dict, got {type(overlay_trace)}")
    merged = copy.deepcopy(base_trace or {})
    if not overlay_trace:
        return merged
    for key, value in overlay_trace.items():
        if key == EVENTS_KEY:
            if not isinstance(value, dict):
                raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} overlay must be a dict, got {type(value)}")
            base_events = merged.get(EVENTS_KEY)
            if base_events is not None and not isinstance(base_events, dict):
                raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} must be a dict, got {type(base_events)}")
            events = copy.deepcopy(base_events or {})
            events.update(copy.deepcopy(value))
            merged[EVENTS_KEY] = events
            continue
        merged[key] = copy.deepcopy(value)
    return merged

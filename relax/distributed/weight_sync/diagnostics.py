# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bound optional observations without making them a dependency of updates."""

import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_slots = threading.BoundedSemaphore(2)
_last_notice = {}
_output_states = weakref.WeakValueDictionary()
_output_lock = threading.Lock()


class _OutputState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sequence = 0


def _missing(label: str, reason: str) -> "Observation":
    now = time.monotonic()
    if now - _last_notice.get(label, -float("inf")) >= 30:
        if len(_last_notice) >= 128:
            _last_notice.clear()
        _last_notice[label] = now
        logger.warning("%s; optional observation unavailable, update continues.", reason)
    return Observation(error=reason)


@dataclass(frozen=True)
class Observation:
    value: object = None
    error: str | None = None


def observe(operation: Callable, label: str, *, timeout: float = 0.2) -> Observation:
    """Return partial coverage when optional I/O fails or exceeds its wait.

    At most two daemon workers may remain in slow kernel calls. Further
    observations report unavailable capacity instead of spawning more work.
    Only side-effect-free reads or independent diagnostic files belong here.
    """
    deadline = time.monotonic() + timeout
    if not _slots.acquire(timeout=timeout):
        return _missing(label, f"{label}: diagnostic workers busy")
    finished = threading.Event()
    outcome = []

    def run() -> None:
        try:
            outcome.append(Observation(value=operation()))
        except Exception as error:
            outcome.append(Observation(error=f"{label}: {type(error).__name__}: {error}"))
        finally:
            _slots.release()
            finished.set()

    try:
        threading.Thread(target=run, name="weight-sync-observation", daemon=True).start()
    except Exception as error:
        _slots.release()
        return _missing(label, f"{label}: {type(error).__name__}: {error}")
    if not finished.wait(max(0, deadline - time.monotonic())):
        return _missing(label, f"{label}: observation pending after {timeout}s")
    if outcome[0].error:
        return _missing(label, outcome[0].error)
    return outcome[0]


def save_diagnostic(path, payload: bytes) -> Observation:
    """Write independent diagnostic bytes with a bounded caller wait."""
    from .store import atomic_write

    key = str(path)
    with _output_lock:
        state = _output_states.get(key)
        if state is None:
            state = _OutputState()
            _output_states[key] = state
        state.sequence += 1
        sequence = state.sequence

    def write():
        with state.lock:
            if sequence == state.sequence:
                atomic_write(path, payload)

    result = observe(write, f"diagnostic output {path}")
    if result.error:
        logger.warning("%s; check diagnostic storage. Functional state is recorded separately.", result.error)
    return result

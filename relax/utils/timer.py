# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import threading
import traceback
from contextlib import contextmanager
from functools import wraps
from time import time

from relax.utils.logging_utils import get_logger

from .misc import SingletonMeta


__all__ = ["Timer", "timer", "TimelineEvent"]

logger = get_logger(__name__)


def _is_dist_rank0() -> bool:
    """Check if torch.distributed is initialized and current rank is 0.

    Returns False if torch is not available or distributed is not initialized.
    """
    try:
        import torch.distributed

        return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    except ImportError:
        return False


def get_default_pid() -> int:
    """Get default process ID for trace events."""
    return os.getpid()


def get_default_tid() -> int:
    """Get default thread ID for trace events."""
    return threading.get_ident()


def get_call_stack() -> str:
    """Get current call stack as a string."""
    return "".join(traceback.format_stack())


class TimelineEvent:
    """Represents a complete Timeline Trace event record.

    Corresponds to Chrome Trace Event format with ph='X' (complete event).
    """

    def __init__(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        pid: int = None,
        tid: int = None,
        step: int = None,
        call_stack: str = None,
    ):
        self.name = name
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.pid = pid if pid is not None else get_default_pid()
        self.tid = tid if tid is not None else get_default_tid()
        self.step = step
        self.call_stack = call_stack

    def to_trace_event(self) -> dict:
        """Convert to Chrome Trace Event format (ph='X' complete event)."""
        event = {
            "name": self.name,
            "ph": "X",
            "ts": int(self.start_ts * 1e6),  # Convert to microseconds
            "dur": int((self.end_ts - self.start_ts) * 1e6),  # Duration in microseconds
            "pid": self.pid,
            "tid": self.tid,
        }
        args = {}
        if self.step is not None:
            args["step"] = self.step
        if self.call_stack:
            args["call_stack"] = self.call_stack
        if args:
            event["args"] = args
        return event

    def __repr__(self):
        return f"TimelineEvent(name={self.name}, duration={self.end_ts - self.start_ts:.3f}s, step={self.step})"


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        self.timers = {}
        self.start_time = {}
        self.start_info = {}  # Store start info (timestamp, call_stack) for each timer
        # Store complete TimelineEvent records
        self.records: list[TimelineEvent] = []

    def start(self, name, **kwargs):
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        if _is_dist_rank0():
            logger.debug(f"Timer {name} start")

        # Store call stack for trace context
        self.start_info[name] = {
            "timestamp": self.start_time[name],
            "call_stack": get_call_stack(),
        }

    def end(self, name, keep: bool = True, **kwargs):
        """
        - keep: wheather to keep this metric and report to WANDB. if False, only record event without reporting.
        """
        assert name in self.start_time, f"Timer {name} not started."
        elapsed_time = time() - self.start_time[name]
        self.add(name, elapsed_time)

        if _is_dist_rank0():
            logger.debug(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")

        # Create TimelineEvent record
        start_info = self.start_info.get(name, {})
        start_ts = start_info.get("timestamp", self.start_time[name])
        end_ts = time()
        call_stack = start_info.get("call_stack", "")

        record = TimelineEvent(
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            pid=get_default_pid(),
            tid=get_default_tid(),
            step=0,  # to be filled
            call_stack=call_stack,
        )
        self.records.append(record)

        del self.start_time[name]
        if name in self.start_info:
            del self.start_info[name]

        if not keep:
            del self.timers[name]

        return elapsed_time

    def reset(self, name=None):
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]

    def add(self, name, elapsed_time):
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self):
        """Return timer name -> elapsed time dict (original behavior)."""
        return self.timers

    def log_record_and_clear(self, step: int):
        """Return records and clear them atomically.

        This is the recommended method for retrieving timeline events to avoid
        potential duplication if log() is called multiple times.
        """
        records = self.records.copy()
        self.records.clear()
        for r in records:
            r.step = step
        return records

    @contextmanager
    def context(self, name, **kwargs):
        self.start(name, **kwargs)
        try:
            yield
        finally:
            self.end(name, **kwargs)


def timer(name_or_func, **kwargs):
    """Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name, **kwargs)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def inverse_timer(name):
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)


def with_defer(deferred_func):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator

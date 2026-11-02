# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, ordered reporting from concurrent producers."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable


class OrderedMetricsReporter:
    def __init__(self, max_pending: int = 16) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics-report")
        self._pending: deque[Future[bool]] = deque()
        self._max_pending = max_pending
        self._success = True
        self._closed = False

    def _collect(self) -> None:
        try:
            result = self._pending.popleft().result()
        except BaseException:
            self._success = False
            raise
        self._success = bool(result) and self._success

    def submit(self, operation: Callable[..., bool], *args: Any) -> bool:
        """Return acceptance, with backpressure when reporting falls behind."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Metrics reporter is closed")
            while self._pending and self._pending[0].done():
                self._collect()
            if len(self._pending) >= self._max_pending:
                self._collect()
            self._pending.append(self._executor.submit(operation, *args))
            return True

    def drain(self) -> bool:
        with self._lock:
            error: BaseException | None = None
            while self._pending:
                try:
                    self._collect()
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            return self._success

    def close(self) -> bool:
        with self._lock:
            try:
                return self.drain()
            finally:
                self._closed = True
                self._executor.shutdown(wait=True)

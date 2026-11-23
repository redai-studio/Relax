# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One bounded CPU-only encoded read, admitted and consumed by the model
thread.

The caller owns layout validation, transport selection and reader leases. It
must drain before releasing a participating reader. Workers only invoke the CPU
loader; GPU transfers and cache publication stay on the model thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _EncodedWindow:
    owner: object
    keys: tuple[str, ...]
    reserved_bytes: int
    future: Future[list[Any]]


class _EncodedReadAhead:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("Encoded prefetch capacity must be nonnegative")
        self.capacity_bytes = capacity_bytes
        self.pending: _EncodedWindow | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    @property
    def reserved_bytes(self) -> int:
        return self.pending.reserved_bytes if self.pending is not None else 0

    def schedule(
        self,
        owner: object,
        keys: tuple[str, ...],
        reserved_bytes: int,
        loader: Callable[[], list[Any]],
    ) -> bool:
        if self._closed:
            raise RuntimeError("Encoded read-ahead is closed")
        if not isinstance(keys, tuple) or not keys or len(set(keys)) != len(keys) or reserved_bytes <= 0:
            raise ValueError("Encoded prefetch requires unique keys and a positive byte reservation")
        if self.pending is not None or reserved_bytes > self.capacity_bytes:
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-encoded-read")

        def load() -> list[Any]:
            import numpy as np

            values = loader()
            if any(not isinstance(value, np.ndarray) for value in values):
                del values
                raise ValueError("Encoded CPU prefetch requires NumPy bodies")
            if sum(value.nbytes for value in values) > reserved_bytes:
                # Do not retain an oversized payload in the exception's frame.
                del values
                raise ValueError("Encoded CPU prefetch body exceeds its byte reservation")
            return values

        future = self._executor.submit(load)
        self.pending = _EncodedWindow(owner, keys, reserved_bytes, future)
        return True

    def take(self, owner: object, keys: tuple[str, ...]) -> list[Any] | None:
        window = self.pending
        if window is None or window.owner is not owner or window.keys != keys:
            return None
        try:
            return window.future.result()
        finally:
            # A failed read cannot be retried from stale data or hold its lease
            # through a forgotten pending window.
            self.pending = None

    def drain(self) -> None:
        window = self.pending
        if window is not None:
            try:
                window.future.result()
            finally:
                self.pending = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.drain()
        finally:
            self._closed = True
            executor, self._executor = self._executor, None
            if executor is not None:
                executor.shutdown(wait=True)

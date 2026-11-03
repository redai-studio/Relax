# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded read-ahead of immutable CPU conditioning leases.

The worker only reads and validates storage; model preparation, GPU copies and
collectives stay on the training thread. Borrowed leases remain charged until
the consumer finishes copying and closes them. This bounds bytes pinned against
LRU eviction, not a second allocation of the shared mmap payload.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(eq=False)
class _Ticket:
    key: str
    future: Future | None = None
    lease: Any = None
    nbytes: int = 0
    discarded: bool = False
    borrowed: bool = False


class _BorrowedLease:
    def __init__(self, owner: ConditioningPrefetch, ticket: _Ticket) -> None:
        self._owner, self._ticket = owner, ticket

    @property
    def data(self) -> Any:
        return self._ticket.lease.data

    def copy_to(self, device: Any) -> tuple[Any, int]:
        return self._ticket.lease.copy_to(device)

    def close(self) -> None:
        self._owner.release(self._ticket)

    def __enter__(self) -> _BorrowedLease:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ConditioningPrefetch:
    """One CPU reader, an explicit future window and a shared lease budget.

    ``set_window`` discards stale work without waiting for a filesystem read.
    ``take`` can wait for its own read and transfers a lease to the consumer;
    ``close`` drains background work before the owning cache may be closed.
    Failed/missing reads are not retained as a negative cache.
    """

    def __init__(self, cache: Any, *, depth: int, max_bytes: int) -> None:
        if depth <= 0 or max_bytes <= 0:
            raise ValueError("Conditioning prefetch requires positive depth and byte capacity")
        self.cache, self.depth, self.max_bytes = cache, depth, max_bytes
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conditioning-prefetch")
        self._pending: dict[str, _Ticket] = {}
        self._tickets: set[_Ticket] = set()
        self._bytes = 0
        self._inflight_limit = 0
        self._closed = False
        self._drained = False
        self._stats: dict[str, int | float] = dict.fromkeys(
            (
                "submitted",
                "loaded",
                "consumed",
                "misses",
                "bypasses",
                "discarded",
                "errors",
                "stale",
                "loaded_bytes",
                "peak_bytes",
                "lookup_seconds",
                "wait_seconds",
            ),
            0,
        )

    def submit(self, key: str) -> bool:
        with self._lock:
            if self._closed or key in self._pending or len(self._tickets) >= self.depth:
                return False
            ticket = _Ticket(key)
            self._pending[key] = ticket
            self._tickets.add(ticket)
            ticket.future = self._pool.submit(self._load, ticket)
            ticket.future.add_done_callback(lambda _: self._finish(ticket))
            self._stats["submitted"] += 1
            return True

    def set_window(self, keys: list[str]) -> int:
        window = list(dict.fromkeys(keys))[: self.depth]
        with self._lock:
            for key, ticket in list(self._pending.items()):
                if key not in window:
                    self._discard(ticket)
        return sum(self.submit(key) for key in window)

    def _load(self, ticket: _Ticket) -> None:
        with self._lock:
            if self._closed or ticket.discarded:
                return
            # Only one reader runs. All completed and consumer-borrowed leases
            # remain in _bytes, so this limit also bounds the in-flight lookup.
            available = self.max_bytes - self._bytes
            if available <= 0:
                self._stats["bypasses"] += 1
                return
            self._inflight_limit = available
        started = time.perf_counter()
        lease = None
        try:
            lease = self.cache.lookup(ticket.key, max_bytes=available)
            with self._lock:
                if lease is None:
                    self._stats["misses"] += 1
                    return
                nbytes = lease.nbytes
                if not 0 <= nbytes <= available:
                    raise ValueError("Conditioning lookup exceeded its prefetch byte reservation")
                ticket.lease, ticket.nbytes = lease, nbytes
                lease = None
                self._inflight_limit = 0
                self._bytes += nbytes
                self._stats["loaded"] += 1
                self._stats["loaded_bytes"] += nbytes
                self._stats["peak_bytes"] = max(self._stats["peak_bytes"], self._bytes)
        except Exception as error:
            with self._lock:
                self._stats["errors"] += 1
            logger.warning("Conditioning prefetch failed; retaining synchronous fallback: %s", error)
        finally:
            if lease is not None:
                lease.close()
            with self._lock:
                self._inflight_limit = 0
                self._stats["lookup_seconds"] += time.perf_counter() - started

    def _finish(self, ticket: _Ticket) -> None:
        with self._lock:
            if ticket.borrowed:
                return
            if ticket.discarded or self._closed or ticket.lease is None:
                self.release(ticket)

    def _discard(self, ticket: _Ticket) -> None:
        ticket.discarded = True
        if self._pending.get(ticket.key) is ticket:
            del self._pending[ticket.key]
        self._stats["discarded"] += 1
        if ticket.future is not None:
            ticket.future.cancel()
            if ticket.future.done():
                self.release(ticket)

    def take(self, key: str) -> _BorrowedLease | None:
        with self._lock:
            ticket = self._pending.pop(key, None)
        if ticket is None:
            return None
        started = time.perf_counter()
        try:
            try:
                ticket.future.result()
            except CancelledError:
                with self._lock:
                    self._discard(ticket)
                return None
            finally:
                with self._lock:
                    self._stats["wait_seconds"] += time.perf_counter() - started
            with self._lock:
                if ticket.discarded or ticket.lease is None or self._closed:
                    self.release(ticket)
                    return None
                unchanged = ticket.lease.is_unchanged()
                if not unchanged:
                    self._stats["stale"] += 1
                    self.release(ticket)
                    return None
                ticket.borrowed = True
                self._stats["consumed"] += 1
                return _BorrowedLease(self, ticket)
        except BaseException:
            # An interrupted consumer no longer owns this popped ticket. A
            # running read must keep its slot/reservation until its callback,
            # while a completed lease can be released immediately.
            try:
                with self._lock:
                    ticket.borrowed = False
                    self._discard(ticket)
            except BaseException as cleanup_error:
                # close() scans all unborrowed tickets and can retry cleanup;
                # preserve the consumer's original cancellation/error here.
                logger.warning("Interrupted conditioning prefetch cleanup: %s", cleanup_error)
            raise

    def release(self, ticket: _Ticket) -> None:
        with self._lock:
            if ticket.lease is not None:
                ticket.lease.close()
                ticket.lease = None
                self._bytes -= ticket.nbytes
                ticket.nbytes = 0
            if self._pending.get(ticket.key) is ticket:
                del self._pending[ticket.key]
            self._tickets.discard(ticket)

    def clear(self) -> None:
        with self._lock:
            for ticket in list(self._pending.values()):
                self._discard(ticket)

    def stats(self) -> dict[str, int | float | bool]:
        with self._lock:
            return {
                **self._stats,
                "bytes": self._bytes,
                "inflight_byte_limit": self._inflight_limit,
                "reserved_bytes": self._bytes + self._inflight_limit,
                "max_bytes": self.max_bytes,
                "depth": self.depth,
                "pending": len(self._pending),
                "outstanding": len(self._tickets),
                "closed": self._closed,
                "drained": self._drained,
            }

    def close(self) -> None:
        with self._lock:
            if self._drained:
                return
            self._closed = True
            # Include tickets popped by a consumer still waiting for a read.
            # Already delivered borrows remain the consumer's responsibility.
            for ticket in list(self._tickets):
                if not ticket.borrowed:
                    self._discard(ticket)
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._drained = True

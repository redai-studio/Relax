# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Best-effort JSONL sinks and bounded daemon writers.

Producers snapshot and enqueue without waiting for disk. Capacity includes in-
flight snapshots. A stalled worker owns its handles beyond the close deadline
and cannot keep the interpreter alive.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import functools
import inspect
import json
import math
import os
import sys
import tempfile
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Bounded so a stalled disk cannot grow the queue without limit. Records are
# small; the cap exists to fail loudly rather than to save memory.
_DEFAULT_QUEUE_CAPACITY = 4096
_DEFAULT_QUEUE_BYTES = 64 * 1024 * 1024
CLOSE_TIMEOUT = 1.0
_WORKERS: weakref.WeakSet[BoundedWorker] = weakref.WeakSet()
_WORKER_SLOTS = threading.BoundedSemaphore(16)


def best_effort(function: Callable) -> Callable:
    """Isolate observer errors without intercepting business cancellation."""
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await function(*args, **kwargs)
            except Exception as error:
                logger.warning("agentic_obs: %s failed: %s", function.__name__, error)

        return async_wrapper

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            logger.warning("agentic_obs: %s failed: %s", function.__name__, error)

    return wrapper


def snapshot_size(value: Any) -> int:
    """Conservative retained size for JSON-compatible snapshots (no tensor
    reads)."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(snapshot_size(k) + snapshot_size(v) for k, v in value.items())
    elif isinstance(value, (list, tuple, set)):
        size += sum(snapshot_size(item) for item in value)
    return size


class BoundedWorker:
    """A single daemon writer; capacity includes queued and in-flight work.

    Neither submission nor closure waits indefinitely on external I/O. After a
    close deadline, queued work is discarded. Only in-flight work remains owned
    by the daemon; callers must not close its active handles. A process-wide
    worker limit also bounds repeated recorder creation against a stalled disk.
    """

    def __init__(
        self,
        *,
        capacity: int = 4096,
        max_bytes: int = _DEFAULT_QUEUE_BYTES,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._capacity = capacity
        self._max_bytes = max_bytes
        self._on_close = on_close
        self._condition = threading.Condition()
        self._queue: deque = deque()
        self._pending = 0
        self._bytes = 0
        self._closing = False
        self._thread: threading.Thread | None = None
        self.done = threading.Event()
        self.dropped_count = 0
        self.error_count = 0
        _WORKERS.add(self)

    def submit(
        self, function: Callable, *args: Any, size_bytes: int = 0, on_discard: Callable[[], None] | None = None
    ) -> bool:
        try:
            with self._condition:
                if self._closing or self._pending >= self._capacity or self._bytes + size_bytes > self._max_bytes:
                    self.dropped_count += 1
                    return False
                if self._thread is None and not self._start_thread():
                    self.dropped_count += 1
                    return False
                self._queue.append((function, args, size_bytes, on_discard))
                self._pending += 1
                self._bytes += size_bytes
                self._condition.notify()
            return True
        except Exception as error:
            self.dropped_count += 1
            logger.warning("agentic_obs: submission failed: %s", error)
            return False

    def _start_thread(self) -> bool:
        """Start under _condition, including close-only work within the slot
        cap."""
        if not _WORKER_SLOTS.acquire(blocking=False):
            return False
        thread = threading.Thread(target=self._run, name="agentic-obs-writer", daemon=True)
        try:
            thread.start()
        except Exception as error:
            _WORKER_SLOTS.release()
            logger.warning("agentic_obs: writer start failed: %s", error)
            return False
        self._thread = thread
        return True

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closing:
                        self._condition.wait()
                    if not self._queue:
                        break
                    function, args, size_bytes, on_discard = self._queue.popleft()
                try:
                    function(*args)
                except Exception as error:
                    self.error_count += 1
                    if self.error_count == 1:
                        logger.warning("agentic_obs: writer failed: %s", error)
                finally:
                    # Release snapshot references as well as the accounting.
                    del function, args, on_discard
                    with self._condition:
                        self._pending -= 1
                        self._bytes -= size_bytes
            if self._on_close is not None:
                try:
                    self._on_close()
                except Exception as error:
                    self.error_count += 1
                    logger.warning("agentic_obs: writer close failed: %s", error)
        finally:
            _WORKER_SLOTS.release()
            self.done.set()

    def request_close(self) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            if self._thread is None and (self._on_close is None or not self._start_thread()):
                self.done.set()
            self._condition.notify()

    def close(self, timeout: float = CLOSE_TIMEOUT) -> bool:
        self.request_close()
        finished = self.done.wait(max(0.0, timeout))
        if not finished:
            self._discard_pending()
        return finished

    async def aclose(self, timeout: float = CLOSE_TIMEOUT) -> bool:
        self.request_close()
        deadline = time.monotonic() + timeout
        try:
            while not self.done.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
        finally:
            if not self.done.is_set():
                self._discard_pending()
        return self.done.is_set()

    def _discard_pending(self) -> None:
        with self._condition:
            pending, self._queue = self._queue, deque()
            self._pending -= len(pending)
            self._bytes -= sum(item[2] for item in pending)
            self.dropped_count += len(pending)
        for _, _, _, on_discard in pending:
            if on_discard is not None:
                try:
                    on_discard()
                except Exception as error:
                    self.error_count += 1
                    logger.warning("agentic_obs: discard failed: %s", error)


_BACKGROUND: BoundedWorker | None = None
_BACKGROUND_LOCK = threading.Lock()


def submit_background(function: Callable, *args: Any, size_bytes: int = 0) -> bool:
    global _BACKGROUND
    with _BACKGROUND_LOCK:
        if _BACKGROUND is None or _BACKGROUND.done.is_set():
            _BACKGROUND = BoundedWorker(capacity=128)
        return _BACKGROUND.submit(function, *args, size_bytes=size_bytes)


async def close_background(timeout: float = CLOSE_TIMEOUT) -> None:
    with _BACKGROUND_LOCK:
        worker = _BACKGROUND
    if worker is not None:
        await worker.aclose(timeout)


def close_observation_writers(timeout: float = CLOSE_TIMEOUT) -> None:
    """One deadline across drain, flush and status for all process writers."""
    deadline = time.monotonic() + timeout
    workers = tuple(_WORKERS)
    for worker in workers:
        worker.request_close()
    for worker in workers:
        worker.close(max(0.0, deadline - time.monotonic()))
    _WORKERS.difference_update(worker for worker in workers if worker.done.is_set())


atexit.register(close_observation_writers)


class _Record(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class _PendingWrite:
    payload: dict[str, Any]
    media: dict[str, str]


def _encode(record: _Record | dict[str, Any]) -> str:
    """Serialize one record as a single self-contained line.

    ``ensure_ascii=False`` keeps CJK prompts readable; the trailing newline is
    part of the same string. A process crash can still leave a partial final
    line; consumers must ignore that incomplete tail until the next scan.
    """

    payload = record if isinstance(record, dict) else record.to_dict()

    def normalize(value: Any, key: str = "") -> Any:
        if key == "sample_export_id" and value is not None:
            return str(value)
        if key == "export_ids" and isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if isinstance(value, dict):
            return {name: normalize(item, name) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return json.dumps(normalize(payload), ensure_ascii=False, allow_nan=False, default=str) + "\n"


class JsonlAppender:
    """Append records to one JSONL file, keeping the handle open."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._handle: Optional[Any] = None
        self._error_count = 0
        self._write_error_logged = False
        self._written_count = 0
        self._dropped_count = 0
        self._used = False
        try:
            previous = json.loads(self._path.with_suffix(".status.json").read_text(encoding="utf-8"))
            if not isinstance(previous, dict):
                raise ValueError("invalid writer status")
            self._error_count = int(previous.get("write_errors", 0))
            self._written_count = int(previous.get("records_written", 0))
            self._dropped_count = int(previous.get("records_dropped", 0))
        except (OSError, ValueError, TypeError):
            pass

    @property
    def error_count(self) -> int:
        return self._error_count

    def _ensure_open(self) -> Any:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        return self._handle

    def append(self, record: _Record | dict[str, Any]) -> None:
        """Write one record.

        Never raises.
        """

        try:
            self._used = True
            handle = self._ensure_open()
            handle.write(_encode(record))
            handle.flush()
            self._written_count += 1
        except Exception as error:  # noqa: BLE001 — observability must not break training
            self._error_count += 1
            if not self._write_error_logged:
                self._write_error_logged = True
                logger.warning("agentic_obs: failed to append to %s: %s", self._path, error)

    def append_many(self, records: list[_Record | dict[str, Any]]) -> None:
        """Write a batch with one syscall."""

        if not records:
            return
        try:
            self._used = True
            handle = self._ensure_open()
            handle.write("".join(_encode(record) for record in records))
            handle.flush()
            self._written_count += len(records)
        except Exception as error:  # noqa: BLE001
            self._error_count += 1
            if not self._write_error_logged:
                self._write_error_logged = True
                logger.warning("agentic_obs: failed to append to %s: %s", self._path, error)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._handle = None
        if self._used:
            self.write_status()

    def write_status(self, *, dropped: int = 0) -> None:
        self._dropped_count += dropped
        atomic_write_json(
            self._path.with_suffix(".status.json"),
            {
                "records_written": self._written_count,
                "write_errors": self._error_count,
                "records_dropped": self._dropped_count,
            },
        )


class AsyncJsonlAppender:
    """Snapshot on the producer; serialize and write only on a bounded
    daemon."""

    def __init__(
        self,
        path: Path,
        *,
        capacity: int = _DEFAULT_QUEUE_CAPACITY,
        media_dir: Optional[Path] = None,
        max_bytes: int = _DEFAULT_QUEUE_BYTES,
    ) -> None:
        self._path = path
        self._appender: JsonlAppender | None = None
        self._media_dir = media_dir
        self._snapshot_drops = 0
        self._worker = BoundedWorker(capacity=capacity, max_bytes=max_bytes, on_close=self._finish_close)

    @property
    def dropped_count(self) -> int:
        return self._snapshot_drops + self._worker.dropped_count

    def start(self) -> None:
        """Kept for callers; the daemon starts lazily on first submission."""

    def submit(self, record: _Record | dict[str, Any], *, media: Optional[dict[str, str]] = None) -> bool:
        try:
            payload = copy.deepcopy(record if isinstance(record, dict) else record.to_dict())
            media = dict(media or {})
            return self._worker.submit(
                self._write_batch,
                [_PendingWrite(payload, media)],
                size_bytes=snapshot_size(payload) + snapshot_size(media),
            )
        except Exception as error:
            self._snapshot_drops += 1
            logger.warning("agentic_obs: record snapshot failed: %s", error)
            return False

    def _write_batch(self, batch: list[_PendingWrite]) -> None:
        if self._appender is None:
            self._appender = JsonlAppender(self._path)
        if self._media_dir is not None:
            for item in batch:
                write_media(self._media_dir, item.media)
        self._appender.append_many([item.payload for item in batch])

    async def aclose(self, timeout: float = CLOSE_TIMEOUT) -> None:
        await self._worker.aclose(timeout)

    def close(self, timeout: float = CLOSE_TIMEOUT) -> None:
        self._worker.close(timeout)

    def _finish_close(self) -> None:
        if self._appender is None and self.dropped_count:
            self._appender = JsonlAppender(self._path)
        if self._appender is not None:
            self._appender.close()
            self._appender.write_status(dropped=self.dropped_count)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Best-effort producer health; never leave a partially updated
    snapshot."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, allow_nan=False)
        os.replace(temporary, path)
    except Exception as error:  # noqa: BLE001
        logger.warning("agentic_obs: status write failed for %s: %s", path, error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def write_media(media_dir: Path, digest_to_uri: dict[str, str]) -> None:
    """Write deduplicated media payloads referenced by stripped messages.

    Best-effort and content-addressed: an existing file is left alone, so the
    same image shared by many samples is stored once.
    """

    if not digest_to_uri:
        return
    try:
        media_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:  # noqa: BLE001
        logger.warning("agentic_obs: cannot create media dir %s: %s", media_dir, error)
        return

    import base64

    for digest, uri in digest_to_uri.items():
        target = media_dir / f"{digest}.bin"
        tmp: Optional[Path] = None
        try:
            if target.exists():
                continue
            _, _, payload = uri.partition(",")
            data = base64.b64decode(payload) if ";base64" in uri.split(",", 1)[0] else payload.encode("utf-8")
            # Different shards can persist the same image concurrently.
            with tempfile.NamedTemporaryFile(
                dir=media_dir, prefix=f"{digest}.", suffix=".tmp", delete=False
            ) as handle:
                tmp = Path(handle.name)
                handle.write(data)
            os.replace(tmp, target)
        except Exception as error:  # noqa: BLE001
            logger.warning("agentic_obs: failed to write media %s: %s", digest, error)
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = ["AsyncJsonlAppender", "JsonlAppender", "write_media"]

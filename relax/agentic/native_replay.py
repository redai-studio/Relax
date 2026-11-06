# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional CPU-only recording of native generation attempts.

Each process owns one stream under ``<root>/streams/<unique-id>``. Requests
keep every native JSON field, but input_ids use little-endian int32 prefix
deltas. Responses are recorded before token sanitization. A COMPLETE marker
means this stream was closed, all accepted records were written, and every
recorded request has a terminal response/error. It never means training or
verification succeeded. Active, killed, overflowing and failed streams retain
INCOMPLETE. No HTTP requests, model initialization or external sampler live
here.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid
from array import array
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Iterator

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
FORMAT_VERSION = 1
_MIB = 1024 * 1024


class NativeReplayError(ValueError):
    """A stream is incomplete, corrupt or uses an unsupported format."""


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return text.encode("utf-8")


def _retained_bytes(value: Any) -> int:
    """Conservative snapshot-memory accounting; input_ids use a fast special
    case."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_retained_bytes(k) + _retained_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(_retained_bytes(v) for v in value)
    if isinstance(value, str):
        return size + len(value.encode("utf-8"))
    return size


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_bytes(tokens: array) -> bytes:
    if tokens.itemsize != 4:
        raise NativeReplayError("native replay requires 32-bit array integers")
    if sys.byteorder != "little":
        tokens = array("i", tokens)
        tokens.byteswap()
    return tokens.tobytes()


def _common_prefix(left: array, right: array) -> int:
    limit = min(len(left), len(right))
    length = 0
    while length + 1024 <= limit and left[length : length + 1024] == right[length : length + 1024]:
        length += 1024
    while length < limit and left[length] == right[length]:
        length += 1
    return length


class NativeReplayRecorder:
    """A single writer with bounded queued snapshot bytes and a bounded prefix
    cache.

    Overflow and recording errors are fail-open for generation, but sticky for
    stream completeness. Snapshot copying happens before enqueue; prefix scans,
    hashes, serialization and file writes happen only on the writer thread.
    Recording calls after close are explicitly rejected (None/False + warning)
    and counted separately; they are outside the immutable stream's coverage.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_queue_bytes: int = 64 * _MIB,
        max_prefix_bytes: int = 64 * _MIB,
    ) -> None:
        if max_queue_bytes <= 0 or max_prefix_bytes < 0:
            raise ValueError("native replay byte budgets must be positive (prefix cache may be zero)")
        self.stream_dir = Path(root) / "streams" / f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        self._max_queue_bytes = max_queue_bytes
        self._max_prefix_bytes = max_prefix_bytes
        self._condition = threading.Condition()
        self._queue: deque[tuple[dict[str, Any], int]] = deque()
        self._queued_bytes = 0
        self._next_seq = 0
        self._closing = False
        self._closed = False
        self._complete = False
        self._failed = False
        self._status_dirty = False
        self._thread: threading.Thread | None = None
        self._open_requests: set[int] = set()
        self._previous: OrderedDict[str, tuple[int, array]] = OrderedDict()
        self._prefix_bytes = 0
        self._stats = dict(
            submitted_records=0,
            written_records=0,
            dropped_records=0,
            capture_errors=0,
            io_errors=0,
            close_timeouts=0,
            queue_peak_bytes=0,
            requests=0,
            responses=0,
            exceptions=0,
            sessions=0,
            input_tokens=0,
            stored_suffix_tokens=0,
            rejected_after_close=0,
        )
        try:
            self.stream_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            self._atomic_json("manifest.json", {"format_version": FORMAT_VERSION, "created_at_ns": time.time_ns()})
            self._write_incomplete("stream is active")
            self._thread = threading.Thread(target=self._writer_loop, name="native-replay-writer", daemon=True)
            self._thread.start()
            atexit.register(self.close)
        except Exception as error:
            self._failed = True
            self._stats["io_errors"] += 1
            logger.error("Native replay initialization failed (%s); generation continues", type(error).__name__)

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {**self._stats, "queued_bytes": self._queued_bytes, "open_requests": len(self._open_requests)}

    def _atomic_json(self, name: str, value: dict[str, Any]) -> None:
        temporary = self.stream_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(_json_bytes(value))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.stream_dir / name)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_incomplete(self, reason: str) -> None:
        self._atomic_json("INCOMPLETE", {"reason": reason, "stats": self.stats()})

    def record_gap(self, error: Exception) -> None:
        with self._condition:
            if self._closed:
                self._reject_after_close()
                late = True
            else:
                late = False
                self._stats["capture_errors"] += 1
                self._status_dirty = True
                self._condition.notify()
        if late:
            logger.warning("Native replay stream is closed; capture call rejected and not covered by COMPLETE")
            return
        logger.warning("Native replay snapshot failed (%s); generation continues", type(error).__name__)

    def _reject_after_close(self) -> None:
        # Called under the condition. A closed stream is immutable: rejected
        # calls are outside its COMPLETE scope and never pretend to be saved.
        self._stats["rejected_after_close"] += 1

    def _submit(self, record: dict[str, Any], size: int) -> int | None:
        with self._condition:
            if self._closed:
                self._reject_after_close()
            elif self._closing or self._failed or self._queued_bytes + size > self._max_queue_bytes:
                self._stats["dropped_records"] += 1
                self._status_dirty = True
                self._condition.notify()
                return None
            else:
                seq = self._next_seq
                self._next_seq += 1
                record["seq"] = seq
                self._queue.append((record, size))
                self._queued_bytes += size
                self._stats["submitted_records"] += 1
                self._stats["queue_peak_bytes"] = max(self._stats["queue_peak_bytes"], self._queued_bytes)
                self._condition.notify()
                return seq
        logger.warning("Native replay stream is closed; capture call rejected and not covered by COMPLETE")
        return None

    def record_request(
        self,
        payload: dict[str, Any],
        *,
        session_id: str | None,
        routing_key: str | None = None,
    ) -> int | None:
        try:
            # Freeze token IDs in their on-disk int32 representation so queued
            # long-context bursts do not retain Python integer lists.
            input_ids = payload["input_ids"]
            tokens = array("i", input_ids if isinstance(input_ids, list) else iter(input_ids))
            rest = copy.deepcopy({key: value for key, value in payload.items() if key != "input_ids"})
            record = {
                "type": "request",
                "submitted_at_ns": time.time_ns(),
                "endpoint": "/generate",
                "session_id": session_id,
                "routing_key": routing_key,
                "payload": rest,
                "input_ids": tokens,
            }
            size = _retained_bytes(rest) + sys.getsizeof(tokens) + 1024
            return self._submit(record, size)
        except Exception as error:
            self.record_gap(error)
            return None

    def record_response(self, request_seq: int | None, response: dict[str, Any], *, elapsed_s: float) -> bool:
        if request_seq is None:
            return False
        try:
            record = {
                "type": "response",
                "request_seq": request_seq,
                "completed_at_ns": time.time_ns(),
                "elapsed_s": elapsed_s,
                "response": copy.deepcopy(response),
            }
            return self._submit(record, _retained_bytes(record)) is not None
        except Exception as error:
            self.record_gap(error)
            return False

    def record_exception(self, request_seq: int | None, error: BaseException, *, elapsed_s: float) -> bool:
        if request_seq is None:
            return False
        try:
            details = {"type": type(error).__name__, "message": str(error)}
            response = getattr(error, "response", None)
            if response is not None:
                details["status_code"] = getattr(response, "status_code", None)
                details["response_text"] = getattr(response, "text", None)
            record = {
                "type": "exception",
                "request_seq": request_seq,
                "completed_at_ns": time.time_ns(),
                "elapsed_s": elapsed_s,
                "error": details,
            }
            return self._submit(record, _retained_bytes(record)) is not None
        except Exception as capture_error:
            self.record_gap(capture_error)
            return False

    def record_session(self, session_id: str, metadata: dict[str, Any]) -> bool:
        try:
            record = {"type": "session", "session_id": session_id, "metadata": copy.deepcopy(metadata)}
            return self._submit(record, _retained_bytes(record)) is not None
        except Exception as error:
            self.record_gap(error)
            return False

    def _can_complete_locked(self) -> bool:
        return (
            self._closing
            and not self._failed
            and not self._open_requests
            and not self._queued_bytes
            and not any(
                self._stats[key] for key in ("dropped_records", "capture_errors", "io_errors", "close_timeouts")
            )
        )

    def _write_record(self, record: dict[str, Any]) -> None:
        record = dict(record)
        kind = record["type"]
        if kind == "request":
            tokens = record.pop("input_ids")
            data = _token_bytes(tokens)
            session_id = record["session_id"]
            previous = self._previous.pop(session_id, None) if session_id is not None else None
            if previous is not None:
                self._prefix_bytes -= len(previous[1]) * 4
            length = _common_prefix(previous[1], tokens) if previous is not None else 0
            suffix = _token_bytes(tokens[length:])
            offset = self._tokens.tell()
            self._tokens.write(suffix)
            self._tokens.flush()
            record["input_delta"] = {
                "base_seq": previous[0] if previous is not None else None,
                "base_length": length,
                "suffix_offset": offset,
                "suffix_count": len(suffix) // 4,
                "input_count": len(tokens),
                "input_sha256": hashlib.sha256(data).hexdigest(),
            }
            if session_id is not None and len(data) <= self._max_prefix_bytes:
                while self._previous and self._prefix_bytes + len(data) > self._max_prefix_bytes:
                    _, evicted = self._previous.popitem(last=False)
                    self._prefix_bytes -= len(evicted[1]) * 4
                self._previous[session_id] = (record["seq"], tokens)
                self._prefix_bytes += len(data)
            self._open_requests.add(record["seq"])
            self._stats["requests"] += 1
            self._stats["input_tokens"] += len(tokens)
            self._stats["stored_suffix_tokens"] += len(suffix) // 4
        elif kind in ("response", "exception"):
            if record["request_seq"] not in self._open_requests:
                raise NativeReplayError("duplicate or unknown native replay request terminal")
            self._open_requests.remove(record["request_seq"])
            self._stats["responses" if kind == "response" else "exceptions"] += 1
        elif kind == "session":
            previous = self._previous.pop(record["session_id"], None)
            if previous is not None:
                self._prefix_bytes -= len(previous[1]) * 4
            self._stats["sessions"] += 1
        else:
            raise NativeReplayError(f"unsupported native replay record: {kind}")
        self._journal.write(_json_bytes(record) + b"\n")
        self._journal.flush()

    def _writer_loop(self) -> None:
        try:
            with (self.stream_dir / "events.jsonl").open("xb") as self._journal:
                with (self.stream_dir / "input-deltas.bin").open("xb") as self._tokens:
                    while True:
                        with self._condition:
                            self._condition.wait_for(lambda: self._queue or self._closing or self._status_dirty)
                            dirty, self._status_dirty = self._status_dirty, False
                            item = self._queue.popleft() if self._queue else None
                            done = self._closing and item is None
                        if dirty:
                            self._write_incomplete("recording gap or capture error")
                        if done:
                            break
                        if item is None:
                            continue
                        record, size = item
                        try:
                            if not self._failed:
                                self._write_record(record)
                                self._stats["written_records"] += 1
                            else:
                                self._stats["dropped_records"] += 1
                        except Exception as error:
                            self._failed = True
                            self._stats["io_errors" if isinstance(error, OSError) else "capture_errors"] += 1
                            logger.error(
                                "Native replay writer failed (%s); stream stays INCOMPLETE", type(error).__name__
                            )
                        finally:
                            with self._condition:
                                self._queued_bytes -= size
                                self._condition.notify_all()
                    self._journal.flush()
                    self._tokens.flush()
                    os.fsync(self._journal.fileno())
                    os.fsync(self._tokens.fileno())
            with self._condition:
                eligible = self._can_complete_locked()
            if eligible:
                # Hashing, fsync and publication may block on shared storage.
                # Never hold the condition across them: close(timeout) must
                # remain bounded and its timeout must invalidate publication.
                hashes = {
                    "events_sha256": _file_hash(self.stream_dir / "events.jsonl"),
                    "input_deltas_sha256": _file_hash(self.stream_dir / "input-deltas.bin"),
                }
                with self._condition:
                    eligible = self._can_complete_locked()
                    stats = self.stats()
            if eligible:
                self._atomic_json(
                    "COMPLETE",
                    {
                        "scope": "accepted calls before close finalized; not training success or later calls",
                        "stats": stats,
                        **hashes,
                    },
                )
                with self._condition:
                    eligible = self._can_complete_locked()
                if eligible:
                    (self.stream_dir / "INCOMPLETE").unlink(missing_ok=True)
                with self._condition:
                    if self._can_complete_locked():
                        # close() treats a finalized stream as closed even if
                        # this thread has not yet returned from its finally.
                        self._complete = True
                        self._closed = True
                        self._condition.notify_all()
                        return
            (self.stream_dir / "COMPLETE").unlink(missing_ok=True)
            self._write_incomplete("stream closed with missing records, pending requests or errors")
        except Exception as error:
            self._failed = True
            self._stats["io_errors"] += 1
            logger.error("Native replay finalization failed (%s); generation continues", type(error).__name__)
            try:
                (self.stream_dir / "COMPLETE").unlink(missing_ok=True)
                self._write_incomplete("writer/finalization failed")
            except Exception:
                pass
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def flush(self, timeout: float = 10.0) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self._queued_bytes == 0 or self._closed, timeout=timeout)

    def close(self, timeout: float = 10.0) -> bool:
        """Drain within timeout; return True only for a finalized, complete
        stream."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                with self._condition:
                    if self._closed:
                        return self._complete
                    self._stats["close_timeouts"] += 1
                    self._status_dirty = True
                    self._condition.notify_all()
                logger.warning("Native replay writer close timed out; stream stays INCOMPLETE")
                return False
        return self._closed and self._complete


def iter_native_replay_events(stream_dir: str | Path, *, require_complete: bool = True) -> Iterator[dict[str, Any]]:
    """Reconstruct and verify native payloads, without contacting an engine."""
    root = Path(stream_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise NativeReplayError("unsupported native replay format")
    complete = root / "COMPLETE"
    if require_complete:
        if not complete.is_file() or (root / "INCOMPLETE").exists():
            raise NativeReplayError("native replay stream is not finalized")
        final = json.loads(complete.read_text())
        for name, key in (("events.jsonl", "events_sha256"), ("input-deltas.bin", "input_deltas_sha256")):
            if _file_hash(root / name) != final[key]:
                raise NativeReplayError(f"native replay checksum mismatch: {name}")
    previous: dict[str, tuple[int, array]] = {}
    open_requests: set[int] = set()
    last_seq = -1
    with (root / "input-deltas.bin").open("rb") as tokens_file:
        with (root / "events.jsonl").open("rb") as journal:
            for line in journal:
                if not line.endswith(b"\n"):
                    raise NativeReplayError("truncated native replay journal")
                record = json.loads(line)
                if not isinstance(record.get("seq"), int) or record["seq"] <= last_seq:
                    raise NativeReplayError("native replay sequence is not increasing")
                last_seq = record["seq"]
                if record["type"] == "request":
                    delta = record["input_delta"]
                    base = previous.get(record["session_id"])
                    if delta["base_seq"] is None:
                        if delta["base_length"] != 0:
                            raise NativeReplayError("native replay root has a nonempty base")
                        prefix = []
                    else:
                        if (
                            base is None
                            or base[0] != delta["base_seq"]
                            or not 0 <= delta["base_length"] <= len(base[1])
                        ):
                            raise NativeReplayError("missing or invalid native replay prefix base")
                        prefix = base[1][: delta["base_length"]].tolist()
                    if delta["suffix_offset"] < 0 or delta["suffix_count"] < 0:
                        raise NativeReplayError("invalid native replay suffix range")
                    tokens_file.seek(delta["suffix_offset"])
                    raw = tokens_file.read(delta["suffix_count"] * 4)
                    if len(raw) != delta["suffix_count"] * 4:
                        raise NativeReplayError("truncated native replay input delta")
                    suffix = array("i")
                    suffix.frombytes(raw)
                    if sys.byteorder != "little":
                        suffix.byteswap()
                    tokens = prefix + suffix.tolist()
                    input_hash = hashlib.sha256(_token_bytes(array("i", tokens))).hexdigest()
                    if len(tokens) != delta["input_count"] or input_hash != delta["input_sha256"]:
                        raise NativeReplayError("reconstructed native replay input checksum mismatch")
                    record["payload"] = {**record["payload"], "input_ids": tokens}
                    if record["session_id"] is not None:
                        previous[record["session_id"]] = (record["seq"], array("i", tokens))
                    open_requests.add(record["seq"])
                elif record["type"] in ("response", "exception"):
                    if record["request_seq"] not in open_requests:
                        raise NativeReplayError("duplicate or unknown native replay request terminal")
                    open_requests.remove(record["request_seq"])
                elif record["type"] == "session":
                    previous.pop(record["session_id"], None)
                yield record
    if require_complete and (open_requests or last_seq + 1 != final["stats"]["written_records"]):
        raise NativeReplayError("finalized native replay stream has missing records or terminals")

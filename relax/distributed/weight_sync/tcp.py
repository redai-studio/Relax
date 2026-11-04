# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded one-object-per-connection TCP pull for trusted networks/tunnels."""

import socket
import socketserver
import struct
import threading
from collections.abc import Callable
from time import monotonic, sleep

from .format import MAX_BLOCK_BYTES, MAX_MANIFEST_BYTES, WeightSyncError, canonical_json, integer, parse_json
from .store import SharedFSStore


MAX_CONTROL_BYTES = 64 * 1024


def _read_exact(
    connection: socket.socket, size: int, deadline: float, on_bytes: Callable[[int], None] | None = None
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("TCP object deadline exceeded")
        connection.settimeout(min(remaining, 5.0))
        try:
            part = connection.recv(min(size - len(chunks), 64 * 1024))
        except socket.timeout:
            # FULL materialization may legitimately hold the response while it
            # verifies/encodes the target. A socket poll is not its total budget.
            continue
        if not part:
            raise ConnectionError("TCP object truncated")
        chunks.extend(part)
        if on_bytes is not None:
            on_bytes(len(part))
    return bytes(chunks)


def _send(
    connection: socket.socket, payload: bytes, deadline: float, on_bytes: Callable[[int], None] | None = None
) -> None:
    offset = 0
    while offset < len(payload):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("TCP response deadline exceeded")
        connection.settimeout(min(remaining, 5.0))
        sent = connection.send(payload[offset : offset + 64 * 1024])
        if not sent:
            raise ConnectionError("TCP send made no progress")
        offset += sent
        if on_bytes is not None:
            on_bytes(sent)


def _frame(
    connection: socket.socket, value: dict, deadline: float, on_bytes: Callable[[int], None] | None = None
) -> None:
    data = canonical_json(value)
    if len(data) > MAX_CONTROL_BYTES:
        raise WeightSyncError("Control frame exceeds limit")
    _send(connection, struct.pack("!I", len(data)) + data, deadline, on_bytes)


def _receive_frame(connection: socket.socket, deadline: float, on_bytes: Callable[[int], None] | None = None) -> dict:
    size = struct.unpack("!I", _read_exact(connection, 4, deadline, on_bytes))[0]
    if size > MAX_CONTROL_BYTES:
        raise WeightSyncError("Control frame exceeds limit")
    result = parse_json(_read_exact(connection, size, deadline, on_bytes), MAX_CONTROL_BYTES)
    if not isinstance(result, dict):
        raise WeightSyncError("Expected control object")
    return result


class ObjectServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        store: SharedFSStore,
        *,
        workers: int = 4,
        queue_size: int = 8,
        request_timeout_s: float = 30,
    ) -> None:
        self.store = store
        self.request_queue_size = integer(queue_size, 1, 128)
        self.request_timeout_s = request_timeout_s
        if request_timeout_s <= 0:
            raise WeightSyncError("TCP request timeout must be positive")
        self._slots = threading.BoundedSemaphore(integer(workers, 1, 128))
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        super().__init__(address, _ObjectHandler)

    def process_request(self, request: socket.socket, client_address: tuple) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                _frame(request, {"status": "BUSY", "length": 0}, monotonic() + 0.1)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return

        def run() -> None:
            try:
                self.finish_request(request, client_address)
            finally:
                self.shutdown_request(request)
                self._slots.release()
                with self._threads_lock:
                    self._threads.discard(threading.current_thread())

        thread = threading.Thread(target=run, daemon=True, name="weight-object-server")
        with self._threads_lock:
            self._threads.add(thread)
        thread.start()

    def server_close(self) -> None:
        super().server_close()
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=self.request_timeout_s + 1)


class _ObjectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        deadline = monotonic() + self.server.request_timeout_s
        sent_header = False
        try:
            command = _receive_frame(self.request, deadline)
            operation = command.get("op")
            store = self.server.store
            if operation == "object":
                data = store.object(command["id"])
                limit = MAX_BLOCK_BYTES
            elif operation == "manifest":
                data = canonical_json(store.manifest(command["id"]))
                limit = MAX_MANIFEST_BYTES
            elif operation == "record":
                data = canonical_json(store.record(command["stream"], command["version"]))
                limit = MAX_CONTROL_BYTES
            elif operation in {"pin", "release"}:
                reference = command.get("reference")
                if not isinstance(reference, str) or not 0 < len(reference) <= 256:
                    raise WeightSyncError("Invalid reference")
                if operation == "pin":
                    store.pin(reference, command["stream"], command["version"], deadline=deadline)
                else:
                    store.release(reference, deadline=deadline)
                data, limit = b"null", MAX_CONTROL_BYTES
            elif operation == "full":
                data = canonical_json(
                    store.ensure_full(
                        command["stream"], command["version"], command.get("compression", True), deadline=deadline
                    )
                )
                limit = MAX_CONTROL_BYTES
            else:
                raise WeightSyncError("Unsupported operation")
            if len(data) > limit:
                raise WeightSyncError("Response exceeds size limit")
            _frame(self.request, {"status": "OK", "length": len(data)}, deadline)
            sent_header = True
            _send(self.request, data, deadline)
        except (OSError, TimeoutError, ConnectionError):
            return
        except (WeightSyncError, KeyError, TypeError) as error:
            if not sent_header:
                try:
                    _frame(self.request, {"status": "ERROR", "length": 0, "message": str(error)[:1024]}, deadline)
                except (OSError, TimeoutError):
                    pass


class TCPTransport:
    def __init__(self, host: str, port: int, *, timeout_s: float = 60, attempts: int = 3) -> None:
        if timeout_s <= 0:
            raise WeightSyncError("TCP timeout must be positive")
        self.address = host, integer(port, 1, 65535)
        self.timeout_s = timeout_s
        self.attempts = integer(attempts, 1, 100)
        self.deadline: float | None = None
        self.metrics = {"payload_bytes": 0, "frame_bytes": 0, "requests": 0, "retries": 0, "waiting_s": 0.0}

    def _count_frame(self, size: int) -> None:
        self.metrics["frame_bytes"] += size

    def _count_payload(self, size: int) -> None:
        self.metrics["payload_bytes"] += size

    def _request(self, command: dict, limit: int) -> bytes:
        deadline = min(monotonic() + self.timeout_s, self.deadline or float("inf"))
        last_error = None
        for attempt in range(self.attempts):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("TCP fetch budget exhausted") from last_error
            try:
                with socket.create_connection(self.address, timeout=min(remaining, 5)) as connection:
                    self.metrics["requests"] += 1
                    _frame(connection, command, deadline, self._count_frame)
                    response = _receive_frame(connection, deadline, self._count_frame)
                    if response.get("status") == "BUSY":
                        raise ConnectionError("WAITING_SERVER: all object worker slots are occupied")
                    if response.get("status") != "OK":
                        raise WeightSyncError(response.get("message", "Remote object error"))
                    size = integer(response.get("length"), 0, limit)
                    payload = _read_exact(connection, size, deadline, self._count_payload)
                    return payload
            except (OSError, ConnectionError) as error:
                last_error = error
                if attempt + 1 < self.attempts:
                    self.metrics["retries"] += 1
                    delay = min(0.05 * 2**attempt, max(0, deadline - monotonic()))
                    sleep(delay)
                    self.metrics["waiting_s"] += delay
        raise ConnectionError("TCP retries exhausted; completed cached objects remain reusable") from last_error

    def record(self, stream: str, version: int) -> dict:
        return parse_json(self._request({"op": "record", "stream": stream, "version": version}, MAX_CONTROL_BYTES))

    def manifest(self, artifact_id: str) -> dict:
        return parse_json(self._request({"op": "manifest", "id": artifact_id}, MAX_MANIFEST_BYTES))

    def object(self, object_id: str) -> bytes:
        return self._request({"op": "object", "id": object_id}, MAX_BLOCK_BYTES)

    def pin(self, reference: str, stream: str, version: int) -> None:
        self._request({"op": "pin", "reference": reference, "stream": stream, "version": version}, MAX_CONTROL_BYTES)

    def release(self, reference: str) -> None:
        self._request({"op": "release", "reference": reference}, MAX_CONTROL_BYTES)

    def ensure_full(self, stream: str, version: int, compression: bool = True) -> str:
        return parse_json(
            self._request(
                {"op": "full", "stream": stream, "version": version, "compression": compression}, MAX_CONTROL_BYTES
            )
        )

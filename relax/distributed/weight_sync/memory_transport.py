# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, authenticated TCP delivery of one immutable in-memory update.

The owning updater holds generation paused. An ACK means the receiver completed
its GPU writes, not just a socket read. No checkpoint or pickle is transferred.
If a fixed-group apply fails, discard the source and encoder and use a new
stream with FULL for every receiver; do not continue the partially applied one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import socket
import socketserver
import struct
import threading
from time import monotonic
from typing import Any, Callable

from .memory_delta import MemoryBucket, validate_bucket


MAX_HEADER = 1 << 20


def _json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _read(sock: socket.socket, size: int, deadline: float) -> bytes:
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Memory transfer deadline exceeded")
        sock.settimeout(remaining)
        received = sock.recv_into(view[offset:])
        if not received:
            raise ConnectionError("Truncated memory transfer")
        offset += received
    return bytes(chunks)


def _header(sock: socket.socket, deadline: float) -> dict:
    size = struct.unpack("!I", _read(sock, 4, deadline))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Memory control header exceeds its budget")
    value = json.loads(_read(sock, size, deadline))
    if not isinstance(value, dict):
        raise ValueError("Memory control header must be an object")
    return value


def _send(sock: socket.socket, payload: bytes, deadline: float) -> None:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Memory transfer deadline exceeded")
    sock.settimeout(remaining)
    sock.sendall(payload)


def _send_header(sock: socket.socket, value: dict, deadline: float) -> None:
    payload = _json(value)
    if len(payload) > MAX_HEADER:
        raise ValueError("Memory control header exceeds its budget")
    _send(sock, struct.pack("!I", len(payload)) + payload, deadline)


class MemorySource:
    """One pending version and a fixed set of single-GPU inference members."""

    def __init__(self, bind_host: str, *, timeout_s: float = 600, host_budget_bytes: int = 8 << 30) -> None:
        if timeout_s <= 0 or host_budget_bytes <= 0:
            raise ValueError("Memory source budgets must be positive")
        self.timeout_s, self.host_budget_bytes = timeout_s, host_budget_bytes
        self._lock = threading.Condition()
        self._pending: dict | None = None
        self._buckets: tuple[MemoryBucket, ...] = ()
        self._members: set[str] = set()
        self._acks: set[str] = set()
        self._active: set[str] = set()
        self._connections: set[socket.socket] = set()
        self._closed = False
        self.last_errors: list[str] = []
        self.last_control_bytes = 0
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                owner._serve(self.request)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False
            # All fixed members connect together; the default backlog of five
            # can force a SYN retry even for eight single-GPU receivers.
            request_queue_size = 128

        self._server = Server((bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def publish(
        self,
        buckets: list[MemoryBucket],
        *,
        stream: str,
        version: int,
        base: int,
        schema: str,
        members: set[str],
        reset: bool = False,
        group_update: str | None = None,
    ) -> dict:
        if not members or any(not isinstance(m, str) or not m for m in members):
            raise ValueError("A memory update requires fixed inference members")
        if not stream or type(version) is not int or type(base) is not int or version != base + 1:
            raise ValueError("Invalid memory stream or version")
        if not buckets or sum(len(b.payload) for b in buckets) > self.host_budget_bytes:
            raise ValueError("Memory update exceeds its host budget")
        if reset and any(b.encoding != "full" for b in buckets):
            raise ValueError("A memory reset requires every bucket in FULL form")
        if any(type(b.payload) is not bytes for b in buckets):
            raise ValueError("Memory payloads must be immutable bytes")
        # A caller may rebind fields on its pending GPU bucket after publish.
        # Retain independent metadata and only its exact immutable CPU bytes.
        buckets = [
            MemoryBucket(
                tuple((name, tuple(shape)) for name, shape in b.names),
                b.elements,
                b.changed,
                b.encoding,
                b.payload,
                None,
                b.dtype,
            )
            for b in buckets
        ]
        # metadata() checks structure and hashes these local payloads once.
        # Network receivers independently verify the complete immutable frame.
        metadata = [b.metadata() for b in buckets]
        manifest = dict(stream=stream, version=version, base=base, schema=schema, reset=reset, buckets=metadata)
        if group_update is not None:
            if not isinstance(group_update, str) or not group_update:
                raise ValueError("Invalid memory group update identity")
            manifest["group_update"] = group_update
        manifest["update"] = hashlib.sha256(_json(manifest)).hexdigest()
        token = secrets.token_hex(32)
        # Application framing only; exclude TCP/IP, HTTP and Ray overhead.
        control_bytes = sum(
            16
            + len(_json(dict(token=token, update=manifest["update"], member=member)))
            + len(_json(manifest))
            + len(_json(dict(ack=manifest["update"])))
            + len(_json(dict(ok=True)))
            for member in members
        )
        with self._lock:
            if self._closed or self._pending is not None:
                raise RuntimeError("A memory delivery is already outstanding")
            self._pending = manifest | {"token": token}
            self._buckets, self._members = tuple(buckets), set(members)
            self._acks, self._active, self.last_errors = set(), set(), []
            self.last_control_bytes = control_bytes
        host, port = self._server.server_address
        return dict(host=host, port=port, token=token, update=manifest["update"])

    def _serve(self, sock: socket.socket) -> None:
        member = None
        deadline = monotonic() + self.timeout_s
        with self._lock:
            if self._closed:
                return
            self._connections.add(sock)
        try:
            request = _header(sock, deadline)
            with self._lock:
                pending = self._pending
                token = request.get("token")
                if (
                    pending is None
                    or not isinstance(token, str)
                    or not hmac.compare_digest(token, pending["token"])
                    or request.get("update") != pending["update"]
                    or request.get("member") not in self._members
                    or request.get("member") in self._active
                ):
                    raise ValueError("Memory delivery identity or member mismatch")
                member = request["member"]
                self._active.add(member)
                manifest = {k: v for k, v in pending.items() if k != "token"}
                buckets = self._buckets
            _send_header(sock, manifest, deadline)
            for bucket in buckets:
                _send(sock, bucket.payload, deadline)
            ack = _header(sock, deadline)
            if ack != {"ack": manifest["update"]}:
                raise ValueError("Receiver did not ACK the applied update")
            # Remove active before exposing completion: the next publication
            # cannot race the previous handler's cleanup of this member name.
            with self._lock:
                self._active.remove(member)
                self._acks.add(member)
                member = None
                self._lock.notify_all()
            _send_header(sock, {"ok": True}, deadline)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            with self._lock:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
                self.last_errors = self.last_errors[-16:]
        finally:
            with self._lock:
                self._connections.discard(sock)
                if member is not None:
                    self._active.discard(member)
                self._lock.notify_all()

    def wait(self, timeout_s: float) -> set[str]:
        deadline = monotonic() + timeout_s
        with self._lock:
            while self._acks != self._members or self._active:
                if self._closed:
                    raise RuntimeError("Memory source was closed before every ACK")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Missing memory ACKs: {sorted(self._members - self._acks)}")
                self._lock.wait(remaining)
            if self._pending is None:
                raise RuntimeError("No outstanding memory delivery")
            return set(self._acks)

    def finish(self) -> None:
        with self._lock:
            if self._active or not self._pending or self._acks != self._members:
                raise RuntimeError("Cannot release an unacknowledged memory version")
            self._pending, self._buckets = None, ()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connections = tuple(self._connections)
            self._lock.notify_all()
        for sock in connections:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        deadline = monotonic() + 2
        with self._lock:
            while self._connections:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Memory source handlers did not release their buffers")
                self._lock.wait(remaining)
            self._pending, self._buckets = None, ()


def receive_memory_update(
    descriptor: dict,
    member: str,
    apply: Callable[[dict, list[bytes]], None],
    *,
    timeout_s: float = 600,
    host_budget_bytes: int = 8 << 30,
) -> dict:
    """Stage and validate all frames before apply; ACK only after it
    returns."""
    deadline = monotonic() + timeout_s
    with socket.create_connection((descriptor["host"], descriptor["port"]), timeout=timeout_s) as sock:
        _send_header(sock, {"token": descriptor["token"], "update": descriptor["update"], "member": member}, deadline)
        manifest = _header(sock, deadline)
        identity = manifest.get("update")
        unsigned = {k: v for k, v in manifest.items() if k != "update"}
        if identity != descriptor["update"] or identity != hashlib.sha256(_json(unsigned)).hexdigest():
            raise ValueError("Memory manifest identity mismatch")
        metadata = manifest.get("buckets")
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("Memory manifest requires buckets")
        payloads, total = [], 0
        for item in metadata:
            size = item.get("bytes")
            if type(size) is not int or size < 0 or size > host_budget_bytes - total:
                raise ValueError("Memory update exceeds its host budget")
            total += size
            payload = _read(sock, size, deadline)
            validate_bucket(item, payload, max_bytes=host_budget_bytes)
            payloads.append(payload)
        apply(manifest, payloads)
        _send_header(sock, {"ack": identity}, deadline)
        if _header(sock, deadline) != {"ok": True}:
            raise RuntimeError("Source did not confirm the memory ACK")
        return manifest

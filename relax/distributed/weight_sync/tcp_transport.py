# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded authenticated delivery of the same immutable raw frames over TCP.

A socket receipt is only data delivery. The original synchronization owner
still waits for receiver apply/fence and final completion before committing.
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


def _read(sock: socket.socket, size: int, deadline: float) -> bytearray:
    data = bytearray(size)
    cursor = 0
    while cursor < size:
        if deadline <= monotonic():
            raise TimeoutError("Weight-sync TCP receive deadline expired")
        sock.settimeout(deadline - monotonic())
        count = sock.recv_into(memoryview(data)[cursor:])
        if not count:
            raise ConnectionError("Truncated weight-sync TCP frame")
        cursor += count
    return data


def _header(sock: socket.socket, deadline: float) -> dict:
    size = struct.unpack("!I", _read(sock, 4, deadline))[0]
    if not 0 < size <= 16384:
        raise ValueError("Weight-sync TCP header exceeds its bound")
    value = json.loads(_read(sock, size, deadline))
    if not isinstance(value, dict):
        raise ValueError("Weight-sync TCP header must be an object")
    return value


def _send(sock: socket.socket, data: bytes, deadline: float) -> None:
    if deadline <= monotonic():
        raise TimeoutError("Weight-sync TCP deadline expired")
    sock.settimeout(deadline - monotonic())
    sock.sendall(data)


def _send_header(sock: socket.socket, value: dict, deadline: float) -> None:
    data = json.dumps(value, separators=(",", ":")).encode()
    if len(data) > 16384:
        raise ValueError("Weight-sync TCP header exceeds its bound")
    _send(sock, struct.pack("!I", len(data)) + data, deadline)


class TcpFrameSource:
    """One immutable frame and fixed recipients; no model/version policy."""

    def __init__(self, bind_host: str, *, timeout_s: float, host_budget_bytes: int) -> None:
        self.timeout_s = timeout_s
        self.host_budget_bytes = host_budget_bytes
        self._condition = threading.Condition()
        self._pending: dict | None = None
        self._active: set[socket.socket] = set()
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                owner._serve(self.request)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            request_queue_size = 128

        self.server = Server((bind_host, 0), Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def publish(self, payload: bytes, members: set[str]) -> dict:
        if type(payload) is not bytes or len(payload) > self.host_budget_bytes or not members:
            raise ValueError("TCP frame requires immutable bounded bytes and fixed recipients")
        token = secrets.token_hex(32)
        digest = hashlib.sha256(payload).hexdigest()
        with self._condition:
            if self._pending is not None:
                raise RuntimeError("TCP frame delivery is already outstanding")
            self._pending = dict(
                payload=payload, members=set(members), started=set(), done=set(), errors=[], token=token
            )
        host, port = self.server.server_address
        return dict(host=host, port=port, token=token, bytes=len(payload), sha256=digest)

    def _serve(self, sock: socket.socket) -> None:
        deadline = monotonic() + self.timeout_s
        with self._condition:
            self._active.add(sock)
        try:
            request = _header(sock, deadline)
            member, token = request.get("member"), request.get("token")
            with self._condition:
                pending = self._pending
                if (
                    pending is None
                    or not isinstance(token, str)
                    or not hmac.compare_digest(token, pending["token"])
                    or member not in pending["members"]
                    or member in pending["started"]
                ):
                    raise ValueError("TCP frame authentication or recipient identity mismatch")
                pending["started"].add(member)
                payload = pending["payload"]
            _send_header(sock, dict(bytes=len(payload)), deadline)
            _send(sock, payload, deadline)
            if _header(sock, deadline) != {"received": True}:
                raise ValueError("TCP frame lacks a complete data receipt")
            with self._condition:
                pending["done"].add(member)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            with self._condition:
                if self._pending is not None:
                    self._pending["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            with self._condition:
                self._active.discard(sock)
                self._condition.notify_all()

    def finish(self) -> None:
        deadline = monotonic() + self.timeout_s
        with self._condition:
            if self._pending is None:
                raise RuntimeError("TCP frame is not pending")
            while self._pending["done"] != self._pending["members"] or self._active:
                if self._pending["errors"]:
                    raise RuntimeError(f"TCP frame delivery failed: {self._pending['errors']}")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("TCP frame recipients did not complete data delivery")
                self._condition.wait(remaining)
            self._pending = None

    def close(self) -> None:
        with self._condition:
            sockets = tuple(self._active)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=2)


def receive(descriptor: dict, member: str, *, timeout_s: float, max_bytes: int) -> bytearray:
    size = descriptor["bytes"]
    if type(size) is not int or not 0 <= size <= max_bytes:
        raise ValueError("TCP frame exceeds the receiver's admitted staging budget")
    deadline = monotonic() + timeout_s
    with socket.create_connection((descriptor["host"], descriptor["port"]), timeout=timeout_s) as sock:
        _send_header(sock, dict(member=member, token=descriptor["token"]), deadline)
        if _header(sock, deadline) != {"bytes": size}:
            raise ValueError("TCP frame header differs from its admitted size")
        payload = _read(sock, size, deadline)
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ValueError("TCP frame integrity check failed")
        _send_header(sock, dict(received=True), deadline)
        return payload

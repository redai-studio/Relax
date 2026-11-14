# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded pull transports. TCP uses one outstanding object per connection.

Use TCP only inside a trusted network or authenticated tunnel. The server is
bounded by worker slots and deadlines, and never accepts filesystem paths.
"""

import math
import socket
import socketserver
import struct
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger

from .codec import MAX_CHUNK_BYTES
from .deadline import checkpoint, time_budget
from .deadline import remaining as operation_remaining
from .diagnostics import save_diagnostic
from .ledger import SocketLedger
from .parallel_io import ordered_map
from .store import (
    MAX_MANIFEST_BYTES,
    VersionStore,
    atomic_write,
    canonical,
    digest,
    durable_directory_batch,
    load_json,
    sync_directory,
    validate_manifest,
    writer_lock,
)


if TYPE_CHECKING:
    from .metrics import SyncMetrics


BUSY_RESPONSE = (1 << 64) - 1
ERROR_RESPONSE = (1 << 64) - 2
CONTROL_OPERATIONS = frozenset(
    ("pin", "unpin", "versions", "register", "acknowledge", "consumer_status", "unregister")
)
logger = get_logger(__name__)
_fetch_target = ContextVar("weight_fetch_target", default=None)


class RemoteStoreError(ValueError):
    """The source returned a bounded operation error rather than a network
    loss."""

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


class ContentError(ValueError):
    """Delivered content failed verification; request same-target repair."""


def receive_exact(sock: socket.socket, size: int, *, progress=None, deadline: float | None = None) -> bytes:
    value = bytearray()
    if deadline is None:
        deadline = time.monotonic() + (sock.gettimeout() or 10)
    while len(value) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Frame deadline exceeded")
        sock.settimeout(remaining)
        part = sock.recv(min(65536, size - len(value)))
        if not part:
            raise ConnectionError("Peer disconnected before completing frame")
        if progress is not None:
            progress(len(part))
        value.extend(part)
    return bytes(value)


class SharedFSTransport:
    def __init__(self, source: VersionStore, *, repair=None) -> None:
        self.source = source
        self.repair = repair
        self.bytes_received = 0
        self._counter_lock = threading.Lock()

    def _received(self, count: int) -> None:
        with self._counter_lock:
            self.bytes_received += count

    def manifest(self, stream: str, version: int, *, full: bool = False) -> bytes:
        if full and self.repair is not None:
            value = self.repair.manifest(stream, version, full=True)
            self._received(len(value))
            return value
        value = canonical(self._full_manifest(stream, version) if full else self.source.manifest(stream, version))
        self._received(len(value))
        return value

    def _full_manifest(self, stream: str, version: int) -> dict:
        from .metrics import SyncMetrics
        from .traffic import capture, current

        parent = current()
        if parent is None:
            return self.source.ensure_full(stream, version)
        target = self.source.manifest(stream, version)
        metrics = SyncMetrics(target.get("update_id") or f"{stream}:{version}")
        failed = True
        try:
            with capture(metrics), metrics.stage("source_full_repair", path=self.source.root):
                manifest = self.source.ensure_full(stream, version)
            failed = False
            return manifest
        finally:
            parent.related(
                {
                    **metrics.report(),
                    "stream": stream,
                    "version": version,
                    "failed": failed,
                    "storage_role": "source",
                    "parent_operation_id": parent.operation_id,
                }
            )

    def pin(self, stream: str, version: int, owner: str) -> None:
        self.source.pin(stream, version, owner)

    def unpin(self, owner: str) -> None:
        self.source.unpin(owner)

    def versions(self, stream: str, *, after: int = -1, limit: int = 100, newest_first: bool = False) -> list[dict]:
        return self.source.versions(stream, after=after, limit=limit, newest_first=newest_first)

    def register(self, consumer: str, stream: str, start_version: int = 0, *, mode: str = "all_versions") -> dict:
        return self.source.register(consumer, stream, start_version, mode=mode)

    def acknowledge(self, consumer: str, receipt: dict) -> dict:
        return self.source.acknowledge(consumer, receipt)

    def consumer_status(self, consumer: str) -> dict:
        return self.source.consumer_status(consumer)

    def unregister(self, consumer: str) -> None:
        self.source.unregister(consumer)

    def object(self, key: str) -> bytes:
        path = self.source.object_path(key)
        if path.stat().st_size > MAX_CHUNK_BYTES:
            raise ValueError("Object exceeds frame limit")
        value = path.read_bytes()
        self._received(len(value))
        return value


class TCPTransport:
    def __init__(
        self,
        address: tuple[str, int],
        *,
        timeout: float = 10,
        retries: int = 2,
        full_timeout: float = 600,
        control_timeout: float = 120,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0 or type(retries) is not int or retries < 0:
            raise ValueError("Invalid network bounds")
        if not math.isfinite(full_timeout) or full_timeout <= 0:
            raise ValueError("Invalid FULL repair operation budget")
        if not math.isfinite(control_timeout) or control_timeout <= 0:
            raise ValueError("Invalid control operation budget")
        self.full_timeout = full_timeout
        self.control_timeout = control_timeout
        self.address, self.timeout, self.retries = address, timeout, retries
        self.bytes_received = 0
        self.bytes_sent = 0
        self._counter_lock = threading.Lock()
        self.ledger = SocketLedger()
        self._target = None

    def _received(self, count: int) -> None:
        with self._counter_lock:
            self.bytes_received += count
        self.ledger.add("received", "C", count)

    def _request(self, request: dict, maximum: int, *, processing_timeout=None) -> bytes:
        if processing_timeout is None and request["op"] in CONTROL_OPERATIONS:
            processing_timeout = self.control_timeout
        if request["op"] == "manifest":
            self._target = {"stream": request["stream"], "version": request["version"]}
        target = _fetch_target.get() or self._target
        if target is not None and request["op"] == "object":
            request = {**request, "target_context": target}
        encoded = canonical(request)
        last = None
        deadline = time.monotonic() + operation_remaining(processing_timeout or self.timeout, "TCP request")
        failures = 0
        while failures <= self.retries:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Transfer deadline exceeded")
                with socket.create_connection(self.address, min(self.timeout, remaining)) as sock:
                    sock.settimeout(max(0.001, deadline - time.monotonic()))
                    frame = memoryview(struct.pack("!I", len(encoded)) + encoded)
                    while frame:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Request deadline exceeded")
                        sock.settimeout(min(self.timeout, remaining))
                        count = sock.send(frame)
                        if count == 0:
                            raise ConnectionError("Peer disconnected during request")
                        with self._counter_lock:
                            self.bytes_sent += count
                        self.ledger.add("sent", "C", count)
                        frame = frame[count:]
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Transfer deadline exceeded")
                    sock.settimeout(remaining)
                    header = receive_exact(sock, 8, progress=self._received, deadline=deadline)
                    size = struct.unpack("!Q", header)[0]
                    if size == BUSY_RESPONSE:
                        last = ConnectionError("Server busy; retry within transfer budget")
                        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                        continue
                    if size == ERROR_RESPONSE:
                        length = struct.unpack(
                            "!I", receive_exact(sock, 4, progress=self._received, deadline=deadline)
                        )[0]
                        if length > 4096:
                            raise ValueError("Remote error exceeds the frame limit")
                        detail = receive_exact(sock, length, progress=self._received, deadline=deadline)
                        error = load_json(detail)
                        raise RemoteStoreError(f"{error['type']}: {error['message']}", error_type=error["type"])
                    if size > maximum:
                        raise ValueError("Response exceeds frame limit")
                    payload_deadline = min(deadline, time.monotonic() + self.timeout)
                    value = bytearray()
                    while len(value) < size:
                        remaining = payload_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Object deadline exceeded")
                        sock.settimeout(remaining)
                        block = sock.recv(min(65536, size - len(value)))
                        if not block:
                            raise ConnectionError("Truncated object")
                        with self._counter_lock:
                            self.bytes_received += len(block)
                        self.ledger.add(
                            "received", "X" if request["op"] in ("object", "manifest") else "C", len(block)
                        )
                        value.extend(block)
                    return bytes(value)
            except (OSError, ConnectionError) as error:
                last = error
                failures += 1
                if time.monotonic() >= deadline:
                    break
        raise ConnectionError("Transfer retry budget exhausted") from last

    def manifest(self, stream: str, version: int, *, full: bool = False) -> bytes:
        return self._request(
            {"op": "manifest", "stream": stream, "version": version, "full": full},
            MAX_MANIFEST_BYTES,
            processing_timeout=self.full_timeout if full else None,
        )

    def pin(self, stream: str, version: int, owner: str) -> None:
        if self._request({"op": "pin", "stream": stream, "version": version, "owner": owner}, 16) != b"OK":
            raise ValueError("Source retention was not confirmed")

    def unpin(self, owner: str) -> None:
        if self._request({"op": "unpin", "owner": owner}, 16) != b"OK":
            raise ValueError("Source retention release was not confirmed")

    def object(self, key: str) -> bytes:
        return self._request({"op": "object", "key": key}, MAX_CHUNK_BYTES)

    def versions(self, stream: str, *, after: int = -1, limit: int = 100, newest_first: bool = False) -> list[dict]:
        return load_json(
            self._request(
                {"op": "versions", "stream": stream, "after": after, "limit": limit, "newest_first": newest_first},
                MAX_MANIFEST_BYTES,
            )
        )

    def register(self, consumer: str, stream: str, start_version: int = 0, *, mode: str = "all_versions") -> dict:
        return load_json(
            self._request(
                {
                    "op": "register",
                    "consumer": consumer,
                    "stream": stream,
                    "start_version": start_version,
                    "mode": mode,
                },
                MAX_MANIFEST_BYTES,
            )
        )

    def acknowledge(self, consumer: str, receipt: dict) -> dict:
        compact = {key: receipt[key] for key in ("phase", "target", "serving")}
        return load_json(
            self._request({"op": "acknowledge", "consumer": consumer, "receipt": compact}, MAX_MANIFEST_BYTES)
        )

    def consumer_status(self, consumer: str) -> dict:
        return load_json(self._request({"op": "consumer_status", "consumer": consumer}, MAX_MANIFEST_BYTES))

    def unregister(self, consumer: str) -> None:
        self._request({"op": "unregister", "consumer": consumer}, 16)


class ObjectServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        store: VersionStore,
        *,
        timeout: float = 10,
        max_workers: int = 4,
        metrics=None,
        full_timeout: float = 600,
        control_timeout: float = 120,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0 or type(max_workers) is not int or max_workers < 1:
            raise ValueError("Invalid server bounds")
        self.store = store
        self.metrics = metrics
        self.ledger = SocketLedger()
        if not math.isfinite(full_timeout) or full_timeout <= 0:
            raise ValueError("Invalid FULL repair operation budget")
        if not math.isfinite(control_timeout) or control_timeout <= 0:
            raise ValueError("Invalid control operation budget")
        self.full_timeout = full_timeout
        self.control_timeout = control_timeout
        self._repairs = {}
        self._repair_lock = threading.Lock()
        self._identity_cache = OrderedDict()
        self._identity_lock = threading.Lock()
        self._request_local = threading.local()
        self.socket_timeout = timeout
        self._slots = threading.BoundedSemaphore(max_workers)
        super().__init__(address, ObjectHandler)

    def target_update_id(self, stream: str, version: int) -> str:
        """Cache validated immutable identities without trusting client labels.

        Object requests need only this identity for their receipt. Re-reading
        and validating the entire manifest for each chunk makes transfer CPU
        work quadratic in a model's tensor count. File identity still detects
        deletion/replacement before a cached update ID can be used.
        """
        path = self.store.manifest_path(stream, version)
        status = path.stat()
        signature = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        key = (stream, version)
        with self._identity_lock:
            cached = self._identity_cache.get(key)
            if cached is not None and cached[0] == signature:
                self._identity_cache.move_to_end(key)
                return cached[1]
            manifest = self.store.manifest(stream, version)
            after = path.stat()
            if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise OSError("Manifest changed during receipt identity validation")
            self._identity_cache[key] = (signature, manifest["update_id"])
            self._identity_cache.move_to_end(key)
            if len(self._identity_cache) > 64:
                self._identity_cache.popitem(last=False)
            return manifest["update_id"]

    def full_manifest(self, stream, version):
        """Coalesce concurrent same-target repairs within bounded handlers.

        The store holds its content lease across reconstruction and GC. The in-
        flight entry is discarded after completion so later requests still
        validate existing FULL objects and can repair subsequent corruption.
        """
        from .traffic import current

        metrics = current()
        key = (stream, version)
        with self._repair_lock:
            owner = key not in self._repairs
            future = self._repairs.setdefault(key, Future())
        if owner:
            try:
                with (
                    time_budget(self.full_timeout),
                    metrics.stage("source_full_repair", path=self.store.root) if metrics else nullcontext(),
                ):
                    future.set_result(canonical(self.store.ensure_full(stream, version)))
            except Exception as error:
                future.set_exception(error)
            finally:
                with self._repair_lock:
                    del self._repairs[key]
        with metrics.stage("source_full_repair_wait", path=self.store.root) if metrics else nullcontext():
            return future.result(timeout=operation_remaining(self.full_timeout, "FULL repair"))

    def send_counted(self, request: socket.socket, data: bytes, stage: str) -> None:
        pending = memoryview(data)
        while pending:
            count = request.send(pending)
            if not count:
                raise ConnectionError("Peer disconnected during response")
            self.ledger.add("sent", stage, count)
            local = getattr(self._request_local, "ledger", None)
            if local is not None:
                local.add("sent", stage, count)
            pending = pending[count:]

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._slots.acquire(blocking=False):
            from .metrics import SyncMetrics
            from .traffic import capture

            metrics = SyncMetrics("unattributed-request")
            ledger = SocketLedger()
            self._request_local.ledger = ledger
            drained = bytearray()
            decoded = None
            try:
                with capture(metrics), metrics.stage("source_tcp_rejection", path=self.store.root):
                    request.settimeout(min(self.socket_timeout, 0.05))
                    self.send_counted(request, struct.pack("!Q", BUSY_RESPONSE), "C")
                    request.shutdown(socket.SHUT_WR)
                    # Drain the bounded request before closing. Closing with unread
                    # request bytes can turn BUSY into a TCP reset on the client.
                    left = 4100
                    until = time.monotonic() + min(self.socket_timeout, 0.05)
                    while left and time.monotonic() < until:
                        request.settimeout(max(0.001, until - time.monotonic()))
                        data = request.recv(left)
                        if not data:
                            break
                        left -= len(data)
                        drained.extend(data)
                        self.ledger.add("received", "C", len(data))
                        ledger.add("received", "C", len(data))
            except OSError:
                pass
            finally:
                self._request_local.ledger = None
                self.shutdown_request(request)
                if len(drained) >= 4:
                    size = struct.unpack("!I", drained[:4])[0]
                    if size <= 4096 and len(drained) == size + 4:
                        try:
                            value = load_json(bytes(drained[4:]))
                            if isinstance(value, dict):
                                decoded = value
                                context = value.get("target_context", value)
                                if isinstance(context, dict) and "stream" in context and "version" in context:
                                    metrics.update_id = self.target_update_id(context["stream"], context["version"])
                        except (KeyError, ValueError, TypeError, OSError):
                            pass
                self._persist_observation(
                    metrics,
                    ledger,
                    decoded,
                    {"type": "BlockingIOError", "message": "Server busy before handler admission"},
                    admission="rejected_busy",
                )
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def _persist_observation(
        self,
        metrics: "SyncMetrics",
        ledger: SocketLedger,
        request: dict | None,
        error: dict | None,
        *,
        admission: str = "handled",
    ) -> None:
        from .traffic import capture

        report = {
            **metrics.report(),
            "storage_role": "source",
            "request": request,
            "error": error,
            "admission": admission,
            "socket_request_counts": ledger.report(),
            "socket_server_instance_id": self.ledger.report()["instance_id"],
            "socket_count_scope": "per_request_detail_of_server_cumulative_ledger",
        }
        if self.metrics is not None:
            self.metrics.related(report)
        try:
            # Observation failure cannot change an already delivered outcome.
            with capture(None):
                save_diagnostic(
                    self.store.root / "transport-observations" / (metrics.operation_id + ".json"), canonical(report)
                )
        except OSError as caught:
            logger.warning("Could not retain TCP request observation: %s", caught)

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        from .traffic import capture

        try:
            with capture(self.metrics):
                super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class ObjectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        from .metrics import SyncMetrics
        from .traffic import capture

        self.attempt_metrics = SyncMetrics("unattributed-request")
        self.attempt_request = None
        self.attempt_error = None
        ledger = SocketLedger()
        self.server._request_local.ledger = ledger
        try:
            with (
                capture(self.attempt_metrics),
                self.attempt_metrics.stage("source_tcp_request", path=self.server.store.root),
            ):
                self._handle()
        finally:
            self.server._request_local.ledger = None
            self.server._persist_observation(self.attempt_metrics, ledger, self.attempt_request, self.attempt_error)

    def _handle(self) -> None:
        deadline = time.monotonic() + self.server.socket_timeout
        self.request.settimeout(self.server.socket_timeout)

        def received(count):
            self.server.ledger.add("received", "C", count)
            self.server._request_local.ledger.add("received", "C", count)

        try:
            size = struct.unpack("!I", receive_exact(self.request, 4, deadline=deadline, progress=received))[0]
            if size > 4096:
                raise ValueError("Request exceeds limit")
            self.request.settimeout(max(0.001, deadline - time.monotonic()))
            request = load_json(receive_exact(self.request, size, deadline=deadline, progress=received))
            self.attempt_request = request
            context = request.get("target_context", request)
            if request.get("op") == "acknowledge":
                context = request["receipt"]["target"]
            if "stream" in context and "version" in context:
                try:
                    self.attempt_metrics.update_id = self.server.target_update_id(
                        context["stream"], context["version"]
                    )
                except (KeyError, ValueError, OSError):
                    pass
            processing_timeout = None
            if request["op"] in CONTROL_OPERATIONS:
                processing_timeout = self.server.control_timeout
            elif request["op"] == "manifest" and request.get("full", False):
                processing_timeout = self.server.full_timeout
            with time_budget(processing_timeout) if processing_timeout is not None else nullcontext():
                payload = self._response(request)
                checkpoint("source request processing")
            if processing_timeout is not None:
                # Storage admission and fsync have a finite processing budget;
                # the response frame retains the independent network bound.
                deadline = time.monotonic() + self.server.socket_timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Object request deadline exceeded")
            self.request.settimeout(remaining)
            self.server.send_counted(self.request, struct.pack("!Q", len(payload)), "C")
            self.server.send_counted(self.request, payload, "X" if request["op"] in ("object", "manifest") else "C")
        except (TimeoutError, ConnectionError) as error:
            self.attempt_error = {"type": type(error).__name__, "message": str(error)}
            return
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.attempt_error = {"type": type(error).__name__, "message": str(error)}
            try:
                self.request.settimeout(min(self.server.socket_timeout, 0.1))
                if isinstance(error, BlockingIOError):
                    self.server.send_counted(self.request, struct.pack("!Q", BUSY_RESPONSE), "C")
                else:
                    detail = canonical({"type": type(error).__name__, "message": str(error)[:800]})
                    self.server.send_counted(
                        self.request, struct.pack("!QI", ERROR_RESPONSE, len(detail)) + detail, "C"
                    )
            except OSError:
                pass

    def _response(self, request: dict) -> bytes:
        source = SharedFSTransport(self.server.store)
        if request["op"] == "manifest":
            if request.get("full", False):
                payload = self.server.full_manifest(request["stream"], request["version"])
            else:
                payload = source.manifest(request["stream"], request["version"])
        elif request["op"] == "object":
            payload = source.object(request["key"])
        elif request["op"] == "pin":
            source.pin(request["stream"], request["version"], request["owner"])
            payload = b"OK"
        elif request["op"] == "unpin":
            source.unpin(request["owner"])
            payload = b"OK"
        elif request["op"] == "versions":
            payload = canonical(
                source.versions(
                    request["stream"],
                    after=request.get("after", -1),
                    limit=request.get("limit", 100),
                    newest_first=request.get("newest_first", False),
                )
            )
        elif request["op"] == "register":
            payload = canonical(
                source.register(
                    request["consumer"],
                    request["stream"],
                    request["start_version"],
                    mode=request.get("mode", "all_versions"),
                )
            )
        elif request["op"] == "acknowledge":
            payload = canonical(source.acknowledge(request["consumer"], request["receipt"]))
        elif request["op"] == "consumer_status":
            payload = canonical(source.consumer_status(request["consumer"]))
        elif request["op"] == "unregister":
            source.unregister(request["consumer"])
            payload = b"OK"
        else:
            raise ValueError("Unknown request")
        return payload


def fetch(
    transport: SharedFSTransport | TCPTransport,
    target: VersionStore,
    stream: str,
    version: int,
    *,
    reuse_objects: bool = True,
    full: bool = False,
    retention_owner: str | None = None,
) -> dict:
    """Fetch a version and its dependencies; validated durable chunks are
    resumed.

    An anchor can initialize an empty receiver. This method materializes bytes;
    installation and policy for serving skipped versions belong to the caller.
    """
    with writer_lock(target.root / ".fetch", timeout=120):
        reconcile_fetches(transport, target)
        owner = "fetch-" + uuid.uuid4().hex
        journal = target.root / "fetch-operations" / (owner + ".json")
        atomic_write(journal, canonical({"owner": owner, "stream": stream, "version": version, "phase": "ACQUIRING"}))
        return _fetch_operation(
            transport,
            target,
            stream,
            version,
            owner,
            journal,
            reuse_objects=reuse_objects,
            full=full,
            retention_owner=retention_owner,
        )


def reconcile_fetches(transport, target: VersionStore) -> None:
    """Replay releases while holding the cache's exclusive fetch lock."""
    for journal in (target.root / "fetch-operations").glob("*.json"):
        record = load_json(journal.read_bytes())
        transport.unpin(record["owner"])
        journal.unlink()
        sync_directory(journal.parent)


def _fetch_operation(transport, target, stream, version, owner, journal, **options):
    try:
        transport.pin(stream, version, owner)
        return _fetch_pinned(
            transport,
            target,
            stream,
            version,
            **options,
        )
    finally:
        try:
            # Cleanup has a small independent bound even after the operation
            # expires, so completed transfers can release source retention.
            with time_budget(5, inherit=False):
                transport.unpin(owner)
                journal.unlink()
                sync_directory(journal.parent)
        except (OSError, ConnectionError, RemoteStoreError) as error:
            logger.warning(
                "Retention release failed for %s: %s. Retry unpin on the source store after connectivity recovers.",
                owner,
                error,
            )


def _fetch_pinned(
    transport: SharedFSTransport | TCPTransport,
    target: VersionStore,
    stream: str,
    version: int,
    *,
    reuse_objects: bool,
    full: bool,
    retention_owner: str | None,
) -> dict:
    with writer_lock(target.root / ".content-writer", timeout=120):
        target.refresh_budget()
        # Retain the independent FULL delivery once this target has used it.
        # A repeated default request must not recreate a dependency on old bases.
        if target.manifest_path(stream, version).exists():
            full = full or target.manifest(stream, version, prefer_full=True)["kind"] == "full"
        pending = []
        requested = version
        while True:
            checkpoint("dependency manifest fetch")
            raw = (
                transport.manifest(stream, requested, full=True)
                if full and requested == version
                else transport.manifest(stream, requested)
            )
            if len(raw) > MAX_MANIFEST_BYTES:
                raise ValueError("Manifest too large")
            try:
                m = validate_manifest(load_json(raw), target.max_weight_bytes)
            except ValueError as error:
                raise ContentError(str(error)) from error
            if m["stream"] != stream or m["version"] != requested:
                raise ValueError("Remote identity mismatch")
            if pending and pending[-1]["base_root"] != m["root"]:
                raise ContentError("Remote chain mismatch")
            pending.append(m)
            if m["kind"] == "full":
                break
            if reuse_objects and target.has_verified_version(stream, m["base_version"], m["base_root"]):
                break
            if len(pending) >= 4096:
                raise ValueError("Catch-up depth exceeded")
            requested = m["base_version"]
        for m in reversed(pending):
            objects = {}
            for tensor in m["tensors"]:
                for chunk in tensor["chunks"]:
                    key = chunk["object"]
                    if objects.setdefault(key, chunk["encoded_size"]) != chunk["encoded_size"]:
                        raise ContentError("One immutable object has conflicting sizes")

            def fetch_object(item):
                key, size = item
                checkpoint("object fetch")
                path = target.object_path(key)
                if (
                    reuse_objects
                    and path.exists()
                    and path.stat().st_size == size
                    and digest(path.read_bytes()) == key
                ):
                    return
                # Complete the deadline-aware capacity scan before receiving
                # bytes. A complete verified download must remain resumable
                # even when the enclosing operation expires during transfer.
                target._reserve(0)
                value = transport.object(key)
                if len(value) != size or digest(value) != key:
                    raise ContentError("Object content differs from manifest")
                target.put_object(key, value)

            # SharedFS benefits from overlapping metadata/read latency. TCP
            # remains serial: concurrent short connections regress throughput
            # through server-slot contention in paired transfer measurements.
            # Workers drain before retention or writer locks are released.
            workers = 1 if isinstance(transport, TCPTransport) else 4
            token = _fetch_target.set({"stream": stream, "version": m["version"]})
            try:
                with durable_directory_batch(target.root / "objects"):
                    for _ in ordered_map(fetch_object, objects.items(), workers):
                        pass
            finally:
                _fetch_target.reset(token)
            target.commit(m)
        try:
            target._materialize(stream, version)
        except ValueError as error:
            raise ContentError(str(error)) from error
        if retention_owner:
            target.pin(stream, version, retention_owner)
        return pending[0]

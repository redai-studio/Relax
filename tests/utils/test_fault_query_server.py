# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""HTTP boundary tests use real loopback sockets; ASGI tests cover cancellation
order."""

import asyncio
import http.client
import json
import logging
import socket
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import uvicorn
from fastapi.exceptions import RequestValidationError

from relax.utils.fault_events import query_server
from relax.utils.fault_events.query_server import FaultQueryServer


class RuntimeStub:
    """Only the HTTP adapter's four runtime methods are substituted in boundary
    tests."""

    def __init__(self) -> None:
        self.run_id = "a" * 32
        self.query_server = None
        self.calls = 0
        self.last_params = None
        self.result = {"run_id": self.run_id, "events": [], "done": True}
        self.failure = None
        self.unknown = False
        self.store = SimpleNamespace(status=lambda: {"run_id": self.run_id, "capacity": 4096, "size": 0})

    def query(self, **kwargs) -> dict:
        self.calls += 1
        self.last_params = kwargs
        if self.failure is not None:
            raise self.failure
        return self.result

    def diagnostics(self) -> dict:
        return {
            "coverage_unknown": self.unknown,
            "query_available": bool(self.query_server and self.query_server.available),
        }

    def status(self) -> dict:
        return {**self.store.status(), **self.diagnostics()}

    def note_unknown(self, reason="OBSERVATION_ERROR") -> None:
        self.unknown = True


def wait_until(predicate, timeout=5) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true within the test budget"
        time.sleep(0.005)


def address(server) -> tuple:
    url = urlsplit(server.url)
    assert url.hostname == "127.0.0.1"
    return url.hostname, url.port


def raw_request(server, request: bytes) -> tuple:
    with socket.create_connection(address(server), timeout=5) as client:
        client.sendall(request)
        response = http.client.HTTPResponse(client, method=request.split(b" ", 1)[0].decode("ascii"))
        response.begin()
        body = response.read()
        return response.status, dict(response.getheaders()), body


def get(server, suffix="/events") -> tuple:
    return raw_request(server, f"GET /fault-events/v1{suffix} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("ascii"))


@pytest.fixture
def start_server():
    servers = []

    def start(runtime=None, port=0):
        runtime = runtime if runtime is not None else RuntimeStub()
        server = FaultQueryServer(runtime, port)
        runtime.query_server = server
        servers.append(server)
        wait_until(lambda: server.available)
        return runtime, server

    yield start
    for server in servers:
        server.close()
    for server in servers:
        server._thread.join(5)
        assert not server._thread.is_alive()
        assert server._socket.fileno() == -1
        assert server.url is None and not server.available


def test_http_get_status_and_read_only_routes_over_socket(start_server):
    runtime, server = start_server()
    status, headers, body = get(server, "/events?limit=1&role=actor&fault_id=" + "b" * 32)
    assert status == 200
    assert json.loads(body) == runtime.result
    assert runtime.last_params == {"limit": 1, "role": "actor", "fault_id": "b" * 32}
    assert headers["connection"] == "close"
    assert "content-encoding" not in headers
    status, _, body = get(server, "/status")
    assert status == 200 and json.loads(body)["query_available"]
    assert json.loads(body)["capacity"] == 4096
    for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"):
        status, _, body = raw_request(
            server,
            f"{method} /fault-events/v1/events HTTP/1.1\r\nHost: localhost\r\nContent-Length: 6\r\n\r\nSECRET".encode(),
        )
        assert status == 405
        assert body == (b"" if method == "HEAD" else b'{"code":"METHOD_NOT_ALLOWED"}')
    for path in ("/docs", "/redoc", "/openapi.json", "/fault-events/v1/events/", "/SECRET"):
        status, _, body = raw_request(server, f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
        assert status == 404 and body == b'{"code":"NOT_FOUND"}'


@pytest.mark.parametrize(
    "suffix",
    [
        "/events?secret=SECRET",
        "/events?limit=1&limit=2",
        "/events?role=actor&%72ole=rollout",
        "/events?cursor=a&cursor=b",
        "/events?limit=0",
        "/events?limit=101",
        "/events?limit=SECRET",
        "/events?limit=1e1",
        "/events?limit=%EF%BC%91",
        "/events?limit=",
        "/events?role=%FF",
        "/events?role=%SECRET",
        "/events?role",
        "/status?limit=1",
    ],
)
def test_http_invalid_query_has_fixed_body_and_does_not_call_runtime(start_server, suffix):
    runtime, server = start_server()
    status, _, body = get(server, suffix)
    assert status == 422 and body == b'{"code":"INVALID_QUERY"}'
    assert runtime.calls == 0


@pytest.mark.parametrize(
    "headers,body",
    [
        (b"Content-Length: 6\r\n", b"SECRET"),
        (b"Content-Length: 1000000000\r\n", b""),
        (b"Transfer-Encoding: chunked\r\n", b"6\r\nSECRET\r\n0\r\n\r\n"),
        (b"Content-Encoding: gzip\r\n", b""),
        (b"Content-Encoding: identity\r\n", b""),
        (b"Content-Encoding: \r\n", b""),
    ],
)
def test_http_body_and_compression_are_rejected_without_reading_payload(start_server, headers, body):
    runtime, server = start_server()
    status, _, result = raw_request(
        server, b"GET /fault-events/v1/events HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n" + body
    )
    assert status == 422 and result == b'{"code":"INVALID_QUERY"}'
    assert runtime.calls == 0


def test_http_uses_raw_target_and_utf8_cursor_byte_limits(start_server):
    runtime, server = start_server()
    prefix = "/events?role="
    padding = query_server.MAX_TARGET_BYTES - len("/fault-events/v1" + prefix)
    assert get(server, prefix + "a" * padding)[0] == 200
    assert get(server, prefix + "a" * (padding + 1))[0] == 422
    # The decoded target is small; percent escapes still count toward the raw target limit.
    before = runtime.calls
    assert get(server, "/events?role=" + "%61" * 700)[0] == 422
    assert runtime.calls == before
    assert get(server, "/events?cursor=" + "a" * 1024)[0] == 200
    before = runtime.calls
    status, _, body = get(server, "/events?cursor=" + "a" * 1025)
    assert status == 422 and body == b'{"code":"INVALID_QUERY"}'
    assert runtime.calls == before


@pytest.mark.parametrize(
    "failure,status,body",
    [
        (ValueError("SECRET"), 422, b'{"code":"INVALID_QUERY"}'),
        (RuntimeError("RUN_MISMATCH"), 409, b'{"code":"RUN_MISMATCH"}'),
        (RuntimeError("SECRET"), 503, b'{"code":"UNAVAILABLE"}'),
        (TimeoutError("SECRET"), 503, b'{"code":"UNAVAILABLE"}'),
        (LookupError("SECRET"), 503, b'{"code":"UNAVAILABLE"}'),
        (RequestValidationError([{"input": "SECRET", "msg": "SECRET"}]), 422, b'{"code":"INVALID_QUERY"}'),
    ],
)
def test_http_runtime_and_framework_errors_never_echo_input(start_server, failure, status, body, caplog):
    runtime, server = start_server()
    runtime.failure = failure
    actual_status, _, actual_body = get(server)
    assert (actual_status, actual_body) == (status, body)
    assert "SECRET" not in caplog.text


def test_http_response_limit_rejects_whole_json(start_server):
    runtime, server = start_server()
    runtime.result = {"x": "a" * (query_server.MAX_RESPONSE_BYTES - len(b'{"x":""}'))}
    status, _, body = get(server)
    assert status == 200 and len(body) == query_server.MAX_RESPONSE_BYTES
    assert json.loads(body) == runtime.result
    runtime.result["x"] += "a"
    status, _, body = get(server)
    assert status == 503 and body == b'{"code":"UNAVAILABLE"}'


def test_http_server_uses_explicit_parser_and_bounded_public_settings(start_server):
    _, server = start_server()
    config = server._server.config
    assert config.http == "h11"
    assert config.limit_concurrency == config.backlog == 8
    assert config.h11_max_incomplete_event_size == 8192
    assert config.timeout_keep_alive == 0
    assert config.workers == 1 and config.ws == "none"
    assert not config.access_log and not config.proxy_headers
    assert query_server.QUERY_TIMEOUT_SECONDS == 3


def test_http_does_not_reconfigure_existing_uvicorn_loggers_or_global_event_loop(start_server, monkeypatch, caplog):
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")

    def logger_state():
        return [
            (logger.level, logger.disabled, logger.propagate, tuple(logger.handlers), tuple(logger.filters))
            for name in names
            if isinstance(logger := logging.Logger.manager.loggerDict.get(name), logging.Logger)
        ]

    with caplog.at_level(5, logger="uvicorn.error"), caplog.at_level(5, logger="uvicorn.asgi"):
        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            monkeypatch.setattr(logging, "logThreads", False)
            access_logger = logging.Logger.manager.loggerDict["uvicorn.access"]
            monkeypatch.setattr(access_logger, "handlers", [logging.NullHandler()])
            monkeypatch.setattr(access_logger, "propagate", True)
            before = logger_state()
            policy = asyncio.get_event_loop_policy()
            runtime, server = start_server()
            assert get(server, "/events?secret=SECRET")[0] == 422
            runtime.failure = LookupError("SECRET")
            assert get(server)[::2] == (503, b'{"code":"UNAVAILABLE"}')
            # This emits on the caller thread, representing an existing service.
            access_logger.info("OTHER_SERVICE_ACCESS")
            assert "OTHER_SERVICE_ACCESS" in caplog.text
            assert "SECRET" not in caplog.text
            server.close()
            server._thread.join(3)
            assert logger_state() == before
            assert asyncio.get_event_loop_policy() is policy


def test_http_port_conflict_keeps_existing_server_and_store_working(start_server):
    first, server = start_server()
    runtime = RuntimeStub()
    conflicting = FaultQueryServer(runtime, address(server)[1])
    try:
        assert conflicting.url is None and not conflicting.available
        assert conflicting._socket.fileno() == -1
        assert runtime.query()["events"] == []
        assert get(server)[0] == 200 and first.calls == 1
    finally:
        conflicting.close()


def test_http_thread_start_failure_releases_bound_socket(monkeypatch):
    bound = []

    def fail_start(thread):
        bound.append(thread._target.__self__._socket.getsockname())
        raise RuntimeError("SECRET")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    server = FaultQueryServer(RuntimeStub())
    assert server.url is None and not server.available
    assert server._socket.fileno() == -1
    assert server._thread is not None and not server._thread.is_alive()
    with socket.socket() as rebound:
        rebound.bind(bound[0])


def test_http_background_initialization_failure_releases_port_and_address(monkeypatch):
    captured = []
    entered = threading.Event()
    release = threading.Event()

    def fail_run(server, sockets):
        captured.append(sockets[0].getsockname())
        entered.set()
        assert release.wait(3)
        raise RuntimeError("SECRET")

    monkeypatch.setattr(uvicorn.Server, "run", fail_run)
    server = FaultQueryServer(RuntimeStub())
    try:
        assert entered.wait(3)
        assert server.url is None and not server.available
        release.set()
        server._thread.join(3)
        assert not server._thread.is_alive()
        assert server._socket.fileno() == -1
        assert server.url is None and not server.available
        with socket.socket() as rebound:
            rebound.bind(captured[0])
    finally:
        release.set()
        server.close()


def test_http_url_is_only_published_after_listening_and_close_never_waits(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_startup = uvicorn.Server.startup

    async def delayed_startup(server, sockets=None):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        await original_startup(server, sockets=sockets)

    monkeypatch.setattr(uvicorn.Server, "startup", delayed_startup)
    server = FaultQueryServer(RuntimeStub())
    try:
        assert entered.wait(3)
        bound_address = server._socket.getsockname()
        assert server.url is None and not server.available
        with pytest.raises(OSError):
            socket.create_connection(bound_address, timeout=0.2)
        start = time.monotonic()
        server.close()
        assert time.monotonic() - start < 0.1
        assert server.url is None and not server.available
    finally:
        release.set()
        server.close()
        server._thread.join(3)
    assert not server._thread.is_alive()
    assert server._socket.fileno() == -1
    with socket.socket() as rebound:
        rebound.bind(bound_address)


@pytest.mark.parametrize("prefix", [b"", b"GET /fault-events/v1/events HTTP/1.1\r\nHost: local"])
def test_http_slow_incomplete_headers_are_reclaimed_on_real_socket(start_server, monkeypatch, prefix):
    monkeypatch.setattr(query_server, "QUERY_TIMEOUT_SECONDS", 0.2)
    runtime, server = start_server()
    with socket.create_connection(address(server), timeout=2) as client:
        if prefix:
            client.sendall(prefix)
        started = time.monotonic()
        assert client.recv(1) == b""
        assert time.monotonic() - started < 1
    assert runtime.calls == 0
    assert get(server)[0] == 200


def test_http_h11_rejects_large_incomplete_header_without_echoing(start_server):
    runtime, server = start_server()
    with socket.create_connection(address(server), timeout=2) as client:
        client.sendall(b"GET /fault-events/v1/events HTTP/1.1\r\nHost: localhost\r\nX-Secret: " + b"SECRET" * 1400)
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 400
        assert b"SECRET" not in response.read()
    assert runtime.calls == 0


def test_http_library_concurrency_rejects_before_asgi_and_recovers(start_server, monkeypatch):
    monkeypatch.setattr(query_server, "QUERY_TIMEOUT_SECONDS", 1)
    protocols = []
    original_connection_made = query_server._ConnectionDeadline.connection_made

    def capture_connection(protocol, transport):
        original_connection_made(protocol, transport)
        protocols.append(protocol)

    monkeypatch.setattr(query_server._ConnectionDeadline, "connection_made", capture_connection)
    runtime, server = start_server()
    clients = []
    try:
        for _ in range(8):
            client = socket.create_connection(address(server), timeout=2)
            client.sendall(b"GET /fault-events/v1/events HTTP/1.1\r\nHost: local")
            clients.append(client)
        wait_until(lambda: len(protocols) == 8)
        status, _, body = get(server, "/events?secret=SECRET")
        # Uvicorn rejects before calling ASGI, using its own fixed text response.
        assert status == 503 and body == b"Service Unavailable"
        assert runtime.calls == 0 and server._app.active == 0
    finally:
        for client in clients:
            client.close()
    wait_until(lambda: all(protocol._transport is None for protocol in protocols))
    assert get(server)[0] == 200


def test_http_keep_alive_and_pipeline_are_closed_after_one_response(start_server):
    runtime, server = start_server()
    request = b"GET /fault-events/v1/events HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n"
    with socket.create_connection(address(server), timeout=2) as client:
        client.sendall(request + request)
        chunks = []
        while chunk := client.recv(8192):
            chunks.append(chunk)
    assert b"".join(chunks).count(b"HTTP/1.1 200") == 1
    assert runtime.calls == 1


def test_http_slow_response_holds_four_slots_then_recovers_without_request_threads(start_server, monkeypatch):
    monkeypatch.setattr(query_server, "QUERY_TIMEOUT_SECONDS", 0.8)
    original_connection_made = query_server._ConnectionDeadline.connection_made

    def small_send_buffer(protocol, transport):
        transport.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        transport.set_write_buffer_limits(high=1024, low=0)
        original_connection_made(protocol, transport)

    monkeypatch.setattr(query_server._ConnectionDeadline, "connection_made", small_send_buffer)
    runtime, server = start_server()
    runtime.result = {"x": "a" * (query_server.MAX_RESPONSE_BYTES - 100)}
    baseline_threads = {thread.ident for thread in threading.enumerate()}
    clients = []
    try:
        for _ in range(4):
            client = socket.socket()
            client.settimeout(2)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            client.connect(address(server))
            client.sendall(b"GET /fault-events/v1/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            clients.append(client)
        wait_until(lambda: server._app.active == 4, timeout=0.5)
        status, _, body = get(server)
        assert status == 503 and body == b'{"code":"UNAVAILABLE"}'
        assert runtime.calls == 4
        assert {thread.ident for thread in threading.enumerate()} == baseline_threads
        wait_until(lambda: server._app.active == 0, timeout=2)
        runtime.result = {"events": []}
        assert get(server)[0] == 200
        for client in clients:
            data = bytearray()
            while True:
                try:
                    chunk = client.recv(16384)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                data.extend(chunk)
            assert len(data) < query_server.MAX_RESPONSE_BYTES - 100
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("phase", ["application", "send"])
def test_http_asgi_slots_remain_owned_during_timeout_cancellation_cleanup(monkeypatch, phase):
    monkeypatch.setattr(query_server, "QUERY_TIMEOUT_SECONDS", 0.03)

    async def run():
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cleanup_count = 0

        async def blocked():
            nonlocal cleanup_count
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_count += 1
                if cleanup_count == 4:
                    cleanup_started.set()
                await finish_cleanup.wait()

        async def slow_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if phase == "application":
                await blocked()
            else:
                await send({"type": "http.response.body", "body": b"response"})

        app = query_server._BoundedQuery(slow_app, SimpleNamespace(_closed=threading.Event()))
        scope = {"type": "http", "method": "GET", "path": "/fault-events/v1/events", "headers": []}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            if message["type"] == "http.response.body":
                await blocked()

        tasks = [asyncio.create_task(app(dict(scope), receive, send)) for _ in range(4)]
        await asyncio.wait_for(cleanup_started.wait(), 1)
        assert all(not task.done() for task in tasks) and app.active == 4
        rejected = []

        async def collect(message):
            rejected.append(message)

        await app(dict(scope), receive, collect)
        assert rejected[0]["status"] == 503
        assert b"".join(message.get("body", b"") for message in rejected) == b'{"code":"UNAVAILABLE"}'
        assert app.active == 4
        finish_cleanup.set()
        await asyncio.gather(*tasks)
        assert app.active == 0

    asyncio.run(run())


def test_http_background_failure_after_accept_aborts_socket_and_cancels_timer(monkeypatch):
    protocols = []
    fail = threading.Event()
    original_connection_made = query_server._ConnectionDeadline.connection_made

    def capture_connection(protocol, transport):
        original_connection_made(protocol, transport)
        protocols.append(protocol)

    async def failing_main_loop(server):
        while not fail.is_set():
            await asyncio.sleep(0.005)
        raise RuntimeError("SECRET")

    monkeypatch.setattr(query_server._ConnectionDeadline, "connection_made", capture_connection)
    monkeypatch.setattr(uvicorn.Server, "main_loop", failing_main_loop)
    server = FaultQueryServer(RuntimeStub())
    try:
        wait_until(lambda: server.available)
        bound_address = address(server)
        with socket.create_connection(bound_address, timeout=2) as client:
            client.sendall(b"GET /fault-events/v1/events HTTP/1.1\r\nHost: local")
            wait_until(lambda: len(protocols) == 1)
            timer = protocols[0]._timer
            fail.set()
            server._thread.join(2)
            assert not server._thread.is_alive()
            assert client.recv(1) == b""
            assert timer.cancelled()
            assert protocols[0]._timer is None and protocols[0]._transport is None
        assert server.url is None and not server.available
        with socket.socket() as rebound:
            rebound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rebound.bind(bound_address)
    finally:
        fail.set()
        server.close()
        server._thread.join(3)


def test_http_real_runtime_snapshot_eviction_and_cross_run_cursor(start_server):
    from relax.utils.fault_events.models import EventDraft, new_id
    from relax.utils.fault_events.runtime import FaultEventRuntime

    runtimes = [FaultEventRuntime(capacity=3), FaultEventRuntime(capacity=3)]
    try:
        runtime, server = start_server(runtimes[0])
        _, other = start_server(runtimes[1])
        fault = new_id()

        def append() -> None:
            assert runtime.record(
                EventDraft(runtime.run_id, fault, new_id(), "actor", 0, "fault_detected", "reported_error")
            )

        for _ in range(3):
            append()
        wait_until(lambda: len(runtime.query()["events"]) == 3)
        status, _, body = get(server, "/events?role=actor&limit=1")
        page = json.loads(body)
        assert status == 200 and [event["sequence"] for event in page["events"]] == [1]
        assert page["snapshot_end_seq"] == 3 and not page["done"]
        cursor = page["next_cursor"]
        append()
        append()
        wait_until(lambda: runtime.query()["snapshot_end_seq"] == 5)
        status, _, body = get(server, "/events?role=actor&cursor=" + cursor)
        page = json.loads(body)
        assert status == 200 and [event["sequence"] for event in page["events"]] == [3]
        assert page["snapshot_end_seq"] == 3 and page["done"]
        assert page["missing_range"] == [2, 2] and page["retention_truncated"]
        assert get(other, "/events?role=actor&cursor=" + cursor)[::2] == (409, b'{"code":"RUN_MISMATCH"}')
        assert get(server, "/events?role=rollout&cursor=" + cursor)[::2] == (422, b'{"code":"INVALID_QUERY"}')
        for suffix in ("/events?role=SECRET", "/events?fault_id=SECRET", "/events?cursor=SECRET"):
            assert get(server, suffix)[::2] == (422, b'{"code":"INVALID_QUERY"}')
        status, _, body = get(server, "/status")
        details = json.loads(body)
        assert status == 200 and details["run_id"] == runtime.run_id and details["query_available"]
        assert details["queue_capacity"] == 512 and not details["coverage_unknown"]
        with runtime.store._lock:
            assert get(server)[::2] == (503, b'{"code":"UNAVAILABLE"}')
        assert get(server)[0] == 200
    finally:
        for runtime in runtimes:
            runtime.close()
            runtime._consumer.join(3)
            assert not runtime._consumer.is_alive()


def test_http_closed_server_port_can_be_reused_after_a_query(start_server):
    _, server = start_server()
    bound_address = address(server)
    assert get(server)[0] == 200
    start = time.monotonic()
    server.close()
    assert time.monotonic() - start < 0.1
    assert server.url is None and not server.available
    server._thread.join(3)
    _, replacement = start_server(port=bound_address[1])
    assert address(replacement) == bound_address
    assert get(replacement)[0] == 200


def test_http_port_failure_preserves_real_recording_coverage(start_server):
    from relax.utils.fault_events.models import EventDraft, new_id
    from relax.utils.fault_events.runtime import FaultEventRuntime

    _, occupied = start_server()
    runtime = FaultEventRuntime()
    failed = FaultQueryServer(runtime, address(occupied)[1])
    runtime.query_server = failed
    try:
        assert not failed.available and failed.url is None
        draft = EventDraft(runtime.run_id, new_id(), new_id(), "actor", 0, "fault_detected", "reported_error")
        assert runtime.record(draft)
        wait_until(lambda: len(runtime.query()["events"]) == 1)
        assert runtime.query()["events"][0]["event_id"] == draft.event_id
        status = runtime.status()
        assert not status["query_available"] and not status["coverage_unknown"]
    finally:
        runtime.close()
        runtime._consumer.join(3)

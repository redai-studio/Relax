# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""A bounded, loopback-only HTTP adapter for the driver's fault history."""

import asyncio
import logging
import re
import socket
import threading
from typing import Any, Callable
from urllib.parse import parse_qsl

from .models import encode


QUERY_TIMEOUT_SECONDS = 3.0
MAX_TARGET_BYTES = 2048
MAX_CURSOR_BYTES = 1024
MAX_RESPONSE_BYTES = 128 * 1024
MAX_ACTIVE_QUERIES = 4
_SEND_CHUNK_BYTES = 16 * 1024
_EVENTS_PATH = "/fault-events/v1/events"
_STATUS_PATH = "/fault-events/v1/status"
_ERROR_CODES = {
    404: b'{"code":"NOT_FOUND"}',
    405: b'{"code":"METHOD_NOT_ALLOWED"}',
    409: b'{"code":"RUN_MISMATCH"}',
    422: b'{"code":"INVALID_QUERY"}',
    503: b'{"code":"UNAVAILABLE"}',
}


class _QueryLogFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return threading.get_ident() != self.thread_id


class _ConnectionDeadline(asyncio.Protocol):
    """Limit incomplete headers and stalled writes using public asyncio
    APIs."""

    def __init__(self, protocol: asyncio.Protocol, on_lost: Callable) -> None:
        self._protocol = protocol
        self._on_lost = on_lost
        self._timer: asyncio.TimerHandle | None = None
        self._transport: asyncio.Transport | None = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        self._transport = transport
        self._timer = asyncio.get_running_loop().call_later(QUERY_TIMEOUT_SECONDS, transport.abort)
        self._protocol.connection_made(transport)

    def connection_lost(self, exc: Exception | None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._transport = None
        self._on_lost(self)
        self._protocol.connection_lost(exc)

    def abort(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._transport is not None:
            self._transport.abort()

    def data_received(self, data: bytes) -> None:
        self._protocol.data_received(data)

    def eof_received(self) -> bool | None:
        return self._protocol.eof_received()

    def pause_writing(self) -> None:
        self._protocol.pause_writing()

    def resume_writing(self) -> None:
        self._protocol.resume_writing()


class _QueryEventLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._listeners: list[asyncio.Server] = []
        self._connections: set[_ConnectionDeadline] = set()

    async def create_server(self, protocol_factory: Callable, *args: Any, **kwargs: Any) -> asyncio.Server:
        def factory() -> _ConnectionDeadline:
            protocol = _ConnectionDeadline(protocol_factory(), self._connections.discard)
            self._connections.add(protocol)
            return protocol

        listener = await super().create_server(factory, *args, **kwargs)
        self._listeners.append(listener)
        return listener

    def close(self) -> None:
        if self.is_closed():
            return
        for listener in self._listeners:
            listener.close()
        self._listeners.clear()
        for protocol in tuple(self._connections):
            protocol.abort()
        if self._connections:
            # abort schedules connection_lost, which closes the fd. Process that
            # callback before closing the loop even when Uvicorn startup/run failed.
            self.run_until_complete(asyncio.sleep(0))
        super().close()


def _query_loop_factory() -> asyncio.AbstractEventLoop:
    # Uvicorn's public custom-loop option creates this loop in its sole server thread.
    return _QueryEventLoop()


async def _send_error(status: int, send: Callable) -> None:
    body = _ERROR_CODES[status]
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _BoundedQuery:
    """Keep a business slot until ASGI sending and cancellation cleanup
    finish."""

    def __init__(self, app: Any, owner: "FaultQueryServer") -> None:
        self.app = app
        self.owner = owner
        self.active = 0

    @staticmethod
    def _validate(scope: dict) -> dict:
        query = scope.get("query_string", b"")
        for name, value in scope["headers"]:
            if name in (b"transfer-encoding", b"content-encoding") or (name == b"content-length" and value != b"0"):
                raise ValueError("INVALID_QUERY")
        if re.search(rb"%(?![0-9a-fA-F]{2})", query):
            raise ValueError("INVALID_QUERY")
        # Python 3.10's strict parser rejects an empty string, but omitting
        # optional filters is a valid request for both endpoints.
        pairs = (
            parse_qsl(
                query.decode("ascii"), keep_blank_values=True, strict_parsing=True, errors="strict", max_num_fields=4
            )
            if query
            else []
        )
        params = dict(pairs)
        allowed = {"role", "fault_id", "limit", "cursor"} if scope["path"] == _EVENTS_PATH else set()
        if len(params) != len(pairs) or params.keys() - allowed:
            raise ValueError("INVALID_QUERY")
        if "cursor" in params and len(params["cursor"].encode("utf-8")) > MAX_CURSOR_BYTES:
            raise ValueError("INVALID_QUERY")
        if "limit" in params:
            value = params["limit"]
            if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 100:
                raise ValueError("INVALID_QUERY")
            params["limit"] = int(value)
        return params

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status = None
        raw_path = scope.get("raw_path", scope["path"].encode("utf-8"))
        query = scope.get("query_string", b"")
        if len(raw_path) + len(query) + bool(query) > MAX_TARGET_BYTES:
            status = 422
        elif scope["method"] != "GET":
            status = 405
        elif scope["path"] not in (_EVENTS_PATH, _STATUS_PATH):
            status = 404
        else:
            try:
                scope["fault_query_params"] = self._validate(scope)
            except (ValueError, UnicodeError):
                status = 422
        if status is None and (self.owner._closed.is_set() or self.active >= MAX_ACTIVE_QUERIES):
            status = 503
        acquired = status is None
        if acquired:
            self.active += 1
        started = False

        async def bounded_send(message: dict) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                await send(message)
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                # A final empty send observes backpressure from the last data chunk.
                # A single large final write could otherwise release its slot early.
                for offset in range(0, len(body), _SEND_CHUNK_BYTES):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": body[offset : offset + _SEND_CHUNK_BYTES],
                            "more_body": True,
                        }
                    )
                await send({"type": "http.response.body", "body": b"", "more_body": message.get("more_body", False)})
            else:
                await send(message)

        async def dispatch() -> None:
            if status is not None:
                await _send_error(status, bounded_send)
                return
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message.get("body") or message.get("more_body"):
                await _send_error(422, bounded_send)
                return
            try:
                await self.app(scope, receive, bounded_send)
            except Exception:
                # Starlette re-raises unexpected errors after its fixed error
                # response. Do not pass the original exception to Uvicorn's log.
                if not started:
                    await _send_error(503, bounded_send)

        try:
            # wait_for waits for the cancelled task's finally blocks before returning.
            await asyncio.wait_for(dispatch(), QUERY_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # The earlier absolute connection deadline also aborts buffered writes.
            # Do not begin another send after the request has exhausted its budget.
            pass
        finally:
            if acquired:
                self.active -= 1


def _make_app(runtime: Any, owner: "FaultQueryServer") -> _BoundedQuery:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import Response
    from starlette.exceptions import HTTPException

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, debug=False, redirect_slashes=False)

    def error(status: int) -> Response:
        return Response(_ERROR_CODES[status], status, media_type="application/json", headers={"Connection": "close"})

    def response(value: dict) -> Response:
        try:
            body = encode(value)
            if len(body) > MAX_RESPONSE_BYTES:
                return error(503)
        except Exception:
            return error(503)
        return Response(body, media_type="application/json", headers={"Connection": "close"})

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValueError)
    async def invalid_query(request: Request, exc: Exception) -> Response:
        return error(422)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, exc: RuntimeError) -> Response:
        return error(409 if exc.args == ("RUN_MISMATCH",) else 503)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        return error(exc.status_code if exc.status_code in _ERROR_CODES else 503)

    @app.exception_handler(Exception)
    async def unavailable(request: Request, exc: Exception) -> Response:
        return error(503)

    @app.get(_EVENTS_PATH)
    async def events(request: Request) -> Response:
        return response(runtime.query(**request.scope["fault_query_params"]))

    @app.get(_STATUS_PATH)
    async def status() -> Response:
        return response(runtime.status())

    return _BoundedQuery(app, owner)


class FaultQueryServer:
    def __init__(self, runtime: Any, port: int = 0) -> None:
        self.runtime = runtime
        self._closed = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._bound_url: str | None = None
        self._announced = False
        self._app: _BoundedQuery | None = None
        try:
            import uvicorn

            if type(port) is not int or not 0 <= port <= 65535:
                raise ValueError("INVALID_PORT")
            self._app = _make_app(runtime, self)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("127.0.0.1", port))
            self._bound_url = f"http://127.0.0.1:{self._socket.getsockname()[1]}/fault-events/v1"
            config = uvicorn.Config(
                self._app,
                host="127.0.0.1",
                loop="relax.utils.fault_events.query_server:_query_loop_factory",
                http="h11",
                ws="none",
                workers=1,
                timeout_keep_alive=0,
                timeout_graceful_shutdown=QUERY_TIMEOUT_SECONDS,
                limit_concurrency=8,
                backlog=8,
                h11_max_incomplete_event_size=8192,
                proxy_headers=False,
                server_header=False,
                date_header=False,
                log_config=None,
                log_level=None,
                # Uvicorn's constructor clears the shared access logger when
                # False. Disable this Config after construction instead.
                access_log=True,
                callback_notify=self._notify_ready,
            )
            config.access_log = False
            self._server = uvicorn.Server(config)
            self._thread = threading.Thread(target=self._serve, daemon=True, name="fault-query")
            self._thread.start()
        except (Exception, SystemExit):
            self.close()
            self._failure()

    @property
    def available(self) -> bool:
        return bool(
            not self._closed.is_set()
            and self._bound_url is not None
            and self._server is not None
            and self._server.started
            and not self._server.should_exit
            and self._socket is not None
            and self._socket.fileno() != -1
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def url(self) -> str | None:
        return self._bound_url if self.available else None

    async def _notify_ready(self) -> None:
        if self.available and not self._announced:
            self._announced = True
            try:
                from relax.utils.logging_utils import get_logger

                get_logger(__name__).info(
                    "Fault event query available at %s (run_id=%s)", self.url, self.runtime.run_id
                )
            except Exception:
                pass

    def _failure(self) -> None:
        try:
            from relax.utils.logging_utils import get_logger

            get_logger(__name__).warning("Fault event query unavailable")
        except Exception:
            pass

    def _serve(self) -> None:
        filtered_loggers = []
        log_filter = _QueryLogFilter()
        try:
            if not self._closed.is_set():
                # Configure third-party loggers only; our own messages use the
                # repository's get_logger. Its lazy Logger upgrade is unsuitable
                # here because these loggers are also owned by other services.
                for name in ("uvicorn.access", "uvicorn.error", "uvicorn.asgi"):
                    logger = logging.getLogger(name)
                    logger.addFilter(log_filter)
                    filtered_loggers.append(logger)
                self._server.run(sockets=[self._socket])
                if not self._closed.is_set():
                    self._failure()
        except (Exception, SystemExit):
            self._failure()
        finally:
            for logger in filtered_loggers:
                logger.removeFilter(log_filter)
            self._closed.set()
            self._bound_url = None
            self._release_socket()

    def _release_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()

    def close(self) -> None:
        """Stop advertising availability immediately; never join the server
        thread."""
        self._closed.set()
        self._bound_url = None
        if self._server is not None:
            self._server.should_exit = True
        # A thread that never started cannot own cleanup of the bound socket.
        if self._thread is None or not self._thread.is_alive():
            self._release_socket()

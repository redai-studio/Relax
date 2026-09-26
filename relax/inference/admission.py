# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from relax.inference.gateway import forward_headers


class AdmissionClosed(RuntimeError):
    pass


class AdmissionGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._ready = False
        self._next_id = 0
        self._active: dict[int, Callable[[], None] | None] = {}
        self._backend_drained = True

    def open(self) -> None:
        with self._condition:
            if self._ready:
                return
            if self._active or not self._backend_drained:
                raise RuntimeError("Cannot reopen inference before HTTP and backend drain are confirmed")
            self._ready = True

    def close(self) -> None:
        with self._condition:
            self._ready = False
            callbacks = [callback for callback in self._active.values() if callback is not None]
        for callback in callbacks:
            callback()

    def enter(self, cancel: Callable[[], None] | None = None) -> int:
        with self._condition:
            if not self._ready:
                raise AdmissionClosed("Inference engine is sleeping or draining")
            self._next_id += 1
            self._active[self._next_id] = cancel
            self._backend_drained = False
            return self._next_id

    def finish(self, token: int, *, completed: bool) -> None:
        with self._condition:
            self._active.pop(token, None)
            self._condition.notify_all()

    def confirm_backend_drained(self) -> None:
        with self._condition:
            if self._ready:
                raise RuntimeError("Backend drain may only be acknowledged behind closed admission")
            self._backend_drained = True
            self._condition.notify_all()

    def wait_drained(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active or not self._backend_drained:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Inference HTTP/backend drain did not complete")
                self._condition.wait(remaining)

    def wait_requests_closed(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Inference proxy requests did not stop")
                self._condition.wait(remaining)

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {"ready": self._ready, "inflight": len(self._active), "backend_drained": self._backend_drained}


class _AdmissionStream(StreamingResponse):
    def __init__(self, upstream: httpx.Response, gate: AdmissionGate, token: int) -> None:
        self._upstream = upstream
        self._gate = gate
        self._token = token
        self._closed = False
        self._completed = False

        async def chunks() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_raw():
                yield chunk
            self._completed = True

        super().__init__(chunks(), status_code=upstream.status_code, headers=forward_headers(upstream.headers))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._upstream.aclose()
        finally:
            self._gate.finish(self._token, completed=self._completed)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.aclose())


def create_admission_app(
    backend_url: str, gate: AdmissionGate, *, http_client: httpx.AsyncClient | None = None
) -> FastAPI:
    client = http_client

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal client
        if client is None:
            client = httpx.AsyncClient(timeout=1800.0, trust_env=False)
        try:
            yield
        finally:
            if http_client is None:
                await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    async def proxy(request: Request, *, inference: bool) -> Response:
        if client is None:
            raise HTTPException(503, "Inference proxy is not initialized")
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        token = None
        if inference:
            try:
                token = gate.enter(lambda: loop.call_soon_threadsafe(task.cancel) if task is not None else None)
            except AdmissionClosed as exc:
                raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
        upstream = None
        handed_off = False
        try:
            body = await request.body()
            headers = forward_headers(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)
            built = client.build_request(
                request.method,
                backend_url.rstrip("/") + request.url.path,
                params=request.query_params,
                content=body,
                headers=headers,
            )
            upstream = await client.send(built, stream=True)
            if token is not None and "text/event-stream" in upstream.headers.get("content-type", ""):
                handed_off = True
                return _AdmissionStream(upstream, gate, token)
            content = await upstream.aread()
            response_headers = forward_headers(upstream.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(content, status_code=upstream.status_code, headers=response_headers)
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Inference backend unavailable") from exc
        finally:
            if not handed_off:
                if upstream is not None:
                    try:
                        await asyncio.shield(upstream.aclose())
                    finally:
                        if token is not None:
                            gate.finish(token, completed=True)
                elif token is not None:
                    gate.finish(token, completed=False)

    @app.post("/generate")
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    @app.post("/v1/completions")
    async def generate(request: Request) -> Response:
        return await proxy(request, inference=True)

    @app.post("/abort_request")
    @app.post("/close_session")
    async def cancel(request: Request) -> Response:
        return await proxy(request, inference=False)

    @app.get("/v1/models")
    async def metadata(request: Request) -> Response:
        return await proxy(request, inference=False)

    async def filtered_metadata(request: Request, endpoint: str, fields: frozenset[str]) -> dict[str, Any]:
        if client is None:
            raise HTTPException(503, "Inference proxy is not initialized")
        headers = forward_headers(request.headers)
        headers.pop("host", None)
        try:
            response = await client.get(backend_url.rstrip("/") + "/" + endpoint, headers=headers)
            if response.status_code == 404:
                response = await client.get(backend_url.rstrip("/") + "/get_" + endpoint, headers=headers)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, "Inference metadata is unavailable") from exc
        if not isinstance(data, dict):
            raise HTTPException(502, "Inference metadata is not an object")
        return {name: value for name, value in data.items() if name in fields}

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def server_info(request: Request) -> dict[str, Any]:
        return await filtered_metadata(
            request,
            "server_info",
            frozenset(
                {
                    "model_id",
                    "model",
                    "model_path",
                    "served_model_name",
                    "tp_size",
                    "dp_size",
                    "load_balance_method",
                    "disaggregation_mode",
                }
            ),
        )

    @app.get("/model_info")
    @app.get("/get_model_info")
    async def model_info(request: Request) -> dict[str, Any]:
        return await filtered_metadata(
            request,
            "model_info",
            frozenset({"model_path", "tokenizer_path", "is_generation", "model_type", "architectures"}),
        )

    @app.get("/health")
    async def liveness(request: Request) -> Response:
        if client is None:
            return Response(status_code=503)
        try:
            response = await client.get(backend_url.rstrip("/") + "/health", timeout=5.0)
        except httpx.HTTPError:
            return Response(status_code=503)
        return Response(status_code=200 if response.is_success else 503)

    @app.get("/health_generate")
    async def health() -> Response:
        return Response(status_code=200 if gate.status()["ready"] else 503)

    return app


class AdmissionServer:
    def __init__(self, backend_url: str, host: str, port: int, *, timeout: float = 30.0) -> None:

        self.gate = AdmissionGate()
        self._socket = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((host, port))
            self._socket.listen()
        except BaseException:
            self._socket.close()
            raise
        self.port = self._socket.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_admission_app(backend_url, self.gate),
                log_config=None,
                access_log=False,
                timeout_graceful_shutdown=timeout,
                lifespan="on",
            )
        )
        self._error: BaseException | None = None

        def run() -> None:
            try:
                self._server.run(sockets=[self._socket])
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=run, name="inference-admission", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if self._error is not None or not self._thread.is_alive():
                self._socket.close()
                raise RuntimeError("Inference admission server failed to start") from self._error
            if time.monotonic() >= deadline:
                self.stop(timeout=timeout)
                raise TimeoutError("Inference admission server startup timed out")
            time.sleep(0.01)

    def stop(self, *, timeout: float = 30.0) -> None:
        self.gate.close()
        self._server.should_exit = True
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Inference admission server did not stop")
        self._socket.close()

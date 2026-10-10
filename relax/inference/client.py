# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import copy
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx

from relax.inference.routing import InferenceRoutingError, RouteResolver, RouteTarget
from relax.inference.specs import validate_snapshot


SnapshotProvider = Callable[[], Awaitable[dict[str, Any]]]
_loop_clients: Any = None


async def _wait_for(awaitable: Awaitable[Any], timeout: float) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError from exc


async def generate_with_discovery(
    discovery_url: str,
    payload: dict[str, Any],
    *,
    model: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 1800.0,
    max_connections: int | None = None,
) -> dict[str, Any]:
    global _loop_clients
    if _loop_clients is None:
        _loop_clients = weakref.WeakKeyDictionary()
    loop = asyncio.get_running_loop()
    cache = _loop_clients.setdefault(loop, {})
    cache_key = (discovery_url, timeout, max_connections)
    if cache_key not in cache:
        cache[cache_key] = InferenceClient(
            discovery_url=discovery_url, timeout=timeout, max_connections=max_connections
        )
    client = cache[cache_key]
    affinity = next((value for key, value in (headers or {}).items() if key.lower() == "x-smg-routing-key"), None)
    return await client.generate(payload, model=model, headers=headers, affinity_key=affinity)


async def close_loop_inference_clients() -> None:
    if _loop_clients is None:
        return
    clients = _loop_clients.pop(asyncio.get_running_loop(), {})
    await asyncio.gather(*(client.aclose() for client in clients.values()))


class _DeadlineStream(httpx.AsyncByteStream):
    def __init__(self, source: httpx.AsyncByteStream, deadline: float, clock: Callable[[], float]) -> None:
        self.source = source
        self.deadline = deadline
        self.clock = clock

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self.source.__aiter__()
        while True:
            try:
                yield await _wait_for(iterator.__anext__(), max(0.0, self.deadline - self.clock()))
            except StopAsyncIteration:
                return

    async def aclose(self) -> None:
        await self.source.aclose()


def endpoint_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Inference URL must be an HTTP(S) URL without embedded credentials")
    if parsed.query or parsed.fragment or not path.startswith("/") or path.startswith("//"):
        raise ValueError("Inference endpoints require an absolute path without a query or fragment in the base URL")
    return base_url.rstrip("/") + path


class InferenceClient:
    def __init__(
        self,
        *,
        discovery_url: str | None = None,
        snapshot_provider: SnapshotProvider | None = None,
        gateway_url: str | None = None,
        timeout: float = 60.0,
        cache_ttl: float = 1.0,
        max_connections: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (discovery_url is None) == (snapshot_provider is None):
            raise ValueError("Provide exactly one discovery URL or snapshot provider")
        if timeout <= 0 or cache_ttl < 0:
            raise ValueError("timeout must be positive and cache_ttl must be nonnegative")
        if max_connections is not None and (
            isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections <= 0
        ):
            raise ValueError("max_connections must be a positive integer")
        self.discovery_url = discovery_url
        self.gateway_url = gateway_url
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._provider = snapshot_provider
        self._clock = clock
        client_kwargs = {"timeout": timeout}
        if max_connections is not None:
            client_kwargs["limits"] = httpx.Limits(max_connections=max_connections)
        self._http = http_client or httpx.AsyncClient(**client_kwargs)
        self._owns_http = http_client is None
        self._snapshot: dict[str, Any] | None = None
        self._expires_at = 0.0
        self._refresh_lock = asyncio.Lock()
        self._resolver = RouteResolver()
        self._closed = False

    def invalidate(self) -> None:
        self._snapshot = None
        self._expires_at = 0.0

    async def snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Inference client is closed")
        async with self._refresh_lock:
            if not refresh and self._snapshot is not None and self._clock() < self._expires_at:
                return copy.deepcopy(self._snapshot)
            self.invalidate()
            if self._provider is not None:
                snapshot = await _wait_for(self._provider(), self.timeout)
            else:
                response = await self._http.get(
                    endpoint_url(self.discovery_url, "/engines"),
                    params={"schema": 2},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                snapshot = response.json()

            validate_snapshot(snapshot)
            self._snapshot = copy.deepcopy(snapshot)
            self._expires_at = self._clock() + self.cache_ttl
            return copy.deepcopy(snapshot)

    async def _request_target(
        self, payload: dict[str, Any], model: str | None, route_key: str | None, affinity_key: str | None
    ) -> Any:
        for attempt in range(2):
            snapshot = await self.snapshot(refresh=attempt > 0)
            try:
                return self._resolver.resolve(
                    snapshot,
                    model=model if model is not None else payload.get("model"),
                    route_key=route_key if route_key is not None else payload.get("route_key"),
                    affinity_key=affinity_key,
                )
            except InferenceRoutingError as exc:
                if exc.status_code != 503 or attempt:
                    raise
        raise AssertionError("Unreachable route selection")

    def _outgoing_payload(self, path: str, payload: dict[str, Any], target: RouteTarget) -> dict[str, Any]:
        outgoing = dict(payload)
        if self.gateway_url is not None:
            outgoing["model"] = target.model_id
        else:
            outgoing.pop("route_key", None)
            if path == "/generate":
                outgoing.pop("model", None)
            if path in {"/chat/completions", "/v1/chat/completions"}:
                if target.served_model_name is None:
                    raise InferenceRoutingError(503, "Discovery does not declare the backend served model name")
                outgoing["model"] = target.served_model_name
        return outgoing

    async def request(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        model: str | None = None,
        route_key: str | None = None,
        affinity_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await _wait_for(
            self._request(path, payload, model, route_key, affinity_key, headers), timeout=self.timeout
        )

    async def _request(
        self,
        path: str,
        payload: dict[str, Any],
        model: str | None,
        route_key: str | None,
        affinity_key: str | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        target = await self._request_target(payload, model, route_key, affinity_key)
        outgoing = self._outgoing_payload(path, payload, target)
        forward_headers = httpx.Headers(headers or {})
        if affinity_key is not None:
            forward_headers.setdefault("X-SMG-Routing-Key", affinity_key)
        try:
            response = await self._http.post(
                endpoint_url(self.gateway_url or target.base_url, path),
                json=outgoing,
                headers=forward_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError, asyncio.CancelledError):
            self.invalidate()
            raise

    async def generate(self, payload: dict[str, Any], **routing: Any) -> dict[str, Any]:
        response = await self.request("/generate", payload, **routing)
        return response.json()

    @asynccontextmanager
    async def stream(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        model: str | None = None,
        route_key: str | None = None,
        affinity_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        deadline = self._clock() + self.timeout
        target = await _wait_for(self._request_target(payload, model, route_key, affinity_key), self.timeout)
        outgoing = self._outgoing_payload(path, payload, target)
        forward_headers = httpx.Headers(headers or {})
        if affinity_key is not None:
            forward_headers.setdefault("X-SMG-Routing-Key", affinity_key)
        response = None
        try:
            request = self._http.build_request(
                "POST",
                endpoint_url(self.gateway_url or target.base_url, path),
                json=outgoing,
                headers=forward_headers,
                timeout=self.timeout,
            )
            response = await _wait_for(self._http.send(request, stream=True), max(0.0, deadline - self._clock()))
            response.raise_for_status()
            response.stream = _DeadlineStream(response.stream, deadline, self._clock)
            yield response
        except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError):
            self.invalidate()
            raise
        finally:
            if response is not None:
                await response.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.invalidate()
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

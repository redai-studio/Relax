# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from relax.inference.client import InferenceClient, SnapshotProvider, _wait_for
from relax.inference.routing import InferenceRoutingError, RouteResolver
from relax.inference.specs import INFERENCE_ROLES, validate_snapshot


GenRMRender = Callable[[str, list[dict[str, Any]], dict[str, Any] | None], Awaitable[dict[str, Any]]]
_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
)


def forward_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    headers = headers or {}
    connection_fields = {
        token.strip().lower()
        for name, value in headers.items()
        if name.lower() == "connection"
        for token in value.split(",")
    }
    return {
        ("X-SMG-Routing-Key" if name.lower() == "x-smg-routing-key" else name): value
        for name, value in headers.items()
        if name.lower() not in _HOP_HEADERS and name.lower() not in connection_fields
    }


@asynccontextmanager
async def _http_errors() -> AsyncIterator[None]:
    try:
        yield
    except InferenceRoutingError as exc:
        headers = {"Retry-After": "1"} if exc.status_code == 503 else None
        raise HTTPException(status_code=exc.status_code, detail=str(exc), headers=headers) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response.is_stream_consumed else "Upstream inference request failed"
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise HTTPException(status_code=504, detail="Inference request timed out") from exc
    except httpx.TransportError as exc:
        raise HTTPException(status_code=502, detail="Inference backend is unavailable") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Inference backend returned invalid JSON") from exc


class _ProxyStreamingResponse(StreamingResponse):
    def __init__(self, upstream: httpx.Response, stack: AsyncExitStack) -> None:
        self._stack = stack
        self._close_task: asyncio.Task | None = None

        async def content() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await self.aclose()

        super().__init__(
            content(),
            status_code=upstream.status_code,
            headers=forward_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._stack.aclose())
        await asyncio.shield(self._close_task)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()


class InferenceGatewayHandler:
    def __init__(
        self,
        role: str,
        snapshot_provider: SnapshotProvider,
        http_client: httpx.AsyncClient | None = None,
        genrm_render: GenRMRender | None = None,
        *,
        timeout: float = 1800.0,
    ) -> None:
        if role not in INFERENCE_ROLES:
            raise ValueError(f"Unsupported inference role: {role!r}")
        self.role = role
        self._provider = snapshot_provider
        self._genrm_render = genrm_render
        self._client = InferenceClient(
            snapshot_provider=self._validated_snapshot, http_client=http_client, timeout=timeout, cache_ttl=0
        )

    @staticmethod
    def _raw_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key not in {"model", "route_key"}}

    async def _validated_snapshot(self) -> dict[str, Any]:
        try:
            snapshot = validate_snapshot(await self._provider())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise InferenceRoutingError(503, "Inference registry is unavailable") from exc
        if snapshot.role != self.role:
            raise InferenceRoutingError(503, "Inference discovery returned the wrong role")
        return snapshot.to_dict()

    async def discovery(self) -> dict[str, Any]:
        async with _http_errors():
            return await self._client.snapshot(refresh=True)

    async def models(self) -> dict[str, Any]:
        snapshot = await self.discovery()
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": self.role}
                for model_id in snapshot["models"]
            ],
        }

    async def health(self) -> dict[str, Any]:
        try:
            snapshot = await self.discovery()
        except HTTPException:
            return {
                "schema_version": 2,
                "role": self.role,
                "gateway_alive": True,
                "registry_available": False,
                "ready": False,
                "models": {},
            }
        resolver = RouteResolver()
        models = {}
        for model_id, model in snapshot["models"].items():
            try:
                resolver.resolve(snapshot, model=model_id, affinity_key="health")
                ready = True
            except InferenceRoutingError:
                ready = False
            models[model_id] = {"state": model["state"], "ready": ready}
        return {
            "schema_version": 2,
            "role": self.role,
            "gateway_alive": True,
            "registry_available": True,
            "ready": bool(models) and all(model["ready"] for model in models.values()),
            "models": models,
        }

    @staticmethod
    def _payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Inference payload must be a JSON object")
        if "stream" in payload and not isinstance(payload["stream"], bool):
            raise HTTPException(status_code=400, detail="stream must be a boolean")
        return payload

    @staticmethod
    def _affinity(headers: Mapping[str, str] | None) -> str | None:
        return next((value for key, value in (headers or {}).items() if key.lower() == "x-smg-routing-key"), None)

    async def _stream(
        self, path: str, payload: dict[str, Any], headers: Mapping[str, str] | None
    ) -> StreamingResponse:
        stack = AsyncExitStack()
        try:
            upstream = await stack.enter_async_context(
                self._client.stream(
                    path,
                    self._raw_payload(payload) if path == "/generate" else payload,
                    model=payload.get("model"),
                    route_key=payload.get("route_key"),
                    headers=forward_headers(headers),
                    affinity_key=self._affinity(headers),
                )
            )
            return _ProxyStreamingResponse(upstream, stack)
        except BaseException:
            await stack.aclose()
            raise

    async def generate(
        self, payload: dict[str, Any], headers: Mapping[str, str] | None = None
    ) -> dict[str, Any] | StreamingResponse:
        async with _http_errors():
            return await _wait_for(self._generate(payload, headers), self._client.timeout)

    async def _generate(
        self, payload: dict[str, Any], headers: Mapping[str, str] | None
    ) -> dict[str, Any] | StreamingResponse:
        payload = self._payload(payload)
        has_messages = "messages" in payload
        has_raw = "input_ids" in payload or "text" in payload
        if has_messages and has_raw:
            raise HTTPException(status_code=400, detail="Specify messages or raw input_ids/text, not both")
        async with _http_errors():
            if has_messages:
                if self.role != "genrm" or self._genrm_render is None:
                    raise HTTPException(status_code=400, detail="messages generation requires a GenRM adapter")
                if not isinstance(payload["messages"], list) or any(
                    not isinstance(message, dict) for message in payload["messages"]
                ):
                    raise HTTPException(status_code=400, detail="messages must be an array of message objects")
                sampling = payload.get("sampling_params")
                if sampling is not None and not isinstance(sampling, dict):
                    raise HTTPException(status_code=400, detail="sampling_params must be an object")
                if payload.get("stream", False):
                    raise HTTPException(status_code=400, detail="Legacy GenRM messages do not support streaming")
                snapshot = await self._client.snapshot(refresh=True)
                target = RouteResolver().resolve(
                    snapshot,
                    model=payload.get("model"),
                    route_key=payload.get("route_key"),
                    affinity_key=self._affinity(headers),
                )
                rendered = await self._genrm_render(target.model_id, payload["messages"], sampling)
                result = await self._client.generate(
                    rendered,
                    model=target.model_id,
                    headers=forward_headers(headers),
                    affinity_key=self._affinity(headers),
                )
                return {"response": result.get("text", "").strip()}
            if not has_raw:
                raise HTTPException(status_code=400, detail="Raw generation requires input_ids or text")
            if payload.get("stream", False):
                return await self._stream("/generate", payload, headers)
            return await self._client.generate(
                self._raw_payload(payload),
                model=payload.get("model"),
                route_key=payload.get("route_key"),
                headers=forward_headers(headers),
                affinity_key=self._affinity(headers),
            )

    async def chat(
        self, payload: dict[str, Any], headers: Mapping[str, str] | None = None
    ) -> httpx.Response | StreamingResponse:
        payload = self._payload(payload)
        if not isinstance(payload.get("messages"), list):
            raise HTTPException(status_code=400, detail="Chat requires a messages array")
        async with _http_errors():
            if payload.get("stream", False):
                return await self._stream("/v1/chat/completions", payload, headers)
            return await self._client.request(
                "/v1/chat/completions",
                payload,
                headers=forward_headers(headers),
                affinity_key=self._affinity(headers),
            )

    @staticmethod
    async def _read_payload(request: Request) -> dict[str, Any]:
        try:
            return InferenceGatewayHandler._payload(await request.json())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON request body") from exc

    async def handle_generate(self, request: Request) -> Response:
        result = await self.generate(await self._read_payload(request), request.headers)
        return result if isinstance(result, StreamingResponse) else JSONResponse(result)

    async def handle_chat(self, request: Request) -> Response:
        result = await self.chat(await self._read_payload(request), request.headers)
        if isinstance(result, StreamingResponse):
            return result
        headers = {
            key: value for key, value in forward_headers(result.headers).items() if key.lower() != "content-encoding"
        }
        return Response(result.content, status_code=result.status_code, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

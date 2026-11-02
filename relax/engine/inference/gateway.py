# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP gateway logic shared by the rollout, GenRM and OPD teacher roles.

``InferenceGateway`` is a plain class, not a Serve deployment: each role's
CPU-only Serve replica holds one instance and exposes its methods as routes.
It never touches a GPU and never changes an engine's state. In particular a
request for a model that is not ready is answered with 503, not by waking the
model up -- who holds GPU memory when is the lifecycle's decision.

Every inference entry acquires a manager-owned permit before proxying. Internal
lookaside clients use the same admission gate and send their payload directly.
Cached topology is used for discovery, never as proof of admission.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable, Mapping

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from relax.engine.inference.client import (
    InferenceClient,
    InferenceRequest,
    InferenceRequestClient,
    SnapshotSource,
    _settle,
)
from relax.engine.inference.discovery import RoleSnapshot
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RouteTarget,
    RoutingError,
    RoutingState,
    select_model,
    select_target,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_HOP_BY_HOP_HEADERS = frozenset({"host", "transfer-encoding", "connection", "keep-alive", "upgrade", "content-length"})

# Seconds a client should wait before retrying a model that is not ready.
DEFAULT_RETRY_AFTER_S = 5
# Upper bound for diagnostic snapshot fetches. Cached discovery is never
# sufficient to admit an inference request.
SNAPSHOT_FETCH_TIMEOUT_S = 5.0


def make_error_chunk(status_code: int, message: str) -> str:
    """An OpenAI-style streaming chunk that carries an upstream error."""
    error_response = {
        "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [],
        "error": {"code": status_code, "message": message},
    }
    return json.dumps(error_response)


def build_forward_headers(original_headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in original_headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS}


class _NativeStreamingResponse(StreamingResponse):
    """Preserve native SSE framing and close upstream even before iteration."""

    def __init__(self, upstream: httpx.Response) -> None:
        self._upstream = upstream
        self._close_task: asyncio.Task[None] | None = None

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except RoutingError as error:
                yield f"data: {make_error_chunk(error.status_code, str(error))}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                await self.aclose()

        # HTTPX decodes content encodings while iterating bytes.
        headers = {
            key: value
            for key, value in build_forward_headers(upstream.headers).items()
            if key.lower() not in {"content-length", "content-encoding"}
        }
        super().__init__(chunks(), status_code=upstream.status_code, headers=headers, media_type="text/event-stream")

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._upstream.aclose())
        await asyncio.shield(self._close_task)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()


class _PermitStreamingResponse(StreamingResponse):
    """Keep the permit until the upstream is closed, even before iteration."""

    def __init__(self, response: StreamingResponse, permit: InferenceRequest | None) -> None:
        self._response = response
        self._permit = permit
        self._close_task: asyncio.Task[None] | None = None

        async def chunks() -> AsyncIterator[str | bytes]:
            try:
                async for chunk in response.body_iterator:
                    yield chunk
            finally:
                await self.aclose()

        super().__init__(
            chunks(), status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type
        )

    async def _close(self) -> None:
        try:
            close = getattr(self._response.body_iterator, "aclose", None)
            if close is not None:
                await close()
            if isinstance(self._response, _NativeStreamingResponse):
                await self._response.aclose()
        finally:
            if self._permit is not None:
                await self._permit.aclose()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await _settle(self._close_task)
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()


class InferenceGateway:
    def __init__(
        self,
        role: str,
        fetch_snapshot: SnapshotSource,
        *,
        upstream_name: str = "inference engine",
        request_client: InferenceRequestClient | None = None,
        proxy_client: httpx.AsyncClient | None = None,
        snapshot_max_age_s: float = 1.0,
        retry_after_s: int = DEFAULT_RETRY_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            role: Role this gateway serves (``rollout`` / ``genrm`` / ``teacher``).
            fetch_snapshot: Synchronous source of the role's topology snapshot,
                typically a Ray call into the role's manager(s).
            upstream_name: What to call the upstream in error messages.
            snapshot_max_age_s: Cache age for diagnostic discovery responses.
                Inference always acquires a permit from the live manager.
        """
        self.role = role
        self._requests = request_client or InferenceRequestClient(role)
        self._upstream_name = upstream_name
        self._retry_after_s = retry_after_s
        self._topology = InferenceClient(fetch_snapshot, max_age_s=snapshot_max_age_s, clock=clock)
        self._routing_state = RoutingState()
        self._proxy_client = proxy_client

    # ------------------------------------------------------------------
    # Discovery.
    # ------------------------------------------------------------------

    async def snapshot(self) -> RoleSnapshot:
        cached = self._topology.last_snapshot
        if cached is not None and not self._topology.needs_refresh():
            return cached
        # The source is a blocking manager call; keep it off the event loop.
        return await asyncio.to_thread(self._topology.snapshot)

    async def engines(self) -> dict[str, Any]:
        return (await self.snapshot()).to_dict()

    async def health(self) -> dict[str, Any]:
        """Report the gateway and each model's state.

        Always answers: a role whose engines are all asleep is healthy and
        unavailable, which is a different thing from a dead gateway.
        """
        snapshot = await self.snapshot()
        return {
            "status": "ok",
            "role": self.role,
            "phase": snapshot.phase,
            "topology_revision": snapshot.topology_revision,
            "models": {model.name: model.state.value for model in snapshot.models},
        }

    async def models(self) -> dict[str, Any]:
        snapshot = await self.snapshot()
        names = [model.name for model in snapshot.models]
        aliases = [model.served_model_name for model in snapshot.models if model.served_model_name]
        names.extend(alias for alias in dict.fromkeys(aliases) if alias not in names and aliases.count(alias) == 1)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "relax"} for name in names],
        }

    async def resolve(
        self, *, model: str | None = None, route_key: str | None = None, affinity_key: Any = None
    ) -> RouteTarget:
        """Pick the upstream for a request, as an HTTP-level outcome.

        Raises 400 for a model this role does not serve and 503 with ``Retry-
        After`` for one that cannot take requests right now.
        """
        snapshot = await self.snapshot()
        try:
            return select_target(
                snapshot, self._routing_state, model=model, route_key=route_key, affinity_key=affinity_key
            )
        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=str(exc), headers={"Retry-After": str(self._retry_after_s)}
            ) from exc
        except RoutingError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Snapshot-routed requests.
    # ------------------------------------------------------------------

    async def generate(self, payload: Mapping[str, Any]) -> Any:
        """Forward a native engine ``/generate`` payload.

        ``model`` / ``route_key`` select the model and are not part of the
        engine's payload, so they are stripped before forwarding.
        """
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="Invalid request body: expected a JSON object")
        forwarded = dict(payload)
        permit = await self._admit(model=forwarded.pop("model", None), route_key=forwarded.pop("route_key", None))
        transferred = False
        try:
            url = f"{permit.target.base_url}/generate"
            if forwarded.get("stream"):
                response = _PermitStreamingResponse(await self._stream_generate(url, forwarded, permit), permit)
                transferred = True
                return response
            return await self.proxy_json(url, forwarded)
        finally:
            if not transferred:
                await permit.aclose()

    async def _admit(self, **selection: Any) -> InferenceRequest:
        permit = self._requests.request(**selection)
        try:
            return await permit.__aenter__()
        except ModelUnavailableError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": str(self._retry_after_s)}) from exc
        except RoutingError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    async def _stream_generate(
        self, url: str, payload: Mapping[str, Any], permit: InferenceRequest
    ) -> StreamingResponse:
        client = self._get_proxy_client()
        response = None
        try:
            response = await client.send(client.build_request("POST", url, json=dict(payload)), stream=True)
            await permit.dispatched()
            response.raise_for_status()
        except RoutingError as exc:
            if response is not None:
                await response.aclose()
            raise HTTPException(
                exc.status_code,
                str(exc),
                headers={"Retry-After": str(self._retry_after_s)} if isinstance(exc, ModelUnavailableError) else None,
            ) from exc
        except httpx.HTTPStatusError as exc:
            try:
                detail = (await exc.response.aread()).decode(errors="replace")
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=detail,
                    headers={"Retry-After": exc.response.headers["Retry-After"]}
                    if "Retry-After" in exc.response.headers
                    else None,
                ) from exc
            finally:
                await exc.response.aclose()
        except httpx.RequestError as exc:
            if response is not None:
                await response.aclose()
            self._on_upstream_unreachable(url, exc)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {exc}") from exc
        except BaseException:
            if response is not None:
                await response.aclose()
            raise
        return _NativeStreamingResponse(response)

    async def chat_completions(self, body: bytes, headers: Mapping[str, str]) -> Any:
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {exc}") from exc

        if request.get("stream"):
            # Validate the public model before sending SSE response headers.
            # This identity lookup grants no permission to dispatch later.
            try:
                select_model(await self.snapshot(), request.get("model"), request.get("route_key"))
            except RoutingError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc

        async def dispatch() -> Any:
            permit = await self._admit(model=request.get("model"), route_key=request.get("route_key"))
            transferred = False
            try:
                original = dict(request)
                request.pop("route_key", None)
                request["model"] = permit.served_model_name or permit.target.model
                result = await self.proxy_chat_completions(
                    permit.target.base_url,
                    body if request == original else json.dumps(request).encode(),
                    headers,
                    stream=bool(request.get("stream")),
                    permit=permit,
                )
                if isinstance(result, StreamingResponse):
                    result = _PermitStreamingResponse(result, permit)
                    transferred = True
                return result
            finally:
                if not transferred:
                    await permit.aclose()

        if not request.get("stream"):
            return await dispatch()

        async def chunks() -> AsyncIterator[str | bytes]:
            response = None
            try:
                # A StreamingResponse can sit unconsumed across offload/resume.
                # Admit in the same coroutine that actually starts the send.
                response = await dispatch()
                async for chunk in response.body_iterator:
                    yield chunk
            except HTTPException as error:
                yield f"data: {make_error_chunk(error.status_code, str(error.detail))}\n\n"
                yield "data: [DONE]\n\n"
            except RoutingError as error:
                yield f"data: {make_error_chunk(error.status_code, str(error))}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if response is not None:
                    await response.aclose()

        return _PermitStreamingResponse(StreamingResponse(chunks(), media_type="text/event-stream"), None)

    # ------------------------------------------------------------------
    # Proxying to an already resolved upstream.
    # ------------------------------------------------------------------

    def _get_proxy_client(self) -> httpx.AsyncClient:
        if self._proxy_client is None:
            # No timeout: generation can legitimately take minutes. The pool is
            # sized for a whole rollout batch of concurrent agentic sessions.
            self._proxy_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(max_connections=4096, max_keepalive_connections=4096, keepalive_expiry=600),
            )
        return self._proxy_client

    async def proxy_get_json(self, url: str) -> Any:
        try:
            response = await self._get_proxy_client().get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text,
                headers={"Retry-After": e.response.headers["Retry-After"]}
                if "Retry-After" in e.response.headers
                else None,
            )
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    async def proxy_json(self, url: str, payload: Mapping[str, Any]) -> Any:
        try:
            response = await self._get_proxy_client().post(url, json=dict(payload))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text,
                headers={"Retry-After": e.response.headers["Retry-After"]}
                if "Retry-After" in e.response.headers
                else None,
            )
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    async def proxy_chat_completions(
        self,
        base_url: str,
        body: bytes,
        headers: Mapping[str, str],
        *,
        stream: bool,
        permit: InferenceRequest | None = None,
    ) -> Any:
        """Forward a chat-completions request body unmodified.

        Returns the upstream JSON, or a ``StreamingResponse`` when ``stream``.
        """
        url = f"{base_url}/v1/chat/completions"
        client = self._get_proxy_client()
        forward_headers = build_forward_headers(headers)
        if stream:
            return self._stream(client, url, body, forward_headers, permit)
        try:
            response = await client.post(url, content=body, headers=forward_headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text,
                headers={"Retry-After": e.response.headers["Retry-After"]}
                if "Retry-After" in e.response.headers
                else None,
            )
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    def _stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: bytes,
        forward_headers: dict[str, str],
        permit: InferenceRequest | None = None,
    ) -> StreamingResponse:
        async def _event_generator():
            response = None
            try:
                req = client.build_request("POST", url, content=body, headers=forward_headers)
                response = await client.send(req, stream=True)
                if permit is not None:
                    await permit.dispatched()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
            except RoutingError as error:
                yield f"data: {make_error_chunk(error.status_code, str(error))}\n\n"
                yield "data: [DONE]\n\n"
            except httpx.HTTPStatusError as e:
                error_body = await e.response.aread()
                error_chunk = make_error_chunk(e.response.status_code, error_body.decode(errors="replace"))
                yield f"data: {error_chunk}\n\n"
                yield "data: [DONE]\n\n"
            except httpx.RequestError as e:
                self._on_upstream_unreachable(url, e)
                error_chunk = make_error_chunk(502, f"Failed to connect to {self._upstream_name}: {e}")
                yield f"data: {error_chunk}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if response is not None:
                    await response.aclose()

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    def _on_upstream_unreachable(self, url: str, error: Exception) -> None:
        logger.error(f"[{self.role}] failed to reach {self._upstream_name} at {url}: {error}")
        # The replica may have been rebuilt elsewhere; look again before the next request.
        self._topology.report_failure()

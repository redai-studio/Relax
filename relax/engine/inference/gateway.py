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
from collections.abc import AsyncIterator, Awaitable, Coroutine
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


async def _run_owned(
    operation: Coroutine[Any, Any, Any], track_cleanup: Callable[[asyncio.Task], asyncio.Task]
) -> Any:
    """Cancel HTTP work once; its unwinding and close remain owned off-
    caller."""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()

        async def finish() -> None:
            try:
                result = await task
            except (asyncio.CancelledError, Exception):
                return
            # Cancellation can win after headers produced a response but before
            # the frontend received it. Close that otherwise unconsumed stream.
            if isinstance(result, _PermitStreamingResponse):
                await result.aclose(cancelled=True)

        track_cleanup(asyncio.create_task(finish()))
        raise


class _UpstreamCloseError(RuntimeError):
    """Upstream exit is unconfirmed; its permit must remain with the owner."""


class _ManagedHTTPStream(httpx.AsyncByteStream):
    """Preserve the original close receipt even if HTTPX sets is_closed
    early."""

    def __init__(self, stream: httpx.AsyncByteStream, track_cleanup: Callable[[asyncio.Task], asyncio.Task]) -> None:
        self._stream = stream
        self._track_cleanup = track_cleanup
        self._close_task: asyncio.Task[None] | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = self._track_cleanup(asyncio.create_task(self._stream.aclose()))
        try:
            await asyncio.shield(self._close_task)
        except Exception as error:
            raise _UpstreamCloseError("Upstream close did not confirm completion; permit remains held") from error


async def _close_upstream(response: httpx.Response) -> None:
    try:
        await response.aclose()
    finally:
        # Response.aclose() may now be a no-op after an interrupted close.
        # The stream's one task, not that flag, is the completion evidence.
        if isinstance(response.stream, _ManagedHTTPStream):
            await response.stream.aclose()


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
            self._close_task = asyncio.create_task(_close_upstream(self._upstream))
        await asyncio.shield(self._close_task)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()


class _PermitStreamingResponse(StreamingResponse):
    """Keep the permit until the upstream is closed, even before iteration."""

    def __init__(
        self,
        response: StreamingResponse,
        permit: InferenceRequest | None,
        track_cleanup: Callable[[asyncio.Task], asyncio.Task],
        close_upstream: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._response = response
        self._permit = permit
        self._track_cleanup = track_cleanup
        self._close_upstream = close_upstream
        self._cancelled = False
        self._close_task: asyncio.Task[None] | None = None

        async def chunks() -> AsyncIterator[str | bytes]:
            try:
                async for chunk in response.body_iterator:
                    yield chunk
            except asyncio.CancelledError:
                self._cancelled = True
                raise
            finally:
                await self.aclose(cancelled=self._cancelled)

        super().__init__(
            chunks(), status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type
        )

    async def _close(self) -> None:
        close = getattr(self._response.body_iterator, "aclose", None)
        if close is not None:
            await close()
        if self._close_upstream is not None:
            await self._close_upstream()
        if isinstance(self._response, _NativeStreamingResponse):
            await self._response.aclose()
        # A failed close is not evidence of upstream exit. Leave the permit
        # fenced in that case; never return it from a finally block.
        if self._permit is not None:
            if self._cancelled:
                self._permit.start_release()
            else:
                await self._permit.aclose()

    async def aclose(self, *, cancelled: bool = False) -> None:
        self._cancelled |= cancelled
        if self._close_task is None:
            self._close_task = self._track_cleanup(asyncio.create_task(self._close()))
        if self._cancelled:
            return
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            self._cancelled = True
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def pump() -> None:
            try:
                await super(_PermitStreamingResponse, self).__call__(scope, receive, send)
            except asyncio.CancelledError:
                self._cancelled = True
                raise
            finally:
                await self.aclose(cancelled=self._cancelled)

        await _run_owned(pump(), self._track_cleanup)


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

        async def dispatch() -> Any:
            permit = await self._admit(model=forwarded.pop("model", None), route_key=forwarded.pop("route_key", None))
            transferred = cancelled = unconfirmed_close = False
            try:
                url = f"{permit.target.base_url}/generate"
                if forwarded.get("stream"):
                    response = _PermitStreamingResponse(
                        await self._stream_generate(url, forwarded, permit), permit, self._requests._track_cleanup
                    )
                    transferred = True
                    return response
                return await self.proxy_json(url, forwarded)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except _UpstreamCloseError:
                unconfirmed_close = True
                raise
            finally:
                if not transferred and not unconfirmed_close:
                    if cancelled:
                        permit.start_release()
                    else:
                        await permit.aclose()

        return await _run_owned(dispatch(), self._requests._track_cleanup)

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
            response = await self._send(client, client.build_request("POST", url, json=dict(payload)))
            await permit.dispatched()
            response.raise_for_status()
        except RoutingError as exc:
            if response is not None:
                await _close_upstream(response)
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
                await _close_upstream(exc.response)
        except httpx.RequestError as exc:
            if response is not None:
                await _close_upstream(response)
            self._on_upstream_unreachable(url, exc)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {exc}") from exc
        except BaseException:
            if response is not None:
                await _close_upstream(response)
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
            transferred = cancelled = unconfirmed_close = False
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
                    if not isinstance(result, _PermitStreamingResponse):
                        result = _PermitStreamingResponse(result, permit, self._requests._track_cleanup)
                    transferred = True
                return result
            except asyncio.CancelledError:
                cancelled = True
                raise
            except _UpstreamCloseError:
                unconfirmed_close = True
                raise
            finally:
                if not transferred and not unconfirmed_close:
                    if cancelled:
                        permit.start_release()
                    else:
                        await permit.aclose()

        if not request.get("stream"):
            return await _run_owned(dispatch(), self._requests._track_cleanup)

        async def chunks() -> AsyncIterator[str | bytes]:
            response = None
            try:
                # A StreamingResponse can sit unconsumed across offload/resume.
                # Admit in the same coroutine that actually starts the send.
                response = await _run_owned(dispatch(), self._requests._track_cleanup)
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

        return _PermitStreamingResponse(
            StreamingResponse(chunks(), media_type="text/event-stream"), None, self._requests._track_cleanup
        )

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

    async def _send(self, client: httpx.AsyncClient, request: httpx.Request) -> httpx.Response:
        # Wrap before aread/iteration: HTTPX may close automatically at EOF.
        response = await client.send(request, stream=True)
        response.stream = _ManagedHTTPStream(response.stream, self._requests._track_cleanup)
        return response

    async def proxy_json(self, url: str, payload: Mapping[str, Any]) -> Any:
        response = None
        try:
            client = self._get_proxy_client()
            response = await self._send(client, client.build_request("POST", url, json=dict(payload)))
            await response.aread()
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
        finally:
            if response is not None:
                await _close_upstream(response)

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
        response = None
        try:
            response = await self._send(
                client, client.build_request("POST", url, content=body, headers=forward_headers)
            )
            await response.aread()
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
        finally:
            if response is not None:
                await _close_upstream(response)

    def _stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: bytes,
        forward_headers: dict[str, str],
        permit: InferenceRequest | None = None,
    ) -> StreamingResponse:
        response = None
        sending: asyncio.Task[httpx.Response] | None = None

        async def close_upstream() -> None:
            nonlocal response
            if response is None and sending is not None:
                try:
                    response = await asyncio.shield(sending)
                except asyncio.CancelledError:
                    if not sending.done():
                        raise
                    # The send's one cancellation finished transport unwinding.
                except Exception:
                    # A failed send did not transfer an HTTP response to us.
                    pass
            if response is not None:
                await _close_upstream(response)

        async def _event_generator():
            nonlocal response, sending
            try:
                req = client.build_request("POST", url, content=body, headers=forward_headers)
                # Starlette's AnyIO scope may cancel iteration repeatedly.
                # Keep the actual send/unwind outside that scope and let the
                # owned close callback await its terminal result before return.
                sending = asyncio.create_task(self._send(client, req))
                try:
                    response = await asyncio.shield(sending)
                except asyncio.CancelledError:
                    sending.cancel()
                    raise
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
                await close_upstream()

        return _PermitStreamingResponse(
            StreamingResponse(_event_generator(), media_type="text/event-stream"),
            permit,
            self._requests._track_cleanup,
            close_upstream,
        )

    def _on_upstream_unreachable(self, url: str, error: Exception) -> None:
        logger.error(f"[{self.role}] failed to reach {self._upstream_name} at {url}: {error}")
        # The replica may have been rebuilt elsewhere; look again before the next request.
        self._topology.report_failure()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP gateway logic shared by the rollout, GenRM and OPD teacher roles.

``InferenceGateway`` is a plain class, not a Serve deployment: each role's
CPU-only Serve replica holds one instance and exposes its methods as routes.
It never touches a GPU and never changes an engine's state. In particular a
request for a model that is not ready is answered with 503, not by waking the
model up -- who holds GPU memory when is the lifecycle's decision.

Two entry styles:

- ``proxy_*`` forward to an upstream the caller already resolved. Rollout uses
  these with its SGLang router, body untouched, so its long-standing
  ``/v1/chat/completions`` contract is preserved byte for byte.
- ``generate`` / ``chat_completions`` resolve the upstream from the role's
  topology snapshot with the shared routing rules.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Mapping

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from relax.engine.inference.client import InferenceClient, SnapshotSource
from relax.engine.inference.discovery import RoleSnapshot
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RouteTarget,
    RoutingError,
    RoutingState,
    select_target,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_HOP_BY_HOP_HEADERS = frozenset({"host", "transfer-encoding", "connection", "keep-alive", "upgrade"})

# Seconds a client should wait before retrying a model that is not ready.
DEFAULT_RETRY_AFTER_S = 5


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


class InferenceGateway:
    def __init__(
        self,
        role: str,
        fetch_snapshot: SnapshotSource,
        *,
        upstream_name: str = "inference engine",
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
            snapshot_max_age_s: How long a fetched snapshot is routed on before
                it is fetched again; bounds how stale an availability answer
                can be without paying a manager call per request.
        """
        self.role = role
        self._upstream_name = upstream_name
        self._retry_after_s = retry_after_s
        self._topology = InferenceClient(fetch_snapshot, max_age_s=snapshot_max_age_s, clock=clock)
        self._routing_state = RoutingState()
        self._proxy_client: httpx.AsyncClient | None = None

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
        return {
            "object": "list",
            "data": [{"id": model.name, "object": "model", "owned_by": "relax"} for model in snapshot.models],
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
        forwarded = dict(payload)
        target = await self.resolve(model=forwarded.pop("model", None), route_key=forwarded.pop("route_key", None))
        return await self.proxy_json(f"{target.base_url}/generate", forwarded)

    async def chat_completions(self, body: bytes, headers: Mapping[str, str]) -> Any:
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {exc}") from exc
        target = await self.resolve(model=request.get("model"), route_key=request.get("route_key"))
        return await self.proxy_chat_completions(target.base_url, body, headers, stream=bool(request.get("stream")))

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
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    async def proxy_json(self, url: str, payload: Mapping[str, Any]) -> Any:
        try:
            response = await self._get_proxy_client().post(url, json=dict(payload))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    async def proxy_chat_completions(
        self, base_url: str, body: bytes, headers: Mapping[str, str], *, stream: bool
    ) -> Any:
        """Forward a chat-completions request body unmodified.

        Returns the upstream JSON, or a ``StreamingResponse`` when ``stream``.
        """
        url = f"{base_url}/v1/chat/completions"
        client = self._get_proxy_client()
        forward_headers = build_forward_headers(headers)
        if stream:
            return self._stream(client, url, body, forward_headers)
        try:
            response = await client.post(url, content=body, headers=forward_headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            self._on_upstream_unreachable(url, e)
            raise HTTPException(status_code=502, detail=f"Failed to connect to {self._upstream_name}: {e}")

    def _stream(
        self, client: httpx.AsyncClient, url: str, body: bytes, forward_headers: dict[str, str]
    ) -> StreamingResponse:
        async def _event_generator():
            response = None
            try:
                req = client.build_request("POST", url, content=body, headers=forward_headers)
                response = await client.send(req, stream=True)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
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

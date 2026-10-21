# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU HTTP gateway shared by rollout, GenRM, and Teacher services."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ray import serve
from starlette.types import Receive, Scope, Send

from relax.engine.inference.executor import AdmittedRequest, InferenceRequestExecutor, assign_backend_rids
from relax.engine.inference.routing import RoutingError
from relax.engine.inference.types import Role, RoleSnapshot
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
GATEWAY_REQUEST_HEADER = "x-relax-inference-gateway"


class _GatewayStreamingResponse(StreamingResponse):
    def __init__(self, content: Any, *, cleanup: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Disconnect can happen before the body iterator is even entered.
            await self._cleanup()


def _json_error(status_code: int, message: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers={"Retry-After": "1"} if status_code == 503 else None,
    )


def _routing_error(exc: RoutingError) -> JSONResponse:
    return _json_error(exc.status_code, str(exc), code="unavailable" if exc.status_code == 503 else "routing_error")


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() not in {"host", "content-length"}
    }
    headers[GATEWAY_REQUEST_HEADER] = "1"
    return headers


def _response_headers(response: httpx.Response, *, body: bool = True) -> dict[str, str]:
    """Upstream headers to relay; ``body=False`` drops those describing a body
    the Gateway rewrites."""
    dropped = _HOP_BY_HOP_HEADERS if body else _HOP_BY_HOP_HEADERS | {"content-length", "content-encoding"}
    return {key: value for key, value in response.headers.items() if key.lower() not in dropped}


class InferenceGateway:
    """Role-neutral HTTP ingress which routes through a Manager snapshot.

    The gateway owns no GPU resources and keeps no availability state of its
    own. The task inference owner behind ``manager_handle`` is the only state
    source. Admitted requests run through the shared
    :class:`~relax.engine.inference.executor.InferenceRequestExecutor`.
    """

    app = FastAPI()

    def __init__(
        self,
        role: str | Role,
        *,
        manager_handle: Any,
        upstream_url: str | None = None,
        genrm_backend_handle: Any | None = None,
        timeout: float | None = None,
    ) -> None:
        if manager_handle is None:
            raise ValueError("InferenceGateway requires the task inference manager handle")
        self.role = Role(role)
        self.manager_handle = manager_handle
        self.upstream_url = upstream_url.rstrip("/") if upstream_url else None
        self.genrm_backend_handle = genrm_backend_handle
        self._executor = InferenceRequestExecutor(
            self.role, manager_handle, httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=2048))
        )
        self._logger = get_logger(__name__)
        self._stream_cleanups: set[asyncio.Task] = set()

    @property
    def _client(self) -> httpx.AsyncClient:
        return self._executor.client

    async def _snapshot(self) -> RoleSnapshot:
        return await self._executor.snapshot()

    async def _target(self, payload: Mapping[str, Any] | None = None) -> str:
        target, _ = await self._resolve_target(payload)
        return target

    async def _resolve_target(self, payload: Mapping[str, Any] | None = None) -> tuple[str, str]:
        payload = payload or {}
        try:
            target, model_id, _ = await self._executor.resolve(
                model=payload.get("model"), route_key=payload.get("route_key")
            )
        except RoutingError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
                headers={"Retry-After": "1"} if exc.status_code == 503 else None,
            ) from exc
        return target, model_id

    async def engines(self, schema_version: int | None = None, status_filter: str | None = None) -> Any:
        """Serve the common discovery schema; ``schema_version=1`` keeps the
        legacy engine-group projection for older clients."""
        if schema_version not in (None, 1, 2):
            raise HTTPException(status_code=400, detail=f"Unsupported discovery schema_version {schema_version}")
        snapshot = await self._snapshot()
        if schema_version == 1:
            return snapshot.to_legacy_dict(status_filter)
        return snapshot.to_dict(status_filter)

    async def health(self) -> JSONResponse:
        """Report gateway liveness separately from model readiness."""
        try:
            snapshot = await self._snapshot()
        except Exception as exc:
            self._logger.warning("Inference discovery unavailable: %s", exc)
            return JSONResponse(
                {"status": "unavailable", "service": self.role.value, "model_status": "unknown"}, status_code=503
            )
        ready = any(model.admission and model.state and model.state.value == "ready" for model in snapshot.models)
        return JSONResponse(
            {
                "status": "healthy",
                "service": self.role.value,
                "model_status": "ready" if ready else "unavailable",
                "manager_epoch": snapshot.manager_epoch,
            }
        )

    async def models(self) -> dict[str, Any]:
        snapshot = await self._snapshot()
        return {
            "object": "list",
            "data": [
                {"id": model.model_id, "object": "model", "created": 0, "owned_by": "relax"}
                for model in snapshot.models
            ],
        }

    async def proxy(self, request: Request, path: str) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            return _json_error(400, f"Invalid JSON request body: {exc}", code="invalid_request")
        if not isinstance(payload, dict):
            return _json_error(400, "Request body must be a JSON object", code="invalid_request")

        messages_request = self.role == Role.GENRM and path == "generate" and "messages" in payload
        if messages_request and ("input_ids" in payload or "text" in payload):
            return _json_error(400, "messages cannot be combined with input_ids or text", code="invalid_request")
        if messages_request and (
            not isinstance(payload["messages"], list)
            or (payload.get("sampling_params") is not None and not isinstance(payload["sampling_params"], dict))
        ):
            return _json_error(400, "Invalid messages or sampling_params", code="invalid_request")

        # The caller's ID only traces the request; the Manager's permit holds
        # its admission.
        trace_id = request.headers.get("x-relax-request-id") or uuid4().hex
        try:
            admitted = await self._executor.admit(
                trace_id, model=payload.get("model"), route_key=payload.get("route_key")
            )
        except RoutingError as exc:
            return _routing_error(exc)
        response: Response | None = None
        try:
            payload.pop("route_key", None)
            wrap_response = False
            if messages_request:
                if self.genrm_backend_handle is None:
                    raise HTTPException(status_code=503, detail="GenRM adapter is not configured")
                payload = await self.genrm_backend_handle.prepare_generate_payload.remote(
                    admitted.model_id, payload["messages"], payload.get("sampling_params")
                )
                wrap_response = True
            if path == "generate":
                payload.pop("model", None)
            # Engines run this execution's own IDs, which abort names. A caller's
            # rid is only traced: equal or prefix rids of concurrent requests
            # would collide in SGLang's abort match.
            caller_rid = payload.get("rid")
            assign_backend_rids(admitted, payload)
            if caller_rid is not None:
                self._logger.info(
                    "Inference request %s (caller rid %s) runs as %s", admitted.trace_id, caller_rid, admitted.rid
                )
            response = await self._forward_admitted(
                request, path, admitted, json.dumps(payload).encode(), wrap_response=wrap_response
            )
            return response
        finally:
            # A stream releases its own request when it ends.
            if not isinstance(response, StreamingResponse):
                await self._executor.release(admitted)

    async def _forward_admitted(
        self, request: Request, path: str, admitted: AdmittedRequest, body: bytes, *, wrap_response: bool
    ) -> Response:
        headers = _forward_headers(request)
        try:
            upstream = await self._executor.send(
                admitted, lambda target: self._build_upstream(request, path, target, body, headers)
            )
        except RoutingError as exc:
            return _routing_error(exc)
        except httpx.RequestError as exc:
            return _json_error(502, f"Failed to connect to inference router: {exc}", code="upstream_unavailable")
        if upstream.headers.get("content-type", "").startswith("text/event-stream"):
            return self._stream_response(upstream, admitted)
        content = await self._executor.read(admitted, upstream)
        if wrap_response and upstream.is_success:
            data = {"response": json.loads(content).get("text", "").strip()}
            return JSONResponse(
                data, status_code=upstream.status_code, headers=_response_headers(upstream, body=False)
            )
        return Response(
            content=content, status_code=upstream.status_code, headers=_response_headers(upstream), media_type=None
        )

    def _build_upstream(
        self, request: Request, path: str, target: str, body: bytes, headers: dict[str, str]
    ) -> httpx.Request:
        upstream_url = f"{target}/{path.lstrip('/')}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"
        return self._client.build_request(request.method, upstream_url, content=body, headers=headers)

    async def proxy_backend(self, request: Request, path: str) -> Response:
        """Forward role control-plane endpoints to the internal service."""
        if self.upstream_url is None:
            return _json_error(404, "Gateway backend is not configured", code="backend_unavailable")
        upstream_request = self._build_upstream(
            request, path, self.upstream_url, await request.body(), _forward_headers(request)
        )
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            return _json_error(502, f"Failed to connect to inference router: {exc}", code="upstream_unavailable")
        if upstream.headers.get("content-type", "").startswith("text/event-stream"):
            return self._stream_response(upstream, None)
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(
            content=content, status_code=upstream.status_code, headers=_response_headers(upstream), media_type=None
        )

    def _stream_response(self, response: httpx.Response, admitted: AdmittedRequest | None) -> StreamingResponse:
        async def finish() -> None:
            try:
                await response.aclose()
            finally:
                if admitted is not None:
                    await self._executor.release(admitted)

        async def body():
            async for chunk in response.aiter_raw():
                yield chunk
            if admitted is not None:
                admitted.finished = True

        async def cleanup() -> None:
            # ASGI disconnect cancels awaits in the response's scope.
            task = asyncio.create_task(finish())
            self._stream_cleanups.add(task)
            task.add_done_callback(self._stream_cleanups.discard)
            await asyncio.shield(task)

        headers = {key: value for key, value in response.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS}
        return _GatewayStreamingResponse(
            body(), cleanup=cleanup, status_code=response.status_code, headers=headers, media_type="text/event-stream"
        )

    async def close(self) -> None:
        try:
            if self._stream_cleanups:
                await asyncio.gather(*self._stream_cleanups)
        finally:
            await self._client.aclose()


gateway_app = FastAPI()


# Routes live in the deployment class: ``serve.ingress`` serves a copy of
# ``gateway_app`` and binds each replica instance to its routes as ``self``, so
# nothing is looked up through the module-level app's state. ``deploy_gateway``
# sizes each role's concurrency; the class keeps a usable default for a direct
# bind instead of Ray Serve's default of 5.
@serve.deployment(ray_actor_options={"num_gpus": 0}, max_ongoing_requests=256)
@serve.ingress(gateway_app)
class InferenceGatewayDeployment(InferenceGateway):
    """Ray Serve deployment form of :class:`InferenceGateway`."""

    @gateway_app.get("/engines")
    async def _engines(self, schema_version: int | None = None, status_filter: str | None = None):
        return await self.engines(schema_version, status_filter)

    @gateway_app.get("/health")
    async def _health(self):
        return await self.health()

    @gateway_app.get("/v1/models")
    async def _models(self):
        return await self.models()

    @gateway_app.api_route("/generate", methods=["POST"])
    async def _generate(self, request: Request):
        return await self.proxy(request, "generate")

    @gateway_app.api_route("/v1/chat/completions", methods=["POST"])
    async def _chat(self, request: Request):
        return await self.proxy(request, "v1/chat/completions")

    @gateway_app.api_route("/chat/completions", methods=["POST"])
    async def _chat_legacy(self, request: Request):
        return await self.proxy(request, "v1/chat/completions")

    @gateway_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def _backend(self, request: Request, path: str):
        return await self.proxy_backend(request, path)


def gateway_deployment_name(role: str | Role) -> str:
    return f"{Role(role).value}_gateway"


def gateway_max_ongoing_requests(role: str | Role) -> int:
    """The concurrency a role's Gateway admits, the limit its callers already
    configure: GenRM and Rollout keep their former Serve ingress limits, and
    the Teacher matches the OPD client's connection limit."""
    limits = {
        Role.ROLLOUT: Envs.ROLLOUT_SERVE_MAX_ONGOING_REQUESTS,
        Role.GENRM: Envs.GENRM_SERVE_MAX_ONGOING_REQUESTS,
        Role.TEACHER: Envs.TEACHER_GATEWAY_MAX_ONGOING_REQUESTS,
    }
    return limits[Role(role)]


def deploy_gateway(
    role: str | Role,
    *,
    manager_handle: Any,
    upstream_url: str | None = None,
    genrm_backend_handle: Any | None = None,
) -> str:
    """Deploy the role's Gateway at ``/<role>`` and return its URL.

    The one way every inference role (Rollout, GenRM, Teacher) gets its
    ingress, whether or not a Service backend sits behind it.
    """
    from relax.utils.utils import get_serve_url

    role = Role(role)
    gateway = InferenceGatewayDeployment.options(max_ongoing_requests=gateway_max_ongoing_requests(role)).bind(
        role.value,
        manager_handle=manager_handle,
        upstream_url=upstream_url,
        genrm_backend_handle=genrm_backend_handle,
    )
    serve.run(gateway, name=gateway_deployment_name(role), route_prefix=f"/{role.value}")
    return get_serve_url(f"/{role.value}")


def delete_gateway(role: str | Role) -> None:
    serve.delete(gateway_deployment_name(role))


__all__ = [
    "InferenceGateway",
    "InferenceGatewayDeployment",
    "delete_gateway",
    "deploy_gateway",
    "gateway_app",
    "gateway_deployment_name",
    "gateway_max_ongoing_requests",
]

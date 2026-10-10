# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU HTTP gateway shared by rollout, GenRM, and Teacher services."""

import asyncio
import inspect
import itertools
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

import httpx
import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ray import serve
from starlette.types import Receive, Scope, Send

from relax.engine.inference.routing import RoutingError, resolve_model, select_target
from relax.engine.inference.types import Role, RoleSnapshot
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

_ABORT_TIMEOUT_S = 10.0


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


class InferenceGateway:
    """Role-neutral HTTP ingress which routes through a Manager snapshot.

    The gateway owns no GPU resources and keeps no availability state of its
    own. The task inference owner behind ``manager_handle`` is the only state
    source.
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
        self._client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=2048))
        # Rotates across direct-eligible replicas; a Router target ignores it.
        self._cursor = itertools.count()
        self._logger = get_logger(__name__)
        self._stream_cleanups: set[asyncio.Task] = set()

    async def _snapshot(self) -> RoleSnapshot:
        return await self._manager_call("snapshot", self.role)

    async def _target(self, payload: Mapping[str, Any] | None = None) -> str:
        target, _ = await self._resolve_target(payload)
        return target

    async def _resolve_target(self, payload: Mapping[str, Any] | None = None) -> tuple[str, str]:
        snapshot = await self._snapshot()
        payload = payload or {}
        requested = payload.get("model")
        # OpenAI clients send the served model's real name (e.g. "Qwen3-8B"), which the
        # router used to accept. A single-model role keeps serving it through its
        # default model; only a multi-model role must be named exactly.
        if (
            requested is not None
            and len(snapshot.models) == 1
            and snapshot.routing.default_model is not None
            and requested != snapshot.models[0].model_id
        ):
            requested = None
        try:
            model = resolve_model(
                snapshot,
                model=requested,
                route_key=payload.get("route_key"),
            )
            return select_target(model, cursor=next(self._cursor)).base_url.rstrip("/"), model.model_id
        except RoutingError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
                headers={"Retry-After": "1"} if exc.status_code == 503 else None,
            ) from exc

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

        try:
            target, model_id = await self._resolve_target(payload)
        except HTTPException as exc:
            return _json_error(
                exc.status_code, str(exc.detail), code="unavailable" if exc.status_code == 503 else "routing_error"
            )

        try:
            request_id = await self._admit(request, model_id)
        except HTTPException as exc:
            return _json_error(exc.status_code, str(exc.detail), code="unavailable")
        rid = request_id
        response: Response | None = None
        finished = False
        try:
            payload.pop("route_key", None)
            wrap_response = False
            if messages_request:
                if self.genrm_backend_handle is None:
                    raise HTTPException(status_code=503, detail="GenRM adapter is not configured")
                payload = await self.genrm_backend_handle.prepare_generate_payload.remote(
                    model_id, payload["messages"], payload.get("sampling_params")
                )
                wrap_response = True
            if path == "generate":
                payload.pop("model", None)
            # Both native and OpenAI requests must carry the ID used by abort.
            rid = payload["rid"] = payload.get("rid") or request_id
            response, finished = await self._forward(
                request,
                path,
                target,
                json.dumps(payload).encode(),
                wrap_response=wrap_response,
                request_id=rid,
                admission_id=request_id,
            )
            return response
        finally:
            # A stream completes its own request when it ends.
            if not isinstance(response, StreamingResponse):
                if not finished:
                    await self._abort_upstream(target, rid)
                await self._manager_call("complete_request", request_id)

    async def _manager_call(self, method: str, *args: Any) -> Any:
        value = getattr(self.manager_handle, method).remote(*args)
        if inspect.isawaitable(value):
            return await value
        return await asyncio.to_thread(ray.get, value)

    async def _admit(self, request: Request, model_id: str) -> str:
        request_id = request.headers.get("x-relax-request-id") or uuid4().hex
        try:
            return await self._manager_call("admit_request", self.role, model_id, request_id)
        except Exception as exc:
            self._logger.warning("Inference request admission failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Inference request is not admitted",
                headers={"Retry-After": "1"},
            ) from exc

    async def _abort_upstream(self, target: str, request_id: str | None) -> None:
        """Send an abort for ``request_id`` to the workers behind ``target``.

        The router does not proxy aborts, so it is broadcast to its workers,
        the same way the rollout and Agentic paths do it. An ACK only proves
        the scheduler received it, never that the request left the running
        batch.
        """
        if not request_id:
            return
        try:
            urls = await self._worker_base_urls(target)
        except Exception as exc:
            self._logger.warning("Could not resolve engines behind %s to abort %s: %s", target, request_id, exc)
            return
        results = await asyncio.gather(
            *(
                self._client.post(f"{url}/abort_request", json={"rid": request_id}, timeout=_ABORT_TIMEOUT_S)
                for url in urls
            ),
            return_exceptions=True,
        )
        for url, result in zip(urls, results, strict=False):
            if isinstance(result, BaseException):
                self._logger.warning("Abort of %s at %s failed: %s", request_id, url, result)

    async def _worker_base_urls(self, target: str) -> list[str]:
        from relax.utils.http_utils import router_worker_base_urls

        base = target.rstrip("/")
        try:
            response = await self._client.get(f"{base}/workers", timeout=_ABORT_TIMEOUT_S)
            response.raise_for_status()
            urls = [worker["url"] for worker in response.json()["workers"] if worker.get("url")]
        except Exception:
            response = await self._client.get(f"{base}/list_workers", timeout=_ABORT_TIMEOUT_S)
            response.raise_for_status()
            urls = list(response.json()["urls"])
        return router_worker_base_urls(urls)

    async def proxy_backend(self, request: Request, path: str) -> Response:
        """Forward role control-plane endpoints to the internal service."""
        if self.upstream_url is None:
            return _json_error(404, "Gateway backend is not configured", code="backend_unavailable")
        response, _ = await self._forward(request, path, self.upstream_url, await request.body())
        return response

    async def _forward(
        self,
        request: Request,
        path: str,
        target: str,
        body: bytes,
        *,
        wrap_response: bool = False,
        request_id: str | None = None,
        admission_id: str | None = None,
    ) -> tuple[Response, bool]:
        """Forward one request; also report whether the upstream finished it.

        A streamed response hands ``request_id`` to the stream, which completes
        or aborts it when the stream ends.
        """
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() not in {"host", "content-length"}
        }
        headers[GATEWAY_REQUEST_HEADER] = "1"
        upstream_url = f"{target}/{path.lstrip('/')}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"
        upstream = self._client.build_request(request.method, upstream_url, content=body, headers=headers)
        response: httpx.Response | None = None
        try:
            response = await self._client.send(upstream, stream=True)
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                stream_response = self._stream_response(response, request_id, admission_id, target)
                response = None
                return stream_response, True
            content = await response.aread()
            response_headers = {
                key: value for key, value in response.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            if wrap_response and response.is_success:
                data = json.loads(content)
                data = {"response": data.get("text", "").strip()}
                response_headers.pop("content-length", None)
                response_headers.pop("content-encoding", None)
                return JSONResponse(data, status_code=response.status_code, headers=response_headers), True
            return (
                Response(content=content, status_code=response.status_code, headers=response_headers, media_type=None),
                True,
            )
        except httpx.RequestError as exc:
            return (
                _json_error(502, f"Failed to connect to inference router: {exc}", code="upstream_unavailable"),
                False,
            )
        finally:
            if response is not None and not response.is_closed:
                await response.aclose()

    def _stream_response(
        self, response: httpx.Response, request_id: str | None, admission_id: str | None, target: str
    ) -> StreamingResponse:
        completed = False

        async def finish() -> None:
            try:
                await response.aclose()
            finally:
                if admission_id is not None:
                    try:
                        if not completed:
                            await self._abort_upstream(target, request_id)
                    finally:
                        await self._manager_call("complete_request", admission_id)

        async def body():
            nonlocal completed
            async for chunk in response.aiter_raw():
                yield chunk
            completed = True

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
# nothing is looked up through the module-level app's state.
@serve.deployment(ray_actor_options={"num_gpus": 0}, max_ongoing_requests=128)
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
    gateway = InferenceGatewayDeployment.bind(
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
]

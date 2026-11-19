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
        try:
            if schema_version == 1:
                return snapshot.to_legacy_dict(status_filter)
            return snapshot.to_dict(status_filter)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

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
        try:
            content = await self._executor.read(admitted, upstream)
        except httpx.RequestError as exc:
            return _json_error(502, f"Failed to read inference response: {exc}", code="upstream_unavailable")
        if wrap_response and upstream.is_success:
            data = {"response": json.loads(content).get("text", "").strip()}
            return JSONResponse(
                data, status_code=upstream.status_code, headers=_response_headers(upstream, body=False)
            )
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream, body=False),
            media_type=None,
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
            content=content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream, body=False),
            media_type=None,
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
                admitted.finished = response.status_code < 500

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


# All public role ingresses use this deployment; role-specific code remains
# payload/workload adaptation and never grants admission itself.
app = FastAPI()


class _LocalGenRMAdapter:
    def __init__(self, gateway: Any) -> None:
        self.prepare_generate_payload = _LocalAdapterMethod(gateway.prepare_generate_payload)


class _LocalAdapterMethod:
    def __init__(self, method: Any) -> None:
        self.remote = method


@serve.deployment(name="InferenceGateway")
@serve.ingress(app)
class InferenceGatewayService(InferenceGateway):
    def __init__(
        self,
        healthy: Any,
        pg: Any,
        num_gpus: int,
        config: Any,
        role: str,
        runtime_env: Any = None,
        data_source: Any = None,
    ) -> None:
        manager = config._rollout_engine_manager
        super().__init__(role, manager_handle=manager)
        self.config = config
        self._workload = None
        self._healthy = True
        self.step = 0
        self._tokenizers: dict[str, Any] = {}
        if self.role is Role.ROLLOUT:
            from relax.components.rollout import RolloutWorkload

            self._workload = RolloutWorkload(healthy, pg, config, data_source, runtime_env)
        elif self.role is Role.GENRM:
            from relax.utils.data.processing_utils import load_tokenizer

            self._tokenizers = {
                key: load_tokenizer(spec["model_path"], trust_remote_code=True)
                for key, spec in config._genrm_instances_resolved.items()
            }
            self.genrm_backend_handle = _LocalGenRMAdapter(self)

    def get_genrm_manager(self, route_key: str | None = None) -> Any:
        managers = self.config._genrm_managers
        if route_key is None and len(managers) == 1:
            return next(iter(managers.values()))
        return managers[route_key or "__default__"]

    def set_barriers(self, *args: Any, **kwargs: Any) -> Any:
        return self._workload.set_barriers(*args, **kwargs)

    @app.get("/evaluate")
    async def evaluate(self, train_step: int) -> Any:
        return await self._workload.evaluate(train_step)

    @app.get("/predict")
    async def predict(self, train_step: int) -> Any:
        return await self._workload.predict(train_step)

    @app.get("/can_do_update_weight_for_async")
    async def can_do_update_weight_for_async(self) -> Any:
        return await self._workload.can_do_update_weight_for_async()

    @app.get("/end_update_weight")
    async def end_update_weight(self) -> Any:
        return await self._workload.end_update_weight()

    @app.get("/is_eval_done")
    async def is_eval_done(self) -> Any:
        return await self._workload.is_eval_done()

    async def prepare_generate_payload(self, model: str, messages: list, sampling: dict | None) -> dict:
        spec = self.config._genrm_instances_resolved[model]
        defaults = spec.get("sampling_config", {})
        tokens = await asyncio.to_thread(
            self._tokenizers[model].apply_chat_template,
            messages,
            return_dict=False,
            tokenize=True,
            add_generation_prompt=True,
            **(defaults.get("chat_template_kwargs") or {}),
        )
        params = {
            "temperature": defaults.get("temperature", 0.2),
            "top_p": defaults.get("top_p", 1.0),
            "top_k": defaults.get("top_k", -1),
            "max_new_tokens": defaults.get("max_response_len", 1024),
        }
        params.update(sampling or {})
        return {"input_ids": list(tokens), "sampling_params": params}

    async def run(self) -> Any:
        if self._workload is not None:
            return await self._workload.run()
        return None

    async def stop(self) -> None:
        if self._workload is not None:
            await self._workload.stop()

    def get_step(self) -> int:
        return self.step if self._workload is None else self._workload.get_step()

    def set_step(self, step: int) -> None:
        self.step = step
        if self._workload is not None:
            self._workload.set_step(step)

    def is_healthy(self) -> bool:
        return self._healthy if self._workload is None else self._workload._healthy

    def get_rollout_manager(self) -> Any:
        if self._workload is None:
            raise RuntimeError("Only Rollout owns a workload")
        return self._workload.get_rollout_manager()

    async def close(self) -> None:
        await self.stop()
        await super().close()

    @app.get("/get_step")
    def http_get_step(self) -> dict:
        return {"step": self.get_step()}

    @app.post("/set_step")
    def http_set_step(self, step: int) -> dict:
        self.set_step(step)
        return {"status": "ok"}

    @app.post("/stop_service")
    async def http_stop(self) -> dict:
        await self.stop()
        return {"status": "ok"}

    @app.post("/scale_out")
    async def scale_out(self, request: Request) -> Any:
        from relax.components.rollout import ScaleOutAPIRequest

        return await self._workload.scale_out(ScaleOutAPIRequest(**await request.json()))

    @app.post("/scale_in")
    async def scale_in(self, request: Request) -> Any:
        from relax.components.rollout import ScaleInAPIRequest

        return await self._workload.scale_in(ScaleInAPIRequest(**await request.json()))

    @app.get("/scale_out/{request_id}")
    async def scale_out_status(self, request_id: str) -> Any:
        return await self._workload.get_scale_out_status(request_id)

    @app.get("/scale_in/{request_id}")
    async def scale_in_status(self, request_id: str) -> Any:
        return await self._workload.get_scale_in_status(request_id)

    @app.get("/scale_out")
    async def list_scale_out_requests(self, model_name: str | None = None, status: str | None = None) -> Any:
        return await self._workload.list_scale_out_requests(model_name, status)

    @app.get("/scale_in")
    async def list_scale_in_requests(self, model_name: str | None = None, status: str | None = None) -> Any:
        return await self._workload.list_scale_in_requests(model_name, status)

    @app.post("/scale_out/{request_id}/cancel")
    async def cancel_scale_out(self, request_id: str) -> Any:
        return await self._workload.cancel_scale_out(request_id)

    @app.post("/scale_out_cancel")
    async def cancel_all_scale_out_requests(self, request: Request) -> Any:
        from relax.components.rollout import CancelAllScaleOutRequestsRequest

        return await self._workload.cancel_all_scale_out_requests(
            CancelAllScaleOutRequestsRequest(**await request.json())
        )

    @app.get("/recover_rollout_engines")
    async def recover_rollout_engines(self) -> Any:
        return await self._workload.recover_rollout_engines()

    @app.get("/engines")
    async def get_engines(self, schema_version: int = 2, status_filter: str | None = None) -> Any:
        return await self.engines(schema_version, status_filter)

    @app.get("/health")
    async def get_health(self) -> JSONResponse:
        return await self.health()

    @app.get("/metrics")
    async def metrics(self) -> dict[str, Any]:
        snapshot = await self._snapshot()
        return {
            "role": self.role.value,
            "manager_epoch": snapshot.manager_epoch,
            "topology_revision": snapshot.topology_revision,
            "models": {
                model.model_id: {
                    "num_engines": len(model.replicas),
                    "admission": model.admission,
                    "state": model.state.value if model.state else "unknown",
                }
                for model in snapshot.models
            },
        }

    @app.get("/v1/models")
    async def get_models(self) -> dict:
        return await self.models()

    @app.post("/generate")
    async def generate(self, request: Request) -> Response:
        return await self.proxy(request, "generate")

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request) -> Response:
        return await self.proxy(request, "v1/chat/completions")

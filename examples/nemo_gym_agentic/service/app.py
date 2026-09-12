# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FastAPI application for the reference shared NeMo Gym Gateway."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..app.protocol import PROTOCOL_VERSION, ProtocolValidationError
from .callback_provider import CallbackProvider, CallbackRequestError, CallbackResponse, CallbackUpstreamError
from .config import GatewayConfigError, GatewaySettings
from .registry import (
    AdmissionRejected,
    CallbackUnavailable,
    GatewayRegistry,
    TrialConflict,
    TrialNotFound,
)
from .run_adapter import HttpNemoGymRunAdapter, RunAdapter
from .verbose_logging import log_verbose_payload


def create_app(
    *,
    settings: GatewaySettings,
    adapter: RunAdapter | None = None,
    callback_transport: Any = None,
) -> FastAPI:
    run_adapter = adapter or HttpNemoGymRunAdapter(
        settings.environments.values(),
        artifact_root=settings.artifact_root,
    )
    registry = GatewayRegistry(settings=settings, adapter=run_adapter)
    callback_provider = CallbackProvider(
        registry,
        transport=callback_transport,
        proxy=settings.callback_proxy,
        timeout_s=settings.callback_timeout_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await registry.start()
        try:
            yield
        finally:
            await registry.close()
            await callback_provider.close()

    app = FastAPI(title="Relax NeMo Gym Gateway", version=PROTOCOL_VERSION, lifespan=lifespan)
    app.state.registry = registry

    @app.exception_handler(CallbackUnavailable)
    @app.exception_handler(CallbackRequestError)
    @app.exception_handler(CallbackUpstreamError)
    async def callback_error(_: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, CallbackUnavailable):
            return JSONResponse(status_code=410, content={"detail": "callback capability is unavailable"})
        status_code = 422 if isinstance(exc, CallbackRequestError) else 502
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return _service_metadata(registry, ready=None)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = await registry.ready()
        return JSONResponse(
            status_code=200 if ready else 503,
            content=_service_metadata(registry, ready=ready),
        )

    @app.post("/v1/trials")
    async def create_trial(request: Request) -> JSONResponse:
        payload = await _json_body(request)
        log_verbose_payload("request", payload, route="/v1/trials")
        try:
            result = await registry.create(payload)
        except ProtocolValidationError as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=422)
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except GatewayConfigError as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=400)
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except TrialConflict as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=409)
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except AdmissionRejected as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=429)
            raise HTTPException(status_code=429, detail=str(exc)) from None
        log_verbose_payload("response", result, route="/v1/trials", status=202)
        return JSONResponse(status_code=202, content=result)

    @app.get("/v1/trials/{request_id}")
    async def get_trial(request_id: str) -> dict[str, Any]:
        try:
            return await registry.get(request_id)
        except TrialNotFound:
            raise HTTPException(status_code=404, detail="trial not found") from None

    @app.post("/v1/trials/{request_id}/renew", status_code=204)
    async def renew_trial(request_id: str) -> Response:
        try:
            await registry.renew(request_id)
        except TrialNotFound:
            raise HTTPException(status_code=404, detail="trial not found") from None
        return Response(status_code=204)

    @app.post("/v1/trials/{request_id}/abort")
    async def abort_trial(request_id: str) -> JSONResponse:
        log_verbose_payload("request", {}, route="/v1/trials/{request_id}/abort", request_id=request_id)
        try:
            result = await registry.abort(request_id)
        except TrialNotFound:
            log_verbose_payload(
                "response",
                {"detail": "trial not found"},
                route="/v1/trials/{request_id}/abort",
                request_id=request_id,
                status=404,
            )
            raise HTTPException(status_code=404, detail="trial not found") from None
        log_verbose_payload(
            "response",
            result,
            route="/v1/trials/{request_id}/abort",
            request_id=request_id,
            status=202,
        )
        return JSONResponse(status_code=202, content=result)

    @app.post("/ng-rollout/{rollout_id}/v1/chat/completions")
    async def callback_chat_completions(rollout_id: str, request: Request) -> Response:
        payload = await _json_body(request)
        log_verbose_payload(
            "request",
            payload,
            route="/ng-rollout/{rollout_id}/v1/chat/completions",
            rollout_id=rollout_id,
        )
        result = await callback_provider.request(rollout_id, payload, resource="chat/completions")
        log_verbose_payload(
            "response",
            result.payload,
            route="/ng-rollout/{rollout_id}/v1/chat/completions",
            rollout_id=rollout_id,
            status=result.status_code,
        )
        return _callback_response(result)

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def callback_responses(rollout_id: str, request: Request) -> Response:
        payload = await _json_body(request)
        log_verbose_payload(
            "request",
            payload,
            route="/ng-rollout/{rollout_id}/v1/responses",
            rollout_id=rollout_id,
        )
        result = await callback_provider.request(rollout_id, payload, resource="responses")
        log_verbose_payload(
            "response",
            result.payload,
            route="/ng-rollout/{rollout_id}/v1/responses",
            rollout_id=rollout_id,
            status=result.status_code,
        )
        return _callback_response(result)

    @app.post("/ng-rollout/{rollout_id}/v1/messages")
    async def callback_messages(rollout_id: str, request: Request) -> Response:
        payload = await _json_body(request)
        log_verbose_payload(
            "request",
            payload,
            route="/ng-rollout/{rollout_id}/v1/messages",
            rollout_id=rollout_id,
        )
        if isinstance(payload, dict) and payload.get("stream"):
            return StreamingResponse(
                _messages_sse(callback_provider, rollout_id, payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = await callback_provider.request(rollout_id, payload, resource="messages")
        log_verbose_payload(
            "response",
            result.payload,
            route="/ng-rollout/{rollout_id}/v1/messages",
            rollout_id=rollout_id,
            status=result.status_code,
        )
        return JSONResponse(status_code=result.status_code, content=result.payload)

    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=GatewaySettings.from_env())


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from None


async def _messages_sse(
    provider: CallbackProvider,
    rollout_id: str,
    payload: dict[str, Any],
) -> AsyncIterator[str]:
    task = asyncio.create_task(provider.request(rollout_id, payload, resource="messages"))
    ping = _sse_event("ping", {"type": "ping"})
    try:
        yield ping
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=15.0)
            if not done:
                yield ping
        try:
            result = task.result()
        except (CallbackUnavailable, CallbackRequestError, CallbackUpstreamError) as exc:
            yield _sse_error(str(exc))
            return
        if result.status_code >= 400:
            if isinstance(result.payload, dict) and result.payload.get("type") == "error":
                yield _sse_event("error", result.payload)
            else:
                yield _sse_error("Upstream model callback failed")
            return
        if not isinstance(result.payload, str):
            yield _sse_error("Upstream model callback returned a non-SSE response")
            return
        yield result.payload
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _callback_response(result: CallbackResponse) -> Response:
    if isinstance(result.payload, str):
        return Response(
            content=result.payload,
            status_code=result.status_code,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(status_code=result.status_code, content=result.payload)


def _sse_error(message: str) -> str:
    return _sse_event("error", {"type": "error", "error": {"type": "api_error", "message": message}})


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _service_metadata(registry: GatewayRegistry, *, ready: bool | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "service_epoch": registry.service_epoch,
        "gym_commit": registry.settings.gym_commit,
        "config_fingerprint": registry.settings.config_fingerprint,
        **registry.stats(),
    }
    if ready is not None:
        payload["ready"] = ready
    return payload

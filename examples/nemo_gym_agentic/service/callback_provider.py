# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Request-scoped model callback bridge from NeMo Gym to Relax."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .registry import CallbackTarget, GatewayRegistry
from .verbose_logging import _redact_payload, log_verbose_payload, logger


_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


def _should_retry(response: httpx.Response) -> bool:
    if response.status_code not in _RETRYABLE_STATUS_CODES:
        return False
    try:
        payload = response.json()
    except ValueError:
        return True
    error = payload.get("error") if isinstance(payload, dict) else None
    # A rendering failure can happen after the response was committed to the
    # SessionForest. Replaying that request would generate a second branch.
    return not (isinstance(error, dict) and error.get("message") == "Internal error while rendering agentic response.")


def _error_summary(response: httpx.Response) -> str:
    try:
        error_payload = response.json()
    except ValueError:
        error_payload = response.text
    if isinstance(error_payload, dict):
        error_payload = error_payload.get("error", error_payload.get("detail", error_payload))
    return json.dumps(_redact_payload(error_payload), ensure_ascii=False)[:4000]


class CallbackRequestError(ValueError):
    pass


class CallbackUpstreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallbackResponse:
    status_code: int
    payload: Any


class CallbackProvider:
    def __init__(
        self,
        registry: GatewayRegistry,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        proxy: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self._registry = registry
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            transport=transport,
            proxy=proxy,
            trust_env=False,
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def request(self, rollout_id: str, payload: Any, *, resource: str) -> CallbackResponse:
        if not isinstance(payload, dict):
            raise CallbackRequestError("Callback body must be a JSON object")
        async with self._registry.callback_target(rollout_id) as target:
            callback_payload = dict(payload)
            callback_payload.update(target.generation.get("sampling_params", {}))
            callback_payload["model"] = target.model
            timeout_s = min(target.remaining_s, self._timeout_s)
            try:
                # One budget covers requests, response reads, and retry backoff.
                # Cancelling this trial also cancels the request or backoff below.
                response = await asyncio.wait_for(
                    self._post(
                        target,
                        callback_payload,
                        resource=resource,
                        rollout_id=rollout_id,
                        deadline=time.monotonic() + timeout_s,
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError as exc:
                logger.warning("NeMo Gym callback deadline exceeded rollout_id=%s resource=%s", rollout_id, resource)
                raise CallbackUpstreamError("Upstream model callback deadline exceeded") from exc
            if resource == "responses" and response.status_code < 400 and isinstance(response.payload, dict):
                for item in response.payload.get("output", []):
                    if item.get("type") == "reasoning" and item.get("content"):
                        item["summary"] = [
                            {"type": "summary_text", "text": content["text"]}
                            for content in item.get("content", [])
                            if content.get("type") == "reasoning_text"
                        ]
                response.payload.setdefault("parallel_tool_calls", callback_payload.get("parallel_tool_calls", True))
                response.payload.setdefault("tool_choice", callback_payload.get("tool_choice", "auto"))
                response.payload.setdefault("tools", callback_payload.get("tools", []))
                usage = response.payload.get("usage")
                if isinstance(usage, dict):
                    usage.setdefault("input_tokens_details", {"cached_tokens": 0})
                    usage.setdefault("output_tokens_details", {"reasoning_tokens": 0})
            return response

    async def _post(
        self,
        target: CallbackTarget,
        payload: dict[str, Any],
        *,
        resource: str,
        rollout_id: str,
        deadline: float,
    ) -> CallbackResponse:
        url = _api_url(target.base_url, resource)
        headers = dict(target.headers)
        headers[target.api_key_header] = f"{target.api_key_prefix}{target.api_key}"
        route = f"upstream/v1/{resource}"
        log_verbose_payload("request", payload, route=route)
        started = time.monotonic()
        attempt = 0
        delay_s = 1.0
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise asyncio.TimeoutError
            timeout = httpx.Timeout(
                remaining_s,
                connect=min(remaining_s, 30.0),
                pool=min(remaining_s, 30.0),
                write=min(remaining_s, 30.0),
            )
            attempt += 1
            try:
                response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
            except httpx.RequestError as exc:
                # Delivery is ambiguous after a transport failure. Only retry
                # explicit server errors, never replay tools or the whole /run.
                raise CallbackUpstreamError("Upstream model callback transport failed") from exc
            if not _should_retry(response):
                break
            await response.aclose()
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise asyncio.TimeoutError
            wait_s = min(delay_s, remaining_s)
            logger.warning(
                "Retrying NeMo Gym model callback rollout_id=%s resource=%s attempt=%d status=%d "
                "wait_s=%.2f remaining_s=%.2f error=%s",
                rollout_id,
                resource,
                attempt,
                response.status_code,
                wait_s,
                remaining_s,
                _error_summary(response),
            )
            await asyncio.sleep(wait_s)
            delay_s = min(delay_s * 2, 8.0)
        if attempt > 1:
            logger.info(
                "NeMo Gym callback retries finished rollout_id=%s resource=%s attempts=%d status=%d elapsed_s=%.2f",
                rollout_id,
                resource,
                attempt,
                response.status_code,
                time.monotonic() - started,
            )
        if response.status_code >= 400:
            logger.warning(
                "NeMo Gym model callback rejected rollout_id=%s resource=%s status=%d error=%s",
                rollout_id,
                resource,
                response.status_code,
                _error_summary(response),
            )
        if payload.get("stream") and response.headers.get("content-type", "").startswith("text/event-stream"):
            response_payload: Any = response.text
        else:
            try:
                response_payload = response.json()
            except ValueError as exc:
                raise CallbackUpstreamError("Upstream model callback returned a non-JSON response") from exc
            if not isinstance(response_payload, dict):
                raise CallbackUpstreamError("Upstream model callback response must be a JSON object")
        log_verbose_payload(
            "response",
            response_payload,
            route=route,
            status=response.status_code,
        )
        return CallbackResponse(status_code=response.status_code, payload=response_payload)


def _api_url(base_url: str, resource: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = f"/{resource}" if normalized.endswith("/v1") else f"/v1/{resource}"
    return f"{normalized}{suffix}"

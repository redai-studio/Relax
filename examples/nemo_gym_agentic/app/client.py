# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-session thin client for a shared, long-lived NeMo Gym gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .protocol import (
    InterruptPolicy,
    ModelEndpoint,
    TrialRequest,
    TrialResult,
    TrialStatus,
    stable_request_id,
)
from .result import to_relax_output, write_relax_output


class GatewayRequestError(RuntimeError):
    """A sanitized gateway transport or HTTP error."""


class TrialCancelled(RuntimeError):
    """The local managed session was cancelled and its remote trial aborted."""


class TrialFailed(RuntimeError):
    """The remote trial reached a failed terminal state."""


@dataclass(frozen=True)
class ClientConfig:
    gateway_url: str
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 30.0
    poll_interval_s: float = 1.0
    abort_timeout_s: float = 2.0
    retry_backoff_s: float = 0.25
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "ClientConfig":
        gateway_url = _required_env("NEMO_GYM_URL").rstrip("/")
        if not gateway_url.startswith(("http://", "https://")):
            raise ValueError("NEMO_GYM_URL must be an absolute http(s) URL")
        return cls(
            gateway_url=gateway_url,
            startup_timeout_s=_positive_float_env("NEMO_GYM_STARTUP_TIMEOUT_S", 30.0),
            request_timeout_s=_positive_float_env("NEMO_GYM_REQUEST_TIMEOUT_S", 30.0),
            poll_interval_s=_positive_float_env("NEMO_GYM_POLL_INTERVAL_S", 1.0),
            abort_timeout_s=_positive_float_env("NEMO_GYM_ABORT_TIMEOUT_S", 2.0),
            retry_backoff_s=_non_negative_float_env("NEMO_GYM_RETRY_BACKOFF_S", 0.25),
            max_retries=_non_negative_int_env("NEMO_GYM_MAX_RETRIES", 2),
        )


class GatewayClient:
    def __init__(self, config: ClientConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.gateway_url,
            timeout=httpx.Timeout(config.request_timeout_s, connect=min(10.0, config.request_timeout_s)),
            transport=transport,
            trust_env=False,
        )

    async def __aenter__(self) -> "GatewayClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self._client.aclose()

    async def create(self, request: TrialRequest) -> TrialResult:
        response = await self._request(
            "POST",
            "/v1/trials",
            operation="create",
            timeout_s=self._config.startup_timeout_s,
            json_payload=request.to_payload(),
        )
        return _parse_result_response(response, expected_request_id=request.request_id)

    async def get(self, request_id: str) -> TrialResult:
        response = await self._request("GET", f"/v1/trials/{request_id}", operation="get")
        return _parse_result_response(response, expected_request_id=request_id)

    async def renew(self, request_id: str) -> None:
        await self._request("POST", f"/v1/trials/{request_id}/renew", operation="renew")

    async def abort(self, request_id: str) -> None:
        await self._request(
            "POST",
            f"/v1/trials/{request_id}/abort",
            operation="abort",
            timeout_s=self._config.abort_timeout_s,
            max_retries=0,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        timeout_s: float | None = None,
        json_payload: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        retry_count = self._config.max_retries if max_retries is None else max_retries
        for attempt in range(retry_count + 1):
            request_kwargs: dict[str, Any] = {}
            if timeout_s is not None:
                request_kwargs["timeout"] = timeout_s
            if json_payload is not None:
                request_kwargs["json"] = json_payload
            try:
                response = await self._client.request(method, path, **request_kwargs)
            except httpx.RequestError as exc:
                if attempt >= retry_count:
                    raise GatewayRequestError(f"Gateway {operation} request failed after transport retries") from exc
            else:
                if response.is_success:
                    return response
                if response.status_code not in {408, 429} and response.status_code < 500:
                    raise GatewayRequestError(f"Gateway {operation} request failed with HTTP {response.status_code}")
                if attempt >= retry_count:
                    raise GatewayRequestError(
                        f"Gateway {operation} request failed with HTTP {response.status_code} after retries"
                    )
            if self._config.retry_backoff_s > 0:
                await asyncio.sleep(self._config.retry_backoff_s * (2**attempt))
        raise AssertionError("Gateway retry loop exhausted without returning or raising")


async def run_trial(
    *,
    client: GatewayClient,
    request: TrialRequest,
    stop_event: asyncio.Event,
    poll_interval_s: float,
) -> TrialResult:
    result = await _request_or_stop(
        request_coro=client.create(request),
        client=client,
        request_id=request.request_id,
        stop_event=stop_event,
    )
    if result.status.is_terminal:
        return _terminal_result_or_raise(result)

    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            client=client,
            request_id=request.request_id,
            lease_s=request.lease_s,
            stop_event=heartbeat_stop,
        )
    )
    try:
        while True:
            await _sleep_or_stop(stop_event=stop_event, delay_s=poll_interval_s)
            if stop_event.is_set():
                await _best_effort_abort(client, request.request_id)
                raise TrialCancelled("Managed session cancelled; remote trial abort requested")
            if heartbeat_task.done():
                try:
                    heartbeat_task.result()
                except GatewayRequestError:
                    await _best_effort_abort(client, request.request_id)
                    raise

            result = await _request_or_stop(
                request_coro=client.get(request.request_id),
                client=client,
                request_id=request.request_id,
                stop_event=stop_event,
            )
            if result.status.is_terminal:
                return _terminal_result_or_raise(result)
    except asyncio.CancelledError:
        await _best_effort_abort(client, request.request_id)
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


async def _heartbeat_loop(
    *,
    client: GatewayClient,
    request_id: str,
    lease_s: float,
    stop_event: asyncio.Event,
) -> None:
    interval_s = max(0.05, lease_s / 3.0)
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            await client.renew(request_id)
            continue
        return


async def _request_or_stop(
    *,
    request_coro: Any,
    client: GatewayClient,
    request_id: str,
    stop_event: asyncio.Event,
) -> TrialResult:
    request_task = asyncio.create_task(request_coro)
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait({request_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and stop_event.is_set():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            await _best_effort_abort(client, request_id)
            raise TrialCancelled("Managed session cancelled; remote trial abort requested")
        return request_task.result()
    except asyncio.CancelledError:
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        await _best_effort_abort(client, request_id)
        raise
    finally:
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


async def _sleep_or_stop(*, stop_event: asyncio.Event, delay_s: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=delay_s)
    except TimeoutError:
        return


async def _best_effort_abort(client: GatewayClient, request_id: str) -> None:
    try:
        await client.abort(request_id)
    except GatewayRequestError:
        return


def _terminal_result_or_raise(result: TrialResult) -> TrialResult:
    if result.status in {TrialStatus.COMPLETED, TrialStatus.TRUNCATED}:
        return result
    if result.status == TrialStatus.ABORTED:
        raise TrialCancelled("Remote trial was aborted")
    error_code = result.error_code or "unknown"
    raise TrialFailed(f"Remote trial failed with error code {error_code!r}")


def build_trial_request(session_input: dict[str, Any], environ: dict[str, str]) -> TrialRequest:
    session_id = _required_mapping_value(environ, "RELAX_SESSION_ID")
    invocation_id = _required_mapping_value(environ, "RELAX_SESSION_IO_DIR")
    metadata = session_input.get("metadata") or {}
    environment = metadata.get("environment") or _required_mapping_value(environ, "NEMO_GYM_ENVIRONMENT")
    config = metadata.get("config") or environ.get("NEMO_GYM_CONFIG") or environment
    model = environ.get("NEMO_GYM_MODEL") or environ.get("OPENAI_MODEL") or "model"
    attempt = _positive_int_mapping_value(environ, "NEMO_GYM_ATTEMPT", 1)
    generation = {
        "responses_create_params": _json_object_env(environ, "NEMO_GYM_RESPONSES_CREATE_PARAMS", {}),
        "sampling_params": _json_object_env(environ, "NEMO_GYM_SAMPLING_PARAMS", {}),
    }
    return TrialRequest(
        request_id=stable_request_id(session_id, attempt, invocation_id=invocation_id),
        session_id=session_id,
        group_id=environ.get("RELAX_GROUP_ID") or "unknown",
        rollout_mode=environ.get("RELAX_ROLLOUT_MODE") or "train",
        environment=environment,
        config=config,
        task=session_input,
        model_endpoint=ModelEndpoint(
            base_url=_required_mapping_value(environ, "RELAX_BASE_URL").rstrip("/"),
            api_key=session_id,
            model=model,
        ),
        generation=generation,
        interrupt_policy=InterruptPolicy(environ.get("NEMO_GYM_INTERRUPT_POLICY", "protected")),
        attempt=attempt,
        deadline_s=_positive_float_mapping_value(environ, "NEMO_GYM_DEADLINE_S", 1800.0),
        lease_s=_positive_float_mapping_value(environ, "NEMO_GYM_LEASE_S", 60.0),
        metadata={
            "integration": "relax",
        },
    )


def read_session_input(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RELAX_INPUT_JSON must contain a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("RELAX_INPUT_JSON must contain a messages array")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("RELAX_INPUT_JSON metadata must be a JSON object")
    payload.setdefault("metadata", {})
    return payload


async def async_main(*, input_json: str, output_json: str) -> None:
    session_input = read_session_input(input_json)
    config = ClientConfig.from_env()
    request = build_trial_request(session_input, dict(os.environ))
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed_signals.append(sig)
    try:
        async with GatewayClient(config) as client:
            result = await run_trial(
                client=client,
                request=request,
                stop_event=stop_event,
                poll_interval_s=config.poll_interval_s,
            )
        write_relax_output(output_json, to_relax_output(result))
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Relax session through a shared NeMo Gym gateway.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(input_json=args.input_json, output_json=args.output_json))
    except (GatewayRequestError, TrialCancelled, TrialFailed, ValueError) as exc:
        raise SystemExit(f"NeMo Gym thin client failed: {exc}") from None


def _parse_result_response(response: httpx.Response, *, expected_request_id: str) -> TrialResult:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GatewayRequestError("Gateway returned a non-JSON response") from exc
    return TrialResult.from_payload(payload, expected_request_id=expected_request_id)


def _required_env(name: str) -> str:
    return _required_mapping_value(dict(os.environ), name)


def _required_mapping_value(environ: dict[str, str], name: str) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be set to a non-empty value")
    return value


def _positive_float_env(name: str, default: float) -> float:
    return _positive_float_mapping_value(dict(os.environ), name, default)


def _positive_float_mapping_value(environ: dict[str, str], name: str, default: float) -> float:
    raw_value = environ.get(name)
    value = default if raw_value is None else float(raw_value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    value = default if raw_value is None else float(raw_value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    value = default if raw_value is None else int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _positive_int_mapping_value(environ: dict[str, str], name: str, default: int) -> int:
    raw_value = environ.get(name)
    value = default if raw_value is None else int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to one")
    return value


def _json_object_env(environ: dict[str, str], name: str, default: dict[str, Any]) -> dict[str, Any]:
    raw_value = environ.get(name)
    if raw_value is None:
        return dict(default)
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


if __name__ == "__main__":
    main()

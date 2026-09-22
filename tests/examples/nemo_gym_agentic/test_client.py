# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from relax_nemo_gym_example.app import client as client_mod
from relax_nemo_gym_example.app.client import (
    ClientConfig,
    GatewayClient,
    GatewayRequestError,
    TrialCancelled,
    build_trial_request,
    read_session_input,
    run_trial,
)
from relax_nemo_gym_example.app.protocol import PROTOCOL_VERSION, TrialStatus


def _config(**overrides):
    values = {
        "gateway_url": "http://gym.example/",
        "startup_timeout_s": 1.0,
        "request_timeout_s": 1.0,
        "poll_interval_s": 0.001,
        "abort_timeout_s": 0.1,
        "retry_backoff_s": 0.0,
        "max_retries": 0,
    }
    values.update(overrides)
    return ClientConfig(**values)


def _environ(**overrides):
    values = {
        "RELAX_SESSION_ID": "session-secret",
        "RELAX_SESSION_IO_DIR": "/tmp/relax-agentic-command-one",
        "RELAX_BASE_URL": "http://relax.example/agentic_api/",
        "RELAX_GROUP_ID": "3",
        "RELAX_ROLLOUT_MODE": "train",
        "NEMO_GYM_ENVIRONMENT": "multi_step",
        "NEMO_GYM_CONFIG": "multi-step-v1",
        "NEMO_GYM_DEADLINE_S": "120",
        "NEMO_GYM_LEASE_S": "30",
    }
    values.update(overrides)
    return values


def _result_payload(request_id, status, **overrides):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": status,
        "reward": None,
        "metrics": {},
        "artifact_ref": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def py310_sleep_or_stop(monkeypatch):
    async def sleep_or_stop(*, stop_event: asyncio.Event, delay_s: float) -> None:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay_s)
        except (TimeoutError, asyncio.TimeoutError):
            return

    monkeypatch.setattr(client_mod, "_sleep_or_stop", sleep_or_stop)


def test_build_request_preserves_session_input_and_uses_request_scoped_callback():
    session_input = {
        "messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]}],
        "metadata": {"task_id": "one"},
    }

    request = build_trial_request(session_input, _environ())
    payload = request.to_payload()

    assert payload["environment"]["task"] == session_input
    assert payload["model_endpoint"] == {
        "base_url": "http://relax.example/agentic_api",
        "api_key": "session-secret",
        "model": "model",
    }
    assert payload["session"]["group_id"] == "3"
    assert payload["deadline_s"] == 120.0


def test_build_request_uses_distinct_id_for_relaunched_managed_session():
    session_input = {
        "messages": [{"role": "user", "content": "solve"}],
        "metadata": {"task_id": "one"},
    }

    first = build_trial_request(
        session_input,
        _environ(RELAX_SESSION_IO_DIR="/tmp/relax-agentic-command-one"),
    )
    relaunched = build_trial_request(
        session_input,
        _environ(RELAX_SESSION_IO_DIR="/tmp/relax-agentic-command-two"),
    )

    assert first.request_id != relaunched.request_id


def test_read_session_input_defaults_metadata_without_changing_messages(tmp_path):
    path = tmp_path / "input.json"
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")

    payload = read_session_input(path)

    assert payload == {"messages": messages, "metadata": {}}


@pytest.mark.parametrize("payload", [[], {}, {"messages": "not-a-list"}, {"messages": [], "metadata": []}])
def test_read_session_input_rejects_invalid_shape(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        read_session_input(path)


async def test_completed_create_uses_single_versioned_endpoint():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())
    seen = []

    def handler(http_request):
        seen.append(http_request)
        return httpx.Response(
            200,
            json=_result_payload(request.request_id, "completed", reward=1.0),
        )

    async with GatewayClient(_config(), transport=httpx.MockTransport(handler)) as client:
        result = await run_trial(
            client=client,
            request=request,
            stop_event=asyncio.Event(),
            poll_interval_s=0.001,
        )

    assert result.status is TrialStatus.COMPLETED
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v1/trials"
    assert json.loads(seen[0].content)["protocol_version"] == PROTOCOL_VERSION


def test_client_exits_with_failure_without_exporting_unscored_deadline(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"messages": [], "metadata": {}}), encoding="utf-8")
    for key, value in _environ(NEMO_GYM_URL="http://gym.example").items():
        monkeypatch.setenv(key, value)

    def handler(http_request):
        request_id = json.loads(http_request.content)["request_id"]
        return httpx.Response(
            200,
            json=_result_payload(
                request_id,
                "truncated",
                error={"code": "deadline_exceeded", "message": "token=session-secret"},
            ),
        )

    monkeypatch.setattr(
        client_mod, "GatewayClient", lambda config: GatewayClient(config, transport=httpx.MockTransport(handler))
    )
    monkeypatch.setattr(
        client_mod, "parse_args", lambda: SimpleNamespace(input_json=str(input_path), output_json=str(output_path))
    )

    with pytest.raises(SystemExit) as caught:
        client_mod.main()

    assert "NeMo Gym thin client failed: NeMo Gym trial has no reward" in str(caught.value.code)
    assert "deadline_exceeded" in str(caught.value.code)
    assert "session-secret" not in str(caught.value.code)
    assert not output_path.exists()


async def test_running_trial_is_polled_to_completion(py310_sleep_or_stop):
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())
    paths = []

    def handler(http_request):
        paths.append(http_request.url.path)
        status = "running" if http_request.method == "POST" else "completed"
        return httpx.Response(200, json=_result_payload(request.request_id, status))

    async with GatewayClient(_config(), transport=httpx.MockTransport(handler)) as client:
        result = await run_trial(
            client=client,
            request=request,
            stop_event=asyncio.Event(),
            poll_interval_s=0.001,
        )

    assert result.status is TrialStatus.COMPLETED
    assert paths == ["/v1/trials", f"/v1/trials/{request.request_id}"]


async def test_renew_and_abort_do_not_send_json_bodies_and_abort_is_not_retried():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())
    seen = []

    def handler(http_request):
        seen.append(http_request)
        if http_request.url.path.endswith("/abort"):
            return httpx.Response(500)
        return httpx.Response(204)

    async with GatewayClient(
        _config(max_retries=5),
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.renew(request.request_id)
        with pytest.raises(GatewayRequestError):
            await client.abort(request.request_id)

    assert [item.url.path for item in seen] == [
        f"/v1/trials/{request.request_id}/renew",
        f"/v1/trials/{request.request_id}/abort",
    ]
    assert all(item.content == b"" for item in seen)


async def test_stop_event_aborts_running_trial():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())
    stop_event = asyncio.Event()
    paths = []

    def handler(http_request):
        paths.append(http_request.url.path)
        if http_request.url.path.endswith("/abort"):
            return httpx.Response(204)
        return httpx.Response(200, json=_result_payload(request.request_id, "running"))

    async def trigger_stop():
        await asyncio.sleep(0.005)
        stop_event.set()

    async with GatewayClient(_config(), transport=httpx.MockTransport(handler)) as client:
        trigger_task = asyncio.create_task(trigger_stop())
        with pytest.raises(TrialCancelled):
            await run_trial(
                client=client,
                request=request,
                stop_event=stop_event,
                poll_interval_s=10.0,
            )
        await trigger_task

    assert paths[-1] == f"/v1/trials/{request.request_id}/abort"


async def test_stop_event_cancels_inflight_status_request_before_abort(py310_sleep_or_stop):
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())

    class BlockingStatusTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.status_entered = asyncio.Event()
            self.status_cancelled = asyncio.Event()
            self.abort_seen = asyncio.Event()

        async def handle_async_request(self, http_request):
            if http_request.method == "POST" and http_request.url.path == "/v1/trials":
                return httpx.Response(200, json=_result_payload(request.request_id, "running"))
            if http_request.method == "GET":
                self.status_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.status_cancelled.set()
                    raise
            if http_request.url.path.endswith("/abort"):
                self.abort_seen.set()
                return httpx.Response(204)
            raise AssertionError(f"Unexpected request: {http_request.method} {http_request.url.path}")

    transport = BlockingStatusTransport()
    stop_event = asyncio.Event()
    async with GatewayClient(_config(), transport=transport) as client:
        run_task = asyncio.create_task(
            run_trial(
                client=client,
                request=request,
                stop_event=stop_event,
                poll_interval_s=0.001,
            )
        )
        await asyncio.wait_for(transport.status_entered.wait(), timeout=1.0)
        stop_event.set()
        with pytest.raises(TrialCancelled):
            await asyncio.wait_for(run_task, timeout=1.0)

    assert transport.status_cancelled.is_set()
    assert transport.abort_seen.is_set()


async def test_gateway_error_does_not_include_response_body_or_token():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())

    def handler(http_request):
        return httpx.Response(500, text=f"leaked token: {request.model_endpoint.api_key}")

    async with GatewayClient(_config(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayRequestError) as exc_info:
            await client.create(request)

    assert request.model_endpoint.api_key not in str(exc_info.value)
    assert "leaked token" not in str(exc_info.value)


async def test_idempotent_create_retries_transport_error_with_same_request_id():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())
    bodies = []

    def handler(http_request):
        bodies.append(json.loads(http_request.content))
        if len(bodies) == 1:
            raise httpx.ConnectError("temporary", request=http_request)
        return httpx.Response(200, json=_result_payload(request.request_id, "completed"))

    async with GatewayClient(
        _config(max_retries=1),
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.create(request)

    assert result.status is TrialStatus.COMPLETED
    assert [body["request_id"] for body in bodies] == [request.request_id, request.request_id]


async def test_non_json_gateway_response_is_rejected():
    request = build_trial_request({"messages": [], "metadata": {}}, _environ())

    async with GatewayClient(
        _config(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not-json")),
    ) as client:
        with pytest.raises(GatewayRequestError, match="non-JSON"):
            await client.create(request)

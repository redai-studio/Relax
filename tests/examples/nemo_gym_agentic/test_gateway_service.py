# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from relax_nemo_gym_example.app.protocol import (
    InterruptPolicy,
    ModelEndpoint,
    TrialRequest,
    TrialStatus,
)
from relax_nemo_gym_example.service import callback_provider as callback_mod
from relax_nemo_gym_example.service.callback_provider import CallbackProvider, CallbackUpstreamError
from relax_nemo_gym_example.service.config import EnvironmentSpec, GatewaySettings
from relax_nemo_gym_example.service.registry import (
    AdmissionRejected,
    CallbackUnavailable,
    GatewayRegistry,
    TrialConflict,
)
from relax_nemo_gym_example.service.run_adapter import AdapterResult, CleanupResult


class ControlledHandle:
    def __init__(self):
        self.result = asyncio.get_running_loop().create_future()
        self.abort_calls = 0
        self.force_cleanup_calls = 0

    async def wait(self):
        return await self.result

    async def abort(self):
        self.abort_calls += 1
        return CleanupResult(confirmed=True)

    async def force_cleanup(self):
        self.force_cleanup_calls += 1
        return CleanupResult(confirmed=True)

    async def probe_cleanup(self):
        return CleanupResult(confirmed=True)

    def complete(self, *, reward=1.0):
        if not self.result.done():
            self.result.set_result(
                AdapterResult(
                    status=TrialStatus.COMPLETED,
                    reward=reward,
                    metrics={"model_calls": 1},
                )
            )


class ControlledAdapter:
    def __init__(self):
        self.contexts = []
        self.handles = []
        self.started = asyncio.Event()
        self.closed = False

    async def start(self, context):
        handle = ControlledHandle()
        self.contexts.append(context)
        self.handles.append(handle)
        self.started.set()
        return handle

    async def ready(self):
        return True

    async def close(self):
        self.closed = True


class BlockingStartAdapter(ControlledAdapter):
    def __init__(self):
        super().__init__()
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()

    async def start(self, context):
        self.start_entered.set()
        await self.release_start.wait()
        return await super().start(context)


def _settings(*, max_concurrency=2, queue_capacity=4, lease_scan_interval_s=0.01):
    spec = EnvironmentSpec(
        environment="multi_step",
        config="multi-step-v1",
        agent_name="example_multi_step_simple_agent",
        agent_url="http://gym-agent.example",
        interrupt_policy=InterruptPolicy.PROTECTED,
        max_concurrency=max_concurrency,
        queue_capacity=queue_capacity,
        max_deadline_s=120.0,
        readiness_urls=("http://gym-agent.example",),
    )
    return GatewaySettings(
        environments={(spec.environment, spec.config): spec},
        callback_allowed_hosts=frozenset({"relax-one.example", "relax-two.example"}),
        gym_commit="test-commit",
        config_fingerprint="test-fingerprint",
        lease_scan_interval_s=lease_scan_interval_s,
        cleanup_grace_s=0.1,
    )


def _request(
    request_id,
    *,
    session_id="secret-one",
    callback_host="relax-one.example",
    callback_path="/agentic_api",
    api_key_header="Authorization",
    api_key_prefix="Bearer ",
    headers=None,
    lease_s=30.0,
    deadline_s=60.0,
):
    return TrialRequest(
        request_id=request_id,
        session_id=session_id,
        group_id="group-1",
        rollout_mode="train",
        environment="multi_step",
        config="multi-step-v1",
        task={"messages": [{"role": "user", "content": "solve"}], "metadata": {}},
        model_endpoint=ModelEndpoint(
            base_url=f"http://{callback_host}{callback_path}",
            api_key=session_id,
            model="policy-model",
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            headers=headers or {},
        ),
        interrupt_policy=InterruptPolicy.PROTECTED,
        deadline_s=deadline_s,
        lease_s=lease_s,
    )


def test_callback_provider_configures_explicit_proxy_without_environment_lookup(monkeypatch):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def aclose(self):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    provider = CallbackProvider(registry, proxy="http://proxy.example:3128")

    assert captured["proxy"] == "http://proxy.example:3128"
    assert captured["trust_env"] is False
    asyncio.run(provider.close())


async def _wait_for_status(registry, request_id, expected, *, timeout=1.0):
    async def poll():
        while True:
            payload = await registry.get(request_id)
            if payload["status"] == expected:
                return payload
            await asyncio.sleep(0.001)

    return await asyncio.wait_for(poll(), timeout=timeout)


async def test_registry_create_is_idempotent_and_rejects_payload_conflict():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    payload = _request("request-one").to_payload()
    try:
        first = await registry.create(payload)
        second = await registry.create(copy.deepcopy(payload))

        assert first["request_id"] == "request-one"
        assert second["request_id"] == "request-one"
        assert len(registry._records) == 1

        conflicting = copy.deepcopy(payload)
        conflicting["environment"]["task"]["metadata"]["variant"] = 2
        with pytest.raises(TrialConflict):
            await registry.create(conflicting)
    finally:
        await registry.close()


async def test_registry_enforces_environment_queue_capacity_before_insert():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(
        settings=_settings(max_concurrency=1, queue_capacity=0),
        adapter=adapter,
    )
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        with pytest.raises(AdmissionRejected):
            await registry.create(_request("request-two", session_id="secret-two").to_payload())
        assert "request-two" not in registry._records
    finally:
        await registry.close()


async def test_abort_is_idempotent_revokes_callback_and_waits_for_confirmed_cleanup():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        rollout_id = adapter.contexts[0].rollout_id

        await registry.abort("request-one")
        await registry.abort("request-one")
        result = await _wait_for_status(registry, "request-one", "aborted")

        assert result["error"]["code"] == "aborted"
        assert adapter.handles[0].abort_calls == 1
        with pytest.raises(CallbackUnavailable):
            async with registry.callback_target(rollout_id):
                pass
    finally:
        await registry.close()


async def test_abort_during_start_cleans_the_registered_handle():
    adapter = BlockingStartAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1.0)
        abort_task = asyncio.create_task(registry.abort("request-one"))
        await asyncio.sleep(0)
        assert not abort_task.done()

        adapter.release_start.set()
        await asyncio.wait_for(abort_task, timeout=1.0)
        await _wait_for_status(registry, "request-one", "aborted")

        assert adapter.handles[0].abort_calls == 1
    finally:
        await registry.close()


async def test_lease_expiry_cancels_a_running_trial():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one", lease_s=0.03).to_payload())
        result = await _wait_for_status(registry, "request-one", "aborted")

        assert result["error"]["code"] == "lease_expired"
        assert adapter.handles[0].abort_calls == 1
    finally:
        await registry.close()


async def test_agent_failure_cleans_the_remote_run_before_marking_failed(caplog):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        with caplog.at_level("ERROR", logger="uvicorn.error"):
            await registry.create(_request("request-one").to_payload())
            await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
            adapter.handles[0].result.set_exception(RuntimeError("agent failed"))

            result = await _wait_for_status(registry, "request-one", "failed")

        assert result["error"]["code"] == "agent_error"
        assert adapter.handles[0].abort_calls == 1
        error_records = [record for record in caplog.records if "NeMo Gym trial failed" in record.message]
        assert len(error_records) == 1
        assert error_records[0].exc_info is not None
        assert "agent failed" in caplog.text
    finally:
        await registry.close()


async def test_two_callback_capabilities_forward_distinct_endpoints_and_tokens():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []

    def handler(request):
        seen.append(
            {
                "host": request.url.host,
                "authorization": request.headers["authorization"],
                "body": request.content,
            }
        )
        return httpx.Response(
            200,
            json={
                "id": f"chat-{request.url.host}",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(_request("request-one").to_payload())
        await registry.create(
            _request(
                "request-two",
                session_id="secret-two",
                callback_host="relax-two.example",
            ).to_payload()
        )
        while len(adapter.contexts) < 2:
            await asyncio.sleep(0.001)

        first, second = adapter.contexts
        await asyncio.gather(
            provider.request(first.rollout_id, {"messages": [], "model": "ignored"}, resource="chat/completions"),
            provider.request(second.rollout_id, {"messages": [], "model": "ignored"}, resource="chat/completions"),
        )

        assert {(item["host"], item["authorization"]) for item in seen} == {
            ("relax-one.example", "Bearer secret-one"),
            ("relax-two.example", "Bearer secret-two"),
        }
        assert all(b'"model":"policy-model"' in item["body"] for item in seen)
    finally:
        await provider.close()
        await registry.close()


async def test_callback_supports_custom_api_key_header_and_standard_v1_base_url():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(
            _request(
                "request-custom-auth",
                callback_path="/v1",
                api_key_header="api-key",
                api_key_prefix="",
                headers={"x-user": "user@example.com", "x-app-id": "app"},
            ).to_payload()
        )
        while not adapter.contexts:
            await asyncio.sleep(0.001)

        await provider.request(adapter.contexts[0].rollout_id, {"messages": []}, resource="chat/completions")

        assert seen[0].url.path == "/v1/chat/completions"
        assert seen[0].headers["api-key"] == "secret-one"
        assert seen[0].headers["x-user"] == "user@example.com"
        assert "authorization" not in seen[0].headers
    finally:
        await provider.close()
        await registry.close()


async def test_callback_timeout_is_capped_by_gateway_setting():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen_timeout = []

    def handler(request):
        seen_timeout.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(
        registry,
        transport=httpx.MockTransport(handler),
        timeout_s=12.0,
    )
    try:
        await registry.create(_request("request-timeout-cap").to_payload())
        while not adapter.contexts:
            await asyncio.sleep(0.001)

        await provider.request(adapter.contexts[0].rollout_id, {"messages": []}, resource="chat/completions")

        assert len(seen_timeout) == 1
        assert seen_timeout[0] == pytest.approx({"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0}, abs=0.1)
    finally:
        await provider.close()
        await registry.close()


@pytest.fixture
def callback_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, waits=[])

    async def sleep(delay):
        clock.waits.append(delay)
        clock.now += delay
        await asyncio.sleep(0)

    # Advance callback time without changing the registry or event-loop clocks.
    monkeypatch.setattr(callback_mod, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        callback_mod,
        "asyncio",
        SimpleNamespace(sleep=sleep, wait_for=asyncio.wait_for, TimeoutError=asyncio.TimeoutError),
    )
    return clock


async def test_callback_retries_keep_tool_history_and_trial_alive(callback_clock, caplog, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VERBOSE", "0")
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []
    statuses = iter([500, 502, 503, 504, 200])

    def handler(request):
        seen.append(request)
        status = next(statuses)
        if status == 500:
            return httpx.Response(
                status,
                json={"error": {"code": "backend_failed", "message": "generation aborted", "api_key": "private-key"}},
            )
        # Proxy errors can be HTML or plain text; retry before decoding JSON.
        if status != 200:
            return httpx.Response(status, text="temporarily unavailable")
        return httpx.Response(200, json={"id": "resp-resumed", "output": []})

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler), timeout_s=30)
    history = [
        {"role": "user", "content": "Create an event, then report the result."},
        {"type": "function_call", "call_id": "call-1", "name": "create_event", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": '{"event_id":"event-1"}'},
    ]
    payload = {"input": history, "model": "ignored", "tools": [], "temperature": 0.7}
    original = copy.deepcopy(payload)
    try:
        await registry.create(_request("request-retry").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        rollout_id = adapter.contexts[0].rollout_id

        response = await provider.request(rollout_id, payload, resource="responses")

        assert response.status_code == 200
        assert response.payload["id"] == "resp-resumed"
        assert callback_clock.waits == [1, 2, 4, 8]
        assert '"code": "backend_failed"' in caplog.text
        assert "generation aborted" in caplog.text
        assert "temporarily unavailable" in caplog.text
        assert "private-key" not in caplog.text
        assert [r.extensions["timeout"]["read"] for r in seen] == [30, 29, 27, 23, 15]
        assert all(r.content == seen[0].content and r.headers == seen[0].headers for r in seen)
        assert json.loads(seen[0].content) == {**original, "model": "policy-model"}
        assert seen[0].headers["authorization"] == "Bearer secret-one"
        assert payload == original
        assert len(adapter.contexts) == 1
        assert adapter.handles[0].abort_calls == 0
        assert (await registry.get("request-retry"))["status"] == "running"
        adapter.handles[0].complete()
        await _wait_for_status(registry, "request-retry", "completed")
    finally:
        await provider.close()
        await registry.close()


async def test_callback_waits_through_sixty_second_router_outage(callback_clock):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    attempts = []

    def handler(request):
        attempts.append(callback_clock.now)
        if callback_clock.now < 60:
            return httpx.Response(503, json={"error": {"code": "no_available_workers"}})
        return httpx.Response(200, json={"output": []})

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler), timeout_s=90)
    try:
        await registry.create(_request("request-router", deadline_s=120).to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        response = await provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        assert response.status_code == 200
        assert attempts == [0, 1, 3, 7, 15, 23, 31, 39, 47, 55, 63]
        assert len(adapter.contexts) == 1
        assert (await registry.get("request-router"))["status"] == "running"
    finally:
        await provider.close()
        await registry.close()


@pytest.mark.parametrize(
    ("status", "message"),
    [(status, "rejected") for status in [400, 401, 403, 404, 409, 422, 429, 499, 501]]
    + [(500, "Internal error while rendering agentic response.")],
)
async def test_callback_does_not_retry_other_http_errors(status, message, callback_clock, caplog, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VERBOSE", "0")
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []
    error = {"error": {"message": message, "code": "invalid_request", "param": "input", "api_key": "private-key"}}

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=error)

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(_request("request-rejected").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        response = await provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        assert response.status_code == status
        assert response.payload == error
        assert len(seen) == 1
        assert callback_clock.waits == []
        assert "NeMo Gym model callback rejected" in caplog.text
        assert f"rollout_id={adapter.contexts[0].rollout_id}" in caplog.text
        assert f"status={status}" in caplog.text
        assert message in caplog.text
        assert '"code": "invalid_request"' in caplog.text
        assert '"param": "input"' in caplog.text
        assert "private-key" not in caplog.text
    finally:
        await provider.close()
        await registry.close()


@pytest.mark.parametrize("callback_timeout,trial_deadline", [(5, 60), (60, 5)])
async def test_callback_retries_share_one_deadline(callback_timeout, trial_deadline, callback_clock):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen_timeouts = []

    def handler(request):
        seen_timeouts.append(request.extensions["timeout"]["read"])
        callback_clock.now += 0.5  # Request duration also consumes the budget.
        return httpx.Response(503, text="unavailable")

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler), timeout_s=callback_timeout)
    try:
        await registry.create(_request("request-deadline", deadline_s=trial_deadline).to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        with pytest.raises(CallbackUpstreamError, match="deadline exceeded"):
            await provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        assert seen_timeouts == pytest.approx([5, 3.5, 1], abs=0.1)
        assert callback_clock.now == pytest.approx(5, abs=0.1)
        assert callback_clock.waits == pytest.approx([1, 2, 0.5], abs=0.1)
    finally:
        await provider.close()
        await registry.close()


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
async def test_callback_does_not_replay_ambiguous_transport_failures(error_type, callback_clock):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []

    def handler(request):
        seen.append(request)
        raise error_type("transport failed", request=request)

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(_request("request-transport").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        with pytest.raises(CallbackUpstreamError, match="transport failed"):
            await provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        assert len(seen) == 1
        assert callback_clock.waits == []
    finally:
        await provider.close()
        await registry.close()


async def test_callback_total_deadline_cancels_a_blocked_response():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    cancelled = asyncio.Event()

    async def handler(request):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler), timeout_s=0.02)
    try:
        await registry.create(_request("request-blocked").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        with pytest.raises(CallbackUpstreamError, match="deadline exceeded"):
            await asyncio.wait_for(
                provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses"), timeout=1
            )
        assert cancelled.is_set()
    finally:
        await provider.close()
        await registry.close()


async def test_abort_cancels_callback_backoff_without_retry_or_trial_restart(monkeypatch):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    sleeping = asyncio.Event()
    sleep_cancelled = asyncio.Event()
    seen = []

    async def backoff(delay):
        sleeping.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sleep_cancelled.set()
            raise

    monkeypatch.setattr(
        callback_mod,
        "asyncio",
        SimpleNamespace(sleep=backoff, wait_for=asyncio.wait_for, TimeoutError=asyncio.TimeoutError),
    )

    def handler(request):
        seen.append(request)
        return httpx.Response(503, text="unavailable")

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(_request("request-backoff").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1)
        callback_task = asyncio.create_task(
            provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        )
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        await registry.abort("request-backoff")
        with pytest.raises(asyncio.CancelledError):
            await callback_task
        await _wait_for_status(registry, "request-backoff", "aborted")
        assert sleep_cancelled.is_set()
        assert len(seen) == len(adapter.contexts) == 1
        assert adapter.handles[0].abort_calls == 1
        with pytest.raises(CallbackUnavailable):
            await provider.request(adapter.contexts[0].rollout_id, {"input": []}, resource="responses")
        assert len(seen) == 1
    finally:
        await provider.close()
        await registry.close()


async def test_one_hundred_concurrent_callbacks_keep_tokens_isolated():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(
        settings=_settings(max_concurrency=100, queue_capacity=0),
        adapter=adapter,
    )
    await registry.start()
    seen_tokens = []

    def handler(request):
        seen_tokens.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        for index in range(100):
            await registry.create(
                _request(
                    f"request-{index}",
                    session_id=f"secret-{index}",
                ).to_payload()
            )
        while len(adapter.contexts) < 100:
            await asyncio.sleep(0.001)

        await asyncio.gather(
            *(
                provider.request(context.rollout_id, {"messages": []}, resource="chat/completions")
                for context in adapter.contexts
            )
        )

        assert set(seen_tokens) == {f"Bearer secret-{index}" for index in range(100)}
    finally:
        await provider.close()
        await registry.close()


async def test_abort_cancels_an_inflight_relax_callback():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()

    class BlockingCallbackTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def handle_async_request(self, request):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    transport = BlockingCallbackTransport()
    provider = CallbackProvider(registry, transport=transport)
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        callback_task = asyncio.create_task(
            provider.request(adapter.contexts[0].rollout_id, {"messages": []}, resource="chat/completions")
        )
        await asyncio.wait_for(transport.entered.wait(), timeout=1.0)

        await registry.abort("request-one")
        with pytest.raises(asyncio.CancelledError):
            await callback_task
        await _wait_for_status(registry, "request-one", "aborted")

        assert transport.cancelled.is_set()
    finally:
        await provider.close()
        await registry.close()


async def test_terminal_completion_cannot_be_overwritten_by_late_abort():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        adapter.handles[0].complete(reward=0.75)
        completed = await _wait_for_status(registry, "request-one", "completed")

        await registry.abort("request-one")
        after_abort = await registry.get("request-one")

        assert completed["reward"] == 0.75
        assert after_abort == completed
        assert adapter.handles[0].abort_calls == 0
    finally:
        await registry.close()

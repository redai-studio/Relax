# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
import json
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest

from relax.inference.client import InferenceClient, endpoint_url
from relax.inference.compat import RoleDiscovery, public_legacy_discovery, teacher_base_url
from relax.inference.registry import InferenceRegistry
from relax.inference.routing import InferenceRoutingError
from relax.inference.specs import ModelSnapshot, ReplicaSnapshot, RoutingSpec


def _snapshot(state="READY", url="http://teacher.example:15000"):
    registry = InferenceRegistry("teacher")
    return registry.publish(
        [ModelSnapshot("math", state, "DIRECT", engines=(ReplicaSnapshot("math/0", url, state, state == "READY"),))],
        RoutingSpec(default_model="math", route_key_map={"math-data": "math"}),
    )


async def test_wait_for_normalizes_python_310_asyncio_timeout(monkeypatch):
    from relax.inference import client as module

    class LegacyAsyncioTimeoutError(Exception):
        pass

    async def legacy_wait_for(_awaitable, *, timeout):
        assert timeout == 1.5
        raise LegacyAsyncioTimeoutError

    monkeypatch.setattr(
        module,
        "asyncio",
        SimpleNamespace(wait_for=legacy_wait_for, TimeoutError=LegacyAsyncioTimeoutError),
    )

    with pytest.raises(TimeoutError) as error:
        await module._wait_for(object(), 1.5)

    assert isinstance(error.value.__cause__, LegacyAsyncioTimeoutError)


async def test_client_caches_snapshots_and_refreshes_expired_endpoints():
    now = [0.0]
    snapshots = [_snapshot(), _snapshot(url="http://replacement.example:15001")]
    reads, requests = [], []

    async def provider():
        reads.append(True)
        return snapshots[min(len(reads) - 1, 1)]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"meta_info": {"input_token_logprobs": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http, clock=lambda: now[0]) as client:
            for _ in range(2):
                await client.generate({"input_ids": [1], "return_logprob": True})
            now[0] = 2.0
            await client.generate({"input_ids": [1]})
    assert len(reads) == 2
    assert [r.url.host for r in requests] == ["teacher.example", "teacher.example", "replacement.example"]


async def test_client_does_not_mutate_payload_or_snapshot_and_forwards_affinity():
    snapshot = _snapshot()
    payload = {"input_ids": [1, 2], "route_key": "math-data", "sampling_params": {"max_new_tokens": 0}}
    original = copy.deepcopy(payload)
    requests = []

    async def provider():
        return snapshot

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http) as client:
            returned = await client.snapshot()
            returned["models"].clear()
            await client.generate(payload, affinity_key="group-4", headers={"x-smg-routing-key": "group-4"})
    assert payload == original
    assert list(snapshot["models"]) == ["math"]
    assert requests[0].headers["X-SMG-Routing-Key"] == "group-4"
    assert json.loads(requests[0].content) == {"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 0}}


async def test_client_sleeping_model_refreshes_once_without_post_or_activation():
    reads, requests = [], []

    async def provider():
        reads.append(True)
        return _snapshot("SLEEPING")

    def handler(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http) as client:
            with pytest.raises(InferenceRoutingError) as exc:
                await client.generate({"input_ids": [1]})
    assert exc.value.status_code == 503
    assert len(reads) == 2
    assert requests == []


async def test_client_response_loss_invalidates_but_never_replays_post():
    requests = []

    async def provider():
        return _snapshot()

    def handler(request):
        requests.append(request)
        raise httpx.ReadError("response lost after acceptance", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http) as client:
            with pytest.raises(httpx.ReadError):
                await client.generate({"input_ids": [1]})
            assert client._snapshot is None
    assert len(requests) == 1


async def test_client_post_deadline_invalidates_cached_endpoint_without_replay():
    requests = []

    async def provider():
        return _snapshot()

    async def handler(request):
        requests.append(request)
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http, timeout=0.05) as client:
            await client.snapshot()
            assert client._snapshot is not None
            with pytest.raises(TimeoutError):
                await client.generate({"input_ids": [1]})
            assert client._snapshot is None
    assert len(requests) == 1


async def test_client_expired_discovery_failure_does_not_use_old_endpoint():
    now, reads = [0.0], []

    async def provider():
        reads.append(True)
        if len(reads) > 1:
            raise ConnectionError("registry unavailable")
        return _snapshot()

    async with InferenceClient(snapshot_provider=provider, clock=lambda: now[0]) as client:
        await client.snapshot()
        now[0] = 10.0
        with pytest.raises(ConnectionError):
            await client.snapshot()
        assert client._snapshot is None


async def test_client_gateway_transport_selects_same_model_before_forwarding():
    requests = []

    async def provider():
        return _snapshot()

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(
            snapshot_provider=provider, gateway_url="http://gateway.example/teacher", http_client=http
        ) as client:
            await client.generate({"input_ids": [1]}, route_key="math-data")
    assert str(requests[0].url) == "http://gateway.example/teacher/generate"
    assert json.loads(requests[0].content)["model"] == "math"


@pytest.mark.parametrize("streaming", [False, True])
async def test_client_chat_uses_backend_name_for_aliases_and_model_arguments(streaming):
    snapshot = _snapshot()
    snapshot["models"]["math"]["served_model_name"] = "teacher-checkpoint"
    snapshot["routing"]["aliases"] = {"teacher-alias": "math"}
    requests = []

    async def provider():
        return snapshot

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http) as client:
            if streaming:
                async with client.stream(
                    "/v1/chat/completions", {"messages": [], "model": "teacher-alias"}
                ) as response:
                    await response.aread()
            else:
                await client.request("/v1/chat/completions", {"messages": []}, model="teacher-alias")
    assert json.loads(requests[0].content)["model"] == "teacher-checkpoint"


async def test_client_http_discovery_uses_schema_two_and_preserves_role_prefix():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_snapshot())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(discovery_url="http://gateway.example/teacher/", http_client=http) as client:
            await client.snapshot()
    assert str(requests[0].url) == "http://gateway.example/teacher/engines?schema=2"


async def test_client_total_deadline_includes_discovery():
    async def provider():
        await asyncio.Event().wait()

    async with InferenceClient(snapshot_provider=provider, timeout=0.01) as client:
        with pytest.raises(TimeoutError):
            await client.generate({"input_ids": [1]})


async def test_client_stream_deadline_closes_backend_without_replay():
    closed, requests = [], []

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def provider():
        return _snapshot()

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=SlowStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with InferenceClient(snapshot_provider=provider, http_client=http, timeout=0.05) as client:
            chunks = []
            with pytest.raises(TimeoutError):
                async with client.stream("/generate", {"input_ids": [1]}) as response:
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
            assert client._snapshot is None
    assert chunks == [b"first"]
    assert len(requests) == 1
    assert closed


async def test_role_discovery_aggregates_models_and_invalidates_same_url_restart():
    current = [_snapshot()]

    async def remote():
        return current[0]

    manager = SimpleNamespace(get_inference_snapshot=SimpleNamespace(remote=remote))
    discovery = RoleDiscovery("teacher", {"math-data": manager})
    initial = await discovery.snapshot()
    unchanged = await discovery.snapshot()
    current[0] = _snapshot()
    restarted = await discovery.snapshot()
    assert initial == unchanged
    assert restarted["topology_revision"] > initial["topology_revision"]
    assert restarted["routing"]["default_model"] == "math-data"
    assert list(restarted["models"]) == ["math-data"]


async def test_role_discovery_preserves_lifecycle_revision_between_reads():
    current = _snapshot()

    async def remote():
        return current

    manager = SimpleNamespace(get_inference_snapshot=SimpleNamespace(remote=remote))
    discovery = RoleDiscovery("teacher", {"math": manager})
    before = await discovery.snapshot()
    current["topology_revision"] += 2
    after = await discovery.snapshot()
    assert before["models"] == after["models"]
    assert after["topology_revision"] > before["topology_revision"]


@pytest.mark.parametrize(
    "url", ["http://teacher.example/generate", "http://teacher.example/generate/", "http://teacher.example"]
)
def test_teacher_url_adapter_preserves_legacy_generate_urls(url):
    assert teacher_base_url(url) == "http://teacher.example"


def test_endpoint_url_does_not_accept_an_absolute_redirect_path():
    with pytest.raises(ValueError):
        endpoint_url("http://gateway.example/teacher", "//different.example/generate")


async def test_managed_rollout_transport_reuses_loop_cache_and_closes(monkeypatch):
    from relax.inference import client as module

    instances, calls = [], []

    class Client:
        def __init__(self, **kwargs):
            instances.append(self)
            self.kwargs = kwargs
            self.closed = False

        async def generate(self, payload, **kwargs):
            calls.append(kwargs)
            return payload

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(module, "InferenceClient", Client)
    monkeypatch.setattr(module, "_loop_clients", None)
    for _ in range(2):
        await module.generate_with_discovery(
            "http://gateway.example/rollout",
            {"input_ids": [1]},
            model="policy",
            headers={"x-smg-routing-key": "session-2"},
            max_connections=512,
        )
    await module.generate_with_discovery(
        "http://gateway.example/rollout",
        {"input_ids": [1]},
        model="policy",
        timeout=900,
        max_connections=1024,
    )
    assert len(instances) == 2
    assert instances[0].kwargs["max_connections"] == 512
    assert instances[1].kwargs == {
        "discovery_url": "http://gateway.example/rollout",
        "timeout": 900,
        "max_connections": 1024,
    }
    assert all(call["model"] == "policy" and call["affinity_key"] == "session-2" for call in calls[:2])
    await module.close_loop_inference_clients()
    assert all(client.closed for client in instances)


async def test_client_connection_limit_allows_configured_rollout_concurrency():
    target = 120
    arrived = 0
    all_arrived = asyncio.Event()
    release = asyncio.Event()

    async def backend(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal arrived
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in header.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            if length:
                await reader.readexactly(length)
            arrived += 1
            if arrived == target:
                all_arrived.set()
            await release.wait()
            body = b'{"ok": true}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    server = await asyncio.start_server(backend, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    async def provider():
        return _snapshot(url=f"http://{host}:{port}")

    try:
        async with InferenceClient(snapshot_provider=provider, timeout=10, max_connections=target) as client:
            requests = [asyncio.create_task(client.generate({"input_ids": [index]})) for index in range(target)]
            await asyncio.wait_for(all_arrived.wait(), timeout=5)
            assert arrived == target
            release.set()
            assert all(result["ok"] for result in await asyncio.gather(*requests))
    finally:
        release.set()
        server.close()
        await server.wait_closed()


def test_public_legacy_discovery_omits_followers_private_metadata_and_pd_workers():
    diagnostics = {
        "total_engines": 3,
        "models": {
            "policy": {
                "total_engines": 3,
                "engine_groups": [
                    {
                        "worker_type": "regular",
                        "engines": [
                            {
                                "rank": 0,
                                "status": "active",
                                "url": "http://head.example",
                                "pid": 12,
                                "node_id": "node-a",
                            },
                            {"rank": 1, "status": "active", "pid": 13, "node_id": "node-b"},
                        ],
                    },
                    {"worker_type": "prefill", "engines": [{"url": "http://prefill.example"}]},
                ],
            },
        },
    }
    public = public_legacy_discovery(diagnostics)
    assert public["total_engines"] == 1
    assert public["models"]["policy"]["engine_groups"][0]["engines"] == [
        {"rank": 0, "status": "active", "url": "http://head.example"}
    ]
    assert public["models"]["policy"]["engine_groups"][1]["engines"] == []
    assert diagnostics["total_engines"] == 3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual runtime adapter with a controlled HTTP transport (no AST
extraction)."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


pytest.importorskip("pytest_asyncio", reason="Agentic transport tests require pytest-asyncio")


def binding():
    return {
        "binding": {
            "cohort_id": "cohort",
            "version_id": "A",
            "digest": "a" * 64,
            "publication_id": "op-a",
            "publication_epoch": 1,
            "lora_path": "relax_policy@A",
            "source_train_step": 3,
        },
        "engine": {"engine_id": "e1", "boot_id": "boot", "endpoint": "http://fixed-engine"},
        "native_lora_id": "uid-a",
    }


def metadata():
    return {
        "adapter_version_id": "A",
        "adapter_digest": "a" * 64,
        "publication_id": "op-a",
        "source_train_step": 3,
        "engine_boot_id": "boot",
        "native_lora_id": "uid-a",
    }


def adapter(client):
    pytest.importorskip("ray", reason="actual Agentic runtime requires Ray")
    pytest.importorskip("torch", reason="actual Agentic compiler requires PyTorch")
    from relax.agentic.pipeline.runtime import SGLangBackendAdapter

    value = object.__new__(SGLangBackendAdapter)
    value._args = SimpleNamespace(
        use_rollout_routing_replay=False, sglang_router_policy="round_robin", slime_router_sticky=False
    )
    value._session_lifecycle = False
    value._publication_manager = SimpleNamespace(
        lora_control=SimpleNamespace(remote=AsyncMock(return_value=binding()))
    )
    value._publication_data = client
    value._publication_commands = client
    value._publication_owner = "owner"
    value._publication_attempts = {}
    value._worker_urls = AsyncMock(side_effect=AssertionError("must use fixed route"))
    value.tokenizer = SimpleNamespace(vocab_size=100)
    value.compiler = SimpleNamespace(processor=None)
    return value


async def test_fixed_route_and_real_identity_propagation():
    calls = []

    def handle(request):
        data = json.loads(request.content)
        calls.append((request.url.path, data))
        assert request.url.host == "fixed-engine"
        if request.url.path == "/generate":
            assert data["lora_path"] == "uid-a"
            assert data["lora_binding"]["native_lora_id"] == "uid-a"
            return httpx.Response(
                200,
                json={
                    "output_ids": [4],
                    "meta_info": {
                        "lora_adapter": {
                            key: metadata()[key] for key in ("native_lora_id", "adapter_digest", "engine_boot_id")
                        },
                        "finish_reason": {"type": "stop"},
                    },
                },
            )
        return httpx.Response(200, json={**data, "state": "REQUEST_FINISHED"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(client)
        assert await backend.bind_adapter_session("s") == binding()
        for rid in ("turn1:0", "turn1:1", "turn2:0"):
            output = await backend.generate(
                input_ids=[1, 2],
                sampling_params={"max_new_tokens": 1},
                session_id="s",
                request_id=rid,
                adapter_binding=binding(),
            )
            assert output.meta_info["lora_adapter"] == metadata()
            await backend.finish_adapter_request(rid)
        assert not backend._publication_attempts
        backend._worker_urls.assert_not_called()
    assert [path for path, _ in calls] == ["/generate", "/lora_attempt_status"] * 3


async def test_timeout_never_reposts_generate_and_cancel_retains_original_identity():
    calls = []

    def handle(request):
        data = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path == "/generate":
            raise httpx.ReadTimeout("response lost", request=request)
        assert data["rid"] == "attempt"
        assert data["native_lora_id"] == "uid-a"
        return httpx.Response(200, json={**data, "state": "REQUEST_FINISHED"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(client)
        with pytest.raises(httpx.ReadTimeout):
            await backend.generate(
                input_ids=[1], sampling_params={}, session_id="s", request_id="attempt", adapter_binding=binding()
            )
        assert "attempt" in backend._publication_attempts
        await backend.finish_adapter_request("attempt")
    assert calls == ["/generate", "/cancel_lora_attempt"]


async def test_wrong_native_identity_cannot_enter_training():
    def handle(request):
        return httpx.Response(
            200,
            json={
                "output_ids": [4],
                "meta_info": {
                    "lora_adapter": {**metadata(), "native_lora_id": "other"},
                    "finish_reason": {"type": "stop"},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(client)
        with pytest.raises(RuntimeError, match="ADAPTER_IDENTITY_MISMATCH"):
            await backend.generate(
                input_ids=[1], sampling_params={}, session_id="s", request_id="attempt", adapter_binding=binding()
            )
        assert not backend._publication_attempts["attempt"]["delivered"]


@pytest.fixture
def limiter():
    module = pytest.importorskip("relax.agentic.session.service", reason="requires complete Ray/Agentic runtime")
    cls = module.AgenticSessionShard.__ray_metadata__.modified_class
    owner = cls.__new__(cls)
    owner.args = SimpleNamespace(_lora_publication_launch={"cohort_id": "cohort"})
    owner._managed_permit_lock = threading.Lock()
    owner._managed_permit_records = {}
    owner._managed_permit_capacity = 4
    owner._managed_permits_held = 0
    owner._managed_permit_epoch = 0
    owner._managed_permit_limit = 32
    owner._sglang_request_semaphore = threading.BoundedSemaphore(4)
    return owner


async def test_local_waiting_permit_cancellation_does_not_consume_lifecycle_capacity(limiter):
    limiter._managed_permit_capacity = 1
    limiter._managed_permit_limit = 2
    await limiter.acquire_sglang_request_permit("held")
    for index in range(4):
        permit_id = f"cancelled-{index}"
        waiter = asyncio.create_task(limiter._acquire_sglang_request_permit(permit_id))
        await asyncio.sleep(0)
        assert limiter._managed_permit_records[permit_id] == "WAITING"
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert limiter._managed_permit_records == {"held": "HELD"}
        assert limiter._managed_permits_held == 1
    await limiter.release_sglang_request_permit("held")
    await limiter._acquire_sglang_request_permit("next")
    assert limiter._managed_permit_records["next"] == "HELD"


async def test_shrinking_budget_retains_old_permits_and_releases_once(limiter):
    for key in ("a", "b", "c"):
        await limiter.acquire_sglang_request_permit(key)
    result = await limiter.set_sglang_request_capacity("cohort", 1, 2)
    assert result == {"topology_epoch": 1, "capacity": 2, "in_use": 3}
    waiter = asyncio.create_task(limiter.acquire_sglang_request_permit("d"))
    await asyncio.sleep(0)
    assert not waiter.done()
    await limiter.release_sglang_request_permit("a")
    await limiter.release_sglang_request_permit("a")  # Lost release ACK / retry.
    assert limiter._managed_permits_held == 2
    assert not waiter.done()
    await limiter.release_sglang_request_permit("b")
    await asyncio.wait_for(waiter, 1)
    assert limiter._managed_permits_held == 2
    assert limiter._managed_permit_records["d"] == "HELD"
    assert (await limiter.set_sglang_request_capacity("cohort", 1, 2))["in_use"] == 2


async def test_late_expansion_cannot_overwrite_newer_shrink(limiter):
    await limiter.set_sglang_request_capacity("cohort", 1, 8)
    await limiter.set_sglang_request_capacity("cohort", 2, 2)
    for owner, epoch, capacity, message in (
        ("cohort", 1, 8, "STALE_TOPOLOGY_EPOCH"),
        ("cohort", 2, 8, "TOPOLOGY_EPOCH_CONFLICT"),
        ("other", 3, 8, "PERMIT_OWNER_MISMATCH"),
    ):
        with pytest.raises(ValueError, match=message):
            await limiter.set_sglang_request_capacity(owner, epoch, capacity)
    assert limiter._managed_permit_capacity == 2


async def test_generation_pool_exhaustion_does_not_block_cancel_or_drain(monkeypatch):
    pytest.importorskip("ray", reason="actual Agentic runtime requires Ray")
    pytest.importorskip("torch", reason="actual Agentic compiler requires PyTorch")
    from relax.agentic.pipeline import runtime

    clients, sent = [], []
    real_client = httpx.AsyncClient

    def client(**kwargs):
        channel = len(clients)

        async def observe(request):
            sent.append((channel, request.url.path))

        # Saturate the real HTTP pool with one unfinished generation. MockTransport
        # alone cannot exercise connection-pool acquisition or starvation.
        kwargs.update(limits=httpx.Limits(max_connections=1), event_hooks={"request": [observe]})
        value = real_client(**kwargs)
        clients.append(value)
        return value

    resources = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=100),
        processor=None,
        processor_pool=None,
        cpu_executor=None,
        shutdown=lambda: None,
    )
    monkeypatch.setattr(runtime, "init_http_client", lambda args: None)
    monkeypatch.setattr(runtime, "load_agentic_compiler_resources", lambda args: resources)
    monkeypatch.setattr(runtime, "SGLangMessageCompiler", lambda **kwargs: SimpleNamespace(processor=None))
    monkeypatch.setattr(runtime.MultimodalConfig, "from_args", lambda args: None)
    monkeypatch.setattr(runtime.httpx, "AsyncClient", client)
    args = SimpleNamespace(
        agentic_session_lifecycle=False,
        _lora_publication_manager=object(),
        apply_chat_template_kwargs={},
        use_audio_in_video=False,
        use_rollout_routing_replay=False,
        sglang_router_policy="round_robin",
        slime_router_sticky=False,
    )
    backend = runtime.SGLangBackendAdapter(args)
    entered, release = asyncio.Event(), asyncio.Event()
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = next(
                int(line.split(":", 1)[1])
                for line in header.split("\r\n")
                if line.lower().startswith("content-length:")
            )
            data = json.loads(await reader.readexactly(length))
            if header.startswith("POST /generate "):
                entered.set()
                await release.wait()
                result = {
                    "output_ids": [4],
                    "meta_info": {
                        "lora_adapter": {
                            key: metadata()[key] for key in ("native_lora_id", "adapter_digest", "engine_boot_id")
                        },
                        "finish_reason": {"type": "stop"},
                    },
                }
            else:
                result = {**data, "state": "REQUEST_FINISHED"}
            body = json.dumps(result).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    route = binding()
    route["engine"]["endpoint"] = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    generation = asyncio.create_task(
        backend.generate(
            input_ids=[1], sampling_params={}, session_id="s", request_id="attempt", adapter_binding=route
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(backend.abort_request("attempt"), 1)
        assert not generation.done()
        release.set()
        await asyncio.wait_for(generation, 2)
        await asyncio.wait_for(backend.finish_adapter_request("attempt"), 1)
        assert sent == [(0, "/generate"), (1, "/cancel_lora_attempt"), (1, "/lora_attempt_status")]
    finally:
        release.set()
        generation.cancel()
        await asyncio.gather(generation, return_exceptions=True)
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        await backend.shutdown()
    assert len(clients) == 2 and all(value.is_closed for value in clients)


@pytest.fixture
def bootstrap_shard(monkeypatch, tmp_path):
    """Run real constructors; replace only compiler IO and process
    launching."""
    pytest.importorskip("ray", reason="actual Agentic startup handoff requires Ray")
    pytest.importorskip("torch", reason="actual Agentic runtime requires PyTorch")
    from relax.agentic.pipeline import runtime
    from relax.agentic.session import service

    resources = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=100),
        processor=None,
        processor_pool=None,
        cpu_executor=None,
        shutdown=lambda: None,
    )
    monkeypatch.setattr(runtime, "init_http_client", lambda args: None)
    monkeypatch.setattr(runtime, "load_agentic_compiler_resources", lambda args: resources)
    monkeypatch.setattr(runtime, "SGLangMessageCompiler", lambda **kwargs: SimpleNamespace(processor=None))
    monkeypatch.setattr(runtime.MultimodalConfig, "from_args", lambda args: None)
    monkeypatch.setattr(service, "resolve_chat_api_base_url", lambda: "http://unused")
    monkeypatch.setattr(service, "ManagedAgentLauncher", lambda *_args: None)
    args = SimpleNamespace(
        lora_publication_config="publication.yaml",
        agentic_session_lifecycle=False,
        apply_chat_template_kwargs={},
        use_audio_in_video=False,
        agent_command="unused",
        agent_cwd=str(tmp_path),
        agent_env=[],
        agent_timeout=30,
    )
    return service.AgenticSessionShard.__ray_metadata__.modified_class(args, 4, None, None)


async def test_publication_startup_handoff_is_idempotent_and_preserves_permits(bootstrap_shard):
    from relax.agentic.pipeline.runtime import RuntimeGroupError

    shard = bootstrap_shard
    backend = shard._generation_backend
    manager = SimpleNamespace(
        _actor_id=b"manager-one",
        lora_control=SimpleNamespace(remote=AsyncMock(side_effect=AssertionError("constructor callback"))),
    )
    launch = {"cohort_id": "cohort", "max_lifecycle_records": 100, "engines_per_gpu": 2}
    owner = backend._publication_owner
    with pytest.raises(RuntimeError, match="PUBLICATION_NOT_READY"):
        await backend.generate(input_ids=[1], sampling_params={}, session_id="s", request_id="early")
    with pytest.raises(RuntimeGroupError, match="PUBLICATION_NOT_READY"):
        await shard.start_group("eval", "early", ())
    assert not shard._groups and not shard._session_records
    try:
        receipt = await shard.configure_lora_publication(manager, launch, 8)
        assert receipt == {"cohort_id": "cohort", "manager_id": manager._actor_id.hex(), "request_capacity": 8}
        assert shard._managed_permit_capacity == 8
        assert shard._managed_permit_limit == 100
        channels = (backend._publication_data, backend._publication_commands)
        await shard.acquire_sglang_request_permit("held")
        await shard.set_sglang_request_capacity("cohort", 1, 2)
        assert await shard.configure_lora_publication(manager, launch, 8) == receipt
        assert shard._managed_permits_held == 1 and shard._managed_permit_capacity == 2
        assert shard._managed_permit_epoch == 1
        assert channels == (backend._publication_data, backend._publication_commands)
        assert backend._publication_owner == owner
        manager.lora_control.remote.assert_not_awaited()
        for conflicting_manager, conflicting_launch, capacity in (
            (SimpleNamespace(_actor_id=b"other-manager"), launch, 8),
            (manager, {**launch, "cohort_id": "other-cohort"}, 8),
            (manager, {**launch, "max_lifecycle_records": 101}, 8),
            (manager, launch, 9),
        ):
            with pytest.raises(ValueError, match="PUBLICATION_CONFIGURATION_CONFLICT"):
                await shard.configure_lora_publication(conflicting_manager, conflicting_launch, capacity)
        await shard.release_sglang_request_permit("held")
    finally:
        await backend.shutdown()


async def test_publication_startup_cannot_adopt_existing_permit_ownership(bootstrap_shard):
    shard = bootstrap_shard
    await shard.acquire_sglang_request_permit("earlier")
    try:
        with pytest.raises(ValueError, match="PUBLICATION_CONFIGURATION_IN_USE"):
            await shard.configure_lora_publication(
                SimpleNamespace(_actor_id=b"manager"), {"cohort_id": "cohort", "max_lifecycle_records": 100}, 8
            )
        assert shard._publication_configuration is None
        assert shard._generation_backend._publication_manager is None
        assert shard._managed_permits_held == 1
    finally:
        await shard.release_sglang_request_permit("earlier")
        await shard._generation_backend.shutdown()

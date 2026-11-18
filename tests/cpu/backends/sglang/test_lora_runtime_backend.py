# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from relax.engine.lora_versions import runtime as runtime_mod
from relax.engine.lora_versions.artifact import write_adapter_artifact
from relax.engine.lora_versions.manager import PublicationError, SessionKey, VersionState
from relax.engine.lora_versions.runtime import LoRAVersionRuntime

from .test_managed_lora import Engine, snapshot
from .test_managed_lora import runtime as receiver_fixture


@pytest.fixture
def receiver_runtime(monkeypatch):
    return receiver_fixture.__wrapped__(monkeypatch)


def install_transport(monkeypatch, engines):
    client_class = httpx.AsyncClient

    def client(*, base_url, **kwargs):
        return client_class(base_url=base_url, transport=httpx.MockTransport(engines[base_url].http), **kwargs)

    monkeypatch.setattr(runtime_mod.httpx, "AsyncClient", client)


async def test_runtime_publication_and_periodic_reclamation(receiver_runtime, tmp_path, monkeypatch):
    engine = Engine(receiver_runtime)
    install_transport(monkeypatch, {"http://engine": engine})
    host = LoRAVersionRuntime.__ray_metadata__.modified_class()
    await host.start({"e": "http://engine"}, [1], reconcile_interval_s=0.01)
    try:
        await host.publish(str(write_adapter_artifact(snapshot(1), tmp_path).path))
        handle = SimpleNamespace(
            lora_heartbeat=SimpleNamespace(remote=AsyncMock()),
            shutdown=SimpleNamespace(remote=AsyncMock()),
        )
        await host.register_shard("s", "boot", handle)
        key = SessionKey("s", "boot", 0)
        binding = await host.bind(key)
        await host.publish(str(write_adapter_artifact(snapshot(2), tmp_path).path))
        assert await host.bind(key) == binding
        assert await host.route(key, binding) == "http://engine"
        assert host._publisher.manager.publication(1).state is VersionState.RETIRED
        await host.release(key)
        async with asyncio.timeout(2):
            while host._publisher.manager.publication(1).state is not VersionState.RECLAIMED:
                await asyncio.sleep(0.01)
        assert (await host.status())["occupied_versions"] == 1
        assert engine.unloads == 1
    finally:
        await host.close()
        await engine.client.http.aclose()


async def test_runtime_replacement_excludes_endpoint_until_verified_retry(receiver_runtime, tmp_path, monkeypatch):
    old, new = Engine(receiver_runtime), Engine(receiver_runtime)
    install_transport(monkeypatch, {"http://old": old, "http://new": new})
    host = LoRAVersionRuntime.__ray_metadata__.modified_class()
    await host.start({"e": "http://old"}, [1], reconcile_interval_s=0.01)
    try:
        await host.publish(str(write_adapter_artifact(snapshot(1), tmp_path).path))
        host._publisher.manager.register_shard("s", "boot")
        key = SessionKey("s", "boot", 0)
        binding = await host.bind(key)
        old_instance = host._publisher.manager.engine_instance("e")
        with pytest.raises(ValueError, match="confirmation"):
            await host.replace_engine("e", old_instance, "http://new")
        new.fail_probe = True
        with pytest.raises(ValueError, match="probe"):
            await host.replace_engine("e", old_instance, "http://new", old_process_exited=True)
        with pytest.raises(PublicationError, match="LORA_VERSION_UNAVAILABLE"):
            await host.route(key, binding)
        assert not (await host.status())["serving_engines"]
        new.fail_probe = False
        await host.replace_engine("e", old_instance, "http://new", old_process_exited=True)
        assert await host.route(key, binding) == "http://new"
        assert await host.bind(key) == binding
        assert new.loads == 1
        await host.replace_engine("e", old_instance, "http://new", old_process_exited=True)
        with pytest.raises(PublicationError, match="ENGINE_INSTANCE_MISMATCH"):
            await host.replace_engine("e", old_instance, "http://stale", old_process_exited=True)
        assert await host.route(key, binding) == "http://new"
        assert (await host.status())["serving_engines"] == ("e",)
    finally:
        await host.release(SessionKey("s", "boot", 0))
        await host.close()
        await old.client.http.aclose()
        await new.client.http.aclose()


async def test_runtime_rejects_duplicate_processes_and_replacement_aliases(receiver_runtime, monkeypatch):
    first, second = Engine(receiver_runtime), Engine(receiver_runtime)
    install_transport(monkeypatch, {"http://a": first, "http://b": second, "http://alias-b": second})
    host = LoRAVersionRuntime.__ray_metadata__.modified_class()
    try:
        with pytest.raises(ValueError, match="distinct process instances"):
            await host.start({"e0": "http://b", "e1": "http://alias-b"}, [1])
        await host.start({"e0": "http://a", "e1": "http://b"}, [1])
        old = host._publisher.manager.engine_instance("e0")
        with pytest.raises(ValueError, match="endpoint belongs"):
            await host.replace_engine("e0", old, "http://b", old_process_exited=True)
        assert not host._replacing
        with pytest.raises(ValueError, match="process belongs"):
            await host.replace_engine("e0", old, "http://alias-b", old_process_exited=True)
        assert host._publisher.manager.engine_instance("e0") == old
        assert first.loads == second.loads == 0
    finally:
        await host.close()
        await first.client.http.aclose()
        await second.client.http.aclose()

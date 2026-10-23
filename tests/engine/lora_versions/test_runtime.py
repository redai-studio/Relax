# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray

from relax.engine.lora_versions import runtime as runtime_mod
from relax.engine.lora_versions.manager import LoRAVersionManager, PublicationError, SessionKey
from relax.engine.lora_versions.runtime import LoRAVersionRuntime
from tests.engine.lora_versions.test_manager import ENGINES, identity, publish


def remote(function):
    return SimpleNamespace(remote=function)


def host_with_version():
    host = LoRAVersionRuntime.__ray_metadata__.modified_class()
    manager = LoRAVersionManager(ENGINES)
    publish(manager, 0)
    host._publisher = SimpleNamespace(manager=manager, reconcile=AsyncMock(return_value={}))
    host._urls = {"e0": "http://engine-0", "e1": "http://engine-1"}
    return host, manager


async def test_runtime_routes_only_live_fixed_bindings():
    host, manager = host_with_version()
    await host.register_shard("s", "boot", None)
    key = SessionKey("s", "boot", 0)
    bound = await host.bind(key)
    endpoint = await host.route(key, bound)
    publish(manager, 1)
    assert await host.bind(key) == bound
    assert await host.route(key, bound) == endpoint
    with pytest.raises(PublicationError, match="SESSION_CLOSED"):
        await host.route(key, identity(1))
    assert await host.release(key)
    assert not await host.release(key)
    with pytest.raises(PublicationError, match="SESSION_CLOSED"):
        await host.route(key, bound)


async def test_runtime_does_not_reclaim_a_timed_out_shard():
    host, manager = host_with_version()
    key = SessionKey("s", "boot", 0)
    handle = SimpleNamespace(lora_heartbeat=remote(AsyncMock(side_effect=TimeoutError("temporary"))))
    await host.register_shard("s", "boot", handle)
    await host.bind(key)
    await host._check_shard(("s", "boot"), handle)
    assert manager.publication(0).session_count == 1
    assert (await host.status())["errors"]
    handle.lora_heartbeat.remote.side_effect = ray.exceptions.ActorDiedError()
    await host._check_shard(("s", "boot"), handle)
    assert manager.publication(0).session_count == 0
    assert not host._shards


async def test_runtime_artifact_read_does_not_block_binding_and_close_joins_it(monkeypatch):
    host, manager = host_with_version()
    entered, resume = threading.Event(), threading.Event()

    def read(_path):
        entered.set()
        assert resume.wait(5)
        return object()

    monkeypatch.setattr(runtime_mod, "read_adapter_artifact", read)
    host._publisher.publish = AsyncMock()
    request = asyncio.create_task(host.publish("artifact"))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), 2)
    # These RPC bodies must run while hashing is blocked.
    manager.register_shard("s", "boot")
    key = SessionKey("s", "boot", 0)
    assert await host.bind(key) == identity(0)
    await host.release(key)
    closing = asyncio.create_task(host.close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not closing.done()
    resume.set()
    with pytest.raises(RuntimeError, match="not accepting"):
        await request
    await closing
    host._publisher.publish.assert_not_awaited()


async def test_runtime_close_orders_sessions_before_clients_and_is_idempotent():
    host, _manager = host_with_version()
    events = []
    key = SessionKey("s", "boot", 0)

    async def shutdown():
        events.append("session")
        await host.release(key)

    async def close_client():
        events.append("client")
        assert (await host.status())["bound_sessions"] == 0

    await host.register_shard("s", "boot", SimpleNamespace(shutdown=remote(shutdown)))
    await host.bind(key)
    host._http.append(SimpleNamespace(aclose=close_client))
    await asyncio.gather(host.close(), host.close())
    await host.close()
    assert events == ["session", "client"]


async def test_runtime_retries_failed_shutdown_without_releasing_early():
    host, manager = host_with_version()
    shutdown = AsyncMock(side_effect=TimeoutError("request still running"))
    await host.register_shard("s", "boot", SimpleNamespace(shutdown=remote(shutdown)))
    await host.bind(SessionKey("s", "boot", 0))
    client = SimpleNamespace(aclose=AsyncMock())
    host._http.append(client)
    with pytest.raises(TimeoutError):
        await host.close()
    assert manager.publication(0).session_count == 1
    client.aclose.assert_not_awaited()
    shutdown.side_effect = None
    await host.close()
    assert manager.publication(0).session_count == 0
    client.aclose.assert_awaited_once()


async def test_training_publication_accepts_only_committed_default():
    host, manager = host_with_version()
    host.publish = AsyncMock(return_value=manager.publication(0))
    assert await host.publish_for_training("artifact") == {
        "status": "published",
        "version_id": 0,
        "default_version": 0,
    }
    # A duplicate reserve can return an existing loading/aborting publication.
    pending = manager.reserve(identity(1), {"a": 1}).publication
    host.publish.return_value = pending
    assert (await host.publish_for_training("artifact"))["status"] == "unconfirmed"
    manager.cancel(1, pending.attempt_id)
    host.publish.return_value = manager.publication(1)
    assert (await host.publish_for_training("artifact"))["status"] == "unconfirmed"


@pytest.mark.parametrize("has_default", [False, True])
async def test_training_capacity_is_nonfatal_only_after_initial_publication(has_default):
    host, manager = host_with_version()
    if not has_default:
        host._publisher.manager = LoRAVersionManager(ENGINES)
    host.publish = AsyncMock(side_effect=PublicationError("CAPACITY_EXHAUSTED", "full"))
    result = await host.publish_for_training("artifact")
    assert result["status"] == ("capacity_skipped" if has_default else "failed")
    assert result["default_version"] == (0 if has_default else None)

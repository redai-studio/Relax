# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from relax.engine.router.router import SlimeRouter


@pytest.fixture
async def router_client():
    args = SimpleNamespace(
        slime_router_sticky=True,
        slime_router_sticky_idle_secs=600,
        slime_router_max_connections=8,
        slime_router_timeout=10,
        slime_router_middleware_paths=[],
        rollout_health_check_interval=0,
        slime_router_health_check_failure_threshold=1,
    )
    router = SlimeRouter(args)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app), base_url="http://router") as client:
        yield router, client
    await router.client.aclose()


@pytest.mark.parametrize("payload", [{"params": {"url": "http://worker"}}, {"json": {"worker_url": "http://worker"}}])
async def test_remove_worker_cleans_state_without_proxying(router_client, payload):
    router, client = router_client
    await client.post("/add_worker", params={"url": "http://worker"})
    await client.post("/add_worker", params={"url": "http://other"})
    router.worker_failure_counts["http://worker"] = 3
    router.dead_workers.add("http://worker")
    router.sticky_map.update(old=["http://worker", 0], keep=["http://other", 0])

    for _ in range(2):
        response = await client.post("/remove_worker", **payload)
        assert response.status_code == 200
    assert router.worker_request_counts == {"http://other": 0}
    assert router.worker_failure_counts == {"http://other": 0}
    assert not router.dead_workers
    assert router.sticky_map == {"keep": ["http://other", 0]}
    assert (await client.get("/list_workers")).json() == {"urls": ["http://other"]}
    assert (await client.get("/workers")).json() == {"workers": [{"url": "http://other"}]}
    assert (await client.post("/remove_worker")).status_code == 400


async def test_reregister_replaces_quarantined_worker_state(router_client):
    router, client = router_client
    await client.post("/add_worker", json={"url": "http://worker"})
    router.worker_failure_counts["http://worker"] = 3
    router.worker_request_counts["http://worker"] = 7
    router.dead_workers.add("http://worker")
    router.sticky_map["old"] = ["http://worker", 0]

    assert (await client.post("/add_worker", params={"url": "http://worker"})).status_code == 200
    assert router.worker_request_counts == {"http://worker": 0}
    assert router.worker_failure_counts == {"http://worker": 0}
    assert not router.dead_workers
    assert not router.sticky_map
    assert router._use_url("new") == "http://worker"


@pytest.mark.parametrize("reregister", [False, True])
async def test_inflight_completion_does_not_change_removed_or_replacement_worker(router_client, reregister):
    router, client = router_client
    entered, release = asyncio.Event(), asyncio.Event()

    async def upstream(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"ok": True})

    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    await client.post("/add_worker", params={"url": "http://worker"})
    pending = asyncio.create_task(client.post("/generate", json={"text": "test"}))
    await entered.wait()
    await client.post("/remove_worker", params={"url": "http://worker"})
    if reregister:
        await client.post("/add_worker", params={"url": "http://worker"})
        router._use_url()
    release.set()
    assert (await pending).status_code == 200
    assert router.worker_request_counts == ({"http://worker": 1} if reregister else {})


async def test_old_health_result_cannot_quarantine_replacement(router_client, monkeypatch):
    router, client = router_client
    await client.post("/add_worker", params={"url": "http://worker"})
    sleeps = 0

    async def sleep_once(_delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    async def stale_health(url: str) -> tuple[str, bool]:
        await client.post("/remove_worker", params={"url": url})
        await client.post("/add_worker", params={"url": url})
        return url, False

    monkeypatch.setattr(asyncio, "sleep", sleep_once)
    monkeypatch.setattr(router, "_check_worker_health", stale_health)
    with pytest.raises(asyncio.CancelledError):
        await router._health_check_loop()
    assert router.worker_failure_counts == {"http://worker": 0}
    assert not router.dead_workers


@pytest.mark.parametrize("path", ["update_weights_from_tensor", "post_process_weights"])
async def test_weight_update_requests_still_proxy_to_registered_worker(router_client, path):
    router, client = router_client
    calls = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), request.content))
        return httpx.Response(200, json={"success": True})

    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    await client.post("/add_worker", params={"url": "http://worker"})
    body = b'{"post_process_quantization":true}'
    response = await client.post(f"/{path}", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert calls == [("POST", f"http://worker/{path}", body)]
    assert router.worker_request_counts == {"http://worker": 0}

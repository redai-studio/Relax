# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout discovery while a real Ray engine actor cannot answer RPCs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import ray
from conftest import HAS_DEPS, create_test_manager, make_engine_group, make_rollout_server

from relax.engine.inference.discovery import RoleSnapshot


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.fixture(scope="module")
def snapshot_ray_cluster():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    if started_here:
        ray.shutdown()


@pytest.mark.parametrize("switch", ["offload", "onload"])
def test_rollout_snapshot_returns_before_engine_memory_switch_finishes(snapshot_ray_cluster, switch):
    @ray.remote(num_cpus=0)
    class Gate:
        def __init__(self):
            self.entered = asyncio.Event()
            self.released = asyncio.Event()

        async def block(self):
            self.entered.set()
            await self.released.wait()

        async def wait_until_entered(self):
            await self.entered.wait()

        async def release(self):
            self.released.set()

    @ray.remote(num_cpus=0.1)
    class Engine:
        def __init__(self, gate):
            self.gate = gate

        def get_url(self):
            return "http://127.0.0.1:18000"

        def release_memory_occupation(self):
            ray.get(self.gate.block.remote())

        def resume_memory_occupation(self, tags=None):
            ray.get(self.gate.block.remote())

    gate = Gate.remote()
    engine = Engine.remote(gate)
    try:
        group = make_engine_group(engines=[engine])
        group._cache_engine_urls()
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        manager.status = "onload" if switch == "offload" else "offload"
        before = RoleSnapshot.from_dict(manager.get_inference_snapshot())
        pending = group.offload() if switch == "offload" else group.onload(tags=["weights"])
        ray.get(gate.wait_until_entered.remote(), timeout=30)

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                # This is the production manager method with real engine handles.
                # A get_url RPC would queue behind the blocked memory switch.
                snapshot = executor.submit(manager.get_inference_snapshot).result(timeout=5)
                model = RoleSnapshot.from_dict(snapshot).model("default")
                assert model.engines[0].base_url == "http://127.0.0.1:18000"
                assert model.state.value == ("draining" if switch == "offload" else "onloading")
                assert snapshot["topology_revision"] == before.topology_revision
                assert ray.wait(pending, timeout=0)[0] == []
            finally:
                ray.get(gate.release.remote(), timeout=30)
        ray.get(pending, timeout=30)
    finally:
        ray.kill(engine)
        ray.kill(gate)

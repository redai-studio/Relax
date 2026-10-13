# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout discovery while a real Ray engine actor cannot answer RPCs."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import ray
from conftest import HAS_DEPS, create_test_manager, make_engine_group, make_rollout_server

from relax.engine.inference.discovery import RoleSnapshot


try:
    from relax.distributed.ray import rollout
except ImportError:
    rollout = None


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

        def init(self):
            return {
                "url": "http://127.0.0.1:18000",
                "pid": os.getpid(),
                "node_id": ray.get_runtime_context().get_node_id(),
            }

        def release_memory_occupation(self):
            ray.get(self.gate.block.remote())

        def resume_memory_occupation(self, tags=None):
            ray.get(self.gate.block.remote())

    gate = Gate.remote()
    engine = Engine.remote(gate)
    try:
        group = make_engine_group(engines=[engine])
        init_ref = engine.init.remote()
        group.engine_init_records[0] = rollout._EngineInitRecord(init_ref)
        ray.get(init_ref, timeout=30)
        group.mark_initialized()
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        manager.status = "onload" if switch == "offload" else "offload"

        def observed():
            return RoleSnapshot.from_dict(manager.get_inference_snapshot())

        # The endpoint shows once the engine's actor is observed alive locally.
        deadline = time.monotonic() + 15
        while (before := observed()).model("default").engines[0].base_url is None:
            assert time.monotonic() < deadline, "engine endpoint never became observable"
            time.sleep(0.025)
        pending = group.offload() if switch == "offload" else group.onload(tags=["weights"])
        ray.get(gate.wait_until_entered.remote(), timeout=30)

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                # This is the production manager method with real engine handles.
                # Any engine RPC would queue behind the blocked memory switch.
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

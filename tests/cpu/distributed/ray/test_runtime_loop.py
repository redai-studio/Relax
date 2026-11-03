# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Persistent rollout locks must see one runtime loop across owner calls."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.placement_planner import plan_placement
from relax.distributed.ray.rollout import RolloutEngineRuntime
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.ownership import OwnershipLedger
from tests.cpu.engine.inference.test_coresidency import Handle


@pytest.mark.parametrize("kind", ["weights", "precheck"])
async def test_owner_reuses_runtime_loop_for_concurrent_port_batches(kind):
    args = SimpleNamespace(resource={"rollout": [1, 1]}, colocate=False)
    plan = plan_placement(args)
    ledger = OwnershipLedger(plan)
    args._inference_authority = Handle(ledger)
    owner = InferenceManager.__ray_metadata__.modified_class(args, "rollout")
    runtime = object.__new__(RolloutEngineRuntime)
    runtime._admission = AdmissionGate()
    owner._runtimes["__default__"] = runtime
    owner._claims["__default__"] = (plan.claim("rollout"),)
    owner._locks["__default__"] = asyncio.Lock()
    ledger.register_manager("rollout", Handle(owner))
    ledger.register_model("rollout", "__default__", Handle(owner), plan.claim("rollout"))
    loops, threads = [], []

    async def batch():
        loops.append(asyncio.get_running_loop())
        threads.append(threading.get_ident())
        entered, finish, contending = asyncio.Event(), asyncio.Event(), asyncio.Event()
        scans = []

        async def remote(**kwargs):
            scans.append(kwargs["start_port"])
            if len(scans) == 1:
                entered.set()
                await finish.wait()
            return "node", kwargs["start_port"]

        seed = SimpleNamespace(_get_current_node_ip_and_free_port=SimpleNamespace(remote=remote))
        allocate = (
            runtime._allocate_weight_sync_ports
            if kind == "weights"
            else runtime._allocate_scale_weight_sync_precheck_ports
        )
        first = asyncio.create_task(allocate(seed, 2))
        await entered.wait()

        async def second():
            contending.set()
            return await allocate(seed, 2)

        other = asyncio.create_task(second())
        await contending.wait()
        finish.set()
        return await asyncio.gather(first, other)

    runtime.get_parallel_allocations = batch
    runtime.dispose = lambda: None
    try:
        first = await owner.call("__default__", "get_parallel_allocations")
        second = await owner.call("__default__", "get_parallel_allocations")
        assert len({port for batch in (first, second) for block in batch for port in block}) == 8
        assert loops[0] is loops[1]
        assert threads[0] == threads[1] != threading.get_ident()
    finally:
        await owner.shutdown()
    assert loops[0].is_closed()

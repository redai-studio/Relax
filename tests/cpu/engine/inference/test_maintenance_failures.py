# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Terminal maintenance failures preserve healthy routing and residency
proof."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.ownership import OwnershipLedger
from tests.cpu.engine.inference.test_coresidency import Handle


class Runtime:
    def __init__(self):
        self._admission = AdmissionGate()
        self.failed_slot = False
        self.offload_entered, self.offload_receipt = threading.Event(), threading.Event()
        self._owned_elastic_pgs = {}

    def snapshot(self):
        return RoleSnapshot(
            "rollout",
            1,
            models=(
                build_model_snapshot(
                    "__default__",
                    [
                        (0, "http://engine-0:1", EngineState.BLOCKED if self.failed_slot else EngineState.READY),
                        (1, "http://engine-1:1", EngineState.READY),
                    ],
                ),
            ),
        )

    def acquire_inference(self, request_id, **selection):
        return self._admission.acquire(self.snapshot, request_id, **selection)

    def release_inference(self, permit):
        self._admission.release(permit)

    def fail(self):
        self.failed_slot = True
        self._owned_elastic_pgs["partial"] = SimpleNamespace(id=SimpleNamespace(hex=lambda: "partial-pg"))
        raise RuntimeError("maintenance failed for one slot")

    async def execute_scale_out(self, request_id):
        self.fail()

    def check_evictions(self):
        self.fail()

    def recover(self):
        self.fail()

    def offload(self):
        self.offload_entered.set()
        assert self.offload_receipt.wait(10), "test did not provide the offload receipt"

    def dispose(self):
        self._admission.close()
        self._owned_elastic_pgs.clear()


def make_owner(colocate=False):
    args = SimpleNamespace(resource={"actor": [1, 1], "rollout": [1, 1]}, colocate=colocate)
    plan = plan_placement(args)
    ledger = OwnershipLedger(plan)
    args._inference_authority = Handle(ledger)
    owner = InferenceManager.__ray_metadata__.modified_class(args, "rollout")
    name = "__default__"
    owner._claims[name] = (plan.claim("rollout"),)
    owner._locks[name] = asyncio.Lock()
    runtime = owner._runtimes[name] = Runtime()
    ledger.register_manager("rollout", Handle(owner))
    ledger.register_model("rollout", name, Handle(owner), plan.claim("rollout"))
    return owner, runtime, ledger, plan


@pytest.mark.parametrize(
    "method,args", [("execute_scale_out", ("scale-1",)), ("check_evictions", ()), ("recover", ())]
)
@pytest.mark.parametrize("leased", [False, True])
async def test_failed_maintenance_preserves_healthy_admission_and_terminal_registration(method, args, leased):
    owner, runtime, ledger, plan = make_owner()
    name = "__default__"
    lease = await ledger.acquire("maintenance", "caller", plan.claims, role="rollout") if leased else None
    try:
        before = await owner.acquire_inference(name, "before")
        await owner.release_inference(name, before)
        with pytest.raises(RuntimeError, match="maintenance failed for one slot"):
            if leased:
                await owner.maintenance(name, method, args, lease=lease)
            else:
                await owner.call(name, method, args)
        assert owner._blocked == {}
        assert ledger.snapshot()["blocked"] == {}
        after = await owner.acquire_inference(name, "after")
        assert after.target.base_url == "http://engine-1:1"
        assert before.generation == after.generation
        await owner.release_inference(name, after)
        assert ledger.snapshot()["resident"]
        assert ledger.snapshot()["resources"]["partial-pg"]["owner"] == "inference:rollout"
        assert not ledger._inflight.get(lease.operation_id if lease else "maintenance")
    finally:
        if lease is not None:
            await asyncio.wait_for(ledger.release(lease), 2)
        await owner.shutdown()


async def test_failed_colocated_recovery_still_requires_offload_receipt_before_training():
    owner, runtime, ledger, plan = make_owner(colocate=True)
    lease = None
    handoff = None
    try:
        with pytest.raises(RuntimeError, match="maintenance failed for one slot"):
            await owner.call("__default__", "recover")
        assert ledger.snapshot()["resident"]
        lease = await ledger.acquire("training", "actor", plan.claims, role="actor")
        handoff = asyncio.create_task(ledger.command(lease, "train", "train"))
        assert await asyncio.to_thread(runtime.offload_entered.wait, 5)
        assert not handoff.done()
        assert ledger.snapshot()["resident"]
        runtime.offload_receipt.set()
        await asyncio.wait_for(handoff, 2)
        assert ledger.snapshot()["resident"] == []
    finally:
        runtime.offload_receipt.set()
        if handoff is not None:
            await asyncio.gather(handoff, return_exceptions=True)
        if lease is not None:
            await ledger.release(lease)
        await owner.shutdown()


@pytest.mark.parametrize("runtime_kind", ["rollout", "static"])
async def test_dispatch_receipt_delegates_through_public_runtime_interface(runtime_kind):
    from relax.distributed.ray.multi_engine_manager import MultiEngineManager
    from relax.distributed.ray.rollout import RolloutEngineRuntime

    runtime = object.__new__(RolloutEngineRuntime if runtime_kind == "rollout" else MultiEngineManager)
    gate = runtime._admission = AdmissionGate()
    owner, transport, ledger, _ = make_owner()
    permit = gate.acquire(transport.snapshot, "sent", dispatch_tracked=True)
    try:
        # The owner's runtime view deliberately has no private gate attribute.
        owner._runtimes["receipt"] = SimpleNamespace(confirm_inference_dispatch=runtime.confirm_inference_dispatch)
        await owner.confirm_inference_dispatch("receipt", permit)
        gate.drain(lambda: None, timeout=0)
        gate.confirm_drained()
        gate.open()
    finally:
        owner._runtimes.pop("receipt", None)
        await owner.shutdown()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cancelled Ray callers cannot release ownership of running mutations."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import ray

from relax.distributed.ray.inference_manager import InferenceManager, ModelManager
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.ownership import LifecycleAuthority


@pytest.fixture(scope="module")
def local_ray():
    started = not ray.is_initialized()
    if started:
        ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    if started:
        ray.shutdown()


def owner_class():
    class Runtime:
        def __init__(self):
            self._admission = AdmissionGate()
            self.entered, self.resume, self.done = threading.Event(), threading.Event(), threading.Event()
            self.resident = False
            self.events = []

        def recover(self):
            self.events.append("mutation entered")
            self.entered.set()
            assert self.resume.wait(30), "test did not release mutation barrier"
            self.resident = True
            self.events.append("mutation terminal")
            self.done.set()

        async def execute_scale_out(self):
            # Exercise the real async dispatch branch as well as sync recover.
            await asyncio.to_thread(self.recover)

        def offload(self):
            self.resident = False
            self.events.append("offload confirmed")

        dispose = offload

    class Owner(InferenceManager.__ray_metadata__.modified_class):
        async def install(self, claim):
            name = claim.model
            self._claims[name] = (claim,)
            self._locks[name] = asyncio.Lock()
            self._runtimes[name] = Runtime()
            await self.authority.register_model.remote(self.role, name, ray.get_runtime_context().current_actor, claim)

        async def wait_entered(self):
            assert await asyncio.to_thread(self._runtimes["__default__"].entered.wait, 20)

        def complete(self):
            self._runtimes["__default__"].resume.set()

        async def wait_done(self):
            assert await asyncio.to_thread(self._runtimes["__default__"].done.wait, 20)

        def summary(self):
            runtime = self._runtimes["__default__"]
            return runtime.resident, self._locks["__default__"].locked(), list(runtime.events)

    return ray.remote(num_cpus=0)(Owner)


@pytest.mark.parametrize("method", ["recover", "execute_scale_out"])
@pytest.mark.parametrize("handoff", ["train", "offload", "shutdown"])
def test_cancelled_maintenance_remains_owned_until_handoff_finishes(local_ray, method, handoff):
    args = SimpleNamespace(resource={"actor": [1, 1], "rollout": [1, 1]}, colocate=True)
    plan = plan_placement(args)
    authority = LifecycleAuthority.remote(plan)
    args._inference_authority = authority
    owner = owner_class().remote(args, "rollout")
    lease = None
    try:
        ray.get(owner.claim_placement.remote(), timeout=10)
        ray.get(owner.install.remote(plan.claim("rollout")), timeout=10)
        model = ModelManager(owner, "__default__")
        request = getattr(model, method).remote()
        ray.get(owner.wait_entered.remote(), timeout=10)
        for _ in range(2):
            ray.cancel(request, force=False, recursive=True)
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(request, timeout=10)
        assert ray.get(owner.summary.remote(), timeout=5)[1], "cancelled caller released the mutation lock"

        if handoff == "shutdown":
            pending = owner.shutdown.remote()
        else:
            lease = ray.get(authority.acquire.remote("training", "train", plan.claims, role="actor"), timeout=10)
            pending = (
                authority.command.remote(lease, "train-phase", "train")
                if handoff == "train"
                else model.offload.remote(lease=lease, command_id="offload")
            )
        assert ray.wait([pending], timeout=0)[0] == []
        assert ray.get(owner.summary.remote(), timeout=5)[2] == ["mutation entered"]
        assert ray.get(authority.snapshot.remote(), timeout=5)["resident"]
        ray.get(owner.complete.remote(), timeout=5)
        ray.get(pending, timeout=15)
        resident, locked, events = ray.get(owner.summary.remote(), timeout=5)
        assert events == ["mutation entered", "mutation terminal", "offload confirmed"]
        assert not resident and not locked
        assert ray.get(authority.snapshot.remote(), timeout=5)["resident"] == []
    finally:
        ray.get(owner.complete.remote(), timeout=5)
        ray.get(owner.wait_done.remote(), timeout=10)
        if lease is not None:
            ray.get(authority.release.remote(lease), timeout=10)
        for handle in (owner, authority):
            ray.kill(handle, no_restart=True)


async def test_cancelled_apply_keeps_begin_receipt_and_lease_return_owned():
    from relax.engine.inference.ownership import OwnershipLedger
    from tests.cpu.engine.inference.test_coresidency import Handle

    args = SimpleNamespace(resource={"rollout": [1, 1]}, colocate=False)
    plan = plan_placement(args)
    ledger = OwnershipLedger(plan)
    registered, receipt = asyncio.Event(), asyncio.Event()
    authority = Handle(ledger)

    async def begin(lease, token, parent_command=None):
        ledger.begin(lease, token, parent_command)
        registered.set()
        await receipt.wait()

    authority.begin = SimpleNamespace(remote=begin)
    args._inference_authority = authority
    owner = InferenceManager.__ray_metadata__.modified_class(args, "rollout")
    name = "__default__"
    owner._claims[name] = (plan.claim("rollout"),)
    owner._locks[name] = asyncio.Lock()
    calls = []
    owner._runtimes[name] = SimpleNamespace(
        onload=lambda **kwargs: calls.append("onloaded"), _admission=AdmissionGate()
    )
    ledger.register_model("rollout", name, Handle(owner), plan.claim("rollout"))
    lease = await ledger.acquire("phase", "caller", plan.claims, role="rollout")
    caller = asyncio.create_task(owner.apply(name, "onload", lease, "load"))
    try:
        await registered.wait()
        caller.cancel()
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert owner._mutations, "accepted begin lost its owner before the receipt arrived"
        returning = asyncio.create_task(ledger.release(lease))
        await asyncio.sleep(0)
        assert not returning.done()
        receipt.set()
        await asyncio.wait_for(returning, 2)
        assert calls == ["onloaded"]
        assert not ledger.snapshot()["owners"]
    finally:
        receipt.set()
        await asyncio.gather(*owner._mutations, return_exceptions=True)
        for token in list(ledger._inflight.get(lease.operation_id, ())):
            await ledger.finish(lease, token)
        await ledger.release(lease)

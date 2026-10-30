# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production ownership boundary with real CPU Ray transports and fake GPUs."""

import asyncio
from types import SimpleNamespace

import pytest
import ray

from relax.distributed.ray.inference_manager import InferenceManager, ModelManager
from relax.distributed.ray.placement_planner import Phase, PlacementClaim, PlacementPlan
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.ownership import LifecycleAuthority, OwnershipError, ResourceBlockedError, StaleLeaseError


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
        def __init__(self, name, gate, fail):
            self.name, self.gate, self.fail = name, gate, fail
            self._admission = AdmissionGate()
            self.calls = []

        def snapshot(self):
            return self._admission.view(
                RoleSnapshot(
                    "genrm", 1, models=(build_model_snapshot(self.name, [(0, "http://engine:1", EngineState.READY)]),)
                )
            )

        def get_inference_snapshot(self):
            return {"state": self.snapshot().models[0].state.value, "engines": [], "topology_revision": 1}

        def acquire_inference(self, request_id, **selection):
            return self._admission.acquire(self.snapshot, request_id, **selection)

        def release_inference(self, permit):
            self._admission.release(permit)

        def offload(self):
            self.calls.append("offload")
            self._admission.close()
            ray.get(self.gate.enter.remote())
            self._admission.drain(lambda: None, timeout=2)
            if self.fail:
                raise RuntimeError("GPU release unconfirmed")

        def onload(self, tags=None):
            self.calls.append("onload")
            self._admission.open()

    class Owner(InferenceManager.__ray_metadata__.modified_class):
        def install_startup_runtime(self, events):
            import sys
            from types import ModuleType

            class StartupRuntime:
                def __init__(self, args, pg):
                    self.args = args
                    ray.get(events.record.remote("initialize"))
                    if getattr(args, "fail_start", False):
                        raise RuntimeError("startup failed after allocating")

                def check_weights(self, action):
                    ray.get(events.record.remote(action))

                def offload(self):
                    ray.get(events.record.remote("offload"))

                def dispose(self):
                    ray.get(events.record.remote("dispose"))

            module = ModuleType("relax.distributed.ray.rollout")
            module.RolloutEngineRuntime = StartupRuntime
            sys.modules[module.__name__] = module

        async def install(self, name, claim, gate, fail=False):
            self._claims[name] = (claim,)
            self._locks[name] = asyncio.Lock()
            self._runtimes[name] = Runtime(name, gate, fail)
            await self.authority.register_model.remote(self.role, name, ray.get_runtime_context().current_actor, claim)

        def calls(self, name):
            return self._runtimes[name].calls

    return ray.remote(num_cpus=0)(Owner)


@pytest.fixture
def cluster(local_ray):
    @ray.remote(num_cpus=0)
    class Gate:
        def __init__(self):
            self.entered = asyncio.Event()

        def enter(self):
            self.entered.set()

        async def wait(self):
            await self.entered.wait()

    claims = tuple(
        PlacementClaim("genrm", name, "actor", i, 1, frozenset({Phase.SCORE})) for i, name in enumerate(("a", "b"))
    )
    authority = LifecycleAuthority.remote(PlacementPlan((), claims))
    owner = owner_class().remote(SimpleNamespace(_inference_authority=authority), "genrm")
    gate = Gate.remote()
    ray.get(owner.claim_placement.remote(), timeout=15)
    for claim in claims:
        ray.get(owner.install.remote(claim.model, claim, gate), timeout=15)
    yield authority, owner, gate, claims
    for actor in (owner, authority, gate):
        ray.kill(actor, no_restart=True)


def test_common_owner_drain_keeps_admission_and_discovery_responsive(cluster):
    authority, owner, gate, claims = cluster
    model = ModelManager(owner, "a")
    permit = ray.get(model.acquire_inference.remote("held"), timeout=10)
    lease = ray.get(authority.acquire.remote("workload", "op", [claims[0]]), timeout=10)
    offload = model.offload.remote(lease=lease, command_id="release")
    retry = model.offload.remote(lease=lease, command_id="release")
    ray.get(gate.wait.remote(), timeout=10)
    assert ray.get(model.get_inference_snapshot.remote(), timeout=3)["state"] == "draining"
    returned = authority.release.remote(lease)
    next_owner = authority.acquire.remote("training", "next", [claims[0]])
    other = ray.get(authority.acquire.remote("other", "independent", [claims[1]]), timeout=3)
    assert ray.wait([offload, retry, returned, next_owner], timeout=0)[0] == []
    ray.get(model.release_inference.remote(permit), timeout=3)
    ray.get([offload, retry, returned], timeout=10)
    assert ray.get(owner.calls.remote("a")) == ["offload"]
    next_lease = ray.get(next_owner, timeout=10)
    assert next_lease.generation > lease.generation
    with pytest.raises(StaleLeaseError):
        ray.get(model.onload.remote(lease=lease), timeout=10)
    with pytest.raises(OwnershipError):
        ray.get(model.onload.remote(), timeout=10)
    # A retry can return its cached receipt without repeating GPU work.
    ray.get(model.offload.remote(lease=lease, command_id="release"), timeout=10)
    ray.get([authority.release.remote(next_lease), authority.release.remote(other)])


def test_common_owner_failed_release_retains_claims_and_blocks_reuse(cluster):
    authority, owner, gate, claims = cluster
    ray.get(
        owner.install.remote(
            "failed", PlacementClaim("genrm", "failed", "actor", 0, 1, frozenset({Phase.SCORE})), gate, True
        )
    )
    lease = ray.get(authority.acquire.remote("workload", "failed-op", [claims[0]]))
    with pytest.raises(RuntimeError, match="unconfirmed"):
        ray.get(owner.apply.remote("failed", "offload", lease, "release"), timeout=10)
    ray.get(authority.release.remote(lease), timeout=10)
    with pytest.raises(ResourceBlockedError):
        ray.get(authority.acquire.remote("new", "new-op", [claims[0]]), timeout=10)
    assert ray.get(owner.get_model_snapshot.remote("failed"))["state"] == "blocked"


def test_lost_owner_cannot_be_replaced_with_an_empty_runtime(cluster):
    authority, owner, _, _ = cluster
    with pytest.raises(OwnershipError, match="already has a resource owner"):
        ray.get(authority.register_manager.remote("genrm", "replacement"), timeout=10)
    assert ray.get(authority.managers.remote()) == [owner]


def test_cancelled_mutation_keeps_lease_until_accepted_work_settles(cluster):
    authority, owner, gate, claims = cluster
    model = ModelManager(owner, "a")
    permit = ray.get(model.acquire_inference.remote("held"), timeout=10)
    lease = ray.get(authority.acquire.remote("workload", "cancel-op", [claims[0]]), timeout=10)
    mutation = model.offload.remote(lease=lease, command_id="release")
    ray.get(gate.wait.remote(), timeout=10)
    ray.cancel(mutation)
    returned = authority.release.remote(lease)
    next_owner = authority.acquire.remote("training", "next", [claims[0]])
    assert ray.wait([returned, next_owner], timeout=0.1)[0] == []
    ray.get(model.release_inference.remote(permit), timeout=3)
    ray.get(returned, timeout=10)
    next_lease = ray.get(next_owner, timeout=10)
    assert ray.get(owner.calls.remote("a")) == ["offload"]
    ray.get(authority.release.remote(next_lease), timeout=10)


@pytest.mark.parametrize("fail_start", [False, True])
def test_model_initialization_is_owned_and_settles_before_return(local_ray, fail_start):
    @ray.remote(num_cpus=0)
    class Events:
        def __init__(self):
            self.events = []

        def record(self, event):
            self.events.append(event)

        def read(self):
            return self.events

    claim = PlacementClaim("rollout", "__default__", "actor", 0, 1, frozenset({Phase.GENERATE}))
    authority = LifecycleAuthority.remote(PlacementPlan((), (claim,)))
    args = SimpleNamespace(
        _inference_authority=authority, offload_rollout=True, check_weight_update_equal=True, fail_start=fail_start
    )
    owner = owner_class().remote(args, "rollout")
    events = Events.remote()
    try:
        ray.get(owner.claim_placement.remote(), timeout=10)
        ray.get(owner.install_startup_runtime.remote(events), timeout=10)
        if fail_start:
            with pytest.raises(RuntimeError, match="startup failed"):
                ray.get(owner.add_model.remote("__default__", args, {}), timeout=10)
            assert ray.get(events.read.remote()) == ["initialize", "dispose"]
            snapshot = RoleSnapshot.from_dict(ray.get(owner.get_inference_snapshot.remote()))
            assert snapshot.models[0].state is EngineState.BLOCKED
        else:
            ray.get(owner.add_model.remote("__default__", args, {}), timeout=10)
            ray.get(owner.add_model.remote("__default__", args, {}), timeout=10)
            assert ray.get(events.read.remote()) == ["initialize", "snapshot", "reset_tensors", "offload"]
        snapshot = ray.get(authority.snapshot.remote())
        assert snapshot["owners"] == snapshot["blocked"] == {}
        assert snapshot["resident"] == []
    finally:
        for actor in (owner, authority, events):
            ray.kill(actor, no_restart=True)

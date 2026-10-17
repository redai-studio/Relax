# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import sys
import threading
import time
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager
from relax.distributed.ray.inference_manager import InferenceManager, ModelHandle
from relax.engine.inference.config import InferenceModelSpec
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RouteMode,
    WeightSource,
)


class FakePool:
    """An engine pool which reports READY while onloaded."""

    def __init__(self, name: str, *, version: str | None = None) -> None:
        policy = version is not None
        self.model_spec = InferenceModelSpec(
            name,
            weight_source=WeightSource.DCS if policy else WeightSource.STATIC,
            route_mode=RouteMode.SGLANG_ROUTER if policy else RouteMode.DIRECT,
        )
        self.onloaded = True
        self.version = version
        self.calls: list[str] = []

    def observe(self, serving: bool = True) -> ModelSnapshot:
        self.serving = serving
        state = (
            LifecycleState.READY
            if self.onloaded and serving
            else (LifecycleState.STARTING if self.onloaded else LifecycleState.SLEEPING)
        )
        replica = ReplicaSnapshot(f"{self.model_spec.name}/replica-0", state, "http://engine-0", self.version)
        return ModelSnapshot(
            self.model_spec.name,
            (replica,),
            "http://router" if self.model_spec.needs_router else None,
            state,
            admission=state == LifecycleState.READY,
            required_weight_version=self.version,
        )

    def onload(self, tags=None) -> None:
        self.calls.append("onload")
        self.onloaded = True

    def offload(self) -> None:
        self.calls.append("offload")
        self.onloaded = False

    def shutdown(self, planner) -> None:
        self.calls.append("shutdown")

    def health_check(self) -> bool:
        return self.onloaded

    def abort_requests(self) -> None:
        self.calls.append("abort")

    def get_urls(self) -> list[str]:
        return ["http://engine-0"]


def _manager(**roles: dict[str, FakePool]) -> InferenceManager:
    manager = InferenceManager()
    for role, pools in roles.items():
        manager.register(role, pools)
    return manager


def test_manager_direct_eligible_only_for_ready_direct_replicas() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")}, rollout={"policy": FakePool("policy", version="1")})

    genrm = manager.snapshot(Role.GENRM)
    assert genrm.routing.default_model == "judge"
    assert genrm.models[0].replicas[0].direct_eligible is True
    assert manager.snapshot(Role.ROLLOUT).models[0].replicas[0].direct_eligible is False

    manager.deactivate(Role.GENRM, "judge")
    replica = manager.snapshot(Role.GENRM).models[0].replicas[0]
    assert replica.state == LifecycleState.SLEEPING
    assert replica.direct_eligible is False


def test_manager_rejects_ready_policy_with_stale_weights() -> None:
    manager = _manager(rollout={"policy": FakePool("policy", version="1")})
    stale = ModelSnapshot(
        "policy",
        (ReplicaSnapshot("policy/replica-0", LifecycleState.READY, "http://engine-0", "0"),),
        "http://router",
        LifecycleState.READY,
        admission=True,
        required_weight_version="1",
    )
    with pytest.raises(ValueError, match="weight version"):
        manager.publish(Role.ROLLOUT, stale)


def test_manager_publish_rejects_admission_on_a_model_that_is_not_ready() -> None:
    manager = _manager(rollout={"policy": FakePool("policy", version="1")})
    starting = ModelSnapshot(
        "policy",
        (ReplicaSnapshot("policy/replica-0", LifecycleState.STARTING, "http://engine-0", "1"),),
        "http://router",
        LifecycleState.STARTING,
        admission=True,
        required_weight_version="1",
    )
    with pytest.raises(ValueError, match="Only READY models may admit requests"):
        manager.publish(Role.ROLLOUT, starting)
    # The rejected observation is not committed.
    assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.READY


def test_manager_topology_revision_bumps_only_on_topology_change() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    revision = manager.snapshot(Role.GENRM).topology_revision
    manager.publish(Role.GENRM, manager.snapshot(Role.GENRM).models[0])
    assert manager.snapshot(Role.GENRM).topology_revision == revision
    manager.deactivate(Role.GENRM, "judge")
    assert manager.snapshot(Role.GENRM).topology_revision == revision
    moved = manager.snapshot(Role.GENRM).models[0]
    replica = ReplicaSnapshot("judge/replica-0", LifecycleState.SLEEPING, "http://engine-1")
    manager.publish(Role.GENRM, ModelSnapshot("judge", (replica,), None, LifecycleState.SLEEPING))
    assert moved.replicas[0].base_url == "http://engine-0"
    assert manager.snapshot(Role.GENRM).topology_revision == revision + 1


def _warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(inference_manager, "logger", SimpleNamespace(warning=logged.append, info=lambda *_: None))
    return logged


def test_manager_drain_closes_admission_and_waits_for_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _warnings(monkeypatch)
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    permit = manager.admit_request(Role.GENRM, "judge", "req-1")

    drainer = _start(manager.drain, [Role.GENRM], 0.01)
    # Without force_abort it warns every timeout and keeps waiting.
    assert drainer.is_alive() and len(warnings) > 1 and "req-1" in warnings[-1]
    assert "abort" not in pool.calls
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.GENRM, "judge")

    manager.complete_request(permit)
    drainer.join(5.0)
    assert not drainer.is_alive()


def test_manager_duplicate_trace_id_on_one_model_holds_each_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = _warnings(monkeypatch)
    manager = _manager(genrm={"judge": FakePool("judge")})
    first = manager.admit_request(Role.GENRM, "judge", "retry")
    second = manager.admit_request(Role.GENRM, "judge", "retry")
    assert first != second and "retry" not in (first, second)

    manager.complete_request(first)
    # A repeated completion and the trace ID itself release nothing.
    manager.complete_request(first)
    manager.complete_request("retry")
    drainer = _start(manager.drain, [Role.GENRM], 0.01)
    assert drainer.is_alive()
    assert f"trace=retry permit={second}" in warnings[-1] and first not in warnings[-1]

    manager.complete_request(second)
    drainer.join(5.0)
    assert not drainer.is_alive()


def test_manager_duplicate_trace_id_across_roles_holds_each_admission() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")}, teacher={"teacher": FakePool("teacher")})
    judge = manager.admit_request(Role.GENRM, "judge", "shared-trace")
    teacher = manager.admit_request(Role.TEACHER, "teacher", "shared-trace")

    manager.complete_request(judge)
    drainer = _start(manager.drain, [Role.TEACHER], 0.01)
    assert drainer.is_alive()

    manager.complete_request(teacher)
    drainer.join(5.0)
    assert not drainer.is_alive()


def test_manager_late_completion_of_an_aborted_permit_keeps_the_new_admission() -> None:
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    old = manager.admit_request(Role.GENRM, "judge", "req-1")
    manager.drain([Role.GENRM], timeout=0.01, force_abort=True)
    manager.activate(Role.GENRM, "judge")
    new = manager.admit_request(Role.GENRM, "judge", "req-1")

    # The aborted request finishes late with its own permit.
    manager.complete_request(old)
    drainer = _start(manager.drain, [Role.GENRM], 0.01)
    assert drainer.is_alive() and pool.calls.count("abort") == 1

    manager.complete_request(new)
    drainer.join(5.0)
    assert not drainer.is_alive()


def test_manager_drain_stays_closed_while_the_pool_reports_ready() -> None:
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    old = manager.admit_request(Role.GENRM, "judge", "old")

    drainer = _start(manager.drain, [Role.GENRM], 5.0)
    for _ in range(3):
        # The pool is still loaded and healthy, and says so.
        assert manager.health_check(Role.GENRM, "judge")
        manager.refresh(Role.GENRM, "judge")
        manager.publish(Role.GENRM, pool.observe())
        model = manager.snapshot(Role.GENRM).models[0]
        assert (model.state, model.admission) == (LifecycleState.DRAINING, False)
        with pytest.raises(RuntimeError, match="not ready"):
            manager.admit_request(Role.GENRM, "judge", "new")
    assert drainer.is_alive()

    manager.complete_request(old)
    drainer.join(5.0)
    assert not drainer.is_alive()
    manager.health_check(Role.GENRM, "judge")
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.GENRM, "judge", "new")

    # Only a controlled activate reopens it.
    manager.activate(Role.GENRM, "judge")
    manager.admit_request(Role.GENRM, "judge", "new")


def test_manager_reactivating_a_drained_model_reopens_without_loading_again() -> None:
    manager, pools = _shared()
    rollout = pools["rollout"]
    manager.activate(Role.ROLLOUT, "policy")
    manager.admit_request(Role.ROLLOUT, "policy", "req-1")
    manager.drain([Role.ROLLOUT], timeout=0.01, force_abort=True)
    assert rollout.calls[-1] == "abort"
    loads = rollout.calls.count("onload")

    manager.activate(Role.ROLLOUT, "policy", ["weights"])
    # Resuming memory that was never released is not idempotent in SGLang.
    assert rollout.calls.count("onload") == loads
    permit = manager.admit_request(Role.ROLLOUT, "policy", "req-2")

    # Once offloaded, it loads again.
    manager.complete_request(permit)
    manager.deactivate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT, "policy")
    assert rollout.calls.count("onload") == loads + 1


def test_manager_drain_of_a_failed_offload_keeps_it_dead_and_loads_on_activate() -> None:
    manager, pools = _shared()
    judge = pools["genrm"]
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    offload = judge.offload
    judge.offload = lambda: (_ for _ in ()).throw(RuntimeError("release failed"))
    with pytest.raises(RuntimeError, match="release failed"):
        manager.deactivate(Role.GENRM, "judge")
    judge.offload = offload

    manager.drain([Role.GENRM])
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DEAD
    loads = judge.calls.count("onload")
    manager.activate(Role.GENRM, "judge")
    assert judge.calls.count("onload") == loads + 1


def test_manager_drops_an_observation_that_spans_offload_and_onload() -> None:
    pool = FakePool("policy", version="1")
    manager = _manager(rollout={"policy": pool})
    observe = pool.observe

    def late_refresh() -> tuple[threading.Thread, threading.Event]:
        taken, land = threading.Event(), threading.Event()

        def slow_observe(serving: bool = True) -> ModelSnapshot:
            snapshot = observe(serving)
            taken.set()
            land.wait(5.0)
            return snapshot

        pool.observe = slow_observe
        thread = threading.Thread(target=manager.refresh, args=(Role.ROLLOUT, "policy"))
        thread.start()
        assert taken.wait(5.0)
        pool.observe = observe
        return thread, land

    # A READY observation taken before the offload lands while asleep.
    thread, land = late_refresh()
    manager.deactivate(Role.ROLLOUT, "policy")
    land.set()
    thread.join(5.0)
    assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.SLEEPING
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.ROLLOUT, "policy")

    # One taken before the offload lands after the next onload with new weights.
    manager.activate(Role.ROLLOUT, "policy")
    thread, land = late_refresh()
    manager.deactivate(Role.ROLLOUT, "policy")
    pool.version = "2"
    manager.activate(Role.ROLLOUT, "policy")
    land.set()
    thread.join(5.0)
    model = manager.snapshot(Role.ROLLOUT).models[0]
    assert (model.state, model.required_weight_version) == (LifecycleState.READY, "2")


def test_lifecycle_switch_offloads_outgoing_before_onloading_incoming() -> None:
    order: list[str] = []
    judge, teacher = FakePool("judge"), FakePool("teacher")
    teacher.onloaded = False
    manager = _manager(genrm={"judge": judge}, teacher={"teacher": teacher})
    judge.offload = lambda: (order.append("judge"), setattr(judge, "onloaded", False))
    teacher.onload = lambda tags=None: (order.append("teacher"), setattr(teacher, "onloaded", True))

    manager.lifecycle.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)

    assert order == ["judge", "teacher"]
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.SLEEPING
    assert manager.snapshot(Role.TEACHER).models[0].state == LifecycleState.READY


def test_manager_role_shutdown_unregisters_the_role(monkeypatch) -> None:
    # The last role's shutdown stops the Routers; the real module needs sglang.
    rollout = ModuleType("relax.distributed.ray.rollout")
    rollout.stop_launched_routers = lambda: None
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout)
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    manager.shutdown(Role.GENRM)
    assert pool.calls == ["shutdown"]
    assert manager.roles() == ()
    manager.register(Role.GENRM, {"judge": FakePool("judge")})


def test_manager_call_only_exposes_lifecycle_and_queries() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.call(Role.GENRM, "judge", "get_urls") == ["http://engine-0"]
    with pytest.raises(ValueError, match="Unsupported"):
        manager.call(Role.GENRM, "judge", "release_memory_occupation")


def test_model_handle_forwards_calls_through_manager() -> None:
    calls = []
    manager = SimpleNamespace(call=SimpleNamespace(remote=lambda *args, **kwargs: calls.append((args, kwargs))))
    ModelHandle(manager, Role.TEACHER, "default").activate.remote(tags=["kv_cache"])
    assert calls == [((Role.TEACHER, "default", "activate"), {"tags": ["kv_cache"]})]


def _shared(**roles: FakePool) -> tuple[InferenceManager, dict[str, FakePool]]:
    """Rollout, GenRM and Teacher on the same GPUs, each in its own phase."""
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest

    pools = {"rollout": FakePool("policy", version="1"), "genrm": FakePool("judge"), "teacher": FakePool("teacher")}
    pools.update(roles)
    manager = _manager(**{role: {pool.model_spec.name: pool} for role, pool in pools.items()})
    view = PlacementGroupView((0, 1), (0, 1), PlacementOwner.CONTROLLER, identity="shared")
    phases = {"rollout": "inference", "genrm": "genrm", "teacher": "teacher"}
    for role, pool in pools.items():
        request = PlacementRequest(f"{role}/{pool.model_spec.name}/group-0", "regular", 2, 1, 2, phases[role], 0)
        manager.plan_placement([request], view)
    manager.lifecycle.switch([Role.ROLLOUT, Role.GENRM, Role.TEACHER], [], timeout=0.01)
    return manager, pools


def _start(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args)
    thread.start()
    thread.join(0.1)
    return thread


def test_lifecycle_switch_keeps_scorers_mutually_exclusive() -> None:
    manager, pools = _shared()
    judge, teacher = pools["genrm"], pools["teacher"]
    both_ready = []

    def onload(pool, tags=None):
        pool.onloaded = True
        both_ready.append(judge.onloaded and teacher.onloaded)

    judge.onload = lambda tags=None: onload(judge)
    teacher.onload = lambda tags=None: onload(teacher)

    manager.lifecycle.switch([Role.ROLLOUT], [Role.GENRM], timeout=0.01)
    assert manager.snapshot(Role.TEACHER).phase == "genrm"
    waiter = _start(manager.lifecycle.switch, [Role.ROLLOUT], [Role.TEACHER], 5.0)
    # The teacher waits while GenRM holds the shared GPUs.
    assert waiter.is_alive() and not teacher.onloaded
    manager.lifecycle.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)

    assert not waiter.is_alive() and teacher.onloaded and not judge.onloaded
    assert both_ready == [False, False]
    assert manager.snapshot(Role.GENRM).phase == "teacher"


def test_lifecycle_switch_times_out_while_another_scorer_holds_the_gpus() -> None:
    manager, _ = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    manager.lifecycle.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)
    assert manager.snapshot(Role.TEACHER).models[0].state == LifecycleState.READY


def test_lifecycle_switch_rejects_activating_two_conflicting_roles() -> None:
    manager, pools = _shared()
    for roles in ([Role.ROLLOUT, Role.GENRM], [Role.ROLLOUT, Role.TEACHER], [Role.GENRM, Role.TEACHER]):
        with pytest.raises(ValueError, match="cannot be active together"):
            manager.lifecycle.switch([], roles, timeout=0.01)
    assert not any(pool.onloaded for pool in pools.values())


def test_manager_lifecycle_operations_are_idempotent() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    manager.activate(Role.GENRM, "judge")
    manager.deactivate(Role.GENRM, "judge")
    manager.deactivate(Role.GENRM, "judge")
    manager.shutdown(Role.GENRM, "judge")
    manager.shutdown(Role.GENRM, "judge")
    assert pools["genrm"].calls == ["offload", "onload", "offload", "shutdown"]


def test_lifecycle_switch_releases_the_gpus_when_a_scorer_fails_to_load() -> None:
    manager, pools = _shared()
    pools["genrm"].onload = lambda tags=None: (_ for _ in ()).throw(RuntimeError("OOM"))

    with pytest.raises(RuntimeError, match="OOM"):
        manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)

    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_snapshot_reports_the_generation_phase_without_a_scorer() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.snapshot(Role.GENRM).phase == "inference"
    assert manager.snapshot(Role.GENRM).to_dict()["phase"] == "inference"


def test_manager_topology_revision_bumps_when_a_replica_restarts_in_place() -> None:
    pool = FakePool("judge")
    pool.recover = lambda: None
    manager = _manager(genrm={"judge": pool})
    revision = manager.snapshot(Role.GENRM).topology_revision

    manager.recover(Role.GENRM, "judge")

    snapshot = manager.snapshot(Role.GENRM)
    assert snapshot.models[0].replicas[0].base_url == "http://engine-0"
    assert snapshot.topology_revision == revision + 1
    # Waking a sleeping replica is not a restart.
    manager.deactivate(Role.GENRM, "judge")
    manager.activate(Role.GENRM, "judge")
    assert manager.snapshot(Role.GENRM).topology_revision == revision + 1


def test_manager_direct_onload_blocks_a_conflicting_switch() -> None:
    manager, pools = _shared()
    manager.activate(Role.GENRM, "judge")

    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert not pools["teacher"].onloaded


def test_manager_direct_offload_releases_the_gpus() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    waiter = _start(manager.lifecycle.switch, [], [Role.TEACHER], 5.0)
    assert waiter.is_alive()

    # Training offloads the scorer directly, not through a switch.
    manager.deactivate(Role.GENRM, "judge")
    waiter.join(5.0)

    assert not waiter.is_alive() and pools["teacher"].onloaded
    assert manager.snapshot(Role.GENRM).phase == "teacher"


def test_manager_recover_waits_for_a_conflicting_role() -> None:
    manager, pools = _shared()
    recovered = []
    pools["teacher"].recover = lambda: recovered.append(True)
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)

    waiter = _start(manager.recover, Role.TEACHER, "teacher")
    assert waiter.is_alive() and not recovered
    manager.lifecycle.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)

    assert not waiter.is_alive() and recovered


def test_manager_onload_outside_a_switch_waits_for_the_scorer() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    # The role that holds the GPUs may still reload itself.
    manager.activate(Role.GENRM, "judge")

    # A weight sync waits for the release.
    waiter = _start(manager.activate, Role.ROLLOUT, "policy")
    assert waiter.is_alive() and not pools["rollout"].onloaded

    manager.lifecycle.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)
    assert not waiter.is_alive() and pools["rollout"].onloaded
    assert manager.snapshot(Role.ROLLOUT).phase == "inference"


def test_manager_split_roles_load_without_waiting() -> None:
    judge, policy = FakePool("judge"), FakePool("policy", version="1")
    manager = _manager(genrm={"judge": judge}, rollout={"policy": policy})
    manager.deactivate(Role.ROLLOUT, "policy")
    manager.activate(Role.GENRM, "judge")
    manager.activate(Role.ROLLOUT, "policy")
    assert judge.onloaded and policy.onloaded


def test_manager_rollout_offload_waits_for_a_request_admitted_by_the_gateway() -> None:
    from starlette.requests import Request
    from starlette.responses import Response

    from relax.components.inference_gateway import InferenceGateway

    manager, pools = _shared()
    policy = pools["rollout"]
    manager.activate(Role.ROLLOUT, "policy")
    paused = []
    manager._rollout_pool = SimpleNamespace(
        health_monitoring_pause=lambda: paused.append(True), weights_changing=lambda: False
    )
    handle = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=partial(asyncio.to_thread, getattr(manager, name)))
            for name in ("snapshot", "admit_request", "complete_request")
        }
    )
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=handle)

    def request() -> Request:
        async def receive():
            return {"type": "http.request", "body": b'{"text": "hi"}', "more_body": False}

        return Request({"type": "http", "method": "POST", "path": "/generate", "headers": []}, receive)

    async def scenario() -> None:
        admitted, forward = asyncio.Event(), asyncio.Event()
        engine_loaded = []

        async def hold_before_forwarding(*args, **kwargs):
            # Admitted, not yet sent to the engine.
            admitted.set()
            await forward.wait()
            engine_loaded.append(policy.onloaded)
            return Response()

        gateway._forward_admitted = hold_before_forwarding
        proxied = asyncio.create_task(gateway.proxy(request(), "generate"))
        await admitted.wait()
        # What components/rollout.py runs after each generate.
        offload = asyncio.create_task(asyncio.to_thread(manager.deactivate, Role.ROLLOUT))
        await asyncio.sleep(0.1)

        assert not offload.done() and policy.calls[-1] == "onload"
        assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.DRAINING
        assert (await gateway.proxy(request(), "generate")).status_code == 503
        with pytest.raises(TimeoutError, match="rollout"):
            await asyncio.to_thread(manager.lifecycle.switch, [], [Role.GENRM], 0.01)

        forward.set()
        await proxied
        await offload
        assert engine_loaded == [True] and policy.calls[-1] == "offload"

    asyncio.run(scenario())
    assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.SLEEPING
    assert manager.rollout_released() and paused == [True]


def test_manager_gateway_retries_with_one_request_id_each_hold_the_drain() -> None:
    from starlette.requests import Request
    from starlette.responses import Response

    from relax.components.inference_gateway import InferenceGateway

    manager = _manager(rollout={"policy": FakePool("policy", version="1")})
    handle = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=partial(asyncio.to_thread, getattr(manager, name)))
            for name in ("snapshot", "admit_request", "complete_request")
        }
    )
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=handle)

    def request() -> Request:
        async def receive():
            return {"type": "http.request", "body": b'{"text": "hi"}', "more_body": False}

        headers = [(b"x-relax-request-id", b"retry")]
        return Request({"type": "http", "method": "POST", "path": "/generate", "headers": headers}, receive)

    async def scenario() -> None:
        released = {"first": asyncio.Event(), "second": asyncio.Event()}
        admitted = asyncio.Queue()
        rids = []

        async def hold(request, path, inflight, *args, **kwargs):
            rids.append(inflight.rid)
            name = "first" if len(rids) == 1 else "second"
            await admitted.put(name)
            await released[name].wait()
            return Response()

        gateway._forward_admitted = hold
        first = asyncio.create_task(gateway.proxy(request(), "generate"))
        assert await admitted.get() == "first"
        second = asyncio.create_task(gateway.proxy(request(), "generate"))
        assert await admitted.get() == "second"
        # The caller's ID only traces; each attempt runs under its own engine rid.
        assert len(set(rids)) == 2 and "retry" not in rids

        released["first"].set()
        await first
        drain = asyncio.create_task(asyncio.to_thread(manager.drain, [Role.ROLLOUT], 0.01))
        await asyncio.sleep(0.1)
        assert not drain.done()

        released["second"].set()
        await second
        await asyncio.wait_for(drain, 5.0)

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(gateway.close())


class BusyEnginePool(FakePool):
    """A pool whose engine keeps running requests its HTTP callers dropped.

    As in SGLang, an abort is only acknowledged: the requests leave the running
    batch ``exit_after`` polls later, or never with ``None``. ``offload`` polls
    until the engine is idle and fails after ``flush_polls``, the way
    ``release_memory_occupation`` waits for ``/flush_cache``.
    """

    def __init__(self, name: str, *, exit_after: int | None = None, flush_polls: int = 3) -> None:
        super().__init__(name)
        self.running: set[str] = set()
        self.aborted = False
        self.exit_after = exit_after
        self.flush_polls = flush_polls

    def _idle(self) -> bool:
        if self.aborted and self.exit_after is not None:
            if self.exit_after == 0:
                self.running.clear()
            self.exit_after -= 1
        return not self.running

    def offload(self) -> None:
        for _ in range(self.flush_polls):
            if self._idle():
                return super().offload()
        raise TimeoutError("Timeout while flushing cache.")


def _engine_gateway(manager: InferenceManager, pool: BusyEnginePool, *, abort_fails: bool = False, hold=None):
    """A GenRM Gateway in front of ``pool``'s engine, whose ``/generate``
    starts running and then loses its HTTP connection, or waits on ``hold``."""
    import json

    import httpx

    from relax.components.inference_gateway import InferenceGateway

    async def engine(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/generate":
            pool.running.add(json.loads(request.content)["rid"])
            if hold is None:
                raise httpx.ReadTimeout("connection lost", request=request)
            await hold.wait()
            return httpx.Response(200, json={})
        if path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine-0"}]})
        if path == "/abort_request":
            if abort_fails:
                raise httpx.ConnectError("abort lost", request=request)
            pool.aborted = True
            return httpx.Response(200, json={})
        raise AssertionError(path)

    handle = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=partial(asyncio.to_thread, getattr(manager, name)))
            for name in ("snapshot", "admit_request", "complete_request")
        }
    )
    gateway = InferenceGateway(Role.GENRM, manager_handle=handle)
    gateway._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    return gateway


def _gateway_request():
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": b'{"text": "hi"}', "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/generate", "headers": []}, receive)


def _record_onload(pool: FakePool, engine: BusyEnginePool) -> list[tuple[set[str], bool]]:
    """Record whether ``engine`` was idle and released when ``pool`` loads."""
    seen: list[tuple[set[str], bool]] = []
    onload = pool.onload

    def recording_onload(tags=None) -> None:
        seen.append((set(engine.running), engine.onloaded))
        onload(tags)

    pool.onload = recording_onload
    return seen


@pytest.mark.parametrize(
    "case", ["abort_acked_then_exits", "abort_acked_never_exits", "abort_fails"], ids=lambda case: case
)
def test_manager_switch_waits_for_the_engine_after_the_gateway_lets_go(case: str) -> None:
    judge = BusyEnginePool("judge", exit_after=1 if case == "abort_acked_then_exits" else None)
    manager, pools = _shared(genrm=judge)
    teacher_loads = _record_onload(pools["teacher"], judge)
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    gateway = _engine_gateway(manager, judge, abort_fails=case == "abort_fails")

    response = asyncio.run(gateway.proxy(_gateway_request(), "generate"))
    asyncio.run(gateway.close())
    # The Gateway let go and returned its permit; the engine still runs.
    assert response.status_code == 502
    assert judge.running and not manager._inflight

    if case == "abort_acked_then_exits":
        manager.lifecycle.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)
        assert teacher_loads == [(set(), False)]
        return

    with pytest.raises(TimeoutError, match="flushing"):
        manager.lifecycle.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)
    # The unconfirmed release keeps the GPUs fenced.
    assert not teacher_loads and judge.onloaded
    assert manager.resident(Role.GENRM)
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DEAD
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)

    # Once the engine finishes, a retried switch hands the GPUs over.
    judge.running.clear()
    manager.lifecycle.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)
    assert teacher_loads == [(set(), False)]


def test_manager_request_cancelled_during_a_scoring_switch_waits_for_the_engine() -> None:
    judge = BusyEnginePool("judge", exit_after=1)
    manager, pools = _shared(genrm=judge)
    teacher_loads = _record_onload(pools["teacher"], judge)
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)

    async def scenario() -> None:
        gateway = _engine_gateway(manager, judge, hold=asyncio.Event())
        try:
            proxied = asyncio.create_task(gateway.proxy(_gateway_request(), "generate"))
            while not judge.running:
                await asyncio.sleep(0.01)
            switch = asyncio.create_task(
                asyncio.to_thread(manager.lifecycle.switch, [Role.GENRM], [Role.TEACHER], 5.0)
            )
            await asyncio.sleep(0.1)
            # The switch waits for the admitted request.
            assert not switch.done() and not teacher_loads

            # The caller gives up mid-switch: the Gateway aborts, completes,
            # and the release still waits for the engine to go idle.
            proxied.cancel()
            with pytest.raises(asyncio.CancelledError):
                await proxied
            await asyncio.wait_for(switch, 5.0)
        finally:
            await gateway.close()

    asyncio.run(scenario())
    assert judge.aborted and teacher_loads == [(set(), False)]


def test_manager_deactivate_keeps_waiting_and_the_gpus_held_without_force_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings = _warnings(monkeypatch)
    manager, pools = _shared()
    judge = pools["genrm"]
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    permit = manager.admit_request(Role.GENRM, "judge", "req-1")

    releaser = _start(manager.deactivate, Role.GENRM, None, 0.01)
    assert releaser.is_alive() and len(warnings) > 1
    # Admission stays closed and the memory is still held.
    assert judge.calls[-1] == "onload"
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DRAINING
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)

    manager.complete_request(permit)
    releaser.join(5.0)
    assert not releaser.is_alive() and judge.calls[-1] == "offload"
    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_force_abort_aborts_the_engines_before_releasing() -> None:
    manager, pools = _shared()
    judge = pools["genrm"]
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    permit = manager.admit_request(Role.GENRM, "judge", "req-1")

    manager.deactivate(Role.GENRM, timeout=0.01, force_abort=True)

    assert judge.calls[-2:] == ["abort", "offload"]
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.SLEEPING
    # The aborted request was forgotten; its late completion is harmless.
    manager.complete_request(permit)
    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_failed_force_abort_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    judge = pools["genrm"]
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    manager.admit_request(Role.GENRM, "judge", "req-1")
    judge.abort_requests = lambda: (_ for _ in ()).throw(RuntimeError("abort failed"))

    with pytest.raises(RuntimeError, match="abort failed"):
        manager.deactivate(Role.GENRM, timeout=0.01, force_abort=True)

    assert judge.calls[-1] == "onload"
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DRAINING
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.GENRM, "judge")
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert not pools["teacher"].onloaded


def test_manager_shutdown_waits_for_admitted_requests_before_stopping() -> None:
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    permit = manager.admit_request(Role.GENRM, "judge", "req-1")

    stopper = _start(manager.shutdown, Role.GENRM, "judge")
    assert stopper.is_alive() and "shutdown" not in pool.calls
    manager.complete_request(permit)
    stopper.join(5.0)
    assert not stopper.is_alive() and pool.calls == ["shutdown"]


def test_manager_shutdown_aborts_requests_after_its_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_manager, "SHUTDOWN_DRAIN_TIMEOUT_S", 0.01)
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    manager.admit_request(Role.GENRM, "judge", "req-1")

    manager.shutdown(Role.GENRM, "judge")
    assert pool.calls == ["abort", "shutdown"]


def test_manager_failed_offload_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    pools["genrm"].offload = lambda: (_ for _ in ()).throw(RuntimeError("release failed"))

    with pytest.raises(RuntimeError, match="release failed"):
        manager.deactivate(Role.GENRM, "judge")

    # DEAD, but the memory may still be held.
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DEAD
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert not pools["teacher"].onloaded


def test_manager_shutdown_releases_the_gpus_only_once_it_completes() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    stopping, stopped = threading.Event(), threading.Event()

    def shutdown(planner) -> None:
        stopping.set()
        stopped.wait(5.0)

    pools["genrm"].shutdown = shutdown
    closer = _start(manager.shutdown, Role.GENRM, "judge")
    assert stopping.wait(5.0)
    waiter = _start(manager.lifecycle.switch, [], [Role.TEACHER], 5.0)
    assert waiter.is_alive() and not pools["teacher"].onloaded

    stopped.set()
    closer.join(5.0)
    waiter.join(5.0)
    assert not waiter.is_alive() and pools["teacher"].onloaded


def test_manager_failed_shutdown_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    pools["genrm"].shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError, match="kill failed"):
        manager.shutdown(Role.GENRM, "judge")
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)


def test_manager_weights_only_onload_holds_the_gpus() -> None:
    manager, pools = _shared()
    rollout = pools["rollout"]

    def onload(tags=None) -> None:
        # Like the engine pool, only a KV-cache onload reports the model loaded.
        rollout.onloaded = tags is None

    rollout.onload = onload
    manager.activate(Role.ROLLOUT, "policy", ["weights"])

    assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.SLEEPING
    with pytest.raises(TimeoutError, match="rollout"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    manager.lifecycle.switch([Role.ROLLOUT], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_failed_role_shutdown_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.lifecycle.switch([], [Role.GENRM], timeout=0.01)
    stop = pools["genrm"].shutdown
    pools["genrm"].shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError, match="genrm"):
        manager.shutdown(Role.GENRM)
    assert "genrm" in manager.roles()
    with pytest.raises(TimeoutError, match="genrm"):
        manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)

    pools["genrm"].shutdown = stop
    manager.shutdown(Role.GENRM)
    assert "genrm" not in manager.roles()
    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_role_shutdown_retry_stops_only_the_failed_models() -> None:
    judge, backup = FakePool("judge"), FakePool("backup")
    manager = _manager(genrm={"judge": judge, "backup": backup})
    backup.shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError):
        manager.shutdown(Role.GENRM)
    assert manager.model_ids(Role.GENRM) == ("backup",)
    assert judge.calls == ["shutdown"]


def test_manager_role_spec_derives_deployment_from_placement() -> None:
    from relax.engine.inference.types import DeploymentMode, WorkloadType

    manager, _ = _shared()
    teacher = manager.role_spec(Role.TEACHER)
    assert teacher.workload == WorkloadType.DISTILLATION
    assert (teacher.deployment.mode, teacher.deployment.phase) == (DeploymentMode.DEFER, "teacher")
    rollout = manager.role_spec(Role.ROLLOUT)
    assert (rollout.deployment.mode, rollout.deployment.phase) == (DeploymentMode.SPLIT, "inference")
    assert [model.name for model in rollout.models] == ["policy"]
    # A role without a placement in the shared ledger runs on its own GPUs.
    solo = _manager(genrm={"judge": FakePool("judge")})
    assert solo.role_spec(Role.GENRM).deployment.mode == DeploymentMode.DECOUPLED


def test_manager_model_placement_names_the_activation_group() -> None:
    manager, _ = _shared()
    teacher = manager.model_placement(Role.TEACHER, "teacher")
    assert (teacher.activation_phase, teacher.bundle_indices) == ("teacher", (0, 1))
    assert teacher.activation_group is not None
    rollout = manager.model_placement(Role.ROLLOUT, "policy")
    assert (rollout.activation_group, rollout.activation_phase) == (None, None)


def test_lifecycle_enter_phase_hands_the_gpus_to_the_phase_roles() -> None:
    manager, pools = _shared()
    manager.activate(Role.ROLLOUT)

    manager.enter_phase("teacher", timeout=0.01)
    assert pools["teacher"].onloaded and not pools["rollout"].onloaded
    assert manager.snapshot(Role.ROLLOUT).phase == "teacher"

    # Generation waits for the scoring phase to be left before it comes back.
    with pytest.raises(TimeoutError, match="teacher"):
        manager.enter_phase("inference", timeout=0.01)
    assert pools["teacher"].onloaded
    manager.leave_phase("teacher", timeout=0.01)
    manager.enter_phase("inference", timeout=0.01)
    assert pools["rollout"].onloaded and not pools["teacher"].onloaded

    manager.enter_phase("genrm", timeout=0.01)
    manager.leave_phase("genrm", timeout=0.01)
    assert not any(pool.onloaded for pool in pools.values())
    assert manager.snapshot(Role.GENRM).phase == "inference"


def test_lifecycle_enter_phase_releases_a_role_activated_while_it_waited() -> None:
    manager, pools = _shared()
    teacher, rollout = pools["teacher"], pools["rollout"]
    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    offloading, finish_offload = threading.Event(), threading.Event()

    def slow_offload() -> None:
        offloading.set()
        finish_offload.wait(5.0)
        teacher.onloaded = False

    teacher.offload = slow_offload
    # Generation comes back while the GenRM stage is already waiting for the
    # GPUs; it must release generation, not wait for it to time out.
    generation = _start(manager.enter_phase, "inference", 5.0)
    assert offloading.wait(5.0)
    genrm = _start(manager.enter_phase, "genrm", 5.0)
    assert genrm.is_alive()
    finish_offload.set()
    generation.join(5.0)
    genrm.join(5.0)

    assert not generation.is_alive() and not genrm.is_alive()
    assert pools["genrm"].onloaded and not rollout.onloaded and not teacher.onloaded


def test_lifecycle_enter_phase_waits_for_a_scoring_phase_to_be_left() -> None:
    manager, pools = _shared()
    manager.enter_phase("teacher", timeout=0.01)

    genrm = _start(manager.enter_phase, "genrm", 5.0)
    # A held scoring phase is not taken over mid-way.
    assert genrm.is_alive() and pools["teacher"].onloaded and not pools["genrm"].onloaded
    manager.leave_phase("teacher", timeout=0.01)
    genrm.join(5.0)

    assert not genrm.is_alive() and pools["genrm"].onloaded and not pools["teacher"].onloaded


def test_lifecycle_enter_phase_rejects_a_phase_without_turn_taking_roles() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    with pytest.raises(ValueError, match="No role takes turns"):
        manager.enter_phase("teacher", timeout=0.01)


def test_lifecycle_rollout_released_follows_rollout_residency() -> None:
    manager, pools = _shared()
    rollout = pools["rollout"]
    assert manager.rollout_released()

    # Closed while a load is still running, even though nothing is READY yet.
    during_load = []
    rollout.onload = lambda tags=None: during_load.append(manager.rollout_released())
    manager.activate(Role.ROLLOUT, "policy", ["weights"])
    assert during_load == [False] and not manager.rollout_released()

    # A failed offload may still hold the memory.
    rollout.offload = lambda: (_ for _ in ()).throw(RuntimeError("release failed"))
    with pytest.raises(RuntimeError, match="release failed"):
        manager.deactivate(Role.ROLLOUT)
    assert not manager.rollout_released()

    rollout.offload = lambda: setattr(rollout, "onloaded", False)
    manager.deactivate(Role.ROLLOUT)
    assert manager.rollout_released()


def test_manager_role_activate_releases_the_models_it_loaded_when_one_fails() -> None:
    first, second, warm = FakePool("first"), FakePool("second"), FakePool("warm")
    manager = _manager(genrm={"warm": warm, "first": first, "second": second})
    manager.deactivate(Role.GENRM, "first")
    manager.deactivate(Role.GENRM, "second")
    second.onload = lambda tags=None: (_ for _ in ()).throw(RuntimeError("OOM"))

    with pytest.raises(RuntimeError, match="OOM"):
        manager.activate(Role.GENRM)

    assert first.calls == ["offload", "onload", "offload"]
    assert second.calls == ["offload", "offload"]
    # A model that was already resident keeps serving.
    assert warm.calls == []
    states = {model.model_id: model.state for model in manager.snapshot(Role.GENRM).models}
    assert states == {
        "warm": LifecycleState.READY,
        "first": LifecycleState.SLEEPING,
        "second": LifecycleState.SLEEPING,
    }
    assert manager.resident(Role.GENRM)
    manager.deactivate(Role.GENRM, "warm")
    assert not manager.resident(Role.GENRM)


def test_manager_model_activate_releases_the_model_when_its_load_fails() -> None:
    manager, pools = _shared()
    pools["genrm"].onload = lambda tags=None: (_ for _ in ()).throw(RuntimeError("OOM"))

    with pytest.raises(RuntimeError, match="OOM"):
        manager.activate(Role.GENRM, "judge")

    assert not manager.resident(Role.GENRM)
    manager.lifecycle.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


class FakeEnginePool:
    """The rollout engine pool primitives the Manager calls; none calls
    back."""

    def __init__(self) -> None:
        self.fenced = False
        self.lease_refused = False
        self.calls: list[str] = []

    def weights_changing(self) -> bool:
        return self.fenced

    def health_monitoring_pause(self) -> None:
        self.calls.append("pause")

    def _set_weight_updating(self, is_updating: bool) -> bool:
        self.calls.append(f"lease={is_updating}")
        return not (is_updating and self.lease_refused)

    def get_rollout_engines_and_lock(self, model_name=None):
        return model_name


def _rollout_role(*names: str) -> tuple[InferenceManager, FakeEnginePool, dict[str, FakePool]]:
    """A Manager owning a rollout role of resident-free, sleeping models."""
    pools = {name: FakePool(name, version="1") for name in names}
    for pool in pools.values():
        pool.onloaded = False
    manager = _manager(rollout=pools)
    engine_pool = FakeEnginePool()
    manager._rollout_pool = engine_pool
    for name in names:
        manager._model(Role.ROLLOUT, name).resident = False
    return manager, engine_pool, pools


def _state(manager: InferenceManager, name: str = "policy") -> ModelSnapshot:
    return next(model for model in manager.snapshot(Role.ROLLOUT).models if model.model_id == name)


def _no_reentry(manager: InferenceManager) -> list[str]:
    """Record every activate/deactivate entry; no pool may add one."""
    calls: list[str] = []
    for method in ("activate", "deactivate"):
        original = getattr(manager, method)

        def wrapper(*args, _original=original, _method=method, **kwargs):
            calls.append(_method)
            return _original(*args, **kwargs)

        setattr(manager, method, wrapper)
    return calls


def test_manager_rollout_staged_onload_publishes_ready_only_with_kv_cache() -> None:
    manager, _, pools = _rollout_role("policy", "draft")
    calls = _no_reentry(manager)

    manager.activate(Role.ROLLOUT, tags=["weights"])
    for name, pool in pools.items():
        assert pool.calls == ["onload"] and not pool.serving
        assert _state(manager, name).state == LifecycleState.STARTING and not _state(manager, name).admission
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.ROLLOUT, "policy")

    manager.activate(Role.ROLLOUT, tags=["kv_cache", "cuda_graph"])
    for name, pool in pools.items():
        assert pool.calls == ["onload", "onload"]
        model = _state(manager, name)
        assert (model.state, model.admission, model.required_weight_version) == (LifecycleState.READY, True, "1")
    assert calls == ["activate", "activate"]


def test_manager_rollout_activating_ready_models_does_not_load_again() -> None:
    manager, _, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT, "policy", ["weights"])
    assert pools["policy"].calls == ["onload"]
    assert _state(manager).state == LifecycleState.READY


def test_manager_rollout_partial_onload_failure_releases_what_it_loaded() -> None:
    manager, _, pools = _rollout_role("policy", "draft")

    def fail(tags=None):
        pools["draft"].calls.append("onload")
        raise RuntimeError("onload failed")

    pools["draft"].onload = fail
    with pytest.raises(RuntimeError, match="onload failed"):
        manager.activate(Role.ROLLOUT)

    assert pools["policy"].calls == ["onload", "offload"]
    assert pools["draft"].calls == ["onload", "offload"]
    assert not manager.resident(Role.ROLLOUT)
    for model in manager.snapshot(Role.ROLLOUT).models:
        assert model.state == LifecycleState.SLEEPING and not model.admission


def test_manager_rollout_repeated_deactivate_and_activate_are_idempotent() -> None:
    manager, engine_pool, pools = _rollout_role("policy", "draft")
    calls = _no_reentry(manager)
    manager.activate(Role.ROLLOUT)

    manager.deactivate(Role.ROLLOUT)
    manager.deactivate(Role.ROLLOUT)
    assert engine_pool.calls == ["pause", "pause"]
    for pool in pools.values():
        assert pool.calls == ["onload", "offload"]
    assert not manager.resident(Role.ROLLOUT)

    manager.activate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT)
    for pool in pools.values():
        assert pool.calls == ["onload", "offload", "onload"]
    assert all(model.state == LifecycleState.READY for model in manager.snapshot(Role.ROLLOUT).models)
    assert len(calls) == 5


def test_manager_rollout_release_goes_by_what_each_model_holds() -> None:
    manager, _, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    manager.deactivate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT, "policy")
    assert _state(manager).state == LifecycleState.READY

    manager.deactivate(Role.ROLLOUT)
    assert pools["policy"].calls == ["onload", "offload", "onload", "offload"]
    assert not manager.resident(Role.ROLLOUT)


def test_manager_backend_weight_update_closes_admission_until_it_ends() -> None:
    manager, _, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    observe = pools["policy"].observe
    taken, land = threading.Event(), threading.Event()

    def slow_observe(serving: bool = True) -> ModelSnapshot:
        snapshot = observe(serving)
        taken.set()
        land.wait(5.0)
        return snapshot

    # A READY observation taken before the update starts lands during it.
    pools["policy"].observe = slow_observe
    late = threading.Thread(target=manager.refresh, args=(Role.ROLLOUT, "policy"))
    late.start()
    assert taken.wait(5.0)
    pools["policy"].observe = observe
    assert manager.begin_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_BACKEND)
    land.set()
    late.join(5.0)
    assert _state(manager).state == LifecycleState.STARTING
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.ROLLOUT, "policy")

    # Neither an observation nor a full reload reopens it meanwhile.
    manager.refresh(Role.ROLLOUT, "policy")
    manager.deactivate(Role.ROLLOUT)
    manager.activate(Role.ROLLOUT)
    assert _state(manager).state == LifecycleState.STARTING and not pools["policy"].serving

    pools["policy"].version = "2"
    manager.end_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_BACKEND)
    model = _state(manager)
    assert (model.state, model.required_weight_version) == (LifecycleState.READY, "2")
    # Idempotent.
    manager.end_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_BACKEND)
    assert _state(manager).state == LifecycleState.READY


def test_manager_fully_async_weight_update_takes_the_pool_lease() -> None:
    manager, engine_pool, _ = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    engine_pool.lease_refused = True
    assert not manager.begin_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_FULLY_ASYNC)
    assert _state(manager).state == LifecycleState.READY

    engine_pool.lease_refused = False
    assert manager.begin_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_FULLY_ASYNC)
    assert _state(manager).state == LifecycleState.STARTING
    manager.end_weight_update(Role.ROLLOUT, inference_manager.WEIGHT_UPDATE_FULLY_ASYNC)
    assert engine_pool.calls == ["lease=True", "lease=True", "lease=False"]
    assert _state(manager).state == LifecycleState.READY


def test_manager_keeps_rollout_closed_while_the_pool_syncs_new_engines() -> None:
    manager, engine_pool, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    engine_pool.fenced = True
    manager.refresh(Role.ROLLOUT, "policy")
    assert _state(manager).state == LifecycleState.STARTING and not pools["policy"].serving
    engine_pool.fenced = False
    manager.refresh(Role.ROLLOUT, "policy")
    assert _state(manager).state == LifecycleState.READY


def test_manager_topology_events_only_record_and_the_manager_observes() -> None:
    manager, _, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    events = manager.topology_events(Role.ROLLOUT)
    observed = threading.Event()
    observe = pools["policy"].observe

    def watched(serving: bool = True) -> ModelSnapshot:
        snapshot = observe(serving)
        observed.set()
        return snapshot

    pools["policy"].observe = watched
    events.changed("policy")
    assert _state(manager).state == LifecycleState.STARTING and not observed.is_set()
    # Events of a model the Manager does not hold yet are ignored.
    events.changed("unregistered")
    events.settled("unregistered")

    events.settled("policy")
    assert observed.wait(5.0)
    deadline = time.monotonic() + 5.0
    while _state(manager).state != LifecycleState.READY and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _state(manager).state == LifecycleState.READY


def test_manager_settled_topology_is_incomplete_until_observed() -> None:
    manager, _, _ = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    manager._refresher = threading.current_thread()  # Keep the refresher thread out of it.

    # A scale-out reports only that it settled: until the Manager observes,
    # discovery must not present the old replicas as the complete view.
    manager.topology_events(Role.ROLLOUT).settled("policy")
    model = _state(manager)
    assert model.issues == (("policy", "topology_unsettled"),)
    assert model.state == LifecycleState.READY and model.admission
    assert not manager.snapshot(Role.ROLLOUT).observation()["complete"]

    manager._refresh_stale()
    assert _state(manager).issues == ()
    assert manager.snapshot(Role.ROLLOUT).observation()["complete"]


def test_manager_rollout_operation_observes_what_the_operation_changed() -> None:
    manager, engine_pool, pools = _rollout_role("policy")
    manager.activate(Role.ROLLOUT)
    events = manager.topology_events(Role.ROLLOUT)
    manager._refresher = threading.current_thread()  # Keep the refresher thread out of it.

    def scale(model_name):
        events.changed(model_name)
        events.settled(model_name)
        return "done"

    engine_pool.execute_scale_out = scale
    assert manager.rollout_operation("execute_scale_out", "policy") == "done"
    assert _state(manager).state == LifecycleState.READY


def test_manager_rollout_operation_refuses_the_weight_update_lease() -> None:
    manager, engine_pool, _ = _rollout_role("policy")

    with pytest.raises(ValueError, match="Unsupported rollout pool method"):
        manager.rollout_operation("_set_weight_updating", True)
    with pytest.raises(ValueError, match="Unsupported rollout pool method"):
        manager.rollout_operation("set_weight_updating", True)
    assert engine_pool.calls == []

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

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

    def observe(self) -> ModelSnapshot:
        state = LifecycleState.READY if self.onloaded else LifecycleState.SLEEPING
        replica = ReplicaSnapshot(f"{self.model_spec.name}/replica-0", state, "http://engine-0", self.version)
        return ModelSnapshot(
            self.model_spec.name,
            (replica,),
            "http://router" if self.model_spec.needs_router else None,
            state,
            admission=self.onloaded,
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


def test_manager_drain_closes_admission_and_waits_for_requests() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.admit_request(Role.GENRM, "judge", "req-1") == "req-1"

    with pytest.raises(TimeoutError):
        manager.drain([Role.GENRM], timeout=0.01)
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.GENRM, "judge")

    manager.complete_request("req-1")
    manager.drain([Role.GENRM], timeout=0.01)


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

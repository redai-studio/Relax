# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The async actor around the InferenceManager: where each call runs."""

import asyncio
import os
import threading
import time

import pytest


ray = pytest.importorskip("ray")

from relax.distributed.ray import inference_manager  # noqa: E402
from relax.distributed.ray.inference_manager import (  # noqa: E402
    _CONCURRENCY_GROUPS,
    _MANAGER_CONCURRENCY,
    InferenceManager,
    InferenceManagerActor,
    _InferenceManagerActor,
)
from relax.engine.inference.types import Role  # noqa: E402


# Every Manager method some caller reaches through the actor handle.
_REMOTE_API = (
    "snapshot",
    "admit_request",
    "complete_request",
    "rollout_released",
    "activate",
    "deactivate",
    "drain",
    "recover",
    "health_check",
    "shutdown",
    "shutdown_all",
    "call",
    "enter_phase",
    "leave_phase",
    "create_role",
    "refresh",
    "model_spec",
    "create_rollout_role",
    "begin_rollout",
    "rollout_operation",
    "begin_weight_update",
    "end_weight_update",
)


def test_inference_manager_actor_exposes_the_remote_api_under_the_same_names() -> None:
    for name in _REMOTE_API:
        assert callable(getattr(InferenceManager, name)), name
        assert callable(getattr(_InferenceManagerActor, name)), name
    groups = {
        name: getattr(_InferenceManagerActor, name).__ray_concurrency_group__
        for name in _REMOTE_API
        if hasattr(getattr(_InferenceManagerActor, name), "__ray_concurrency_group__")
    }
    assert groups["snapshot"] == groups["admit_request"] == inference_manager.REQUEST_PERMIT_GROUP
    assert groups["complete_request"] == inference_manager.REQUEST_CONTROL_GROUP
    assert "activate" not in groups and "deactivate" not in groups
    assert InferenceManagerActor is not None


@pytest.fixture(scope="module")
def local_ray():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    if started:
        ray.shutdown()


def _probe_actor(sleep_s: float):
    """A real actor over static models; ``health_check`` blocks ``sleep_s``."""
    from relax.engine.inference.config import InferenceModelSpec
    from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot, RouteMode, WeightSource

    class Pool:
        def __init__(self, name: str) -> None:
            self.model_spec = InferenceModelSpec(name, weight_source=WeightSource.STATIC, route_mode=RouteMode.DIRECT)
            self.onloaded = True

        def observe(self, serving: bool = True) -> ModelSnapshot:
            state = LifecycleState.READY if self.onloaded and serving else LifecycleState.SLEEPING
            replica = ReplicaSnapshot(f"{self.model_spec.name}/replica-0", state, "http://engine-0", None)
            return ModelSnapshot(self.model_spec.name, (replica,), None, state, state == LifecycleState.READY)

        def onload(self, tags=None) -> None:
            self.onloaded = True

        def offload(self) -> None:
            self.onloaded = False

        def health_check(self) -> bool:
            time.sleep(sleep_s)
            return True

        def abort_requests(self) -> None:
            pass

    class Probe(_InferenceManagerActor):
        def __init__(self) -> None:
            super().__init__()
            self.manager.register("genrm", {"judge": Pool("judge")})
            self.manager.register("teacher", {"teacher": Pool("teacher")})

        async def threads(self) -> int:
            return len(os.listdir(f"/proc/{os.getpid()}/task"))

    actor_cls = ray.remote(num_cpus=0, concurrency_groups=_CONCURRENCY_GROUPS)(Probe)
    return actor_cls.options(max_concurrency=_MANAGER_CONCURRENCY).remote()


def _timed(ref) -> float:
    started = time.monotonic()
    ray.get(ref, timeout=30)
    return time.monotonic() - started


def test_inference_manager_actor_serves_admission_while_lifecycle_calls_block(local_ray) -> None:
    actor = _probe_actor(sleep_s=3.0)
    permit = ray.get(actor.admit_request.remote("genrm", "judge", "held"))
    # A release waits for that request, and more blocking calls than there are
    # lifecycle threads queue behind it: they serialize on the teacher's model
    # lock, so every lifecycle thread is taken.
    release = actor.deactivate.remote("genrm", "judge")
    [actor.health_check.remote("teacher", "teacher") for _ in range(100)]
    ready, _ = ray.wait([release], timeout=0.5)
    assert not ready

    # Discovery, admission and completion of another role stay fast.
    assert _timed(actor.snapshot.remote("teacher")) < 2.0
    other = ray.get(actor.admit_request.remote("teacher", "teacher", "other"), timeout=2.0)
    assert _timed(actor.complete_request.remote(other)) < 2.0
    assert _timed(actor.rollout_released.remote()) < 2.0

    # The completion the release waits for gets in and ends it.
    ray.get(actor.complete_request.remote(permit), timeout=2.0)
    ray.get(release, timeout=30)
    snapshot = ray.get(actor.snapshot.remote("genrm"))
    assert not snapshot.models[0].admission

    # Slots are coroutines; only the bounded pools add threads.
    assert ray.get(actor.threads.remote()) < 300
    ray.kill(actor)


# ---------------------------------------------------------------------------
# Waiting holds no thread: the actor class driven in-process, with thread pools
# of one, where a call that waited on a thread would deadlock.
# ---------------------------------------------------------------------------


class _Pool:
    """A static engine pool; ``onload`` and ``health_check`` can be held."""

    def __init__(self, name: str, *, onloaded: bool = True, check_s: float = 0.0) -> None:
        from relax.engine.inference.config import InferenceModelSpec
        from relax.engine.inference.types import RouteMode, WeightSource

        self.model_spec = InferenceModelSpec(name, weight_source=WeightSource.STATIC, route_mode=RouteMode.DIRECT)
        self.onloaded = onloaded
        self.check_s = check_s
        self.hold_onload: threading.Event | None = None
        self.onloading = threading.Event()
        self.calls: list[str] = []

    def observe(self, serving: bool = True):
        from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot

        state = LifecycleState.READY if self.onloaded and serving else LifecycleState.SLEEPING
        replica = ReplicaSnapshot(f"{self.model_spec.name}/replica-0", state, "http://engine-0", None)
        return ModelSnapshot(self.model_spec.name, (replica,), None, state, state == LifecycleState.READY)

    def onload(self, tags=None) -> None:
        self.onloading.set()
        if self.hold_onload is not None:
            assert self.hold_onload.wait(10.0)
        self.calls.append("onload")
        self.onloaded = True

    def offload(self) -> None:
        self.calls.append("offload")
        self.onloaded = False

    def health_check(self) -> bool:
        time.sleep(self.check_s)
        return True

    def abort_requests(self) -> None:
        self.calls.append("abort")

    def shutdown(self, planner) -> None:
        self.calls.append("shutdown")


def _sharing_actor(monkeypatch, threads: int = 1, **pools: _Pool) -> _InferenceManagerActor:
    """GenRM resident and Teacher asleep on the same GPUs, in their own
    phases."""
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest

    monkeypatch.setattr(inference_manager, "_LIFECYCLE_THREADS", threads)
    monkeypatch.setattr(inference_manager, "_ROLLOUT_THREADS", threads)
    actor = _InferenceManagerActor()
    manager = actor.manager
    pools = {"genrm": _Pool("judge"), "teacher": _Pool("teacher", onloaded=False), **pools}
    view = PlacementGroupView((0, 1), (0, 1), PlacementOwner.CONTROLLER, identity="shared")
    for role, pool in pools.items():
        name = pool.model_spec.name
        manager.register(role, {name: pool})
        manager.plan_placement([PlacementRequest(f"{role}/{name}/group-0", "regular", 2, 1, 2, role, 0)], view)
    manager._model(Role.TEACHER, "teacher").resident = False
    return actor


def _run(coro, timeout: float = 10.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_inference_manager_actor_load_waiting_for_a_release_holds_no_thread(monkeypatch) -> None:
    actor = _sharing_actor(monkeypatch)

    async def scenario() -> None:
        # The load waits for GenRM's GPUs; the release that frees them needs
        # the only lifecycle thread.
        load = asyncio.create_task(actor.activate(Role.TEACHER))
        await asyncio.sleep(0.2)
        assert not load.done()
        await actor.deactivate(Role.GENRM)
        await load

    _run(scenario())
    assert actor.manager.resident(Role.TEACHER) and not actor.manager.resident(Role.GENRM)
    assert actor.manager.lifecycle._gpu_owner is None


def test_inference_manager_actor_drain_waits_for_completions_without_a_thread(monkeypatch) -> None:
    actor = _sharing_actor(monkeypatch)
    permit = actor.manager.admit_request(Role.GENRM, "judge", "held")

    async def scenario() -> None:
        release = asyncio.create_task(actor.deactivate(Role.GENRM))
        await asyncio.sleep(0.2)
        assert not release.done()
        # The only lifecycle thread is free for other work meanwhile.
        assert await actor.health_check(Role.TEACHER, "teacher")
        await actor.complete_request(permit)
        await release

    _run(scenario())
    assert not actor.manager.resident(Role.GENRM)


def test_inference_manager_actor_phase_switch_drains_on_the_loop(monkeypatch) -> None:
    actor = _sharing_actor(monkeypatch)
    permit = actor.manager.admit_request(Role.GENRM, "judge", "held")

    async def scenario() -> None:
        switch = asyncio.create_task(actor.enter_phase("teacher"))
        await asyncio.sleep(0.2)
        assert not switch.done()
        assert await actor.health_check(Role.TEACHER, "teacher")
        await actor.complete_request(permit)
        await switch
        assert actor.manager.resident(Role.TEACHER) and not actor.manager.resident(Role.GENRM)
        await actor.leave_phase("teacher")

    _run(scenario())
    assert not actor.manager.resident(Role.TEACHER)
    assert actor.manager.lifecycle._gpu_owner is None


def test_inference_manager_actor_calls_on_one_model_queue_without_threads(monkeypatch) -> None:
    actor = _sharing_actor(monkeypatch, threads=2, genrm=_Pool("judge", check_s=0.3))

    async def scenario() -> float:
        busy = [asyncio.create_task(actor.health_check(Role.GENRM, "judge")) for _ in range(6)]
        await asyncio.sleep(0.05)
        started = time.monotonic()
        # Without the gates, both threads would wait on the judge's lock.
        assert await actor.health_check(Role.TEACHER, "teacher")
        elapsed = time.monotonic() - started
        await asyncio.gather(*busy)
        return elapsed

    assert _run(scenario()) < 0.9


def test_inference_manager_actor_cancelled_loads_give_the_lease_back(monkeypatch) -> None:
    teacher = _Pool("teacher", onloaded=False)
    teacher.hold_onload = threading.Event()
    actor = _sharing_actor(monkeypatch, teacher=teacher)
    lifecycle = actor.manager.lifecycle

    async def scenario() -> None:
        # Cancelled while waiting for the lease: nothing is held.
        waiting = asyncio.create_task(actor.activate(Role.TEACHER))
        await asyncio.sleep(0.1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert lifecycle._gpu_owner is None

        # Cancelled while its load runs: the lease stays until the load ends.
        await actor.deactivate(Role.GENRM)
        running = asyncio.create_task(actor.activate(Role.TEACHER))
        assert await asyncio.to_thread(teacher.onloading.wait, 5.0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert lifecycle._gpu_owner is not None
        teacher.hold_onload.set()
        deadline = time.monotonic() + 5.0
        while lifecycle._gpu_owner is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert lifecycle._gpu_owner is None and teacher.onloaded

    _run(scenario())


def test_inference_manager_actor_rollout_recovery_waits_for_the_lease(monkeypatch) -> None:
    from types import SimpleNamespace

    actor = _sharing_actor(monkeypatch)
    manager = actor.manager
    policy = _Pool("policy")
    manager.register(Role.ROLLOUT, {"policy": policy})
    leased = []
    policy.recover = lambda: leased.append(manager.lifecycle._owns_lease())
    manager._rollout_pool = SimpleNamespace(
        health_monitoring_pause=lambda: None,
        get_rollout_engines_and_lock=lambda name: "engines",
        weights_changing=lambda: False,
    )

    # Before the first generate pass there is nothing to recover.
    assert _run(actor.rollout_operation("recover_rollout_engines")) == "engines"
    assert leased == []
    manager._rollout_started = True
    assert _run(actor.rollout_operation("recover_rollout_engines", "policy")) == "engines"
    assert leased == [True] and manager.lifecycle._gpu_owner is None

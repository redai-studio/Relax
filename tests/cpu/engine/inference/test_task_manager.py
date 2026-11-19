# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot, Role


class MemoryPool:
    def __init__(self, role, args, binding, options):
        self.model_id = options["model_id"]
        self.start_sleeping = False
        self.backend = SimpleNamespace(engines=[], offload=self.offload, onload=self.onload)
        self.resident = True
        self.url = "http://engine"
        self.disposed = False

    def record_placement(self, planner):
        self.planner = planner

    def observe(self, *, ready, state):
        return (
            ModelSnapshot(
                self.model_id, (ReplicaSnapshot(self.model_id, state, self.url),), state=state, admission=ready
            ),
        )

    def offload(self):
        self.resident = False

    def onload(self):
        self.resident = True

    def unhealthy_ranks(self):
        return []

    def dispose(self):
        self.disposed = True


@pytest.mark.asyncio
async def test_task_registers_all_models_in_one_request_and_placement_domain(monkeypatch):
    from relax.distributed.ray import static_inference_pool

    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class()
    await task.register_static("teacher", None, None, {"model_id": "math"})
    await task.register_static("teacher", None, None, {"model_id": "code"})
    snapshots = await task.snapshot("teacher")
    assert {model.model_id for model in snapshots.models} == {"math", "code"}
    first = await task.admit_request("teacher", "math")
    second = await task.admit_request("teacher", "code")
    assert set(task.requests.permits) == {first, second}
    assert all(
        runtime.ledger.root is task.requests and runtime.placement is task.placement
        for runtime in task.models.values()
    )
    offload = asyncio.create_task(task.dispatch("teacher", "math", "offload"))
    for _ in range(100):
        if "memory" in task.models[(Role.TEACHER, "math")].ledger.fences:
            break
        await asyncio.sleep(0)
    assert not offload.done()
    await task.complete_request(first)
    await asyncio.wait_for(offload, 1)
    assert second in task.requests.permits
    assert task.models[(Role.TEACHER, "math")].state is LifecycleState.SLEEPING
    assert task.models[(Role.TEACHER, "code")].memory_ready
    await task.complete_request(second)
    await task.dispose()
    assert all(runtime.pool is None for runtime in task.models.values())


@pytest.mark.asyncio
async def test_phase_handoff_releases_partial_policy_weights_even_from_startup(monkeypatch):
    from relax.distributed.ray import static_inference_pool

    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class()
    task.shared_memory = True
    task.pool = MemoryPool(Role.ROLLOUT, None, None, {"model_id": "default"})
    task.pool.backend.onload_weights = task.pool.onload
    task.pool.resident = False
    task.ledger.fences.add("memory")
    await task.register_static("teacher", None, None, {"model_id": "default"})
    teacher = task.models[(Role.TEACHER, "default")]
    await teacher.pool_call("offload")
    await task.configure_phases(
        {"startup": (), "inference": (("rollout", "default"),), "teacher": (("teacher", "default"),)}, "startup"
    )
    await task.pool_call("onload_weights")
    assert task.resident and not task.memory_ready
    assert task.pool.resident
    await task.switch("teacher")
    assert not task.resident
    assert not task.pool.resident
    assert teacher.pool.resident and teacher.memory_ready


@pytest.mark.asyncio
async def test_retired_deployment_cannot_publish_late_model_start(monkeypatch):
    from threading import Event

    from relax.distributed.ray import static_inference_pool

    entered, release = Event(), Event()
    original = MemoryPool.__init__

    def initialize(pool, *args):
        entered.set()
        assert release.wait(2)
        original(pool, *args)

    monkeypatch.setattr(MemoryPool, "__init__", initialize)
    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class(startup_token="deployment-1")
    with pytest.raises(RuntimeError, match="token"):
        await task.register_static("teacher", None, None, {"model_id": "wrong"}, startup_token="old")
    pending = asyncio.create_task(
        task.register_static(
            "teacher",
            None,
            None,
            {"model_id": "default"},
            startup_token="deployment-1",
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    await task.invalidate_startup()
    release.set()
    with pytest.raises(RuntimeError, match="retired"):
        await pending
    assert task.models[(Role.TEACHER, "default")].pool is not None
    await task.dispose()
    assert task.models[(Role.TEACHER, "default")].pool is None
    with pytest.raises(RuntimeError, match="retired"):
        await task.start(None, None, startup_token="deployment-1")


@pytest.mark.asyncio
async def test_task_rejects_cross_role_overlap_before_pool_constructor(monkeypatch):
    from relax.distributed.ray import static_inference_pool
    from relax.engine.inference.placement import (
        PlacementConflictError,
        PlacementGroupView,
        PlacementOwner,
        PlacementRequest,
        ProbedPlacementGroup,
    )

    constructed = []
    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", lambda *args: constructed.append(args))
    task = InferenceManager.__ray_metadata__.modified_class()
    pg = SimpleNamespace(bundle_count=2)
    binding = ProbedPlacementGroup(pg, [0, 1], [0, 1], ("node", "node"), (0, 1))
    view = PlacementGroupView(
        (0, 1),
        (0, 1),
        PlacementOwner.CONTROLLER,
        identity=pg,
        node_ids=("node", "node"),
        local_gpu_ids=(0, 1),
        bundle_count=2,
    )
    task.placement.plan([PlacementRequest("rollout/policy", "regular", 1, 1, 2, bundle_offset=0)], view)
    args = SimpleNamespace(
        genrm_model_path="judge",
        genrm_num_gpus=1,
        genrm_num_gpus_per_engine=1,
        genrm_engine_config={},
        genrm_sampling_config={},
        num_gpus_per_node=2,
    )
    with pytest.raises(PlacementConflictError, match="overlap"):
        await task.register_static("genrm", args, binding, {"model_id": "judge"})
    assert constructed == []
    assert task.models == {}


@pytest.mark.asyncio
async def test_decoupled_models_keep_residency_when_policy_releases_for_training(monkeypatch):
    from relax.distributed.ray import static_inference_pool

    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class()
    task.pool = MemoryPool(Role.ROLLOUT, None, None, {"model_id": "default"})
    task.memory_ready = task.resident = True
    task.state = LifecycleState.READY
    await task.register_static("teacher", None, None, {"model_id": "default"})
    teacher = task.models[(Role.TEACHER, "default")]
    await task.configure_phases(
        {"startup": (), "inference": (("rollout", "default"),), "resident": (("teacher", "default"),)}, "inference"
    )
    await task.switch("startup")
    assert not task.memory_ready and not task.pool.resident
    assert teacher.memory_ready and teacher.pool.resident
    assert (await task.snapshot("teacher")).models[0].admission


@pytest.mark.asyncio
async def test_role_route_snapshot_preserves_task_phase_epoch_and_role_revision(monkeypatch):
    from relax.distributed.ray import static_inference_pool

    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class()
    await task.register_static("teacher", None, None, {"model_id": "math"})
    await task.register_static("teacher", None, None, {"model_id": "code"})
    await task.configure_phases({"startup": (), "teacher": (("teacher", "math"), ("teacher", "code"))}, "teacher")
    whole = await task.snapshot("teacher")
    single = await task.dispatch("teacher", "code", "snapshot", "teacher")
    assert single.manager_epoch == whole.manager_epoch == task.epoch
    assert single.phase == whole.phase == "teacher"
    assert single.topology_revision == whole.topology_revision
    assert [model.model_id for model in single.models] == ["code"]
    assert single.routing.default_model == "code"
    task.models[(Role.TEACHER, "math")].pool.url = "http://replacement"
    changed = await task.snapshot("teacher")
    still_code = await task.dispatch("teacher", "code", "snapshot")
    assert changed.topology_revision > whole.topology_revision
    assert still_code.topology_revision == changed.topology_revision
    assert [model.model_id for model in still_code.models] == ["code"]


@pytest.mark.asyncio
async def test_failed_aux_disposal_retires_queued_handoffs_but_allows_cleanup_retry(monkeypatch):
    from relax.distributed.ray import static_inference_pool

    monkeypatch.setattr(static_inference_pool, "StaticInferencePool", MemoryPool)
    task = InferenceManager.__ray_metadata__.modified_class()
    await task.register_static("genrm", None, None, {"model_id": "judge"})
    await task.register_static("teacher", None, None, {"model_id": "default"})
    judge = task.models[(Role.GENRM, "judge")]
    teacher = task.models[(Role.TEACHER, "default")]
    await judge.pool_call("offload")
    await teacher.pool_call("offload")
    phases = {"startup": (), "genrm": (("genrm", "judge"),), "teacher": (("teacher", "default"),)}
    await task.configure_phases(phases, "startup")
    original_dispose = judge.pool.dispose
    onloads = []
    original_onload = teacher.pool.backend.onload

    def fail_dispose():
        raise TimeoutError("judge subprocess exit unconfirmed")

    def onload():
        onloads.append("teacher")
        original_onload()

    judge.pool.dispose = fail_dispose
    teacher.pool.backend.onload = onload
    await task._phase_switch.acquire()
    queued = asyncio.create_task(task.switch("teacher"))
    await asyncio.sleep(0)
    try:
        with pytest.raises(TimeoutError, match="exit unconfirmed"):
            await task.dispose()
    finally:
        task._phase_switch.release()
    with pytest.raises(RuntimeError, match="retired"):
        await queued
    with pytest.raises(RuntimeError, match="retired"):
        await task.configure_phases(phases, "startup")
    with pytest.raises(RuntimeError, match="retired"):
        await task.switch("teacher")
    assert onloads == []
    assert not teacher.pool.resident
    assert teacher.pool is not None
    await task.switch("startup")
    judge.pool.dispose = original_dispose
    await task.dispose()
    assert judge.pool is None and teacher.pool is None


@pytest.mark.asyncio
async def test_inference_phase_resumes_only_kv_after_policy_weights():
    task = InferenceManager.__ray_metadata__.modified_class()
    task.shared_memory = True
    task.pool = MemoryPool(Role.ROLLOUT, None, None, {"model_id": "default"})
    loaded = set()

    def weights():
        assert "weights" not in loaded
        loaded.add("weights")

    def kv():
        assert loaded == {"weights"}
        loaded.add("kv")

    def all_memory():
        assert not loaded, "Cannot resume weights twice"
        loaded.update(("weights", "kv"))

    task.pool.backend.onload_weights = weights
    task.pool.backend.onload_kv = kv
    task.pool.backend.onload = all_memory
    task.resident = False
    task.memory_ready = False
    await task.configure_phases({"startup": (), "inference": (("rollout", "default"),)}, "startup")
    await task.pool_call("onload_weights")
    await task.pool_call("onload_kv")
    assert loaded == {"weights", "kv"}
    assert task.memory_ready

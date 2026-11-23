# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from relax.distributed.ray.inference_handle import ManagedRolloutHandle, use_managed_rollout
from relax.distributed.ray.inference_manager import InferenceManager, _blocking
from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot, Role


class FakeBackend:
    def __init__(self):
        self._is_weight_updating = False
        self.reject_update = False
        self.fail_offload = False
        self.rollout_engines = []
        self.status = "onload"

    async def offload(self):
        if self.fail_offload:
            raise RuntimeError("engine release failed")
        self.status = "offload"

    async def onload(self, tags=None):
        self.status = "onload"

    async def onload_kv(self):
        await self.onload()

    async def onload_weights(self):
        await self.onload(["weights"])

    def set_weight_updating(self, updating):
        if updating and self.reject_update:
            return False
        self._is_weight_updating = updating
        return True

    def get_status(self):
        return self.status


class FakePool:
    def __init__(self):
        self.backend = FakeBackend()
        self.generation = 0
        self.synced_actor_ids = set()

    def mark_synced(self, actor_ids):
        self.synced_actor_ids.update(actor_ids)

    def observe(self, *, ready, state):
        return (
            ModelSnapshot(
                "default",
                (ReplicaSnapshot(f"replica-{self.generation}", state, "http://engine"),),
                "http://router",
                LifecycleState.READY if ready else state,
                admission=ready,
            ),
        )

    def dispose(self):
        self.backend.status = "disposed"


def manager():
    instance = InferenceManager.__ray_metadata__.modified_class()
    instance.pool = FakePool()
    instance.memory_ready = True
    instance.state = LifecycleState.READY
    return instance


@pytest.mark.asyncio
async def test_drain_closes_admission_but_allows_completions():
    owner = manager()
    permit = await owner.admit_request(Role.ROLLOUT, "default")
    offload = asyncio.create_task(owner.pool_call("offload"))
    await asyncio.sleep(0)
    assert not (await owner.snapshot()).models[0].admission
    assert not offload.done()
    with pytest.raises(RuntimeError):
        await owner.admit_request(Role.ROLLOUT, "default")
    await owner.complete_request(permit)
    await asyncio.wait_for(offload, 2)
    assert owner.pool.backend.status == "offload"
    await owner.pool_call("onload_weights")
    assert not (await owner.snapshot()).models[0].admission
    await owner.pool_call("onload_kv")
    assert (await owner.snapshot()).models[0].admission


@pytest.mark.asyncio
async def test_offload_failure_keeps_admission_closed():
    owner = manager()
    owner.pool.backend.fail_offload = True
    with pytest.raises(RuntimeError, match="release failed"):
        await owner.pool_call("offload")
    assert owner.state is LifecycleState.DEAD
    assert owner.ledger.fences == {"memory"}
    assert not (await owner.snapshot()).models[0].admission


@pytest.mark.asyncio
async def test_overlapping_weight_updates_do_not_reopen_each_other():
    owner = manager()
    first = await owner.begin_weight_update("backend")
    second = await owner.begin_weight_update("backend")
    assert first != second
    await owner.end_weight_update(first)
    assert not (await owner.snapshot()).models[0].admission
    await owner.end_weight_update(second)
    assert (await owner.snapshot()).models[0].admission
    with pytest.raises(ValueError):
        await owner.end_weight_update(first)


@pytest.mark.asyncio
async def test_fully_async_rejected_update_rolls_back_only_its_fence():
    owner = manager()
    token = await owner.begin_weight_update("backend")
    owner.pool.backend.reject_update = True
    assert await owner.pool_call("set_weight_updating", True) is False
    assert owner.ledger.fences == {token}
    await owner.end_weight_update(token)
    assert (await owner.snapshot()).models[0].admission


@pytest.mark.asyncio
async def test_fully_async_end_does_not_clear_backend_fence():
    owner = manager()
    token = await owner.begin_weight_update("backend")
    assert await owner.pool_call("set_weight_updating", True)
    await owner.pool_call("set_weight_updating", False)
    assert not (await owner.snapshot()).models[0].admission
    await owner.end_weight_update(token)
    assert (await owner.snapshot()).models[0].admission


@pytest.mark.asyncio
async def test_blocking_cancellation_waits_for_engine_mutation():
    from threading import Event

    entered, release = Event(), Event()

    def mutate():
        entered.set()
        assert release.wait(2)

    task = asyncio.create_task(_blocking(mutate))
    await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_pool_operation_cancellation_does_not_spin_the_actor_loop():
    async def cancel_self():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(_blocking(cancel_self), timeout=2)


@pytest.mark.parametrize(
    "legacy",
    [
        {"_genrm_instances_resolved": {"judge": {}}},
        {"use_opd": True},
        {"loss_type": "sft"},
        {"use_agentic_rollout": True},
        {"rollout_engine_class_path": "custom.Engine"},
        {"custom_reward_post_process_path": "custom.reward"},
        {"debug_train_only": True},
        {"custom_convert_samples_to_train_data_path": "custom.convert"},
        {"rollout_external": True},
        {"custom_generate_function_path": "custom.generate"},
        {"custom_dataset_class_path": "custom.Data"},
        {"rollout_function_path": "custom.rollout"},
        {"eval_function_path": "custom.eval"},
        {"eval_datasets": [SimpleNamespace(custom_generate_function_path="custom.eval")]},
    ],
)
def test_unmigrated_tasks_select_complete_legacy_path(legacy):
    assert not use_managed_rollout(SimpleNamespace(**legacy))


@pytest.mark.parametrize("mode", [{}, {"fully_async": True}, {"hybrid": True}, {"prefill_num_servers": 2}])
def test_standard_rollout_selects_managed_path(mode):
    assert use_managed_rollout(SimpleNamespace(**mode))


def test_handle_survives_serialization_and_routes_by_owner():
    class Remote:
        def remote(self, *args, **kwargs):
            return args, kwargs

    manager_handle = SimpleNamespace(pool_call=Remote(), dispose=Remote())
    worker = SimpleNamespace(generate=Remote(), set_train_parallel_config=Remote())
    handle = ray.cloudpickle.loads(ray.cloudpickle.dumps(ManagedRolloutHandle(manager_handle, worker)))
    assert handle.generate.remote(3) == ((3,), {})
    assert handle.offload.remote() == (("offload",), {})
    assert handle.set_train_parallel_config.remote({"tp": 2}) == (({"tp": 2},), {})


@pytest.mark.asyncio
async def test_snapshot_rereads_background_generation_changes():
    owner = manager()
    before = await owner.snapshot()
    owner.pool.generation += 1
    after = await owner.snapshot()
    assert after.topology_revision > before.topology_revision
    assert replace(after, models=before.models, topology_revision=before.topology_revision) == before


def test_real_cpu_ray_drain_can_receive_permit_completions():
    # Runs the actual production actor API; only the GPU pool is synthetic.
    # No existing Ray cluster is attached or modified.
    if ray.is_initialized():
        pytest.skip("requires an isolated local Ray runtime")

    class Harness(InferenceManager.__ray_metadata__.modified_class):
        def __init__(self):
            super().__init__()
            self.pool = FakePool()
            self.memory_ready = True
            self.state = LifecycleState.READY

    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parent)}},
    )
    actor = ray.remote(num_cpus=1, max_concurrency=4096)(Harness).remote()
    try:
        permits = ray.get([actor.admit_request.remote("rollout", "default") for _ in range(64)], timeout=20)
        drain = actor.pool_call.remote("offload")
        # Async Ray methods are deliberately unordered; observe the fence
        # before releasing permits, without relying on submission order.
        import time

        deadline = time.monotonic() + 10
        while ray.get(actor.snapshot.remote(), timeout=10).models[0].admission:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        ray.get([actor.complete_request.remote(token) for token in permits], timeout=20)
        ray.get(drain, timeout=20)
        assert ray.get(actor.pool_call.remote("get_status"), timeout=10) == "offload"
        ray.get(actor.pool_call.remote("onload_kv"), timeout=10)
        assert ray.get(actor.snapshot.remote(), timeout=10).models[0].admission
        from relax.distributed.ray.inference_manager import dispose_managed_rollout

        ray.get(dispose_managed_rollout.remote(actor, None), timeout=20)
        with pytest.raises(ray.exceptions.ActorDiedError):
            ray.get(actor.snapshot.remote(), timeout=10)
    finally:
        ray.kill(actor)
        ray.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_topology_operation_before_disposing():
    """An engine created after shutdown starts must still belong to cleanup."""
    import threading

    owner = manager()
    started = threading.Event()
    finish = threading.Event()
    published = []

    def scale(request_id):
        started.set()
        assert finish.wait(5)
        published.append(request_id)

    owner.pool.backend.execute_scale_out = scale
    owner.pool.dispose = lambda: published.clear()
    scaling = asyncio.create_task(owner.pool_call("execute_scale_out", "new-engine"))
    assert await asyncio.to_thread(started.wait, 5)
    shutdown = asyncio.create_task(owner.dispose())
    await asyncio.sleep(0)
    assert not shutdown.done()
    with pytest.raises(RuntimeError, match="closing"):
        await owner.pool_call("execute_scale_out", "too-late")
    # The release side must stay usable while scale-in/out waits for weights.
    await owner.pool_call("set_weight_updating", False)
    finish.set()
    await asyncio.wait_for(asyncio.gather(scaling, shutdown), 5)
    assert not published
    assert owner.pool is None


@pytest.mark.asyncio
async def test_shutdown_cancellation_failure_still_disposes_pool():
    owner = manager()
    pool = owner.pool

    def fail_cancel():
        raise RuntimeError("cancellation RPC failed")

    pool.backend.cancel_all_scale_out_requests = fail_cancel
    await owner.dispose()
    assert owner.pool is None
    assert pool.backend.status == "disposed"


@pytest.mark.asyncio
async def test_internal_generation_waits_for_weight_update_then_resumes():
    import httpx

    from relax.engine.inference.executor import InferenceRequestExecutor

    owner = manager()
    token = await owner.begin_weight_update("backend")
    remote = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(owner, name))
            for name in ("snapshot", "admit_request", "complete_request", "wait_ready")
        }
    )
    sent = []

    def upstream(request):
        sent.append(request)
        return httpx.Response(200, json={"text": "continued"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        executor = InferenceRequestExecutor(Role.ROLLOUT, remote, client)
        generation = asyncio.create_task(executor.generate_json({"text": "hello"}))
        await asyncio.sleep(0.01)
        assert not generation.done()
        assert not sent
        await owner.end_weight_update(token)
        assert await asyncio.wait_for(generation, 2) == {"text": "continued"}
    assert not owner.ledger.permits


@pytest.mark.asyncio
async def test_internal_wait_exits_when_manager_closes():
    owner = manager()
    await owner.begin_weight_update("backend")
    waiting = asyncio.create_task(owner.wait_ready(Role.ROLLOUT, "default"))
    await owner.dispose()
    with pytest.raises(RuntimeError, match="closing"):
        await asyncio.wait_for(waiting, 2)


@pytest.mark.asyncio
async def test_failed_residency_does_not_leave_internal_generation_waiting():
    owner = manager()
    owner.state = LifecycleState.DEAD
    owner.memory_ready = False
    with pytest.raises(RuntimeError, match="residency"):
        await owner.wait_ready(Role.ROLLOUT)


@pytest.mark.asyncio
async def test_dispose_is_idempotent():
    owner = manager()
    await owner.dispose()
    await owner.dispose()
    assert owner.pool is None
    assert owner.state is LifecycleState.DEAD


@pytest.mark.asyncio
async def test_cancelled_start_retains_pool_ownership_until_shutdown(monkeypatch):
    """A cancelled startup RPC must not lose engines created in its thread."""
    import threading

    import relax.distributed.ray.inference_pool as pool_module

    started = threading.Event()
    finish = threading.Event()
    created = []

    class StartupPool(FakePool):
        def __init__(self, args, binding):
            super().__init__()
            started.set()
            assert finish.wait(5)
            self.backend.servers = {"policy": None}
            self.backend.get_router_address = lambda name: {"router_ip": "router", "router_port": 3000}
            created.append(self)

    monkeypatch.setattr(pool_module, "plan_rollout", lambda *args: None)
    monkeypatch.setattr(pool_module, "RolloutPool", StartupPool)
    owner = InferenceManager.__ray_metadata__.modified_class()
    startup = asyncio.create_task(owner.start(SimpleNamespace(), None))
    assert await asyncio.to_thread(started.wait, 5)
    startup.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(startup, 5)
    assert owner.pool is created[0]
    await owner.dispose()
    assert created[0].backend.status == "disposed"
    assert owner.pool is None


@pytest.mark.asyncio
async def test_concurrent_dispose_closes_worker_only_once(monkeypatch):
    owner = manager()
    entered = asyncio.Event()
    finish = asyncio.Event()
    closes = []
    killed = []

    async def close():
        closes.append("closed")
        entered.set()
        await finish.wait()

    worker = SimpleNamespace(close=SimpleNamespace(remote=close))
    monkeypatch.setattr(ray, "kill", killed.append)
    first = asyncio.create_task(owner.dispose(worker))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(owner.dispose(worker))
    await asyncio.sleep(0)
    finish.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert closes == ["closed"]
    assert killed == [worker]
    assert owner.pool is None


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_fails", [False, True])
async def test_shutdown_closes_worker_with_dead_slots_or_abort_failure(monkeypatch, abort_fails):
    from unittest.mock import AsyncMock, MagicMock

    owner = manager()
    pool = owner.pool
    engine = MagicMock()
    owner.pool.backend.rollout_engines = [None, engine]
    worker = MagicMock()
    worker.close.remote = AsyncMock()

    def abort_result(*args, **kwargs):
        if abort_fails:
            raise RuntimeError("abort failed")

    monkeypatch.setattr(ray, "get", abort_result)
    kill = MagicMock()
    monkeypatch.setattr(ray, "kill", kill)
    await owner.dispose(worker)
    engine.abort_requests.remote.assert_called_once()
    worker.close.remote.assert_awaited_once()
    kill.assert_called_once_with(worker)
    assert pool.backend.status == "disposed"

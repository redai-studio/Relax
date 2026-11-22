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
        self.router = SimpleNamespace(is_healthy=lambda: True)
        self._is_weight_updating = False
        self.reject_update = False
        self.fail_offload = False
        self.rollout_engines = []
        self.engines = []
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
async def test_weight_sync_and_generation_handoff_resume_policy_tags_once():
    from relax.engine.inference.phases import PhaseCoordinator

    owner = manager()
    allocated = {"weights", "kv"}
    events = []

    def resume(tags):
        assert not allocated.intersection(tags)
        allocated.update(tags)

    def activate_judge():
        events.append("judge onload")

    owner.pool.backend.offload = allocated.clear
    owner.pool.backend.onload_weights = lambda: resume({"weights"})
    owner.pool.backend.onload_kv = lambda: resume({"kv"})
    owner.pool.backend.onload = lambda: resume({"weights", "kv"})
    judge = SimpleNamespace(onload=activate_judge, offload=lambda: events.append("judge offload"))

    async def call(target, method):
        if target is owner:
            await getattr(owner, method)()
        else:
            getattr(target, method)()

    coordinator = PhaseCoordinator({"startup": (), "inference": (owner, judge)}, "startup", call)
    await owner.offload()
    for _ in range(2):
        await owner.pool_call("onload_weights")
        assert allocated == {"weights"}
        await owner.pool_call("onload_kv")
        await coordinator.switch("inference")
        assert allocated == {"weights", "kv"}
        await coordinator.switch("startup")
        assert not allocated
    assert events == ["judge onload", "judge offload"] * 2


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
        {"_genrm_instances_resolved": {"judge": {}}, "colocate": True},
        {"use_opd": True, "colocate": True},
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
def test_inference_paths_use_task_control_plane(legacy):
    assert use_managed_rollout(SimpleNamespace(**legacy)) is (legacy.get("loss_type") != "sft")


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


@pytest.mark.parametrize("role", list(Role))
def test_real_cpu_ray_drain_can_receive_permit_completions(role):
    # Runs the actual production actor API; only the GPU pool is synthetic.
    # No existing Ray cluster is attached or modified.
    if ray.is_initialized():
        pytest.skip("requires an isolated local Ray runtime")

    class Harness(InferenceManager.__ray_metadata__.modified_class):
        def __init__(self):
            super().__init__()
            self.pool = FakePool()
            self.role = role
            self._last_health = float("inf")
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
        permits = ray.get([actor.admit_request.remote(role, "default") for _ in range(64)], timeout=20)
        drain = actor.pool_call.remote("offload")
        # Async Ray methods are deliberately unordered; observe the fence
        # before releasing permits, without relying on submission order.
        import time

        deadline = time.monotonic() + 10
        while ray.get(actor.snapshot.remote(role), timeout=10).models[0].admission:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        ray.get([actor.complete_request.remote(token) for token in permits], timeout=20)
        ray.get(drain, timeout=20)
        if role is Role.ROLLOUT:
            assert ray.get(actor.pool_call.remote("get_status"), timeout=10) == "offload"
        ray.get(actor.pool_call.remote("onload_kv" if role is Role.ROLLOUT else "onload"), timeout=10)
        assert ray.get(actor.snapshot.remote(role), timeout=10).models[0].admission
        from relax.distributed.ray.inference_manager import dispose_managed_rollout

        ray.get(dispose_managed_rollout.remote(actor, None), timeout=20)
        with pytest.raises(ray.exceptions.ActorDiedError):
            ray.get(actor.snapshot.remote(role), timeout=10)
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
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("confirmed", [False, True])
async def test_disconnected_poster_requires_stop_receipt_before_returning_permit(monkeypatch, cancelled, confirmed):
    import httpx

    from relax.engine.inference.executor import InferenceRequestExecutor
    from relax.utils import http_utils

    owner = manager()
    completed = []

    async def complete(permit):
        completed.append(permit)
        await owner.complete_request(permit)

    remote = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(owner, name))
            for name in ("snapshot", "admit_request", "fail_request_transport")
        },
        complete_request=SimpleNamespace(remote=complete),
    )
    started, disconnected, stopping, stopped = (asyncio.Event() for _ in range(4))
    paths = []
    poster_calls = []

    async def do_request(*args):
        poster_calls.append(args)
        started.set()
        await disconnected.wait()
        if cancelled:
            raise ray.exceptions.TaskCancelledError()
        raise ray.exceptions.ActorUnavailableError("connection lost", None)

    async def no_cancel_receipt(request_id):
        raise ray.exceptions.ActorDiedError()

    async def stop_requests():
        stopping.set()
        await stopped.wait()
        if not confirmed:
            raise ray.exceptions.ActorDiedError()

    def upstream(request):
        paths.append(request.url.path)
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine"}]})
        return httpx.Response(200)

    poster = SimpleNamespace(
        do_request=SimpleNamespace(remote=do_request),
        stop_requests=SimpleNamespace(remote=stop_requests),
        cancel_request=SimpleNamespace(remote=no_cancel_receipt),
    )
    monkeypatch.setattr(http_utils, "_post_actors", [poster])
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        monkeypatch.setattr(http_utils, "_http_client", client)
        executor = InferenceRequestExecutor(
            Role.ROLLOUT, remote, client, send_request=http_utils.send_distributed_request
        )
        generation = asyncio.create_task(executor.generate_json({"text": "generate"}))
        await asyncio.wait_for(started.wait(), 1)
        if cancelled:
            generation.cancel()
        disconnected.set()
        try:
            await asyncio.wait_for(stopping.wait(), 1)
            assert not completed and not paths
            fallback = await http_utils.send_distributed_request(
                client.build_request("POST", "http://engine/fallback")
            )
            await fallback.aclose()
            assert paths == ["/fallback"] and len(poster_calls) == 1
        finally:
            stopped.set()
        expected = (
            (asyncio.CancelledError if cancelled else ray.exceptions.ActorUnavailableError)
            if confirmed
            else http_utils.DistributedPostUnconfirmed
        )
        with pytest.raises(expected):
            await asyncio.wait_for(generation, 2)

    if confirmed:
        assert len(completed) == 1
        assert paths == ["/fallback", "/workers", "/abort_request"]
    else:
        assert not completed and paths == ["/fallback"]
        assert not (await owner.snapshot()).models[0].admission
        with pytest.raises(RuntimeError, match="cannot confirm distributed POST"):
            await owner.wait_ready("rollout")


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
    args = SimpleNamespace(
        colocate=True,
        _genrm_instances_resolved={"judge": {}},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=False,
    )
    startup = asyncio.create_task(owner.start(args, None))
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


@pytest.mark.parametrize("role", ["genrm", "teacher"])
def test_independent_auxiliary_tasks_select_managed_path(role):
    from relax.distributed.ray.inference_handle import use_managed_auxiliary

    args = SimpleNamespace(
        colocate=False, use_opd=role == "teacher", _genrm_instances_resolved={"judge": {}} if role == "genrm" else {}
    )
    assert use_managed_auxiliary(args)
    assert use_managed_rollout(args)
    args.colocate = True
    assert use_managed_auxiliary(args)
    assert use_managed_rollout(args)


@pytest.mark.asyncio
async def test_static_role_recovery_drains_before_replacement_and_refreshes_snapshot():
    owner = manager()
    owner.role = Role.TEACHER
    owner._last_health = asyncio.get_running_loop().time()
    owner.pool.unhealthy_ranks = lambda: [0]
    recovered = []

    def recover(dead):
        recovered.extend(dead)
        owner.pool.generation += 1

    owner.pool.recover = recover
    owner.pool.record_placement = lambda planner: None
    before = await owner.snapshot(Role.TEACHER)
    permit = await owner.admit_request(Role.TEACHER, "default")
    recovery = asyncio.create_task(owner.recover_static())
    for _ in range(100):
        if "health" in owner.ledger.fences:
            break
        await asyncio.sleep(0.001)
    assert "health" in owner.ledger.fences
    assert not recovered
    assert not (await owner.snapshot(Role.TEACHER)).models[0].admission
    await owner.complete_request(permit)
    await asyncio.wait_for(recovery, 2)
    after = await owner.snapshot(Role.TEACHER)
    assert recovered == [0]
    assert after.role is Role.TEACHER
    assert after.topology_revision > before.topology_revision
    assert after.models[0].admission
    with pytest.raises(ValueError, match="Static"):
        await owner.begin_weight_update("actor")


@pytest.mark.asyncio
async def test_static_healthy_probe_does_not_drain_requests():
    owner = manager()
    owner.role = Role.GENRM
    owner._last_health = asyncio.get_running_loop().time()
    owner.pool.unhealthy_ranks = lambda: []
    permit = await owner.admit_request(Role.GENRM, "default")
    await asyncio.wait_for(owner.recover_static(), 2)
    assert owner.ledger.permits
    assert not owner.ledger.fences
    await owner.complete_request(permit)


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


def test_real_phase_coordinator_switch_allows_public_manager_permit_completion():
    """One isolated CPU Ray runtime covers actor serialization and cross-owner
    drain.

    Local coordinator tests cannot exercise the factory's captured actor
    handles or concurrent Ray RPC completion. This adds two zero-GPU actors and
    one coordinator for a single bounded transition (roughly 10-20 seconds in
    CI).
    """
    import os
    import subprocess
    import sys
    import textwrap
    from tempfile import TemporaryDirectory

    script = textwrap.dedent("""
        import sys
        import time
        import ray
        from test_managed_rollout import FakePool
        from relax.distributed.ray.inference_manager import InferenceManager, _ModelRuntime
        from relax.distributed.ray.inference_handle import InferenceRoleHandle
        from relax.engine.inference.lifecycle import AdmissionScope
        from relax.engine.inference.phases import create_phase_coordinator
        from relax.engine.inference.types import LifecycleState, Role

        class Harness(InferenceManager.__ray_metadata__.modified_class):
            def __init__(self):
                super().__init__()
                self.pool = FakePool()
                self.memory_ready = True
                self.state = LifecycleState.READY
                self._last_health = float("inf")
                teacher = _ModelRuntime()
                teacher.role = Role.TEACHER
                teacher.pool = FakePool()
                teacher.ledger = AdmissionScope(self.requests, "teacher/default")
                teacher.memory_ready = False
                teacher.state = LifecycleState.SLEEPING
                teacher.ledger.fences.add("memory")
                teacher._last_health = float("inf")
                self.models[(Role.TEACHER, "default")] = teacher

        ray.init(address="local", num_cpus=2, include_dashboard=False,
                 log_to_driver=False, _temp_dir=sys.argv[1])
        owners = []
        coordinator = None
        try:
            Actor = ray.remote(num_cpus=0, max_concurrency=100)(Harness)
            policy = Actor.remote()
            teacher = InferenceRoleHandle(policy, "teacher", "default")
            owners = [policy]
            permit = ray.get(policy.admit_request.remote("rollout", "default"), timeout=20)
            coordinator = create_phase_coordinator({"inference": (policy,), "teacher": (teacher,)}, "inference")
            assert coordinator == policy
            transition = coordinator.switch.remote("teacher")
            deadline = time.monotonic() + 15
            while ray.get(policy.snapshot.remote("rollout"), timeout=10).models[0].admission:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert not ray.get(teacher.snapshot.remote("teacher"), timeout=10).models[0].admission
            assert not ray.wait([transition], timeout=0)[0]
            ray.get(policy.complete_request.remote(permit), timeout=10)
            ray.get(transition, timeout=20)
            assert not ray.get(policy.snapshot.remote("rollout"), timeout=10).models[0].admission
            assert ray.get(teacher.snapshot.remote("teacher"), timeout=10).models[0].admission
            ray.get(coordinator.switch.remote("inference"), timeout=20)
            assert ray.get(policy.snapshot.remote("rollout"), timeout=10).models[0].admission
            assert not ray.get(teacher.snapshot.remote("teacher"), timeout=10).models[0].admission
        finally:
            for owner in owners:
                ray.kill(owner)
            ray.shutdown()
    """)
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).resolve().parent), str(Path.cwd()), env.get("PYTHONPATH", ""))
    )
    # Ray's UNIX sockets require a short path, independent of pytest node IDs.
    with TemporaryDirectory(prefix="rphase-", dir="/tmp") as ray_dir:
        result = subprocess.run(
            [sys.executable, "-c", script, ray_dir],
            env=env,
            capture_output=True,
            text=True,
            timeout=100,
        )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_memory_transitions_do_not_change_topology_revision():
    owner = manager()
    initial = await owner.snapshot()
    await owner.pool_call("offload")
    sleeping = await owner.snapshot()
    assert not sleeping.models[0].admission
    assert sleeping.topology_revision == initial.topology_revision
    await owner.pool_call("onload")
    ready = await owner.snapshot()
    assert ready.models[0].admission
    assert ready.topology_revision == initial.topology_revision


@pytest.mark.parametrize("shared", [False, True])
def test_manual_genrm_hook_uses_task_owner_in_all_layouts(shared):
    from relax.distributed.ray.inference_handle import shared_task_enabled

    args = SimpleNamespace(
        colocate=True,
        _genrm_instances_resolved={"judge": {}},
        _genrm_colocate_with_rollout=shared,
        defer_reward_to_post_process=False,
        custom_reward_post_process_path="custom.manual_genrm",
    )
    assert use_managed_rollout(args)
    assert shared_task_enabled(args)


@pytest.mark.parametrize("distributed_post", [False, True])
def test_shared_teacher_rejects_unmarked_deferred_genrm_before_placement(monkeypatch, distributed_post):
    from relax.distributed.ray.inference_handle import shared_task_enabled
    from relax.utils import misc

    def hook(args, samples):
        return [0.0] * len(samples)

    monkeypatch.setattr(misc, "load_function", lambda path: hook)
    args = SimpleNamespace(
        colocate=True,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        resource={"actor": [1, 2], "rollout": [1, 2], "teacher": [1, 2]},
        _genrm_instances_resolved={"judge": {}},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=True,
        custom_reward_post_process_path="test.legacy_hook",
        use_distributed_post=distributed_post,
    )
    with pytest.raises(ValueError, match="shared OPD Teacher with deferred GenRM requires a scoring-only"):
        shared_task_enabled(args)
    hook.managed_genrm_scoring = True
    assert shared_task_enabled(args)


@pytest.mark.parametrize(
    "scorer,split,distributed_post",
    [
        ("teacher", False, False),
        ("teacher", True, False),
        ("teacher", True, True),
        ("genrm", False, False),
        ("genrm", False, True),
        ("genrm", True, False),
        ("genrm", True, True),
    ],
)
def test_scoring_on_training_bundles_requires_rollout_offload(scorer, split, distributed_post):
    from relax.distributed.ray.inference_handle import shared_task_enabled

    args = SimpleNamespace(
        colocate=True,
        offload_rollout=False,
        use_distributed_post=distributed_post,
        use_opd=scorer == "teacher",
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        resource={"actor": [1, 2], "rollout": [1, 1 if split else 2], "teacher": [1, 1]},
        _genrm_instances_resolved={"judge": {}} if scorer == "genrm" else {},
        _genrm_colocate_with_rollout=scorer == "genrm" and not split,
    )
    with pytest.raises(ValueError, match="requires --offload-rollout"):
        shared_task_enabled(args)
    args.debug_rollout_only = True
    assert shared_task_enabled(args)
    args.debug_rollout_only = False
    args.offload_rollout = True
    assert shared_task_enabled(args)
    args.colocate = False
    args.offload_rollout = False
    assert shared_task_enabled(args)


def test_training_only_replay_skips_startup_with_shared_teacher():
    from relax.distributed.ray.inference_handle import shared_task_enabled

    args = SimpleNamespace(
        colocate=True,
        debug_train_only=True,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        resource={"actor": [1, 2], "rollout": [1, 2], "teacher": [1, 1]},
        offload_rollout=False,
    )
    assert not shared_task_enabled(args)

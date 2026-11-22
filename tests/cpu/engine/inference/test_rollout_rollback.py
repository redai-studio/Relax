# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager as module


def test_partial_rollout_cleanup_failure_preserves_locks_and_routers(monkeypatch):
    from relax.distributed.ray import policy_engine_pool as rollout
    from relax.distributed.ray.inference_pool import RolloutPool

    implementation = rollout.PolicyEnginePool
    lock = object()
    released = []

    def partial_init(backend, *args, **kwargs):
        backend._eviction_monitor_stop = object()
        backend.rollout_engine_lock = lock
        raise RuntimeError("startup failed")

    def failed_dispose(backend):
        raise TimeoutError("engine exit unconfirmed")

    monkeypatch.setattr(implementation, "__init__", partial_init)
    monkeypatch.setattr(implementation, "dispose", failed_dispose)
    monkeypatch.setattr(rollout, "stop_launched_routers", lambda: released.append("routers"))
    monkeypatch.setattr(module.ray, "kill", released.append)
    pool = RolloutPool.__new__(RolloutPool)
    with pytest.raises(TimeoutError, match="exit unconfirmed"):
        RolloutPool.__init__(pool, None, None)
    assert pool.backend.rollout_engine_lock is lock
    assert released == []
    monkeypatch.setattr(implementation, "dispose", lambda backend: None)
    pool.dispose()
    assert released == [lock]


def test_managed_rollout_disposal_retains_owner_on_failure(monkeypatch):
    owner = SimpleNamespace(dispose=SimpleNamespace(remote=lambda worker: object()))
    killed = []
    monkeypatch.setattr(module.ray, "kill", killed.append)

    def failed_get(*args, **kwargs):
        raise TimeoutError("cleanup unconfirmed")

    monkeypatch.setattr(module.ray, "get", failed_get)
    dispose = module.dispose_managed_rollout._function
    try:
        with pytest.raises(TimeoutError):
            dispose(owner, None)
        assert killed == []
        assert module._FAILED_ROLLBACK_OWNERS[id(owner)] is owner
        monkeypatch.setattr(module.ray, "get", lambda *args, **kwargs: None)
        dispose(owner, None)
        assert killed == [owner]
    finally:
        module._FAILED_ROLLBACK_OWNERS.pop(id(owner), None)


def test_actual_rollout_second_lock_failure_retains_pool_until_shutdown_retry(monkeypatch):
    from relax.distributed.ray import policy_engine_pool as rollout
    from relax.distributed.ray.inference_pool import RolloutPool

    lock = object()
    killed = []
    calls = []
    shutdown_attempts = []

    class Locks:
        @staticmethod
        def options(**kwargs):
            return Locks

        @staticmethod
        def remote():
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("second lock failed")
            return lock

    def shutdown():
        shutdown_attempts.append(None)
        if len(shutdown_attempts) == 1:
            raise TimeoutError("engine still running")

    engine = SimpleNamespace(shutdown=SimpleNamespace(remote=shutdown))
    group = SimpleNamespace(rank_offset=0, all_engines=[engine])
    server = SimpleNamespace(engine_groups=[group])
    monkeypatch.setattr(module.ray, "get", lambda *a, **k: None)
    monkeypatch.setattr(rollout, "Lock", Locks)
    monkeypatch.setattr(rollout, "init_tracking", lambda *a, **k: None)
    monkeypatch.setattr(rollout, "init_http_client", lambda *a: None)
    monkeypatch.setattr(rollout, "start_rollout_servers", lambda *a: {"policy": server})
    monkeypatch.setattr(rollout, "with_control_plane_affinity", lambda args, options: options)
    monkeypatch.setattr(module.ray, "kill", killed.append)
    args = SimpleNamespace(use_agentic_rollout=False, debug_train_only=False, use_fault_tolerance=False)
    pool = RolloutPool.__new__(RolloutPool)
    with pytest.raises(RuntimeError, match="engine still running"):
        RolloutPool.__init__(pool, args, None)
    assert pool.backend.rollout_engine_lock is lock
    assert killed == []
    pool.dispose()
    assert len(shutdown_attempts) == 2
    assert killed == [lock]


@pytest.mark.parametrize("startup_failure", [False, True])
def test_managed_sft_startup_uses_resolved_steps_and_keeps_task_owner(monkeypatch, startup_failure):
    from unittest.mock import Mock

    from relax.distributed.ray import placement_group, rollout_worker

    manager = Mock()
    manager.start.remote.return_value = {"router_ip": "127.0.0.1", "router_port": 30000}
    worker = Mock()
    if startup_failure:
        worker.configure.remote.side_effect = RuntimeError("worker startup failed")
    worker_factory = Mock()
    worker_factory.options.return_value.remote.return_value = worker
    monkeypatch.setattr(rollout_worker, "RolloutWorker", worker_factory)
    monkeypatch.setattr(placement_group, "_get_head_node_id", lambda: "ab" * 28)
    monkeypatch.setattr(module.ray, "get", lambda ref, **kwargs: ref)
    kill = Mock()
    monkeypatch.setattr(module.ray, "kill", kill)
    monkeypatch.setattr("relax.utils.agentic_obs.startup.record_run_startup", Mock())
    args = SimpleNamespace(
        _rollout_engine_manager=manager,
        _inference_phase_coordinator=manager,
        _inference_startup_token="deployment",
        loss_type="sft",
        rollout_global_dataset=False,
        num_epoch=2,
        num_rollout=12,
        num_rollout_per_epoch=6,
        offload_rollout=False,
        check_weight_update_equal=False,
    )
    if startup_failure:
        with pytest.raises(RuntimeError, match="worker startup failed"):
            module.create_managed_rollout(args, None, None, None)
        worker.close.remote.assert_called_once_with()
        kill.assert_called_once_with(worker)
        manager.attach_workload.remote.assert_not_called()
    else:
        handle, count = module.create_managed_rollout(args, None, None, None)
        assert handle.worker is worker and count == 6
        manager.attach_workload.remote.assert_called_once_with(worker, "deployment")
        kill.assert_not_called()
    worker.get_num_rollout_per_epoch.remote.assert_not_called()
    worker.configure.remote.assert_called_once_with(12, 6)
    manager.dispose.remote.assert_not_called()

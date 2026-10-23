# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU control-flow checks; these do not validate NCCL or live recovery."""

import asyncio
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    pytest.importorskip("megatron.core")
    from relax.backends.megatron import actor

    monkeypatch.setattr(actor, "dist", Mock(get_rank=Mock(return_value=0)))
    monkeypatch.setattr(actor, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor, "ray", SimpleNamespace(get=lambda value, **kwargs: value))
    monkeypatch.setattr(actor.time, "sleep", Mock())
    return actor


def _actor(backend, **args):
    args.setdefault("inference_recovery_state", object())
    worker = SimpleNamespace(args=SimpleNamespace(inference_recovery=True, **args))
    worker._inference_recovery = worker.args.inference_recovery_state
    for name in ("_begin_inference_sync", "_end_inference_sync"):
        setattr(worker, name, MethodType(getattr(backend.MegatronTrainRayActor, name), worker))
    return worker


@pytest.mark.parametrize("mode", ["disabled", "paused", "ready", "error"])
def test_sync_reservation(backend, mode):
    state = SimpleNamespace(
        begin_sync=SimpleNamespace(remote=Mock(side_effect=[False, True])),
        snapshot=SimpleNamespace(remote=Mock(return_value={"paused": mode == "paused"})),
    )
    worker = _actor(backend, inference_recovery_state=state)
    if mode == "disabled":
        worker.args.inference_recovery = False
        worker._inference_recovery = None
    if mode == "error":
        state.begin_sync.remote.side_effect = ValueError("state unavailable")
        with pytest.raises(RuntimeError, match="state unavailable"):
            worker._begin_inference_sync()
    else:
        assert worker._begin_inference_sync() is (mode != "paused")
    if mode == "disabled":
        state.begin_sync.remote.assert_not_called()
        backend.dist.broadcast_object_list.assert_not_called()
    else:
        backend.dist.broadcast_object_list.assert_called_once()
        assert backend.dist.broadcast_object_list.call_args.kwargs == {"src": 0, "group": "gloo"}
    if mode == "ready":
        assert state.begin_sync.remote.call_count == 2
        backend.time.sleep.assert_called_once_with(0.1)


@pytest.mark.parametrize("can_sync,error", [(True, None), (False, None), (False, "ValueError: state unavailable")])
def test_sender_rank_uses_broadcast_sync_decision(backend, can_sync, error):
    state = Mock()
    worker = _actor(backend, inference_recovery_state=state)
    backend.dist.get_rank.return_value = 1
    backend.dist.broadcast_object_list.side_effect = lambda payload, **kwargs: payload.__setitem__(
        0, {"can_sync": can_sync, "error": error}
    )

    if error is None:
        assert worker._begin_inference_sync() is can_sync
    else:
        with pytest.raises(RuntimeError, match="state unavailable"):
            worker._begin_inference_sync()
    state.begin_sync.remote.assert_not_called()
    backend.dist.broadcast_object_list.assert_called_once()


@pytest.mark.parametrize("mode", ["normal", "paused", "sender_error"])
def test_train_async_sync_tail(backend, monkeypatch, mode):
    state = SimpleNamespace(sync=False)

    def begin():
        state.sync = mode != "paused"
        return state.sync

    state.begin_sync = SimpleNamespace(remote=Mock(side_effect=begin))
    state.snapshot = SimpleNamespace(remote=Mock(return_value={"paused": True}))
    state.end_sync = SimpleNamespace(remote=Mock(side_effect=lambda: setattr(state, "sync", False)))
    worker = _actor(
        backend,
        inference_recovery_state=state,
        use_routing_replay=False,
        num_rollout=1,
        true_on_policy_mode=False,
        kl_coef=0,
        use_kl_loss=False,
        multimodal_keys=None,
        use_opd=False,
        advantage_estimator="grpo",
    )
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    buffer = [({"total_lengths": [2], "loss_masks": [[1]]}, None)]
    worker.data_iterator = [Mock(get_buffer=Mock(return_value=buffer))]
    worker.num_microbatches = 1
    worker.prof, worker.tokenizer, worker.flops_counter = Mock(), None, None
    worker._wait_for_previous_eval = Mock()
    worker._check_services_health = Mock(return_value=(False, False))
    worker.update_weights_fully_async = Mock()
    worker._run_step_evaluation = Mock()
    if mode == "sender_error":
        worker.update_weights_fully_async.side_effect = RuntimeError("sender failed")
    for name in ("timer", "inverse_timer"):
        monkeypatch.setattr(backend, name, lambda *args: nullcontext())
    for name in ("train", "log_rollout_data", "log_perf_data"):
        monkeypatch.setattr(backend, name, Mock())
    monkeypatch.setattr(backend, "train_dump_utils", Mock())
    monkeypatch.setattr(backend, "tracking_utils", Mock())
    monkeypatch.setattr(backend, "compute_rollout_step", lambda *args: 0)
    monkeypatch.setattr(backend, "evaluation_step_for_rollout", lambda *args: 0)
    monkeypatch.setattr(backend, "Timer", lambda: SimpleNamespace())
    monkeypatch.setattr(backend, "mpu", Mock(get_data_parallel_world_size=Mock(return_value=1)))
    backend.dist.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
    if mode == "sender_error":
        with pytest.raises(RuntimeError, match="sender failed"):
            backend.MegatronTrainRayActor.train_async(worker, 0)
    else:
        backend.MegatronTrainRayActor.train_async(worker, 0)
        backend.tracking_utils.flush_metrics.assert_called_once()
    backend.train.assert_called_once()
    assert backend.train.call_args.args[1:4] == (worker.model, worker.optimizer, worker.opt_param_scheduler)
    assert state.sync is (mode == "sender_error")
    assert state.end_sync.remote.call_count == (mode == "normal")
    assert worker._run_step_evaluation.call_count == (mode == "normal")
    if mode == "paused":
        worker._wait_for_previous_eval.assert_not_called()
        worker._check_services_health.assert_not_called()
        worker.update_weights_fully_async.assert_not_called()
    elif mode == "normal":
        worker._run_step_evaluation.assert_called_once_with(0, end_update_weight=True)
        assert worker._wait_for_previous_eval.call_count == 2


@pytest.mark.parametrize("mode", ["success", "export_error", "cleanup_error"])
def test_colocate_cpu_export_preserves_training_state(backend, monkeypatch, mode):
    events = []
    pool_alive = False

    @contextmanager
    def temporary_pool():
        nonlocal pool_alive
        pool_alive = True
        events.append("enter")
        try:
            yield
        finally:
            pool_alive = False
            events.append("exit")

    def clear():
        assert pool_alive, "temporary buffers must be collected before their memory pool exits"
        events.append("clear")
        if mode == "cleanup_error":
            raise RuntimeError("cleanup failed")

    weights = {"weight": torch.zeros(1)}
    getter = Mock(return_value=weights)
    updater = Mock(spec=backend.UpdateWeightFromTensor)
    updater.weights_getter = getter
    updater.lora_enabled = False
    updater._model_update_groups = "old_sender"
    updater._ipc_engine = object()

    def update():
        assert updater.weights_getter() is weights
        if mode == "export_error":
            raise RuntimeError("export failed")
        events.append("export")

    updater.update_weights.side_effect = update
    worker = _actor(
        backend,
        colocate=True,
        fully_async=False,
        offload_train=True,
        enable_weights_backuper=True,
        offload_rollout=True,
        distributed_timeout_minutes=1,
    )
    worker.role = "actor"
    worker._inference_recovery_parked = True
    worker.weight_updater = updater
    worker.weights_backuper = Mock(get=Mock(return_value=weights))
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    identities = (worker.model, worker.optimizer, worker.opt_param_scheduler)
    worker.wake_up = Mock()
    worker._train_state_offloader = Mock(disable_during_update=Mock(side_effect=temporary_pool))
    manager = Mock()
    manager.get_rollout_engines_and_lock.remote.return_value = (["new"], "lock", 1, [1], [0])
    manager.onload_kv.remote.side_effect = lambda: events.append("kv")
    worker.rollout_manager = manager
    monkeypatch.setattr(backend, "reload_process_groups", Mock())
    monkeypatch.setattr(backend, "destroy_process_groups", lambda: events.append("destroy"))
    monkeypatch.setattr(backend, "clear_memory", clear)
    monkeypatch.setattr(backend, "device_module", Mock())
    if mode != "success":
        with pytest.raises(RuntimeError, match="export failed|cleanup failed"):
            backend.MegatronTrainRayActor.recovery_update_weights_colocate(worker)
        assert events == ["enter"] + (["export"] if mode == "cleanup_error" else []) + ["clear", "exit", "destroy"]
        manager.onload_kv.remote.assert_not_called()
        manager.clear_num_new_engines.remote.assert_not_called()
    else:
        backend.MegatronTrainRayActor.recovery_update_weights_colocate(worker)
        assert events == ["enter", "export", "clear", "exit", "destroy", "kv"]
        manager.clear_num_new_engines.remote.assert_called_once()
    assert (worker.model, worker.optimizer, worker.opt_param_scheduler) == identities
    assert updater.weights_getter is getter
    assert updater._model_update_groups is None
    assert updater._ipc_engine is None
    worker.weights_backuper.backup.assert_not_called()
    worker.weights_backuper.copy.assert_not_called()
    worker.wake_up.assert_not_called()
    worker._train_state_offloader.reload.assert_not_called()
    backend.dist.destroy_process_group.assert_called_once_with("old_sender")


@pytest.mark.parametrize("mode", ["success", "http_error", "wrong_role"])
def test_unregister_is_strict(backend, monkeypatch, mode):
    response = Mock()
    if mode == "http_error":
        response.raise_for_status.side_effect = RuntimeError("unregister rejected")
    client = SimpleNamespace(
        role="actor" if mode == "wrong_role" else "actor_fwd",
        rank=2,
        coordinator_url="http://coordinator.test",
        _registered=True,
        _http_client=SimpleNamespace(delete=AsyncMock(return_value=response)),
    )
    monkeypatch.setattr(backend, "run", asyncio.run)
    worker = SimpleNamespace(checkpoint_engine_client=client)
    if mode == "success":
        backend.MegatronTrainRayActor.unregister_inference_transport(worker)
    else:
        with pytest.raises(RuntimeError, match="Only the replaced|unregister rejected"):
            backend.MegatronTrainRayActor.unregister_inference_transport(worker)
    assert client._registered is (mode != "success")
    if mode == "wrong_role":
        client._http_client.delete.assert_not_called()
    else:
        client._http_client.delete.assert_awaited_once_with(
            "http://coordinator.test/unregister", params={"role": "actor_fwd", "rank": 2}
        )
        response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("fail_push", [False, True])
def test_hybrid_recovery_push_preserves_training_and_snapshot_tags(backend, fail_push):
    snapshot = {"weight": torch.zeros(1)}
    updater = Mock(spec=backend.UpdateWeightFromTensor)
    updater.lora_enabled = False
    updater._model_update_groups = "old transport"
    updater._ipc_engine = object()
    updater.weights_getter = lambda: snapshot
    updater.update_weights.side_effect = RuntimeError("push failed") if fail_push else None
    worker = _actor(backend, hybrid=True, offload_train=False, offload_rollout=False)
    worker.role = "actor"
    worker.weight_updater = updater
    worker.weights_backuper = Mock(get=Mock(return_value=snapshot))
    worker._train_state_offloader = SimpleNamespace(disable_during_update=nullcontext)
    worker.rollout_manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: (["new engine"], "lock", 1, [2], [0])),
        clear_num_new_engines=SimpleNamespace(remote=Mock()),
    )
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    identities = (worker.model, worker.optimizer, worker.opt_param_scheduler)
    if fail_push:
        with pytest.raises(RuntimeError, match="push failed"):
            backend.MegatronTrainRayActor.recovery_update_weights_hybrid(worker)
        worker.rollout_manager.clear_num_new_engines.remote.assert_not_called()
    else:
        backend.MegatronTrainRayActor.recovery_update_weights_hybrid(worker)
        worker.rollout_manager.clear_num_new_engines.remote.assert_called_once()
    assert (worker.model, worker.optimizer, worker.opt_param_scheduler) == identities
    worker.weights_backuper.backup.assert_not_called()
    worker.weights_backuper.copy.assert_not_called()
    assert updater.weights_getter() is snapshot
    updater.connect_rollout_engines.assert_called_once_with(
        ["new engine"], "lock", engine_gpu_counts=[2], engine_gpu_offsets=[0]
    )
    assert updater._model_update_groups is None and updater._ipc_engine is None

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU control-flow checks; these do not validate NCCL or live recovery."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    pytest.importorskip("megatron.core")
    from relax.backends.megatron import actor
    from relax.backends.megatron.weight_update import update_weight_from_tensor

    monkeypatch.setattr(actor, "dist", Mock(get_rank=Mock(return_value=0)))
    monkeypatch.setattr(update_weight_from_tensor, "dist", actor.dist)
    monkeypatch.setattr(actor, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor, "ray", SimpleNamespace(get=lambda value, **kwargs: value))
    monkeypatch.setattr(actor.time, "sleep", Mock())
    return actor


def _actor(backend, **args):
    return SimpleNamespace(args=SimpleNamespace(**args))


@pytest.mark.parametrize("mode", ["normal", "deferred", "tail_error"])
def test_train_async_sync_tail(backend, monkeypatch, mode):
    from relax.utils import timer as timing
    from relax.utils.training import train_metric_utils

    clock = [0.0]
    timing_state = timing.Timer()
    monkeypatch.setattr(timing_state, "timers", {})
    monkeypatch.setattr(timing_state, "start_time", {})
    monkeypatch.setattr(timing_state, "start_info", {})
    monkeypatch.setattr(timing_state, "records", [])
    monkeypatch.setattr(timing, "time", lambda: clock[0])
    monkeypatch.setattr(timing, "_is_dist_rank0", lambda: False)

    def advance(seconds):
        clock[0] += seconds

    timing_state.start("train_wait")
    worker = _actor(
        backend,
        use_routing_replay=False,
        num_rollout=1,
        true_on_policy_mode=False,
        kl_coef=0,
        use_kl_loss=False,
        multimodal_keys=None,
        fully_async=True,
        hybrid=False,
        wandb_always_use_train_step=False,
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
    worker.update_weights_fully_async = Mock(side_effect=lambda *args, **kwargs: advance(2.0))
    worker._run_step_evaluation = Mock()
    worker.evaluate_after_training = Mock()
    worker._log_async_training_perf = lambda step, data: backend.MegatronTrainRayActor._log_async_training_perf(
        worker, step, data
    )
    if mode == "tail_error":
        worker.evaluate_after_training.side_effect = RuntimeError("sender failed")
    monkeypatch.setattr(backend, "train", Mock(side_effect=lambda *args: advance(1.0)))
    monkeypatch.setattr(backend, "log_rollout_data", Mock())
    monkeypatch.setattr(
        backend,
        "log_perf_data",
        Mock(side_effect=lambda step, args, **kwargs: train_metric_utils.log_perf_data_raw(step, args, True)),
    )
    monkeypatch.setattr(backend, "train_dump_utils", Mock())
    monkeypatch.setattr(backend, "tracking_utils", Mock())
    monkeypatch.setattr(train_metric_utils, "tracking_utils", backend.tracking_utils)
    monkeypatch.setattr(backend, "compute_rollout_step", lambda *args: 0)
    monkeypatch.setattr(backend, "evaluation_step_for_rollout", lambda *args: 0)
    monkeypatch.setattr(backend, "mpu", Mock(get_data_parallel_world_size=Mock(return_value=1)))
    backend.dist.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
    if mode == "tail_error":
        with pytest.raises(RuntimeError, match="sender failed"):
            backend.MegatronTrainRayActor.train_async(worker, 0, defer_inference=mode == "deferred")
    else:
        backend.MegatronTrainRayActor.train_async(worker, 0, defer_inference=mode == "deferred")
        if mode == "deferred":
            backend.log_perf_data.assert_not_called()
            backend.tracking_utils.flush_metrics.assert_not_called()
            worker.update_weights_fully_async(0)
            backend.MegatronTrainRayActor.finish_training_metrics(worker, 0)
        backend.tracking_utils.flush_metrics.assert_called_once()
        metrics = backend.tracking_utils.log.call_args.args[1]
        assert metrics["perf/train_time"] == pytest.approx(3.0)
        assert metrics["perf/train_wait_time"] == pytest.approx(0.0)
        assert "train" not in timing_state.start_time
    if mode == "tail_error":
        backend.tracking_utils.flush_metrics.assert_not_called()
    backend.train.assert_called_once()
    assert backend.train.call_args.args[1:4] == (worker.model, worker.optimizer, worker.opt_param_scheduler)
    assert worker.evaluate_after_training.call_count == (mode != "deferred")


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
    updater.args = SimpleNamespace(offload_rollout=True)
    updater.prepare_for_recovery.side_effect = lambda *topology: backend.UpdateWeightFromTensor.prepare_for_recovery(
        updater, *topology
    )
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
    worker._training_state_parked = True
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
            backend.MegatronTrainRayActor.recovery_update_weights_colocate(worker, ["new"], "lock", [1], [0])
        assert events == ["enter"] + (["export"] if mode == "cleanup_error" else []) + ["clear", "exit", "destroy"]
        manager.onload_kv.remote.assert_not_called()
        manager.clear_num_new_engines.remote.assert_not_called()
    else:
        backend.MegatronTrainRayActor.recovery_update_weights_colocate(worker, ["new"], "lock", [1], [0])
        assert events == ["enter", "export", "clear", "exit", "destroy"]
        manager.clear_num_new_engines.remote.assert_not_called()
    assert (worker.model, worker.optimizer, worker.opt_param_scheduler) == identities
    assert updater.weights_getter is getter
    assert updater._model_update_groups is None
    assert updater._ipc_engine is None
    worker.weights_backuper.backup.assert_not_called()
    worker.weights_backuper.copy.assert_not_called()
    worker.wake_up.assert_not_called()
    worker._train_state_offloader.reload.assert_not_called()
    backend.dist.destroy_process_group.assert_called_once_with("old_sender")

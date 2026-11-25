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


@pytest.mark.parametrize("failed_updates", [0, 2], ids=["successful_sync", "retry_failed_sync"])
def test_train_async_metrics_include_the_component_owned_tail(backend, monkeypatch, failed_updates):
    from relax.distributed.ray import inference_sync
    from relax.distributed.ray.actor_group import RayTrainGroup
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

    def update_weights(step, **kwargs):
        advance(2)
        if step < failed_updates:
            raise RuntimeError("Failed to fetch rollout topology: DCS coordinator unavailable")

    timing_state.start("train_wait")
    args = SimpleNamespace(
        use_routing_replay=False,
        num_rollout=failed_updates + 2,
        true_on_policy_mode=True,
        kl_coef=0,
        use_kl_loss=False,
        multimodal_keys=None,
        fully_async=True,
        hybrid=False,
        debug_train_only=False,
        debug_rollout_only=False,
        wandb_always_use_train_step=False,
        use_opd=False,
        advantage_estimator="grpo",
    )
    worker = object.__new__(backend.MegatronTrainRayActor)
    worker.args = args
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    buffer = [({"total_lengths": [2], "loss_masks": [[1]]}, None)]
    worker.data_iterator = [Mock(get_buffer=Mock(return_value=buffer))]
    worker.num_microbatches = 1
    worker.prof, worker.tokenizer, worker.flops_counter = Mock(), None, None
    worker.update_weights_fully_async = Mock(side_effect=update_weights)
    rank_handle = Mock()
    for method in ("record_inference_wait", "update_weights_fully_async", "finish_training_metrics"):
        getattr(rank_handle, method).remote.side_effect = getattr(worker, method)
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [rank_handle]
    monkeypatch.setattr(inference_sync, "check_services_health", Mock(return_value=(True, False)))
    monkeypatch.setattr(inference_sync, "wait_for_previous_eval", Mock())
    monkeypatch.setattr(inference_sync, "_evaluate_rollout", Mock())
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
    for step in range(failed_updates + 1):
        advance(4)
        worker.train_async(step)
        backend.log_perf_data.assert_not_called()
        backend.tracking_utils.flush_metrics.assert_not_called()
        if step < failed_updates:
            with pytest.raises(RuntimeError, match="Failed to fetch rollout topology: DCS coordinator unavailable"):
                inference_sync.complete_inference_step(
                    args, group, step, has_rollout=True, managed=False, settle=lambda refs: refs
                )
        else:
            inference_sync.complete_inference_step(
                args, group, step, has_rollout=True, managed=False, settle=lambda refs: refs
            )

    backend.tracking_utils.flush_metrics.assert_called_once()
    metrics = backend.tracking_utils.log.call_args.args[1]
    # Each attempt computes for one second. Only the successful two-second
    # sync is finalized. Failed tails stay in the interval before the retry.
    assert metrics["perf/train_time"] == 3 + failed_updates
    assert metrics["perf/train_wait_time"] == 4 + 6 * failed_updates
    trace = [event.to_trace_event() for event in timing_state.log_record_and_clear(step)]
    for name in ("train", "train_wait"):
        duration_us = sum(event["dur"] for event in trace if event["name"] == name)
        assert duration_us == metrics[f"perf/{name}_time"] * 1_000_000
    advance(5)
    worker.train_async(step + 1)
    inference_sync.complete_inference_step(
        args, group, step + 1, has_rollout=True, managed=False, settle=lambda refs: refs
    )
    assert backend.tracking_utils.flush_metrics.call_count == 2
    next_metrics = backend.tracking_utils.log.call_args.args[1]
    assert (next_metrics["perf/train_time"], next_metrics["perf/train_wait_time"]) == (3, 5)
    assert backend.train.call_count == args.num_rollout
    assert backend.train.call_args.args[1:4] == (worker.model, worker.optimizer, worker.opt_param_scheduler)


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

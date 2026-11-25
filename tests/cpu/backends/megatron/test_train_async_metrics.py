# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify fully-async step timing across training and synchronization."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def backend(monkeypatch):
    pytest.importorskip("megatron.core")
    from relax.backends.megatron import actor

    monkeypatch.setattr(actor, "dist", Mock(get_rank=Mock(return_value=0)))
    monkeypatch.setattr(actor, "get_gloo_group", lambda: "gloo")
    return actor


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
    monkeypatch.setattr(inference_sync.ray, "get", lambda refs: refs)
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
                inference_sync.complete_inference_step(args, group, step, has_rollout=True, managed=False)
        else:
            inference_sync.complete_inference_step(args, group, step, has_rollout=True, managed=False)

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
    inference_sync.complete_inference_step(args, group, step + 1, has_rollout=True, managed=False)
    assert backend.tracking_utils.flush_metrics.call_count == 2
    next_metrics = backend.tracking_utils.log.call_args.args[1]
    assert (next_metrics["perf/train_time"], next_metrics["perf/train_wait_time"]) == (3, 5)
    assert backend.train.call_count == args.num_rollout
    assert backend.train.call_args.args[1:4] == (worker.model, worker.optimizer, worker.opt_param_scheduler)

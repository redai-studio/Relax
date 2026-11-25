# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify fully-async step timing across training and group-owned synchronization."""

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


def test_train_async_metrics_include_the_component_owned_tail(backend, monkeypatch):
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
    args = SimpleNamespace(
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
    worker = SimpleNamespace(args=args)
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    buffer = [({"total_lengths": [2], "loss_masks": [[1]]}, None)]
    worker.data_iterator = [Mock(get_buffer=Mock(return_value=buffer))]
    worker.num_microbatches = 1
    worker.prof, worker.tokenizer, worker.flops_counter = Mock(), None, None
    worker.update_weights_fully_async = Mock(side_effect=lambda *args, **kwargs: advance(2.0))
    worker._log_async_training_perf = lambda step, data: backend.MegatronTrainRayActor._log_async_training_perf(
        worker, step, data
    )
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
    backend.MegatronTrainRayActor.train_async(worker, 0)
    backend.log_perf_data.assert_not_called()
    backend.tracking_utils.flush_metrics.assert_not_called()
    worker.update_weights_fully_async.assert_not_called()
    worker.update_weights_fully_async(0)
    backend.MegatronTrainRayActor.finish_training_metrics(worker, 0)
    backend.tracking_utils.flush_metrics.assert_called_once()
    metrics = backend.tracking_utils.log.call_args.args[1]
    assert metrics["perf/train_time"] == pytest.approx(3.0)
    assert metrics["perf/train_wait_time"] == pytest.approx(0.0)
    assert "train" not in timing_state.start_time
    backend.train.assert_called_once()
    assert backend.train.call_args.args[1:4] == (worker.model, worker.optimizer, worker.opt_param_scheduler)

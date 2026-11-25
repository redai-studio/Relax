# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise training and post-training synchronization together."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.components.actor import Actor
from relax.distributed.ray import inference_sync
from relax.distributed.ray.actor_group import RayTrainGroup


@pytest.mark.parametrize(
    "mode,fail_at",
    [("colocate", None), ("hybrid", None), ("fully_async", None), ("colocate", "weights"), ("hybrid", "train")],
)
def test_component_owns_sync_after_all_training_ranks_settle(monkeypatch, mode, fail_at):
    events = []

    def settle(refs):
        (event,) = refs
        events.append(event)
        if event == fail_at:
            raise RuntimeError("rank failed")

    def send_weights(rollout_id, rollout_only=False, actor_fwd_only=False):
        assert (rollout_id, rollout_only, actor_fwd_only) == (4, True, False)
        return "weights"

    monkeypatch.setattr(inference_sync.ray, "get", settle)
    monkeypatch.setattr(inference_sync, "_evaluate_rollout", lambda *args, **kwargs: events.append("evaluate"))
    monkeypatch.setattr(inference_sync, "wait_for_previous_eval", Mock())
    monkeypatch.setattr(inference_sync, "check_services_health", Mock(return_value=(True, False)))
    rank = Mock()
    train_method = {"colocate": "train", "hybrid": "train_hybrid", "fully_async": "train_async"}[mode]
    getattr(rank, train_method).remote.return_value = "train"
    if mode == "fully_async":
        rank.update_weights_fully_async.remote.side_effect = send_weights
    else:
        rank.update_weights.remote.return_value = "weights"
    rank.record_inference_wait.remote.return_value = "handshake"
    rank.finish_training_metrics.remote.return_value = "flush"
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [rank]
    actor = SimpleNamespace(
        config=SimpleNamespace(
            hybrid=mode == "hybrid",
            fully_async=mode != "colocate",
            num_critic_only_steps=0,
            num_rollout=5,
            train_backend="megatron",
            debug_train_only=False,
            debug_rollout_only=False,
        ),
        step=4,
        actor_model=group,
        rollout_manager=object(),
        _managed_lora_publication=False,
        _maybe_save_model=lambda: events.append("save"),
    )
    if fail_at:
        with pytest.raises(RuntimeError, match="rank failed"):
            Actor.func_or_class._execute_training(actor)
    else:
        assert Actor.func_or_class._execute_training(actor)
    expected = {
        "colocate": ["train", "weights", "flush", "evaluate"],
        "hybrid": ["train", "handshake", "weights", "evaluate", "flush"],
        "fully_async": ["train", "handshake", "weights", "evaluate", "flush", "save"],
    }[mode]
    if fail_at:
        expected = expected[: expected.index(fail_at) + 1]
    assert events == expected
    getattr(rank, train_method).remote.assert_called_once_with(4)


def test_fsdp_component_does_not_dispatch_megatron_tail(monkeypatch):
    group = SimpleNamespace(async_train=Mock(return_value=["fsdp-step"]))
    get = Mock()
    monkeypatch.setattr(inference_sync.ray, "get", get)
    actor = SimpleNamespace(
        config=SimpleNamespace(train_backend="fsdp", hybrid=False, fully_async=False, num_critic_only_steps=0),
        step=2,
        actor_model=group,
    )
    assert Actor.func_or_class._execute_training(actor)
    group.async_train.assert_called_once_with(2)
    get.assert_called_once_with(["fsdp-step"])

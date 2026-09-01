# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import torch


try:
    from relax.backends.megatron import actor as actor_module
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {exc}", allow_module_level=True)


def test_sft_prepacked_iterator_close_releases_device_batch_and_event():
    iterator = actor_module._SFTPrepackedDeviceIterator(
        packed_cpu=[],
        first_device_micro_batch=(actor_module.PrepackedBatch(), None),
        first_ready_event=object(),
        copy_stream=object(),
        device=torch.device("cpu"),
    )

    iterator.close()

    assert iterator._packed_cpu == []
    assert iterator._next_device_micro_batch is None
    assert iterator._next_ready_event is None


def test_sft_peer_error_agreement_propagates_peer_failure(monkeypatch):
    groups = []

    def mark_peer_error(error_flag, *, op, group):
        groups.append(group)
        error_flag.fill_(1)

    monkeypatch.setattr(actor_module.dist, "all_reduce", mark_peer_error)

    with pytest.raises(RuntimeError, match="failed on a peer rank"):
        actor_module._raise_if_sft_peer_error(
            None,
            phase="validation",
            device=torch.device("cpu"),
            tp_group="tp",
            dp_group="dp",
        )

    assert groups == ["tp", "dp"]


def test_sft_lookahead_pauses_on_checkpoint_boundary(monkeypatch):
    monkeypatch.setattr(actor_module, "should_run_sft_eval", lambda *_args: False)
    monkeypatch.setattr(actor_module, "should_run_sft_predict", lambda *_args: False)
    args = Namespace(num_rollout=100, save="/checkpoint", rotate_ckpt=False, save_interval=20)

    assert actor_module._should_pause_sft_prepack_lookahead(args, rollout_id=19) is True
    assert actor_module._should_pause_sft_prepack_lookahead(args, rollout_id=18) is False


def test_sft_prepack_fetch_concatenates_ready_tq_shards(monkeypatch):
    partition_ids = ["sft_3_shard_0_of_2", "sft_3_shard_1_of_2"]
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_rank", lambda **_kwargs: 1)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 2)
    monkeypatch.setattr(actor_module, "run", lambda value: value)

    calls = []

    def _get_data_from_transfer_queue(**kwargs):
        calls.append(kwargs)
        shard_id = len(calls) - 1
        base = shard_id * 10
        return (
            {
                "tokens": [[base], [base + 1]],
                "total_lengths": [base, base + 1],
            },
            None,
        )

    monkeypatch.setattr(actor_module, "get_data_from_transfer_queue", _get_data_from_transfer_queue)

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = Namespace(global_batch_size=8, loss_type="sft", sft_async_prepack=True)
    actor.data_system_client = MagicMock()
    actor.data_system_client.async_get_partition_list.return_value = partition_ids

    batch = actor._fetch_sft_prepack_rollout_once("sft_train", rollout_id=3, data_fields=["tokens"])

    assert batch == {
        "tokens": [[0], [1], [10], [11]],
        "total_lengths": [0, 1, 10, 11],
    }
    assert [call["partition_id"] for call in calls] == partition_ids
    assert [call["batch_size"] for call in calls] == [2, 2]
    assert all(call["sampling_config"]["dp_rank"] == 1 for call in calls)


def test_sft_prepack_fetch_waits_until_all_tq_shards_are_ready(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "2")
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_rank", lambda **_kwargs: 0)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 2)
    monkeypatch.setattr(actor_module, "run", lambda value: value)
    fetch = MagicMock()
    monkeypatch.setattr(actor_module, "get_data_from_transfer_queue", fetch)

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = Namespace(global_batch_size=8, loss_type="sft", sft_async_prepack=True)
    actor.data_system_client = MagicMock()
    actor.data_system_client.async_get_partition_list.return_value = ["sft_3_shard_0_of_2"]

    assert actor._fetch_sft_prepack_rollout_once("sft_train", rollout_id=3, data_fields=["tokens"]) is None
    fetch.assert_not_called()

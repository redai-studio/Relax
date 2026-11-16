# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Image SFT must not consume RL trajectories or invoke rollout-only hooks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.backends.fsdp import actor, sft


@pytest.mark.parametrize("task", ["i2i", "t2i"])
def test_image_sft_actor_dispatches_training_and_saving_to_sft(task):
    worker = SimpleNamespace(args=SimpleNamespace(loss_type="sft", generation_task=task), sft=Mock())
    actor.FSDPTrainRayActor.train(worker, 3)
    actor.FSDPTrainRayActor.save_model(worker, 3, force_sync=True)
    worker.sft.train.assert_called_once_with(3)
    worker.sft.save.assert_called_once_with()


@pytest.mark.parametrize("task", ["i2i", "t2i"])
def test_image_sft_partition_fetch_requires_sample_ids_only(task):
    args = SimpleNamespace(loss_type="sft", generation_task=task, global_batch_size=8)
    setattr(args, f"_{task}_sft_attempt_id", "test-attempt")
    client = Mock()
    client.get_meta.return_value = SimpleNamespace(size=8)
    worker = SimpleNamespace(args=args, _tq_client=lambda: client)
    result = actor.FSDPTrainRayActor._read_train_partition(worker, 3)
    assert result is client.get_data.return_value
    assert client.get_meta.call_args.kwargs["data_fields"] == ["sample_id"]
    assert client.get_meta.call_args.kwargs["partition_id"] == "sft_test-attempt_3"


def test_t2i_sft_rejects_rl_only_adapter_before_model_loading(monkeypatch):
    class RLAdapter:
        supported_tasks = ("t2i",)

        def load_train_model(self, args):
            pytest.fail("Wrong adapter must fail before loading weights")

    monkeypatch.setattr(actor, "load_function", lambda _: RLAdapter)
    args = SimpleNamespace(loss_type="sft", generation_task="t2i", model_adapter_path="test.adapter")
    with pytest.raises(ValueError, match="GenerativeSFTAdapter implementing prepare_for_sharding"):
        actor.FSDPTrainRayActor._load_adapter(args)


def test_sft_rejects_adapter_task_mismatch(monkeypatch):
    monkeypatch.setattr(actor, "load_function", lambda _: lambda: SimpleNamespace(supported_tasks=("i2i",)))
    args = SimpleNamespace(loss_type="sft", generation_task="t2i", model_adapter_path="test.adapter")
    with pytest.raises(ValueError, match="does not support generation_task=t2i"):
        actor.FSDPTrainRayActor._load_adapter(args)


def test_rl_adapter_loading_does_not_require_sft_methods(monkeypatch):
    adapter = object()
    monkeypatch.setattr(actor, "load_function", lambda _: lambda: adapter)
    args = SimpleNamespace(loss_type="policy_loss", generation_task="t2i", model_adapter_path="test.adapter")
    assert actor.FSDPTrainRayActor._load_adapter(args) is adapter


@pytest.mark.parametrize("task", ["i2i_sft", "t2i_sft"])
def test_sft_resume_uses_own_namespace_and_restores_rng(monkeypatch, tmp_path, task):
    trainer = sft.FSDPSFTTrainer.__new__(sft.FSDPSFTTrainer)
    trainer.task, trainer.rank = task, 0
    trainer.args = SimpleNamespace(load=str(tmp_path), _start_rollout_id_explicit=True, start_rollout_id=3)
    trainer.contract = {"family": "test", "config": {"dataset_size": 8}}
    trainer.actor = Mock()
    trainer.actor._trainable_mode = "lora"
    trainer.scheduler = Mock()
    latest = Mock(return_value=3)
    contract = Mock(return_value=trainer.contract)
    restore = Mock()
    load = Mock(return_value={"completed_steps": 3, "scheduler": {"last_epoch": 3}})
    monkeypatch.setattr(sft.checkpoint, "find_latest_committed", latest)
    monkeypatch.setattr(sft.checkpoint, "read_adapter_contract", contract)
    monkeypatch.setattr(sft.checkpoint, "load_checkpoint", load)
    monkeypatch.setattr(sft, "restore_rng", restore)
    trainer._resume()
    latest.assert_called_once_with(str(tmp_path), task)
    contract.assert_called_once_with(str(tmp_path), task, 3)
    assert load.call_args.args[1] == task
    assert load.call_args.kwargs["adapter_only"] is True
    restore.assert_called_once_with(tmp_path / task / sft.checkpoint.iter_dir_name(3), 0)
    assert trainer.step == 3


def test_t2i_resume_rejects_changed_data_or_optimizer_contract_before_load(monkeypatch, tmp_path):
    trainer = sft.FSDPSFTTrainer.__new__(sft.FSDPSFTTrainer)
    trainer.task = "t2i_sft"
    trainer.args = SimpleNamespace(load=str(tmp_path))
    trainer.contract = {"config": {"dataset_size": 8}}
    load = Mock()
    monkeypatch.setattr(sft.checkpoint, "find_latest_committed", lambda *_args: 3)
    monkeypatch.setattr(sft.checkpoint, "read_adapter_contract", lambda *_args: {"config": {"dataset_size": 16}})
    monkeypatch.setattr(sft.checkpoint, "load_checkpoint", load)
    with pytest.raises(ValueError, match="contract differs"):
        trainer._resume()
    load.assert_not_called()

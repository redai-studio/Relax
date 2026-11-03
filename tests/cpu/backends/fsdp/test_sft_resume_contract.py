# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The cache extension must resume Qwen online-only checkpoints safely."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.backends.fsdp import sft


def _trainer(monkeypatch, tmp_path):
    saved = {
        "family": "qwen_image",
        "generation_task": "t2i",
        "config": {"dataset_size": 8, "text_encoder_placement": "sharded"},
        "metadata_sha256": "original-data",
        "optimizer": {"lr": 0.0001},
    }
    trainer = sft.FSDPSFTTrainer.__new__(sft.FSDPSFTTrainer)
    trainer.task, trainer.rank, trainer.step = "t2i_sft", 0, 0
    trainer.args = SimpleNamespace(load=str(tmp_path), _start_rollout_id_explicit=True, start_rollout_id=3)
    trainer.contract = deepcopy(saved)
    trainer.contract["config"]["cache_vae_posterior"] = False
    trainer.actor = SimpleNamespace(model=object(), optimizer=object(), _trainable_mode="lora")
    trainer.scheduler = Mock()
    load = Mock(return_value={"completed_steps": 3, "scheduler": {"last_epoch": 3}})
    restore = Mock()
    monkeypatch.setattr(sft.checkpoint, "find_latest_committed", lambda *_args: 3)
    monkeypatch.setattr(sft.checkpoint, "read_adapter_contract", lambda *_args: saved)
    monkeypatch.setattr(sft.checkpoint, "load_checkpoint", load)
    monkeypatch.setattr(sft, "restore_rng", restore)
    return trainer, saved, load, restore


@pytest.mark.parametrize("reverse", [False, True])
def test_qwen_resume_accepts_missing_and_false_cache_option_without_mutation(monkeypatch, tmp_path, reverse):
    trainer, saved, load, restore = _trainer(monkeypatch, tmp_path)
    if reverse:
        saved["config"]["cache_vae_posterior"] = False
        del trainer.contract["config"]["cache_vae_posterior"]
    before = deepcopy((saved, trainer.contract))

    trainer._resume()

    assert (saved, trainer.contract) == before
    assert trainer.step == 3
    load.assert_called_once_with(
        str(tmp_path), "t2i_sft", 3, trainer.actor.model, trainer.actor.optimizer, adapter_only=True
    )
    trainer.scheduler.load_state_dict.assert_called_once_with({"last_epoch": 3})
    restore.assert_called_once_with(tmp_path / "t2i_sft" / sft.checkpoint.iter_dir_name(3), 0)


@pytest.mark.parametrize(
    "change", ["enabled", "previously_enabled", "null", "zero", "optimizer", "data", "other_family", "other_task"]
)
def test_qwen_resume_rejects_inequivalent_contract_before_loading(monkeypatch, tmp_path, change):
    trainer, saved, load, restore = _trainer(monkeypatch, tmp_path)
    if change == "enabled":
        trainer.contract["config"]["cache_vae_posterior"] = True
    elif change == "previously_enabled":
        saved["config"]["cache_vae_posterior"] = True
    elif change == "null":
        trainer.contract["config"]["cache_vae_posterior"] = None
    elif change == "zero":
        trainer.contract["config"]["cache_vae_posterior"] = 0
    elif change == "optimizer":
        trainer.contract["optimizer"]["lr"] *= 2
    elif change == "data":
        trainer.contract["metadata_sha256"] = "different-data"
    elif change == "other_family":
        saved["family"] = trainer.contract["family"] = "other"
    else:
        saved["generation_task"] = trainer.contract["generation_task"] = "i2i"
        trainer.task = "i2i_sft"
    before = deepcopy((saved, trainer.contract))

    with pytest.raises(ValueError, match="contract differs"):
        trainer._resume()

    assert (saved, trainer.contract) == before
    assert trainer.step == 0
    load.assert_not_called()
    restore.assert_not_called()

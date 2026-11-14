# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Overlap contracts: published backing, reader completion, and safe
rejection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.megatron.weight_update.utils import (
    validate_actor_weight_retention,
)
from relax.backends.sglang.weight_loader import fence_native_weights


def test_retention_checks_contracts_and_integration_gaps_not_training_recipe():
    args = dict(
        colocate=True,
        offload_train=True,
        offload_rollout=True,
        enable_weights_backuper=True,
        megatron_to_hf_mode="bridge",
        resource={"actor": (1, 4), "rollout": (1, 4)},
        rollout_num_gpus_per_engine=2,
    )

    def check(**changes):
        return validate_actor_weight_retention(
            SimpleNamespace(**(args | changes)), role="actor", with_ref=False, with_teacher=False, requested=True
        )

    assert check(optimizer="adam", decoupled_weight_decay=True) is None
    assert check(optimizer="adam", decoupled_weight_decay=False) is None
    for optimizer in ("sgd", "muon", "lion", "soap", "adaptive_muon", "dist_muon", "adamw", "unknown"):
        assert "optimizer=adam" in check(optimizer=optimizer)

    # Representative unvalidated settings are not a recipe whitelist.
    assert check(bf16=False, pipeline_model_parallel_size=2, update_weights_interval=2, use_critic=True) is None
    assert check(lora_rank=8, lora_merge_mode=True, use_opd=True, opd_type="sglang") is None
    assert "equal Actor and rollout" in check(resource={"actor": (1, 4), "rollout": (1, 2)})
    assert "whole rollout engines" in check(rollout_num_gpus_per_engine=3)
    assert "offload_train" in check(offload_train=False)
    assert "single actor snapshot" in check(keep_old_actor=True)
    assert "snapshot keys" in check(megatron_to_hf_mode="raw")
    assert "resume generation" in check(lora_rank=8, lora_adapter_mode=True)
    assert "teacher onload" in check(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        resource=args["resource"] | {"teacher": (1, 1)},
    )


def test_native_receiver_fence_rejects_payload_and_waits_all_streams(monkeypatch):
    from relax.backends.megatron.weight_update import train_offload

    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    with pytest.raises(ValueError):
        fence_native_weights(None, [("weight", torch.ones(1))])
    sync.assert_not_called()
    fence_native_weights(None, [])
    sync.assert_called_once_with()
    # The sender must also await D2H before releasing its actual backing storage.
    flat = torch.ones(4)
    snapshot = {"weight": flat.clone()}
    offloader = train_offload._SelectiveOffloadStrategy(
        [SimpleNamespace(buffers=[SimpleNamespace(param_data=flat)])], None, SimpleNamespace()
    )
    offloader._actor_weights_pending_release, offloader._cpu_weight_snapshot = True, snapshot

    def wait_d2h():
        assert flat.untyped_storage().nbytes() > 0

    offloader._cpu_backup_done = Mock(synchronize=Mock(side_effect=wait_d2h))
    monkeypatch.setattr(train_offload.device_module, "empty_cache", lambda: None)
    assert offloader.finish_weight_sync() is snapshot
    offloader._cpu_backup_done.synchronize.assert_called_once_with()
    assert flat.untyped_storage().nbytes() == 0 and not offloader._actor_weights_pending_release


def test_bridge_export_waits_only_for_its_preparation_stream(monkeypatch):
    pytest.importorskip("megatron.core", reason="Bridge export requires Megatron Core", exc_type=ImportError)
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge

    stream = Mock()
    monkeypatch.setattr(bridge.device_module, "current_stream", Mock(return_value=stream))
    monkeypatch.setattr(bridge.device_module, "synchronize", Mock(side_effect=AssertionError("waited for D2H backup")))
    info = SimpleNamespace(name="model.weight", src_rank=0, shape=(2, 2), dtype=torch.float32, attrs={})
    source = SimpleNamespace(to=Mock(return_value=torch.ones(2, 2)))
    params = bridge._load_to_gpu([info], {"weight": source}, {"model.weight": "weight"}, torch.device("cuda"), rank=0)
    assert torch.equal(params[0], torch.ones(2, 2))
    stream.synchronize.assert_called_once_with()
    stream.reset_mock()
    bridge._load_to_gpu([info], {"weight": torch.ones(2, 2)}, {"model.weight": "weight"}, torch.device("cpu"), rank=0)
    stream.synchronize.assert_not_called()


def test_actor_snapshot_is_unreadable_until_complete_backing_is_adopted():
    from relax.utils.training.tensor_backper import _TensorBackuperNormal

    tensor = torch.arange(4).float()
    backup = _TensorBackuperNormal(lambda: [("weight", tensor)])
    backup._backups["actor"] = {"weight": tensor.clone()}
    backup.mark_actor_backup_pending()
    with pytest.raises(RuntimeError, match="awaiting completion"):
        backup.get("actor")
    with pytest.raises(ValueError, match="shape/dtype"):
        backup.reuse_actor_cpu_backup({"weight": torch.zeros(5)})
    with pytest.raises(RuntimeError, match="awaiting completion"):
        backup.get("actor")
    backing = {"weight": tensor.clone() + 1}
    backup.reuse_actor_cpu_backup(backing)
    assert backup.get("actor") is backing
    assert torch.equal(backup.get("actor")["weight"], tensor + 1)


def test_optimizer_offload_preserves_native_policy_and_rejects_storage_aliases(monkeypatch):
    from relax.backends.megatron.weight_update import train_offload
    from relax.backends.megatron.weight_update.train_offload import _SelectiveOffloadStrategy

    master = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([master])
    wrapped = SimpleNamespace(optimizer=optimizer, shard_fp32_from_float16_groups=[[master]])
    grad = torch.zeros(4)
    buffer = SimpleNamespace(param_data=torch.ones(4, dtype=torch.bfloat16), grad_data=grad)
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    offloader = _SelectiveOffloadStrategy([model], wrapped, SimpleNamespace(optimizer_cpu_offload=False))
    monkeypatch.setattr(offloader, "_iter_optimizers", lambda: iter([wrapped]))
    assert offloader.validate_optimizer_storage_aliases() is None  # Lazy state is valid.
    master.grad = torch.ones_like(master)
    optimizer.step()
    assert offloader.validate_optimizer_storage_aliases() is None
    optimizer.state[master]["exp_avg"] = grad
    assert "aliases" in offloader.validate_optimizer_storage_aliases()
    assert grad.untyped_storage().nbytes() == 16
    optimizer.state[master]["exp_avg"] = torch.zeros_like(master)
    extra_state = optimizer.state[master]["extra_state"] = torch.tensor(7)
    assert offloader.validate_optimizer_storage_aliases() is None
    offloader._offload_training_state(retain_parameters=True)
    assert grad.untyped_storage().nbytes() == 0
    assert buffer.param_data.untyped_storage().nbytes() > 0
    assert optimizer.state[master]["extra_state"] is extra_state
    # Native master groups may hold a tensor directly instead of a list.
    wrapped.shard_fp32_from_float16_groups = [buffer.param_data]
    assert "aliases" in offloader.validate_optimizer_storage_aliases()
    wrapped.shard_fp32_from_float16_groups = [[master]]
    # HDO owns its state placement; the ordinary offloader must not touch it.
    offloader._skip_optimizer = True
    assert offloader.validate_optimizer_storage_aliases() is None
    buffer.grad_data = torch.zeros(4)
    offloader._offload_training_state(retain_parameters=True)
    assert buffer.grad_data.untyped_storage().nbytes() == 0

    # Ordinary offload must free one buffer before allocating the next CPU copy.
    model.buffers = [
        SimpleNamespace(param_data=torch.full((4,), value), grad_data=torch.ones(4)) for value in (1.0, 2.0)
    ]
    expected = [buffer.param_data.clone() for buffer in model.buffers]
    allocate, copied = torch.empty_like, []

    def allocate_cpu(tensor, **kwargs):
        for previous in model.buffers[: len(copied)]:
            assert previous.param_data.untyped_storage().nbytes() == 0
            assert previous.grad_data.untyped_storage().nbytes() == 0
        copied.append(tensor)
        return allocate(tensor, **kwargs)

    monkeypatch.setattr(train_offload, "_PIN_MEMORY", False)
    monkeypatch.setattr(torch, "empty_like", allocate_cpu)
    offloader._offload_training_state()
    assert len(copied) == 2
    for buffer, value in zip(model.buffers, expected, strict=True):
        assert torch.equal(buffer.param_data.cpu_data, value)
        assert buffer.param_data.untyped_storage().nbytes() == buffer.grad_data.untyped_storage().nbytes() == 0

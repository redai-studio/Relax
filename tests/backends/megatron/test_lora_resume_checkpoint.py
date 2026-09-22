# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.training.checkpointing")

from relax.backends.megatron import checkpoint  # noqa: E402


class _AdapterModel(torch.nn.Module):
    def __init__(self, *, train_base: bool = False):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.adapter = torch.nn.Module()
        self.layer.adapter.linear_in = torch.nn.Linear(2, 2, bias=False)
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(train_base)


def _runtime_args():
    return SimpleNamespace(
        hf_checkpoint="/models/base",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["linear_in"],
        lora_scope="all",
        lora_merge_mode=True,
        lora_adapter_mode=False,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        world_size=4,
    )


def test_filter_lora_checkpoint_keeps_resume_state_and_marks_args():
    marker = {"format_version": 1}
    optimizer = object()
    scheduler = object()
    rng = object()
    state = {
        "args": SimpleNamespace(),
        "model": {
            "decoder.base.weight": object(),
            "decoder.adapter.linear_in.weight": object(),
            "decoder.adapter.linear_out.weight": object(),
        },
        "model1": {
            "vision.base.weight": object(),
            "vision.adapter.linear_in.weight": object(),
        },
        "optimizer": optimizer,
        "opt_param_scheduler": scheduler,
        "rng_state": rng,
        "iteration": 7,
    }

    filtered = checkpoint._filter_lora_checkpoint_state_dict(state, metadata=marker)

    assert tuple(filtered["model"]) == (
        "decoder.adapter.linear_in.weight",
        "decoder.adapter.linear_out.weight",
    )
    assert tuple(filtered["model1"]) == ("vision.adapter.linear_in.weight",)
    assert filtered["optimizer"] is optimizer
    assert filtered["opt_param_scheduler"] is scheduler
    assert filtered["rng_state"] is rng
    assert filtered["iteration"] == 7
    assert getattr(filtered["args"], checkpoint._LORA_CHECKPOINT_METADATA_ATTR) == marker


def test_lora_checkpoint_rejects_non_lora_trainable_parameters():
    with pytest.raises(RuntimeError, match="non-LoRA trainable parameter"):
        checkpoint._global_trainable_lora_parameter_names([_AdapterModel(train_base=True)])


def test_lora_checkpoint_accepts_only_trainable_adapters():
    names = checkpoint._global_trainable_lora_parameter_names([_AdapterModel()])

    assert names == ("layer.adapter.linear_in.weight",)


def test_save_lora_only_filters_model_after_optimizer_state_is_built(monkeypatch):
    import megatron.training.checkpointing as mcore_checkpointing

    seen = {}

    def fake_generate_state_dict(*_args, **_kwargs):
        seen["optimizer_saw_base"] = True
        return {
            "args": SimpleNamespace(),
            "model": {
                "layer.to_wrap.weight": object(),
                "layer.adapter.linear_in.weight": object(),
            },
            "optimizer": {"state": "kept"},
            "rng_state": "kept",
        }

    def fake_save(*_args, **_kwargs):
        seen["state"] = mcore_checkpointing.generate_state_dict()

    monkeypatch.setattr(mcore_checkpointing, "generate_state_dict", fake_generate_state_dict)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", fake_save)
    monkeypatch.setattr(checkpoint, "get_args", _runtime_args)
    monkeypatch.setattr(
        checkpoint,
        "_global_trainable_lora_parameter_names",
        lambda _model: ("layer.adapter.linear_in.weight",),
    )

    checkpoint.save_checkpoint(3, [object()], None, None, lora_only=True)

    assert seen["optimizer_saw_base"] is True
    assert tuple(seen["state"]["model"]) == ("layer.adapter.linear_in.weight",)
    assert seen["state"]["optimizer"] == {"state": "kept"}
    assert seen["state"]["rng_state"] == "kept"
    metadata = getattr(seen["state"]["args"], checkpoint._LORA_CHECKPOINT_METADATA_ATTR)
    assert metadata["base_hf_checkpoint"] == {"config_sha256": None, "index_sha256": None, "path": "/models/base"}


def test_lora_model_load_allows_missing_base_but_requires_adapter():
    model = _AdapterModel()
    adapter_state = {"layer.adapter.linear_in.weight": torch.ones_like(model.layer.adapter.linear_in.weight)}

    with checkpoint._validate_lora_model_state_load([model]):
        model.load_state_dict(adapter_state, strict=False)

    with checkpoint._validate_lora_model_state_load([model]):
        with pytest.raises(RuntimeError, match="missing_adapters"):
            model.load_state_dict({}, strict=False)


def test_hf_checkpoint_identity_detects_shard_replacement(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    checkpoint._hf_checkpoint_identity.cache_clear()
    before = checkpoint._hf_checkpoint_identity(str(tmp_path))

    shard.write_bytes(b"replacement")
    checkpoint._hf_checkpoint_identity.cache_clear()
    after = checkpoint._hf_checkpoint_identity(str(tmp_path))

    assert before["shard_stat_sha256"] != after["shard_stat_sha256"]


def test_hf_checkpoint_identity_covers_pytorch_bin(tmp_path):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    checkpoint._hf_checkpoint_identity.cache_clear()

    identity = checkpoint._hf_checkpoint_identity(str(tmp_path))

    assert identity["shard_count"] == "1"
    assert identity["shard_stat_sha256"]


def test_lightweight_resume_loads_hf_base_before_lora_dcp(monkeypatch, tmp_path):
    resume_path = tmp_path / "checkpoint"
    resume_path.mkdir()
    (resume_path / "latest_checkpointed_iteration.txt").write_text("7")
    args = SimpleNamespace(
        load=str(resume_path), hf_checkpoint="/models/base", dist_ckpt_strictness="assume_ok_unexpected"
    )
    marker = {"format_version": 1}
    calls = []

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_alias_renamed_transfer_queue_enum", lambda: None)
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", lambda _path: marker)
    monkeypatch.setattr(checkpoint, "_validate_lora_model_state_load", lambda _model: nullcontext())
    monkeypatch.setattr(
        checkpoint,
        "_validate_lora_checkpoint_metadata",
        lambda current_args, model, metadata: calls.append(("validate", current_args, model, metadata)),
    )
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: calls.append(("hf",)))
    monkeypatch.setattr(
        checkpoint,
        "_load_checkpoint_megatron",
        lambda **kwargs: calls.append(("dcp", kwargs["strict"])) or (7, 0),
    )

    assert checkpoint.load_checkpoint("model", "optimizer", "scheduler", {}, False) == (7, 0)
    assert [call[0] for call in calls] == ["validate", "hf", "dcp"]
    assert calls[-1] == ("dcp", True)
    assert args.dist_ckpt_strictness == "assume_ok_unexpected"


def test_lightweight_resume_preserves_hybrid_optimizer_step(monkeypatch):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    captured = {}
    parameter = torch.nn.Parameter(torch.zeros(1))
    sub_optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
    )
    optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
        sub_optimizers=[sub_optimizer],
    )

    def fake_load_state_dict(_optimizer, state_dict):
        captured.update(state_dict)

    monkeypatch.setattr(HybridDeviceOptimizer, "load_state_dict", fake_load_state_dict)
    state_dict = {
        "param_groups": [{"params": [0], "step": 7}],
        "state": {
            0: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        },
    }

    with checkpoint._preserve_hybrid_optimizer_steps_on_load():
        HybridDeviceOptimizer.load_state_dict(optimizer, state_dict)

    assert captured["state"][0]["step"].item() == 7
    assert optimizer.state[parameter]["step"].item() == 7
    assert optimizer.param_groups[0]["step"] == 7
    assert sub_optimizer.state[parameter]["step"].item() == 7
    assert sub_optimizer.param_groups[0]["step"] == 7
    assert HybridDeviceOptimizer.load_state_dict is fake_load_state_dict


def test_lightweight_save_syncs_hybrid_optimizer_checkpoint_step():
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    optimizer = object.__new__(HybridDeviceOptimizer)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer.state = {parameter: {"step": torch.tensor(7.0)}}
    optimizer.param_groups = [{"params": [parameter], "step": 1}]
    optimizer.cpu_optimizers = [SimpleNamespace(param_groups=[{"params": [parameter], "step": 1}])]
    optimizer.gpu_optimizer = None
    outer = SimpleNamespace(optimizer=optimizer)

    checkpoint._sync_hybrid_optimizer_checkpoint_steps(outer)

    assert optimizer.param_groups[0]["step"] == 7
    assert optimizer.cpu_optimizers[0].param_groups[0]["step"] == 7


def test_lightweight_resume_restores_step_after_distributed_parameter_load(monkeypatch):
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    parameter = torch.nn.Parameter(torch.zeros(1))
    sub_optimizer = SimpleNamespace(
        state={parameter: {"step": torch.tensor(1.0)}},
        param_groups=[{"params": [parameter], "step": 1}],
    )
    hybrid_optimizer = object.__new__(HybridDeviceOptimizer)
    hybrid_optimizer.state = {parameter: {"step": torch.tensor(1.0)}}
    hybrid_optimizer.param_groups = [{"params": [parameter], "step": 1}]
    hybrid_optimizer.cpu_optimizers = [sub_optimizer]
    hybrid_optimizer.gpu_optimizer = None
    distributed_optimizer = object.__new__(DistributedOptimizer)
    distributed_optimizer.optimizer = hybrid_optimizer

    def fake_distributed_load(_optimizer, _state_dict):
        # Model the later parameter-shard phase leaving the bootstrap step.
        hybrid_optimizer.state[parameter]["step"] = torch.tensor(1.0)

    monkeypatch.setattr(DistributedOptimizer, "load_state_dict", fake_distributed_load)
    with checkpoint._preserve_hybrid_optimizer_steps_on_load():
        DistributedOptimizer.load_state_dict(
            distributed_optimizer,
            {"optimizer": {"param_groups": [{"step": 7}]}},
        )

    assert hybrid_optimizer.state[parameter]["step"].item() == 7
    assert hybrid_optimizer.param_groups[0]["step"] == 7
    assert sub_optimizer.state[parameter]["step"].item() == 7
    assert sub_optimizer.param_groups[0]["step"] == 7
    assert DistributedOptimizer.load_state_dict is fake_distributed_load

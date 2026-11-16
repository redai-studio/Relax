# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Target isolation and post-freeze guards without a Megatron runtime."""

import re
import sys
from argparse import Namespace
from types import ModuleType

import pytest
import torch

from relax.models.common.lora import finalize_lora_models, resolve_lora_targets
from relax.models.qwen3_8.lora import HC_TARGETS, PLE_TARGETS


class _HC(torch.nn.Module):
    def __init__(self, mixer: bool = False) -> None:
        super().__init__()
        for name in HC_TARGETS:
            setattr(self, name, None if mixer and name == "block_inject_weight" else torch.nn.Linear(2, 2, bias=False))


class _PLE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in PLE_TARGETS:
            setattr(self, name, torch.nn.Linear(2, 2, bias=False))
        self.norm = torch.nn.Parameter(torch.ones(2))


class _Adapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = torch.nn.Linear(2, 2, bias=False)
        self.linear_out = torch.nn.Linear(2, 2, bias=False)


@pytest.fixture
def case(monkeypatch):
    matcher = ModuleType("megatron.bridge.peft.utils")
    matcher.wildcard_match = lambda pattern, name: re.fullmatch(pattern.replace("*", "(.*)"), name) is not None
    monkeypatch.setitem(sys.modules, matcher.__name__, matcher)
    for path, name, cls in (
        ("megatron.core.transformer.hyper_connection", "GatedResidualHyperConnection", _HC),
        ("megatron.core.transformer.per_layer_embedding", "PerLayerEmbedding", _PLE),
        ("megatron.bridge.peft.lora_layers", "LinearAdapter", _Adapter),
    ):
        module = ModuleType(path)
        setattr(module, name, cls)
        monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    model = torch.nn.Module()
    model.config = Namespace(mhc_variant="gated_residual")
    model.attention_hc = _HC()
    model.final_layernorm = _HC(mixer=True)
    model.mtp = torch.nn.Module()
    model.mtp.hc = _HC()
    model.ple = _PLE()
    model.vision_model = torch.nn.Module()
    model.vision_model.key_proj = torch.nn.Linear(2, 2)
    args = Namespace(lora_target_modules=[*HC_TARGETS, *PLE_TARGETS], lora_dropout=0.0, lora_rank=2)
    return model, args


def _inject(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    for name in model._relax_qwen38_lora_targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        wrapper = torch.nn.Module()
        wrapper.to_wrap = getattr(parent, leaf)
        wrapper.adapter = _Adapter()
        setattr(parent, leaf, wrapper)


def test_qwen38_lora_targets_isolate_owners_and_keep_export_names(case):
    model, args = case
    args.lora_dropout = 0.25
    original = list(args.lora_target_modules)
    result = resolve_lora_targets(model, ["linear_qkv", *original], args)

    assert result[0] == "linear_qkv"
    assert len(result) == 11  # Three HC + two mixer + three MTP HC + two PLE.
    assert "vision_model.key_proj" not in result
    assert "final_layernorm.block_inject_weight" not in result
    assert "mtp.hc.block_inject_weight" in result
    assert args.lora_target_modules == original


def test_qwen38_lora_empty_pipeline_chunk_does_not_select_all_linears(case):
    model, args = case
    empty = torch.nn.Module()
    empty.config = model.config
    assert resolve_lora_targets(empty, args.lora_target_modules, args) == ["__relax_lora_no_match__"]
    assert empty._relax_qwen38_lora_targets == ()
    with pytest.raises(ValueError, match="no Qwen3.8 HC/PLE projection matched"):
        finalize_lora_models([empty], args)


def test_qwen38_lora_does_not_change_other_models(case, monkeypatch):
    model, args = case
    model.config.mhc_variant = "other"
    monkeypatch.setitem(sys.modules, "relax.models.qwen3_8.lora", None)
    assert resolve_lora_targets(model, args.lora_target_modules, args) == args.lora_target_modules
    finalize_lora_models([model], args)
    assert not hasattr(model, "_relax_qwen38_lora_targets")

    del model.config
    assert resolve_lora_targets(model, args.lora_target_modules, args) == args.lora_target_modules
    finalize_lora_models([model], args)


@pytest.mark.parametrize(
    "setting,value,message",
    [
        ("lora_scope", "vision", "scope"),
        ("lora_adapter_mode", True, "merge mode"),
    ],
)
def test_qwen38_lora_rejects_unsupported_modes(case, setting, value, message):
    model, args = case
    setattr(args, setting, value)
    with pytest.raises(ValueError, match=message):
        resolve_lora_targets(model, args.lora_target_modules, args)


def test_qwen38_lora_rejects_streamed_ple_only_when_requested(case, monkeypatch):
    model, args = case
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    resolve_lora_targets(model, list(HC_TARGETS), args)
    with pytest.raises(ValueError, match="PLE_STREAMED=0"):
        resolve_lora_targets(model, args.lora_target_modules, args)


def test_qwen38_lora_validates_after_freeze_and_sync_inheritance(case):
    model, args = case
    resolve_lora_targets(model, args.lora_target_modules, args)
    _inject(model)
    finalize_lora_models([model], args)
    projection = model.ple.key_proj
    projection.adapter.linear_in.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="requested adapter was frozen"):
        finalize_lora_models([model], args)
    projection.adapter.linear_in.weight.requires_grad_(True)
    projection.to_wrap.weight.sequence_parallel = True
    with pytest.raises(ValueError, match="TP SUM"):
        finalize_lora_models([model], args)
    for parameter in projection.adapter.parameters():
        parameter.sequence_parallel = True
    finalize_lora_models([model], args)
    model.ple.norm.requires_grad_(True)
    with pytest.raises(ValueError, match="base parameters must stay frozen"):
        finalize_lora_models([model], args)


def test_qwen38_lora_validates_targets_across_all_virtual_pipeline_chunks(case):
    model, args = case
    ple_chunk = torch.nn.Module()
    ple_chunk.config = model.config
    ple_chunk.ple = model.ple
    del model.ple
    empty_chunk = torch.nn.Module()
    empty_chunk.config = model.config
    chunks = [empty_chunk, model, ple_chunk]
    for chunk in chunks:
        resolve_lora_targets(chunk, args.lora_target_modules, args)
        _inject(chunk)

    finalize_lora_models(chunks, args)
    ple_chunk.ple.key_proj.adapter.linear_in.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="requested adapter was frozen"):
        finalize_lora_models(chunks, args)


def test_qwen38_lora_rejects_filtered_and_globally_missing_targets(case):
    model, args = case
    resolve_lora_targets(model, args.lora_target_modules, args)
    with pytest.raises(ValueError, match="expected a plain LinearAdapter"):
        finalize_lora_models([model], args)
    del model.ple
    resolve_lora_targets(model, args.lora_target_modules, args)
    _inject(model)
    with pytest.raises(ValueError, match="no Qwen3.8 HC/PLE projection matched"):
        finalize_lora_models([model], args)


@pytest.mark.parametrize("target", ["ple.key_proj", "*.input_mix_weight_*", "*.key_proj", "*ple*key_proj"])
def test_qwen38_lora_paths_cannot_bypass_validation(case, target):
    model, args = case
    args.lora_target_modules = [target]
    args.lora_dropout = 0.05
    resolve_lora_targets(model, args.lora_target_modules, args)
    with pytest.raises(ValueError, match="use canonical HC/PLE target names"):
        finalize_lora_models([model], args)

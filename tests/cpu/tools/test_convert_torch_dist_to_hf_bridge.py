# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
import weakref
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file


_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/tools/convert_torch_dist_to_hf_bridge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_torch_dist_to_hf_bridge_under_test", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load_module()


def _metadata(entries):
    return SimpleNamespace(state_dict_metadata={key: SimpleNamespace(size=size) for key, size in entries.items()})


def test_build_lora_checkpoint_spec_detects_dense_and_per_expert_adapters():
    metadata = _metadata(
        {
            "language_model.decoder.layers.0.in_proj.adapter.linear_in.weight": (32, 4096),
            "language_model.decoder.layers.0.in_proj.adapter.linear_out.weight": (8192, 32),
            "language_model.decoder.layers.0.mlp.experts.experts.linear_fc1.adapter.linear_in.weight": (
                512,
                32,
                4096,
            ),
            "language_model.decoder.layers.0.mlp.experts.experts.linear_fc1.adapter.linear_out.weight": (
                512,
                2048,
                32,
            ),
            "language_model.decoder.layers.0.weight": (4096, 4096),
        }
    )

    spec = converter._build_lora_checkpoint_spec(metadata)

    assert spec["rank"] == 32
    assert spec["share_expert_adapters"] is False
    assert spec["target_modules"] == [
        "language_model.decoder.layers.0.in_proj",
        "language_model.decoder.layers.0.mlp.experts.linear_fc1",
    ]
    assert len(spec["adapter_keys"]) == 4


def test_build_lora_checkpoint_spec_expands_collapsed_vision_layer_key():
    metadata = _metadata(
        {
            "vision_model.decoder.layers.self_attention.linear_qkv.adapter.linear_in.weight": (24, 32, 4096),
            "vision_model.decoder.layers.self_attention.linear_qkv.adapter.linear_out.weight": (24, 4096, 32),
            "vision_model.decoder.deepstack_merger_list.linear_fc1.adapter.linear_in.weight": (3, 32, 4096),
            "vision_model.decoder.deepstack_merger_list.linear_fc1.adapter.linear_out.weight": (3, 4096, 32),
            "language_model.decoder.layers.0.mlp.experts.experts.linear_fc1.adapter.linear_in.weight": (32, 4096),
            "language_model.decoder.layers.0.mlp.experts.experts.linear_fc1.adapter.linear_out.weight": (2048, 32),
            "vision_model.merger.linear_fc1.adapter.linear_in.weight": (32, 4096),
            "vision_model.merger.linear_fc1.adapter.linear_out.weight": (4096, 32),
        }
    )

    spec = converter._build_lora_checkpoint_spec(metadata)

    assert spec["target_modules"] == [
        "language_model.decoder.layers.0.mlp.experts.linear_fc1",
        "vision_model.decoder.deepstack_merger_list.*.linear_fc1",
        "vision_model.decoder.layers.*.self_attention.linear_qkv",
        "vision_model.merger.linear_fc1",
    ]
    assert spec["share_expert_adapters"] is True


def test_build_lora_checkpoint_spec_returns_none_for_full_checkpoint():
    assert converter._build_lora_checkpoint_spec(_metadata({"decoder.layers.0.weight": (4, 4)})) is None


def test_build_lora_checkpoint_spec_rejects_half_pair():
    metadata = _metadata({"decoder.layers.0.in_proj.adapter.linear_in.weight": (32, 4096)})
    with pytest.raises(ValueError, match="incomplete LoRA adapter pair"):
        converter._build_lora_checkpoint_spec(metadata)


def test_assert_adapter_coverage_rejects_silent_adapter_drop():
    expected = {
        "decoder.layers.0.in_proj.adapter.linear_in.weight",
        "decoder.layers.0.in_proj.adapter.linear_out.weight",
    }
    with pytest.raises(ValueError, match="missing=1"):
        converter._assert_adapter_coverage(expected, {"decoder.layers.0.in_proj.adapter.linear_in.weight"})


def test_adapter_coverage_uses_persistent_sharded_tensor_key():
    persistent_key = "decoder.layers.0.mlp.experts.experts.linear_fc1.adapter.linear_in.weight"
    chunk = SimpleNamespace(
        sharded_state_dict=lambda: {
            "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight": SimpleNamespace(key=persistent_key),
            "decoder.layers.0.mlp.experts.linear_fc1.to_wrap.weight": SimpleNamespace(key="ignored"),
        }
    )
    assert converter._adapter_checkpoint_keys_from_model([chunk]) == {persistent_key}


def test_checkpoint_precision_overrides_fp32_provider_default():
    provider = SimpleNamespace(bf16=False, fp16=False, params_dtype=torch.float32)
    converter._apply_checkpoint_precision(provider, SimpleNamespace(bf16=True, fp16=False))
    assert provider.bf16 is True
    assert provider.fp16 is False
    assert provider.params_dtype == torch.bfloat16
    converter._apply_checkpoint_precision(provider, SimpleNamespace(params_dtype=torch.float32))
    assert provider.bf16 is False
    assert provider.params_dtype == torch.float32


def test_export_dtype_cast_keeps_ple_shards_lazy_and_integer_buffers_intact():
    loaded = []
    live = []

    class Shards(Mapping):
        def __iter__(self):
            return iter(("shard_0", "shard_1", "offsets"))

        def __len__(self):
            return 3

        def __getitem__(self, name):
            assert not any(ref() is not None for ref in live)
            loaded.append(name)
            tensor = torch.arange(8, dtype=torch.int64 if name == "offsets" else torch.float32)
            live.append(weakref.ref(tensor))
            return tensor

    source = Shards()
    assert converter._cast_export_weight_dtype(source, None) is source
    converted = converter._cast_export_weight_dtype(source, torch.bfloat16)
    assert loaded == []
    for name, tensor in converted.items():
        assert tensor.dtype == (torch.int64 if name == "offsets" else torch.bfloat16)
        torch.testing.assert_close(tensor.float(), torch.arange(8, dtype=torch.float32))
        del tensor
    assert loaded == ["shard_0", "shard_1", "offsets"]


def test_bf16_writer_streams_out_of_order_views_and_filters_reference_keys(tmp_path, monkeypatch):
    import safetensors.torch

    from relax.utils.quant_cast.fp8_checkpoint import StreamingHFWriter

    save_file = safetensors.torch.save_file
    written = []

    def save_compact_shard(tensors, filename, metadata=None):
        sizes = [tensor.numel() * tensor.element_size() for tensor in tensors.values()]
        assert sum(sizes) <= 12
        for tensor, size in zip(tensors.values(), sizes):
            assert tensor.untyped_storage().nbytes() == size
            assert not tensor.requires_grad
        written.extend(tensors)
        save_file(tensors, filename, metadata=metadata)

    monkeypatch.setattr(safetensors.torch, "save_file", save_compact_shard)
    names = ["ple.shard_0", "layer.weight", "ple.shard_1"]
    expected = {name: torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + i for i, name in enumerate(names)}
    refs = []

    def tensors():
        for name in names:
            assert len(written) == len(refs)
            if len(refs) >= 2:
                # Once written, a slice must not keep the preceding full table alive.
                assert refs[-2]() is None
            backing = torch.zeros(100, 3, dtype=torch.bfloat16, requires_grad=True)
            with torch.no_grad():
                backing[:2].copy_(expected[name])
            refs.append(weakref.ref(backing))
            yield name, backing[:2]
            del backing

    source_map = {name: "original.safetensors" for name in [*names, "mtp.weight", "ple.shard_0_scale_inv"]}
    writer = StreamingHFWriter(source_map, max_shard_size=12)
    writer.save_generator(
        tensors(),
        tmp_path,
        ignored_source_key_prefixes=("mtp.",),
        ignored_source_key_suffixes=("_scale_inv",),
    )
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(names)
    assert index["metadata"]["total_size"] == 36
    assert len(set(index["weight_map"].values())) == 3
    for name, filename in index["weight_map"].items():
        torch.testing.assert_close(load_file(tmp_path / filename)[name], expected[name])
    assert all(ref() is None for ref in refs)


def test_bf16_writer_serializes_tied_storage_and_noncontiguous_tensors(tmp_path):
    from relax.utils.quant_cast.fp8_checkpoint import StreamingHFWriter

    weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    tensors = {"embedding": weight, "head": weight, "transposed": weight.T}
    writer = StreamingHFWriter(dict.fromkeys(tensors, "source.safetensors"), max_shard_size=1024)
    writer.save_generator(iter(tensors.items()), tmp_path)
    actual = load_file(tmp_path / "model.safetensors")
    for name, expected in tensors.items():
        torch.testing.assert_close(actual[name], expected)


def test_bf16_writer_rejects_missing_backbone_without_replacing_output(tmp_path):
    from relax.utils.quant_cast.fp8_checkpoint import StreamingHFWriter

    existing = tmp_path / "model.safetensors"
    existing.write_bytes(b"previous export")
    writer = StreamingHFWriter({"a": "source.safetensors", "b": "source.safetensors"}, max_shard_size=8)
    with pytest.raises(KeyError, match="did not yield 1 tensors"):
        writer.save_generator(iter([("a", torch.ones(4, dtype=torch.bfloat16))]), tmp_path)
    assert existing.read_bytes() == b"previous export"
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_export_checkpoint_loads_on_cpu_and_merges_without_autograd(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.training.model_load_save",
        SimpleNamespace(temporary_distributed_context=lambda backend: nullcontext()),
    )
    weight = torch.nn.Parameter(torch.ones(2, 2))
    saved = []

    class Bridge:
        def load_megatron_model(self, path, **kwargs):
            assert kwargs["use_cpu_initialization"] is True
            return weight

        def save_hf_pretrained(self, model, path, **kwargs):
            assert kwargs["merge_adapter_weights"] is True
            assert kwargs["weight_dtype"] == torch.bfloat16
            merged = model + model @ model
            assert not merged.requires_grad
            saved.append(merged)

    converter._export_checkpoint(Bridge(), "checkpoint", tmp_path, strict=True, weight_dtype=torch.bfloat16)
    torch.testing.assert_close(saved[0], torch.full((2, 2), 3.0))

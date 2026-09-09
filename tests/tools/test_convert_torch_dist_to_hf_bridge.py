# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/tools/convert_torch_dist_to_hf_bridge.py"


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
            "vision_model.decoder.layers.self_attention.linear_qkv.adapter.linear_in.weight": (32, 4096),
            "vision_model.decoder.layers.self_attention.linear_qkv.adapter.linear_out.weight": (4096, 32),
            "vision_model.merger.linear_fc1.adapter.linear_in.weight": (32, 4096),
            "vision_model.merger.linear_fc1.adapter.linear_out.weight": (4096, 32),
        }
    )

    spec = converter._build_lora_checkpoint_spec(metadata)

    assert spec["target_modules"] == [
        "vision_model.decoder.layers.*.self_attention.linear_qkv",
        "vision_model.merger.linear_fc1",
    ]


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

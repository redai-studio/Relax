# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression for V4.1's unquantized train/rollout payload contract.

Use real Bridge mappings and conversion, with prebuilt local tasks instead of a
distributed model. TP/EP gathering and SGLang's GPU expert loader are not
covered.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge.models.conversion.model_bridge")

from megatron.bridge.models.conversion.model_bridge import WeightConversionTask  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    GatedMLPMapping,
    ReplicatedMapping,
)

from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter  # noqa: E402
from relax.models.deepseekv41.bridge import DeepSeekV41Bridge  # noqa: E402


def _convert_bf16(megatron_name: str, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Supply an already gathered CPU tensor without mocking conversion or
    quantization."""
    mapping = DeepSeekV41Bridge().mapping_registry().megatron_to_hf_lookup(megatron_name)
    assert isinstance(mapping, (ReplicatedMapping, GatedMLPMapping))
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(weight.clone()))
    module.config = SimpleNamespace()
    task = WeightConversionTask(
        param_name=megatron_name,
        global_param_name=megatron_name,
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
    )
    converter = BridgeConverter(Namespace(vocab_size=128), [module], quantization_config=None)
    # Provider tests cover checkpoint validation; only task construction is bypassed here.
    converter._bridge_task_map = {megatron_name: task}
    converter._hf_alias_checkpoint_keys = frozenset()
    converted = converter.convert("module.module." + megatron_name, weight)
    assert not any("scale" in name or name.endswith("_packed") for name, _ in converted)
    assert all(tensor.dtype == torch.bfloat16 for _, tensor in converted)
    return dict(converted)


def _weight() -> torch.Tensor:
    # Distinct, exactly representable BF16 values expose casts, stale scales, and reordering.
    return ((torch.arange(32 * 64).reshape(32, 64) % 127 - 63) / 64).to(torch.bfloat16)


def test_v41_bf16_sync_engram_wkv_preserves_full_weight() -> None:
    weight = _weight()
    converted = _convert_bf16("decoder.layers.1.v41_engram.linear_wkv.weight", weight)

    assert set(converted) == {"layers.1.engram.wkv.weight"}
    actual = converted["layers.1.engram.wkv.weight"]
    assert actual.shape == weight.shape
    torch.testing.assert_close(actual, weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("megatron_expert", "hf_expert"),
    [
        ("experts.linear_fc1.weight7", "experts.7"),
        ("experts.local_experts.7.linear_fc1.weight", "experts.7"),
        ("shared_experts.linear_fc1.weight", "shared_experts"),
    ],
)
def test_v41_bf16_sync_expert_gate_up_preserves_full_weights(megatron_expert: str, hf_expert: str) -> None:
    gate = _weight()
    up = -gate * 2 + 0.625
    fused = torch.cat((gate, up), dim=0)
    converted = _convert_bf16(f"decoder.layers.2.mlp.{megatron_expert}", fused)

    prefix = f"layers.2.ffn.{hf_expert}."
    assert set(converted) == {prefix + "w1.weight", prefix + "w3.weight"}
    for suffix, expected in (("w1.weight", gate), ("w3.weight", up)):
        actual = converted[prefix + suffix]
        assert actual.shape == expected.shape  # No FP4 packing halves the input width.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v41_bf16_sync_sglang_engram_loader_preserves_values() -> None:
    """Exercise the installed SGLang ReplicatedLinear loader, not an emulated
    copy."""
    linear = pytest.importorskip("sglang.srt.layers.linear")
    weight = _weight()
    converted = _convert_bf16("decoder.layers.1.v41_engram.linear_wkv.weight", weight)
    layer = linear.ReplicatedLinear(64, 32, bias=False, params_dtype=torch.bfloat16, quant_config=None)
    assert set(layer.state_dict()) == {"weight"}, "BF16 rollout must have no stale quantization scale"

    layer.weight_loader(layer.weight, converted["layers.1.engram.wkv.weight"])

    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.shape == weight.shape
    torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)

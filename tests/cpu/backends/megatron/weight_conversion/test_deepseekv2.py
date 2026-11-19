# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Public raw conversion used by ordinary NCCL, including Bridge-trained V2."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron import weight_conversion


def _args(q_lora_rank=None):
    return SimpleNamespace(
        vocab_size=16, hidden_size=8, num_attention_heads=2, num_query_groups=2, kv_channels=4, q_lora_rank=q_lora_rank
    )


def _same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("self_attention.linear_q_proj.weight", "self_attn.q_proj.weight"),
        ("self_attention.linear_kv_down_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),
        ("self_attention.linear_kv_up_proj.layer_norm_weight", "self_attn.kv_a_layernorm.weight"),
        ("self_attention.linear_kv_up_proj.weight", "self_attn.kv_b_proj.weight"),
    ],
)
def test_v2_lite_public_conversion_preserves_mla_weights_without_pairing(source, target):
    value = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    result = weight_conversion.convert_to_hf(
        _args(),
        "deepseekv2config",
        "module.module.decoder.layers.2." + source,
        value,
    )
    assert len(result) == 1
    assert result[0][0] == "model.layers.2." + target
    _same_bits(result[0][1], value)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("mlp", "mlp"),
        ("mlp.shared_experts", "mlp.shared_experts"),
        ("mlp.experts", "mlp.experts.7"),
    ],
)
def test_v2_public_conversion_keeps_dense_shared_and_global_expert_gate_order(source, target):
    expert_suffix = "7" if source == "mlp.experts" else ""
    gate = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + 1
    up = -gate
    result = weight_conversion.convert_to_hf(
        _args(),
        "deepseekv2config",
        f"module.module.decoder.layers.3.{source}.linear_fc1.weight{expert_suffix}",
        torch.cat((gate, up)),
    )
    assert [name for name, _ in result] == [
        f"model.layers.3.{target}.gate_proj.weight",
        f"model.layers.3.{target}.up_proj.weight",
    ]
    _same_bits(result[0][1], gate)
    _same_bits(result[1][1], up)
    down = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    result = weight_conversion.convert_to_hf(
        _args(),
        "deepseekv2config",
        f"module.module.decoder.layers.3.{source}.linear_fc2.weight{expert_suffix}",
        down,
    )
    assert result[0][0] == f"model.layers.3.{target}.down_proj.weight"
    _same_bits(result[0][1], down)


def test_v3_low_rank_q_and_kv_still_pair_before_emission():
    cache = weight_conversion._cached_tensors
    assert not cache
    kv = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    q = -kv
    try:
        assert (
            weight_conversion.convert_to_hf(
                _args(4),
                "deepseekv3config",
                "module.module.decoder.layers.1.self_attention.linear_kv_down_proj.weight",
                kv,
            )
            == []
        )
        result = weight_conversion.convert_to_hf(
            _args(4),
            "deepseekv3config",
            "module.module.decoder.layers.1.self_attention.linear_q_down_proj.weight",
            q,
        )
        assert [name for name, _ in result] == [
            "model.layers.1.self_attn.q_a_proj.weight",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        ]
        _same_bits(result[0][1], q)
        _same_bits(result[1][1], kv)
        assert not cache
    finally:
        cache.clear()


@pytest.mark.parametrize("model_name", ["deepseekv4config", "unknownconfig"])
def test_raw_conversion_rejects_architectures_outside_the_shared_mla_map(model_name):
    with pytest.raises(ValueError, match="Unsupported model"):
        weight_conversion.convert_to_hf(_args(), model_name, "module.module.output_layer.weight", torch.ones(4, 4))

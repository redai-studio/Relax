"""Legacy distributed sync must export the native Bridge vision tower
correctly."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_conversion.qwen3_vl import convert_qwen3vl_to_hf
from relax.backends.megatron.weight_conversion.qwen3_vl_moe import convert_qwen3vlmoe_to_hf


@pytest.fixture(params=[convert_qwen3vl_to_hf, convert_qwen3vlmoe_to_hf])
def convert(request, monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(
        misc, "get_hf_config", lambda _: SimpleNamespace(vision_config=SimpleNamespace(num_heads=2, hidden_size=4))
    )
    return lambda name, value: request.param(
        SimpleNamespace(hf_checkpoint="unused"), "module.module.vision_model." + name, value
    )


@pytest.mark.parametrize("bias", [False, True])
def test_native_qkv_uses_complete_q_k_v_head_blocks(convert, bias):
    # Each pair of rows is one head; labels distinguish both head and Q/K/V.
    native = torch.tensor([10, 11, 20, 21, 30, 31, 12, 13, 22, 23, 32, 33], dtype=torch.float32)
    expected = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33], dtype=torch.float32)
    if not bias:
        native = native[:, None].expand(-1, 4).contiguous()
        expected = expected[:, None].expand(-1, 4).contiguous()
    suffix = "bias" if bias else "weight"
    [(name, value)] = convert(f"decoder.layers.7.self_attention.linear_qkv.{suffix}", native)
    assert name == f"model.visual.blocks.7.attn.qkv.{suffix}"
    assert torch.equal(value, expected)


@pytest.mark.parametrize(
    "native,expected",
    [
        ("decoder.layers.2.self_attention.linear_proj.bias", "blocks.2.attn.proj.bias"),
        ("decoder.layers.2.self_attention.linear_qkv.layer_norm_weight", "blocks.2.norm1.weight"),
        ("decoder.layers.2.mlp.linear_fc1.layer_norm_bias", "blocks.2.norm2.bias"),
        ("decoder.layers.2.mlp.linear_fc2.weight", "blocks.2.mlp.linear_fc2.weight"),
        ("decoder.deepstack_merger_list.1.patch_norm.weight", "deepstack_merger_list.1.norm.weight"),
        ("decoder.deepstack_merger_list.0.linear_fc1.bias", "deepstack_merger_list.0.linear_fc1.bias"),
        ("merger.patch_norm.bias", "merger.norm.bias"),
        ("patch_embed.proj.weight", "patch_embed.proj.weight"),
        ("pos_embed.weight", "pos_embed.weight"),
        ("blocks.3.attn.qkv.weight", "blocks.3.attn.qkv.weight"),
        ("deepstack_merger_list.2.norm.bias", "deepstack_merger_list.2.norm.bias"),
    ],
)
def test_native_names_and_legacy_hf_tower_remain_distinct(convert, native, expected):
    original = torch.arange(16).reshape(4, 4)
    [(name, value)] = convert(native, original)
    assert name == "model.visual." + expected
    assert value is original


def test_unknown_native_weight_and_invalid_qkv_shape_fail(convert):
    with pytest.raises(ValueError, match="Unknown native"):
        convert("decoder.layers.0.missing.weight", torch.zeros(4))
    with pytest.raises(ValueError, match="QKV shape"):
        convert("decoder.layers.0.self_attention.linear_qkv.weight", torch.zeros(6, 4))

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.backends.megatron.weight_conversion.qwen2 import convert_qwen2_to_hf


@pytest.mark.parametrize(
    ("native_name", "hf_name"),
    [
        (
            "module.module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "module.module.decoder.layers.0.input_layernorm.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "module.module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
        (
            "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
    ],
)
def test_qwen2_layer_norm_aliases(native_name, hf_name):
    args = Namespace(
        kv_channels=4,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
    )
    weight = torch.arange(8, dtype=torch.float32)

    assert convert_qwen2_to_hf(args, native_name, weight) == [(hf_name, weight)]

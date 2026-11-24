# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


def test_minimax_loads_vision_qkv_across_weight_updates():
    model_module = pytest.importorskip("sglang.srt.models.minimax_m3_vl")
    from sglang.srt.layers.linear import QKVParallelLinear

    cls = model_module.MiniMaxM3SparseForConditionalGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(num_local_experts=0))
    prefix = "vision_tower.vision_model.encoder.layers.0.self_attn"
    parent = model
    for part in prefix.split("."):
        child = torch.nn.Module()
        parent.add_module(part, child)
        parent = child
    projection = QKVParallelLinear(4, 2, 4, tp_rank=1, tp_size=2, params_dtype=torch.float32)
    parent.add_module("qkv_proj", projection)

    weights = torch.arange(96, dtype=torch.float32).reshape(3, 8, 4)
    biases = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    with torch.no_grad():
        projection.weight.fill_(-1)
        projection.bias.fill_(-1)
        for i, shard in [(1, "k"), (0, "q"), (2, "v")]:
            model.load_weights(
                [(f"{prefix}.{shard}_proj.weight", weights[i]), (f"{prefix}.{shard}_proj.bias", biases[i])]
            )

    torch.testing.assert_close(projection.weight, weights[:, 4:].reshape(12, 4), rtol=0, atol=0)
    torch.testing.assert_close(projection.bias, biases[:, 4:].reshape(12), rtol=0, atol=0)

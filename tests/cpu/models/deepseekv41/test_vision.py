# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from relax.models.deepseekv41.vision import FrozenVisionInput, VisionAttention, VisionConfig, VisionInput


@torch.no_grad()
def test_tiled_vision_queries_still_attend_to_every_patch() -> None:
    config = VisionConfig(8, 1, 8, 1, 2, 10000.0, 3)
    attention = VisionAttention(config)
    # Uniform attention with identity value/output projections gives the
    # global image mean on every row, including the first query tile.
    attention.wqkv.weight.zero_()
    attention.wqkv.weight[-8:].copy_(torch.eye(8))
    attention.wqkv.bias.zero_()
    attention.wo.weight.copy_(torch.eye(8))
    attention.wo.bias.zero_()
    patches = torch.zeros(769, 8)
    patches[-1] = len(patches)
    cos = torch.ones(len(patches), 1, 4)
    torch.testing.assert_close(attention(patches, cos, torch.zeros_like(cos)), torch.ones_like(patches))


def test_frozen_image_span_preserves_row_order_and_text_gradients_across_shards(tmp_path: Path) -> None:
    config = VisionConfig(
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=20,
        num_hidden_layers=1,
        patch_size=2,
        rope_theta=10000.0,
        downsample_ratio=3,
    )
    reference = VisionInput(config, language_width=12).eval()
    for value, param in enumerate((reference.image_start, reference.image_end, reference.image_newline), start=1):
        torch.nn.init.constant_(param, value)
    weights = reference.state_dict()
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
    )

    # A 4x5 patch grid produces 2x2 aligned rows. Each output row has its own newline.
    patches = torch.randn(20, 3, 2, 2)
    with torch.no_grad():
        features = reference.aligner(reference.vision(patches, 4, 5), 4, 5)
        expected = torch.stack(
            [
                reference.image_start,
                features[0],
                features[1],
                reference.image_newline,
                features[2],
                features[3],
                reference.image_newline,
                reference.image_end,
            ]
        )

    frozen = FrozenVisionInput(str(tmp_path), asdict(config), 12)
    text = torch.randn(12, 1, 12, requires_grad=True)
    shards = [frozen.merge(text[i : i + 3], patches, [(4, 5)], [(2, 8)], i) for i in range(0, len(text), 3)]
    actual = torch.cat(shards)
    torch.testing.assert_close(actual[2:10, 0], expected)
    torch.testing.assert_close(actual[:2], text[:2])
    torch.testing.assert_close(actual[10:], text[10:])
    actual.sum().backward()
    expected_grad = torch.ones_like(text)
    expected_grad[2:10] = 0
    torch.testing.assert_close(text.grad, expected_grad)
    assert all(not param.requires_grad and param.grad is None for param in frozen.module.parameters())
    frozen.release()
    torch.testing.assert_close(frozen.merge(text, patches, [(4, 5)], [(2, 8)], 0), actual)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


bridge_vision = pytest.importorskip("megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl")

from relax.models.minimax_m31.vlm import MiniMaxM31VLModel  # noqa: E402


def test_minimax_packed_vision_keeps_media_from_each_sample_together():
    config = SimpleNamespace(
        hidden_size=24,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        patch_size=2,
        temporal_patch_size=2,
    )
    model = MiniMaxM31VLModel.__new__(MiniMaxM31VLModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(spatial_merge_size=2)
    model.vision_tower = bridge_vision.MiniMaxM3VisionTower(config, spatial_merge_size=2)
    model.multi_modal_projector = bridge_vision.MiniMaxM3ProjectorMLP(24, 32, 24, bias=True)
    model.patch_merge_mlp = bridge_vision.MiniMaxM3ProjectorMLP(96, 32, 24, bias=True)
    model.eval()
    pixels = torch.randn(20, 24, generator=torch.Generator().manual_seed(42))
    grids = [[[1, 2, 2], [1, 2, 4]], [[2, 2, 2]]]

    with torch.no_grad():
        packed = model._project_vision(pixels, grids)
        first = bridge_vision.MiniMaxM3VLModel._project_vision(model, pixels[:12], torch.tensor(grids[0]))
        second = bridge_vision.MiniMaxM3VLModel._project_vision(model, pixels[12:], torch.tensor(grids[1]))
        torch.testing.assert_close(packed, torch.cat([first, second]), rtol=0, atol=0)

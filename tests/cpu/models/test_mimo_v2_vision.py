# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from torch import nn


pytest.importorskip("megatron.bridge")
from relax.models.mimo_v2.vision import MiMoV2ImageModel


class _Embedding(nn.Embedding):
    def forward(self, input_ids, position_ids):
        return super().forward(input_ids).transpose(0, 1)


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 1
        self.patch_embed = nn.ModuleDict({"proj": nn.Linear(2, 3, bias=False)})

    def forward(self, pixel_values, grid_thw):
        return self.patch_embed.proj(pixel_values)


@pytest.fixture
def model():
    model = MiMoV2ImageModel.__new__(MiMoV2ImageModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_token_id=3, sequence_parallel=False, freeze_vision_model=False, freeze_vision_projection=False
    )
    model.visual = _Visual()
    model.embedding = _Embedding(4, 3)
    return model


def test_image_slots_preserve_row_order_and_route_gradients(model):
    ids = torch.tensor([[0, 3, 3], [3, 1, 2]])
    pixels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    grid = torch.tensor([[1, 1, 2], [1, 1, 1]])
    output = model.image_embeddings(ids, None, pixels, grid).transpose(0, 1)
    torch.testing.assert_close(output[ids == 3], model.visual(pixels, grid))
    torch.testing.assert_close(output[ids != 3], model.embedding.weight[ids[ids != 3]])
    output.sum().backward()
    torch.testing.assert_close(model.embedding.weight.grad[3], torch.zeros(3))
    torch.testing.assert_close(model.visual.patch_embed.proj.weight.grad, pixels.detach().sum(0).expand(3, 2))


def test_text_only_rank_has_zero_visual_gradient(model):
    ids = torch.tensor([[0, 1]])
    output = model.image_embeddings(ids, None, None, None)
    torch.testing.assert_close(output, model.embedding(ids, None))
    output.sum().backward()
    grad = model.visual.patch_embed.proj.weight.grad
    torch.testing.assert_close(grad, torch.zeros_like(model.visual.patch_embed.proj.weight))


def test_rejects_a_missing_or_surplus_image(model):
    ids = torch.tensor([[3]])
    with pytest.raises(RuntimeError, match="require pixel_values"):
        model.image_embeddings(ids, None, None, None)
    with pytest.raises(RuntimeError, match="counts must match"):
        model.image_embeddings(ids, None, torch.ones(2, 2), torch.tensor([[1, 1, 2]]))


def test_merger_only_creates_biases_present_in_checkpoint(model, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers import dynamic_module_utils

    class Visual(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.merger = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2))

    save_file({"visual.merger.1.bias": torch.ones(2)}, tmp_path / "model.safetensors")
    monkeypatch.setattr(dynamic_module_utils, "get_class_from_dynamic_module", lambda *args: Visual)
    model.config.hf_model_path = str(tmp_path)
    model.config.vision_config = {}
    model.config.params_dtype = torch.float32
    model.config.recompute_granularity = None
    model.build_visual()
    assert model.visual.merger[0].bias is None
    assert model.visual.merger[1].bias is not None

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline regressions for packed media isolation and trainable K3.1 vision."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.cpu.models.test_kimi_k3 import k3_modules as k3_modules


@pytest.fixture
def vision_module():
    target = Path(__file__).parents[3] / "relax/models/kimi_k3/vision.py"
    spec = importlib.util.spec_from_file_location("kimi_k31_vision_target", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vision_config():
    return SimpleNamespace(
        vt_hidden_size=16,
        vt_num_attention_heads=3,
        vt_num_hidden_layers=2,
        vt_intermediate_size=32,
        qkv_hidden_size=24,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        init_pos_emb_time=4,
        patch_size=2,
        patch_embed_proj_bias=False,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode="bilinear",
        norm_type="rmsnorm",
        mlp_type="mlp2",
        attn_bias=False,
        linear_bias=False,
        merge_type="sd2_tpool",
        merge_kernel_size=[2, 2],
        mm_hidden_size=16,
        text_hidden_size=12,
        mm_projector_type="patchmergerv2",
        projector_hidden_act="gelu",
        projector_ln_eps=1e-5,
    )


def test_kimi_k31_vision_packed_media_matches_separate_outputs_and_gradients(vision_module, vision_config):
    torch.manual_seed(31)
    tower = vision_module.MoonViT(vision_config)
    projector = vision_module.PatchMergerMLPV2(vision_config)
    separate_tower, separate_projector = deepcopy(tower), deepcopy(projector)
    pixels = torch.randn(28, 3, 2, 2, requires_grad=True)
    separate_pixels = pixels.detach().clone().requires_grad_()
    # Mix an image and an interpolated, three-frame video in the same batch.
    grids = torch.tensor([[1, 2, 2], [3, 2, 4]])
    features = tower(pixels, grids)
    assert [feature.shape for feature in features] == [(1, 4, 16), (2, 4, 16)]
    packed_output = torch.cat(projector(features))
    separate_output = torch.cat(
        [
            separate_projector(separate_tower(sample, grid[None]))[0]
            for sample, grid in zip(separate_pixels.split([4, 24]), grids, strict=True)
        ]
    )
    cotangent = torch.randn_like(packed_output)
    (packed_output * cotangent).sum().backward()
    (separate_output * cotangent).sum().backward()
    torch.testing.assert_close(packed_output, separate_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(pixels.grad, separate_pixels.grad, atol=2e-6, rtol=2e-5)
    for packed, separate in ((tower, separate_tower), (projector, separate_projector)):
        for name, parameter in packed.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
            torch.testing.assert_close(
                parameter.grad, dict(separate.named_parameters())[name].grad, atol=4e-6, rtol=3e-5
            )


def test_kimi_k31_vision_temporal_pooling_preserves_spatial_patch_order(vision_module, vision_config):
    vision_config.vt_num_hidden_layers = 0
    tower = vision_module.MoonViT(vision_config)
    with torch.no_grad():
        tower.patch_embed.pos_emb.weight.zero_()
        tower.patch_embed.pos_emb.time_weight.zero_()
    pixels = torch.randn(16, 3, 2, 2, requires_grad=True)
    # Both frames have 2x4 spatial patches; each 2x2 tile merges independently.
    actual = tower(pixels, torch.tensor([[2, 2, 4]]))[0]
    patches = tower.encoder.final_layernorm(tower.patch_embed.proj(pixels).flatten(1))
    first_tile = torch.stack((patches[[0, 1, 4, 5]], patches[[8, 9, 12, 13]])).mean(0)
    second_tile = torch.stack((patches[[2, 3, 6, 7]], patches[[10, 11, 14, 15]])).mean(0)
    torch.testing.assert_close(actual, torch.stack((first_tile, second_tile)))
    actual.square().sum().backward()
    assert torch.isfinite(pixels.grad).all()
    assert (pixels.grad.flatten(1).abs().sum(1) > 0).all()


@pytest.mark.parametrize(
    ("grids", "patches", "message"),
    [
        ([[1, 2, 3]], 6, "divisible"),
        ([[1, 2, 2]], 8, "patch count"),
        ([[5, 2, 2]], 20, "at most 4 frames"),
    ],
)
def test_kimi_k31_vision_rejects_invalid_media_geometry(vision_module, vision_config, grids, patches, message):
    tower = vision_module.MoonViT(vision_config)
    with pytest.raises(ValueError, match=message):
        tower(torch.randn(patches, 3, 2, 2), torch.tensor(grids))


def test_kimi_k31_vision_builder_runs_without_checkpoint_modeling_code(monkeypatch, k3_modules, vision_config):
    hooked = []
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.utils.common_utils",
        SimpleNamespace(hook_hf_module_setattr_for_tp_grad_sync=hooked.append),
    )
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(is_flash_attn_2_available=lambda: False))
    # The K3.1 path must not attempt to import nonexistent remote modeling code.
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", None)
    config = SimpleNamespace(
        hf_model_path="configuration-only-k31-checkpoint",
        trust_remote_code=True,
        kimi_dsa_config={"index_topk": 2048},
        vision_config=vision_config,
        params_dtype=torch.float32,
    )
    tower, projector = k3_modules.model._build_vision_modules(config)
    assert hooked == [tower, projector]
    output = projector(tower(torch.randn(4, 3, 2, 2), torch.tensor([[1, 2, 2]])))[0]
    output.square().sum().backward()
    assert output.shape == (1, 12)
    assert all(parameter.grad is not None for module in hooked for parameter in module.parameters())

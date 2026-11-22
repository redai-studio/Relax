# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


bridge_vision = pytest.importorskip("megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl")

from relax.models.minimax_m31.bridge import MiniMaxM31ModelProvider  # noqa: E402
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
    grids = [[[1, 2, 2], [1, 2, 4]], [[1, 2, 4]]]

    with torch.no_grad():
        packed = model._project_vision(pixels, grids)
        first = bridge_vision.MiniMaxM3VLModel._project_vision(model, pixels[:12], torch.tensor(grids[0]))
        second = bridge_vision.MiniMaxM3VLModel._project_vision(model, pixels[12:], torch.tensor(grids[1]))
        torch.testing.assert_close(packed, torch.cat([first, second]), rtol=0, atol=0)


def test_minimax_forward_returns_tensor_when_labels_are_absent():
    model = MiniMaxM31VLModel.__new__(MiniMaxM31VLModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    groups = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1))
    model.config = SimpleNamespace(_pg_collection=groups, sequence_parallel=False)
    logits = torch.zeros(1, 3, 8)
    mask = torch.tensor([[0, 1, 1]])
    labels = torch.tensor([[1, 2, 3]])
    expected_logits = logits.clone()
    expected_mask = mask.clone()

    def language_forward(*, labels, loss_mask, **kwargs):
        if labels is None:
            assert loss_mask is None
            return logits
        torch.testing.assert_close(loss_mask, expected_mask, rtol=0, atol=0)
        return torch.zeros_like(labels, dtype=torch.float32)

    model.language_model = language_forward
    output = model(loss_mask=mask)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, expected_logits, rtol=0, atol=0)

    token_losses, returned_mask = model(labels=labels, loss_mask=mask)
    assert token_losses.shape == labels.shape
    torch.testing.assert_close(returned_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize(
    "multimodal_keys,lora_rank,lora_scope,frozen_vision,allowed",
    [
        ({"image": "images"}, 0, "all", False, False),
        ({"image": "images"}, 16, "all", False, False),
        ({"image": "images"}, 16, "language", False, True),
        (None, 0, "all", False, False),
        (None, 0, "all", True, True),
    ],
)
def test_minimax_overlapped_grad_reduce_checks_trainable_vision(
    multimodal_keys, lora_rank, lora_scope, frozen_vision, allowed
):
    provider = SimpleNamespace(freeze_vision_model=frozen_vision, freeze_vision_projection=frozen_vision)
    args = SimpleNamespace(
        loss_type="sft",
        multimodal_keys=multimodal_keys,
        overlap_grad_reduce=True,
        lora_rank=lora_rank,
        lora_scope=lora_scope,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        dynamic_context_parallel=False,
        qkv_format="thd",
        allgather_cp=False,
        tensor_model_parallel_size=2,
        apply_rope_fusion=False,
    )
    if allowed:
        MiniMaxM31ModelProvider.configure_runtime(provider, args)
    else:
        with pytest.raises(ValueError, match="trainable vision parameters"):
            MiniMaxM31ModelProvider.configure_runtime(provider, args)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("layout,cp_rank", [("packed", 0), ("rows", 0), ("rows", 1)])
def test_generated_image_tokens_keep_text_embeddings_positions_and_gradients(monkeypatch, layout, cp_rank):
    from relax.backends.megatron.cp_utils import gdn_cp_slice
    from relax.models.keye_vl2 import model as model_module

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(100, 4)

        def embedding(self, input_ids, position_ids):
            return self.table(input_ids).transpose(0, 1)

        def forward(self, **kwargs):
            self.received = kwargs
            return kwargs["decoder_input"]

    model = model_module.KeyeVLModel.__new__(model_module.KeyeVLModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hf_config=SimpleNamespace(image_token_id=99))
    model.language_model = LanguageModel()
    model.visual = nn.Identity()
    model.mlp_AR = nn.Identity()
    cp_size = 1 if layout == "packed" else 2
    model.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank))
    # The response contains both a complete vision wrapper and a bare image ID.
    rows = torch.tensor([[1, 99, 99, 99, 99, 2, 97, 99, 98, 99, 3, 0]] * 2)
    original = rows.clone()
    ids = rows.reshape(1, -1) if layout == "packed" else rows
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, 12, 24], prompt_lengths=[6, 6], num_valid_tokens=24)
    grids = torch.tensor([[1, 4, 4], [1, 4, 4]])
    features = torch.randn(8, 4, requires_grad=True)
    monkeypatch.setattr(model_module, "image_features", lambda *args: features)
    result = model.forward(ids, packed_seq_params=packed, pixel_values=torch.empty(0), grid_thws=grids)

    expected = model.language_model.table(rows).reshape(24, 1, 4)
    expected = expected.clone()
    expected[1:5, 0] = features[:4]
    expected[13:17, 0] = features[4:]
    if cp_size > 1:
        expected = gdn_cp_slice(expected, packed.cu_seqlens_q_cpu, cp_size, cp_rank)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(rows, original)
    # Generated IDs use ordinary consecutive text positions after the image.
    positions = model.language_model.received["position_ids"][:, 0]
    for start in (0, 12):
        torch.testing.assert_close(positions[:, start + 5 : start + 11], torch.arange(3, 9).expand(3, -1))
    result.sum().backward()
    assert features.grad is not None
    # CP rank 0 owns the bare response image at offset 9; rank 1 owns offset 7.
    assert model.language_model.table.weight.grad[99].abs().sum() > 0


def test_generated_image_token_without_input_image_is_text():
    from relax.models.keye_vl2.positions import image_positions

    ids = torch.tensor([[1, 2, 99, 3]])
    positions = image_positions(ids, None, 99, torch.zeros_like(ids, dtype=torch.bool))
    torch.testing.assert_close(positions, torch.arange(4).expand(3, 1, -1))


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("loss_type", ["policy_loss", "sft"])
def test_packing_preserves_rl_prompt_boundary_for_visual_embedding(monkeypatch, cp_size, loss_type):
    from tests.cpu.backends.megatron.test_data_vpp import _load_data_module

    data = _load_data_module(monkeypatch)
    monkeypatch.setattr(data, "PackedSeqParams", SimpleNamespace)
    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(loss_type=loss_type))
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data, "slice_with_cp", lambda values, *_args, **_kwargs: values)
    batch = {
        "tokens": [torch.arange(12), torch.arange(8)],
        "total_lengths": [12, 8],
        "response_lengths": [4, 3],
        "loss_masks": [torch.ones(4), torch.ones(3)],
    }
    packed = data.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)
    for key in ("packed_seq_params", "vlm_packed_seq_params"):
        if key in packed:
            if loss_type == "sft":
                assert not hasattr(packed[key], "prompt_lengths")
            else:
                assert packed[key].prompt_lengths == [8, 5]

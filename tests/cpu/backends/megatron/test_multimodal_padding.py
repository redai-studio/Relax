# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from tests.cpu.backends.megatron.test_data_vpp import _load_data_module


@pytest.mark.parametrize("chunk_counts", [(1, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize("feature_shape,transpose", [((128,), (0, 2)), ((), (0, 1))])
def test_audio_padding_preserves_variable_chunks(monkeypatch, chunk_counts, feature_shape, transpose):
    data = _load_data_module(monkeypatch)
    inputs = [
        torch.full((count, *feature_shape, length), float(index + 1), requires_grad=True)
        for index, (count, length) in enumerate(zip(chunk_counts, (4, 7), strict=True))
    ]
    output, counts = data.pad_and_flatten(inputs, transpose=transpose, padding_value=0)
    assert counts == list(chunk_counts)
    assert output.shape == (sum(chunk_counts), *feature_shape, 7)
    offset = 0
    for original in inputs:
        restored = output[offset : offset + original.size(0)]
        torch.testing.assert_close(restored[..., : original.size(-1)], original)
        assert torch.count_nonzero(restored[..., original.size(-1) :]) == 0
        offset += original.size(0)
    output.sum().backward()
    for original in inputs:
        torch.testing.assert_close(original.grad, torch.ones_like(original))


def test_audio_padding_single_sample_keeps_chunk_count(monkeypatch):
    data = _load_data_module(monkeypatch)
    features = torch.randn(3, 128, 7)
    output, counts = data.pad_and_flatten([features], **data.PAD_RULES["input_features"])
    assert output is features
    assert counts == [3]


@pytest.mark.parametrize("layout", ["text", "vl_text", "media", "unsplit_text"])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_nullable_agentic_multimodal_column_preserves_cp_layout(monkeypatch, layout, cp_rank):
    data = _load_data_module(monkeypatch)
    args = SimpleNamespace(loss_type="policy_loss", uses_unsplit_forward=layout == "unsplit_text")
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(data, "PackedSeqParams", SimpleNamespace)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: cp_rank, raising=False)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    pixels = torch.ones(4, 8)
    media = {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 2, 2]])}
    rows = [None, media if layout == "media" else None]
    batch = {
        "tokens": [torch.arange(5), torch.arange(9)],
        "total_lengths": [5, 9],
        "response_lengths": [2, 2],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "multimodal_train_inputs": rows,
    }
    result = data.get_batch(iter([(batch, None)]), list(batch), pad_multiplier=1, is_vl_model=layout == "vl_text")
    # Text GPT consumes one CP-sharded THD stream, not a two-row Bridge input.
    assert result["tokens"].shape == (1, 10)
    assert result["packed_seq_params"].cu_seqlens_q.tolist() == [0, 8, 20]
    assert rows[0] is None and (rows[1] is media if layout == "media" else rows[1] is None)
    if layout == "text":
        assert "unsplit_tokens" not in result
        assert "vlm_packed_seq_params" not in result
        assert "padded_total_lengths" not in result
    else:
        assert result["unsplit_tokens"].shape == (2, 16)
        assert result["vlm_packed_seq_params"].cu_seqlens_q.tolist() == [0, 8, 24]
        assert result["padded_total_lengths"] == [8, 16]
        assert result["unsplit_attention_mask"].sum(dim=1).tolist() == [5, 9]
    if layout == "media":
        torch.testing.assert_close(result["multimodal_train_inputs"]["pixel_values"], pixels)
    else:
        assert result["multimodal_train_inputs"] is None

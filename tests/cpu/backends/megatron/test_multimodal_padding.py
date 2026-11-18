# Copyright (c) 2026 Relax Authors. All Rights Reserved.

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

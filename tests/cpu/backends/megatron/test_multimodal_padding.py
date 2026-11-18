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


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_dummy_batch_masks_main_and_mtp_without_mutating_real_rows(monkeypatch, cp_size, cp_rank):
    from types import SimpleNamespace

    data = _load_data_module(monkeypatch)
    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(enable_mtp_training=True, uses_unsplit_forward=True))
    monkeypatch.setattr(data, "PackedSeqParams", SimpleNamespace)
    for name, value in {
        "get_context_parallel_world_size": cp_size,
        "get_context_parallel_rank": cp_rank,
        "get_tensor_model_parallel_world_size": 1,
        "get_context_parallel_group": None,
    }.items():
        monkeypatch.setattr(data.mpu, name, lambda value=value: value, raising=False)
    masks = [torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])]
    rows = {"tokens": [torch.arange(8)], "loss_masks": masks, "total_lengths": [8], "response_lengths": [8]}
    iterator = data.DataIterator(rows, micro_batch_indices=[[0], [0]], dummy_micro_batch_offsets={0})
    dummy = data.get_batch(iterator, list(rows), pad_multiplier=1, pack_device=torch.device("cpu"))
    real = data.get_batch(iterator, list(rows), pad_multiplier=1, pack_device=torch.device("cpu"))
    assert dummy["__is_dummy__"]
    assert torch.count_nonzero(dummy["full_loss_masks"]) == 0
    assert torch.count_nonzero(dummy["loss_masks"][0]) == 0
    assert torch.count_nonzero(real["full_loss_masks"]) > 0
    if cp_size > 1:
        assert torch.count_nonzero(dummy["unsplit_mtp_loss_mask"]) == 0
        assert torch.count_nonzero(real["unsplit_mtp_loss_mask"]) > 0
    assert masks[0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]


def test_dummy_pre_shifted_labels_preserve_mask_contract(monkeypatch):
    from types import SimpleNamespace

    data = _load_data_module(monkeypatch)
    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(loss_type="sft", uses_unsplit_forward=True))
    monkeypatch.setattr(data, "PackedSeqParams", SimpleNamespace)
    for name, value in {
        "get_context_parallel_world_size": 1,
        "get_context_parallel_rank": 0,
        "get_tensor_model_parallel_world_size": 1,
    }.items():
        monkeypatch.setattr(data.mpu, name, lambda value=value: value, raising=False)
    labels = torch.tensor([-100, -100, 3, 4, 5, 6, 7, -100])
    rows = {
        "tokens": [torch.arange(8)],
        "loss_masks": [labels.ne(-100).int()],
        "pre_shifted_labels": [labels],
        "total_lengths": [8],
        "response_lengths": [8],
    }
    iterator = data.DataIterator(rows, micro_batch_indices=[[0], [0]], dummy_micro_batch_offsets={0})
    dummy = data.get_batch(iterator, list(rows), pad_multiplier=1, pack_device=torch.device("cpu"))
    real = data.get_batch(iterator, list(rows), pad_multiplier=1, pack_device=torch.device("cpu"))
    assert dummy["full_loss_masks"].count_nonzero() == 0
    assert dummy["unconcat_pre_shifted_labels"][0].eq(-100).all()
    torch.testing.assert_close(real["unconcat_pre_shifted_labels"][0], labels)
    assert real["full_loss_masks"].count_nonzero() > 0

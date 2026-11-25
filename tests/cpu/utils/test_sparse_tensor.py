# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.sparse_tensor import FullTensorRequired, SparseTensor, storage_identity


@pytest.mark.parametrize(
    "dtype,words,nan1,nan2,minus_zero",
    [
        (torch.bfloat16, torch.int16, 0x7FC1, 0x7FC2, -32768),
        (torch.float32, torch.int32, 0x7FC00001, 0x7FC00002, -2147483648),
    ],
)
def test_sparse_layout_preserves_nan_payload_and_signed_zero(dtype, words, nan1, nan2, minus_zero):
    raw = torch.tensor([nan1, nan2, minus_zero, 0] * 4, dtype=words).reshape(4, 4)
    mask = torch.arange(16).reshape(4, 4) % 3 == 0
    before = torch.arange(16, dtype=words).reshape(4, 4)
    destination = before.clone().view(dtype)
    patch = SparseTensor(raw.view(dtype), mask, frozenset([storage_identity(destination)]))
    shaped = torch.cat([patch[:, :2], patch[:, 2:]], dim=1).transpose(0, 1).contiguous()
    destination.copy_(shaped)
    assert torch.equal(destination.view(words), torch.where(mask.T, raw.T, before))


def test_sparse_layout_rejects_write_overloads_and_source_destination_alias():
    value = torch.tensor([10.0, 20.0])
    mask = torch.tensor([True, False])
    destination = torch.tensor([-1.0, -2.0])
    patch = SparseTensor(value, mask, frozenset([storage_identity(destination)]))
    with pytest.raises(FullTensorRequired, match="out="):
        torch.cat([patch], out=destination)
    assert torch.equal(destination, torch.tensor([-1.0, -2.0]))
    with pytest.raises(FullTensorRequired, match="also be a live destination"):
        SparseTensor(value, mask, frozenset([storage_identity(value)]))
    with pytest.raises(FullTensorRequired):
        value.copy_(patch)
    assert torch.equal(value, torch.tensor([10.0, 20.0]))


def test_sparse_layout_requires_full_for_arithmetic_and_dtype_conversion():
    patch = SparseTensor(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(FullTensorRequired):
        patch.add(1)
    with pytest.raises(FullTensorRequired):
        patch.to(torch.bfloat16)
    bf16 = SparseTensor(torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(2, 2, dtype=torch.bool))
    for other in (patch, torch.ones(2, 2)):
        with pytest.raises(FullTensorRequired, match="implicitly promote"):
            torch.cat([bf16, other])
        with pytest.raises(FullTensorRequired, match="implicitly promote"):
            torch.stack([bf16, other])

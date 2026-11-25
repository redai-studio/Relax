# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.backends.sglang import verl_nan_mask


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.cuda is None, reason="requires CUDA")
@pytest.mark.parametrize(
    ("dtype", "word", "infinity", "sign"),
    [
        (torch.float16, torch.int16, 0x7C00, -(1 << 15)),
        (torch.bfloat16, torch.int16, 0x7F80, -(1 << 15)),
        (torch.float32, torch.int32, 0x7F800000, -(1 << 31)),
    ],
)
def test_native_cuda_mask_writes_only_present_words(dtype, word, infinity, sign):
    from relax.backends.sglang.verl_masked_copy_kernel import copy_present

    # Include both signs of zero, subnormals, infinities and NaNs with different payloads.
    words = torch.tensor(
        [0, sign, 1, sign + 1, infinity, sign + infinity, infinity + 1, sign + infinity + 3], dtype=word
    )
    source = words.repeat(65)[:513].cuda().view(dtype)
    original = torch.full((515,), infinity + 7, dtype=word)
    model = torch.nn.Module()
    model.register_buffer("weight", original.cuda().view(dtype))
    expected = original.clone()
    present = torch.arange(source.numel()) % len(words) < 6
    expected[1:-1][present] = words.repeat(65)[:513][present]

    destination = model.weight[1:-1]
    version = destination._version
    assert copy_present(destination, source)
    assert destination._version == version + 1
    assert torch.equal(model.weight.view(word).cpu(), expected)

    # Overlap needs a snapshot of the source before any destination words change.
    assert not copy_present(model.weight[1:], model.weight[:-1])
    expected[1:] = torch.where(torch.isnan(expected[:-1].view(dtype)), expected[1:], expected[:-1])
    with verl_nan_mask._masked_copy(model):
        model.weight[1:].copy_(model.weight[:-1])
    assert torch.equal(model.weight.view(word).cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.cuda is None, reason="requires CUDA")
def test_cuda_virtual_zero_and_forward_ad_use_native_copy():
    model = torch.nn.Module()
    model.register_buffer("weight", torch.ones(3, device="cuda"))
    zeros = torch._efficientzerotensor((3,), dtype=torch.float32, device="cuda")
    with verl_nan_mask._masked_copy(model):
        model.weight.copy_(zeros)
    assert torch.equal(model.weight.cpu(), torch.zeros(3))

    with torch.no_grad(), torch.autograd.forward_ad.dual_level(), verl_nan_mask._masked_copy(model):
        values = torch.tensor([3.0, float("nan"), -0.0], device="cuda")
        dual = torch.autograd.forward_ad.make_dual(values, torch.ones_like(values))
        model.weight.copy_(dual)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(model.weight)
        assert torch.equal(primal.cpu().view(torch.int32), torch.tensor([3.0, 0.0, -0.0]).view(torch.int32))
        assert torch.equal(tangent.cpu(), torch.tensor([1.0, 0.0, 1.0]))

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Standalone GPU correctness of the experimental bitmap-native QSA kernel."""

import pytest
import torch


def _oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bits: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    outputs = []
    group = q.shape[1] // k.shape[1]
    for start, end in zip(bounds, bounds[1:]):
        if start == end:
            continue
        positions = torch.arange(end - start, device=q.device)
        blocks = positions // 4
        complete = (positions + 1) // 4
        selected = ((bits[start:end, blocks // 32] >> (blocks % 32)[None, :]) & 1).bool()
        mask = (positions[None, :] <= positions[:, None]) & (
            ((blocks[None, :] < complete[:, None]) & selected) | (blocks[None, :] == complete[:, None])
        )
        keys = k[start:end].float().repeat_interleave(group, 1)
        values = v[start:end].float().repeat_interleave(group, 1)
        scores = torch.einsum("qhd,khd->hqk", q[start:end].float(), keys) / 16
        scores = scores.masked_fill(~mask[None], -float("inf"))
        safe = torch.where(mask.any(-1)[None, :, None], scores, 0.0)
        probabilities = safe.softmax(-1) * mask[None]
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, values))
    return torch.cat(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA BF16 attention kernels")
@pytest.mark.parametrize(
    "lengths,hq,hk,zero",
    [
        ([0, 1, 2, 3, 4, 5, 17, 130, 513], 6, 1, False),
        ([2053], 6, 1, False),
        ([7, 129, 1025], 12, 2, False),
        ([17, 129], 8, 1, False),
        ([7, 130], 3, 1, False),
        ([3, 4, 5, 65], 6, 1, True),
    ],
)
def test_bitmap_attention_output_and_qkv_gradients_match_fp32_oracle(lengths, hq, hk, zero):
    from relax.models.qwen3_8.ops.qsa_mixed_ops.bitmap_attention import attention
    from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import attention_from_bits

    torch.manual_seed(71)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    total = bounds[-1]
    words = (max(lengths) // 4 + 32) // 32
    bits = torch.randint(-(2**31), 2**31 - 1, (total, words), device="cuda", dtype=torch.int32)
    bits[:, 0] |= 1
    if zero:
        bits.zero_()
    q = torch.randn(total, hq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, hk, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    grad = torch.randn_like(q)
    expected = _oracle(q, k, v, bits, bounds)
    expected_grad = torch.autograd.grad(expected, (q, k, v), grad.float())
    errors = []
    for function in (attention, attention_from_bits):
        output = function(q, k, v, bits, 1 / 16, bounds)
        gradients = torch.autograd.grad(output, (q, k, v), grad)
        current = []
        for actual, reference in zip((output, *gradients), (expected, *expected_grad), strict=True):
            assert torch.isfinite(actual).all()
            diff = actual.detach().float() - reference.detach().float()
            current.append((diff.norm() / reference.float().norm().clamp_min(1e-10), diff.abs().max()))
        errors.append(current)
    # BF16 products and output rounding differ from the FP32 math oracle.
    # Require accuracy comparable to the production Flex implementation.
    for direct, flex in zip(*errors, strict=True):
        assert direct[0] <= max(flex[0] * 1.3, 0.004)
        assert direct[1] <= max(flex[1] * 1.5, 0.015625)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA attention inputs")
def test_bitmap_attention_empty_input_and_invalid_word_stride():
    from relax.models.qwen3_8.ops.qsa_mixed_ops.bitmap_attention import attention

    q = torch.empty(0, 6, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.empty(0, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.empty_like(k, requires_grad=True)
    output = attention(q, k, v, torch.empty(0, 0, device="cuda", dtype=torch.int32), 1 / 16, [0, 0])
    gradients = torch.autograd.grad(output.sum(), (q, k, v))
    assert all(g.numel() == 0 for g in gradients)
    q = torch.empty(129, 6, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(129, 1, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="enough int32 words"):
        attention(q, k, k, torch.empty(129, 1, device="cuda", dtype=torch.int32), 1 / 16, [0, 129])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA attention inputs")
def test_bitmap_attention_rejects_tiles_crossing_bitmap_words():
    from relax.models.qwen3_8.ops.qsa_mixed_ops.bitmap_attention import attention

    q = torch.empty(1, 6, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 1, 256, device="cuda", dtype=torch.bfloat16)
    bits = torch.ones(1, 1, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="128-token word"):
        attention(q, k, k, bits, 1 / 16, [0, 1], bn=256)

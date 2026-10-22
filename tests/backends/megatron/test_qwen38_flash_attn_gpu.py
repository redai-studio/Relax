# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention 4 regression requires CUDA")
@pytest.mark.parametrize(
    ("head_dim", "query_heads", "kv_heads", "causal", "softmax_scale"),
    [(72, 4, 4, False, None), (72, 4, 4, False, 0.2), (256, 6, 1, True, None)],
    ids=["vision-head72", "vision-head72-explicit-scale", "language-cp1-head256-gqa"],
)
def test_qwen38_flash_attn_packed_forward_backward(
    head_dim: int, query_heads: int, kv_heads: int, causal: bool, softmax_scale: float | None
) -> None:
    # Import the CuTe API directly: a TE fallback must not mask an FA4 regression.
    fa4 = pytest.importorskip("flash_attn.cute.interface", reason="FlashAttention 4 is not installed")
    lengths = (129, 257)
    offsets = (0, lengths[0], sum(lengths))
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    torch.manual_seed(42)
    inputs = [
        torch.randn(sum(lengths), heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for heads in (query_heads, kv_heads, kv_heads)
    ]
    query, key, value = [tensor.detach().requires_grad_() for tensor in inputs]
    actual, _ = fa4.flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=causal,
        softmax_scale=softmax_scale,
    )
    upstream = torch.randn_like(actual)
    actual.backward(upstream)

    reference_inputs = [tensor.float().detach().requires_grad_() for tensor in inputs]
    outputs = []
    # Separate documents explicitly to detect accidental attention across packed boundaries.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            q, k, v = [tensor[start:end].transpose(0, 1).unsqueeze(0) for tensor in reference_inputs]
            output = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True, scale=softmax_scale)
            outputs.append(output.squeeze(0).transpose(0, 1))
    expected = torch.cat(outputs)
    expected.backward(upstream.float())

    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
    for tensor, reference in zip((query, key, value), reference_inputs, strict=True):
        assert tensor.grad is not None and reference.grad is not None
        torch.testing.assert_close(tensor.grad.float(), reference.grad, atol=3e-2, rtol=3e-2)

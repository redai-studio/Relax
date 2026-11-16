# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate real GLM DSA CUDA kernels against independent dense references."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GLM DSA kernels require CUDA")


def test_glm_indexer_selects_sparse_history_and_masks_packed_sequences() -> None:
    from relax.models.glm_moe_dsa.ops.indexer import lighting_indexer

    torch.manual_seed(17)
    q = torch.randn(128, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2304, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(128, 32, 1, device="cuda", dtype=torch.float32) / 32
    starts = torch.cat((torch.zeros(64, device="cuda"), torch.full((64,), 1152, device="cuda"))).int()
    ends = torch.arange(2177, 2305, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        scores, indices = lighting_indexer(q, k, weights, starts, ends, 2048)
        # FP64 keeps the oracle independent of TF32 matmul policy.
        logits = (torch.einsum("qhd,kd->qhk", q.double(), k.double()).relu() * weights.double()).sum(1)
        positions = torch.arange(k.shape[0], device="cuda")[None, :]
        logits.masked_fill_((positions < starts[:, None]) | (positions >= ends[:, None]), -torch.inf)
        expected_scores = logits.topk(2048, dim=-1).values
        torch.testing.assert_close(scores.double(), expected_scores, rtol=2e-4, atol=2e-4)
        valid = indices >= 0
        assert valid[:64].all()  # More than 2048 legal keys: actual sparse selection.
        assert (~valid[64:]).any()  # The second packed sequence needs -1 padding.
        assert torch.all((indices >= starts[:, None]) | ~valid)
        assert torch.all((indices < ends[:, None]) | ~valid)
        gathered = logits.gather(1, indices.long().clamp_min(0)).masked_fill(~valid, -torch.inf)
        torch.testing.assert_close(scores.double(), gathered, rtol=2e-4, atol=2e-4)


def test_glm_sparse_mla_short_packed_forward_and_gradients() -> None:
    from relax.models.glm_moe_dsa.ops.indexer import pytorch_topk_with_invalid_padding
    from relax.models.glm_moe_dsa.ops.sparse_mla import SparseMLA

    torch.manual_seed(23)
    q = (torch.randn(96, 32, 576, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    kv = (torch.randn(96, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    positions = torch.arange(96, device="cuda")
    starts = torch.where(positions < 32, 0, 32)
    allowed = (positions[None, :] >= starts[:, None]) & (positions[None, :] <= positions[:, None])
    logits = torch.randn(96, 96, device="cuda").masked_fill(~allowed, -torch.inf)
    _, indices = pytorch_topk_with_invalid_padding(logits, 2048)
    scale = 256**-0.5
    actual, lse = SparseMLA.apply(q, kv, indices[:, None, :], scale)
    ref_q, ref_kv = q.detach().float().requires_grad_(), kv.detach().float().requires_grad_()
    attention = torch.einsum("qhd,kd->qhk", ref_q, ref_kv[:, 0]) * scale
    attention = attention.masked_fill(~allowed[:, None, :], -torch.inf)
    expected = torch.einsum("qhk,kd->qhd", attention.softmax(-1), ref_kv[:, 0, :512])
    upstream = torch.randn_like(actual) * 0.1
    actual_grads = torch.autograd.grad(actual, (q, kv), upstream)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_kv), upstream.float())
    assert torch.isfinite(lse).all()
    for value, reference in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        assert torch.isfinite(value).all()
        error = (value.float() - reference).norm() / reference.norm().clamp_min(1e-10)
        assert error < 0.02, f"Relative L2 error: {error.item()}"

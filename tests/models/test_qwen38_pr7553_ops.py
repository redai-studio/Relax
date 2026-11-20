# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real Blackwell correctness checks, intentionally not run on CPU builders."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="PR7553 CuTe numerical validation requires a Blackwell GPU",
)


@pytest.mark.parametrize("blocks,budget,impl", [(37, 8, "triton"), (4096, 512, "auto")])
def test_qsa_radix_gpu_matches_cpu_oracle_with_ties(blocks, budget, impl):
    from relax.models.qwen3_8.ops.qsa_cute.selector import select_ids

    generator = torch.Generator().manual_seed(71)
    q = torch.randint(-2, 3, (65, 4, 128), generator=generator).to(torch.bfloat16)
    k = torch.randint(-2, 3, (blocks, 128), generator=generator).to(torch.bfloat16)
    q[:3].zero_()
    positions = torch.arange(65) * max(3, blocks // 16)
    expected = select_ids(q, k, positions, budget)
    actual = select_ids(q.cuda(), k.cuda(), positions.cuda(), budget, impl=impl)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("local_queries", [False, True])
def test_qsa_cute_packed_forward_and_all_gradients_match_dense_oracle(local_queries):
    from relax.models.qwen3_8.ops.qsa_cute.adapter import sparse_attention

    torch.manual_seed(17)
    bounds = (0, 1, 9, 73)
    rows = (
        torch.tensor([0, 1, 4, 8, 9, 12, 40, 72], device="cuda") if local_queries else torch.arange(73, device="cuda")
    )
    q = torch.randn(1, rows.numel(), 12, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(1, 73, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    cu = torch.tensor(bounds, device="cuda")
    docs = torch.searchsorted(cu[1:], rows, right=True)
    relative = rows - cu[docs]
    block_ids = torch.arange(4, device="cuda").expand(rows.numel(), -1)
    ids = torch.where(block_ids < (relative[:, None] + 1) // 4, block_ids, -1).int()[None]
    out = sparse_attention(q, k, v, ids, bounds, 256**-0.5, rows)
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (q, k, v), grad)
    qr, kr, vr = [value.detach().float().requires_grad_() for value in (q, k, v)]
    tokens = torch.arange(73, device="cuda")
    key_docs = torch.searchsorted(cu[1:], tokens, right=True)
    relative_keys = tokens[None] - cu[docs, None]
    selected = ((relative_keys[:, :, None] // 4) == ids[0, :, None]).any(-1)
    tail = relative_keys // 4 == (relative[:, None] + 1) // 4
    mask = (docs[:, None] == key_docs[None]) & (tokens[None] <= rows[:, None]) & (selected | tail)
    logits = torch.einsum("bqhd,bkhd->bhqk", qr, kr.repeat_interleave(12, dim=2)) * 256**-0.5
    expected = torch.einsum(
        "bhqk,bkhd->bqhd", logits.masked_fill(~mask, -torch.inf).softmax(-1), vr.repeat_interleave(12, dim=2)
    )
    expected_grads = torch.autograd.grad(expected, (qr, kr, vr), grad.float())
    torch.testing.assert_close(out.float(), expected, atol=0.035, rtol=0.035)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual.float(), reference, atol=0.055, rtol=0.055)
    repeated = sparse_attention(q, k, v, ids, bounds, 256**-0.5, rows)
    repeated_grads = torch.autograd.grad(repeated, (q, k, v), grad)
    for first, second in zip(actual_grads, repeated_grads):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_qsa_cute_mixed_short_long_matches_grouped(monkeypatch):
    from types import SimpleNamespace

    from relax.models.qwen3_8.ops.qsa_mixed_ops import attention_for_selection

    torch.manual_seed(41)
    bounds = (0, 31, 2335)
    length = bounds[-1]
    positions = torch.cat((torch.arange(31), torch.arange(2304))).cuda()
    doc_ids = torch.cat((torch.zeros(31), torch.ones(2304))).cuda().int()
    scores = torch.rand(length, 576, device="cuda")
    valid = torch.arange(576, device="cuda")[None] < (positions[:, None] + 1) // 4
    values, ids = scores.masked_fill(~valid, -torch.inf).topk(512, dim=-1)
    selection = SimpleNamespace(
        compress_ratio=4,
        selected_ids=torch.where(values.isfinite(), ids, -1).int(),
        positions=positions[None],
        doc_ids=doc_ids[None],
        _relax_qsa_mixed_bounds=bounds,
    )
    q = torch.randn(1, length, 12, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(1, length, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", "2051")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "0")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "grouped")
    reference = attention_for_selection(q, k, v, selection, 256**-0.5, True)
    grad = torch.randn_like(reference)
    expected_grads = torch.autograd.grad(reference, (q, k, v), grad)
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "cute")
    actual = attention_for_selection(q, k, v, selection, 256**-0.5, True)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    torch.testing.assert_close(actual, reference, atol=0.035, rtol=0.035)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.055, rtol=0.055)


def test_qsa_cute_dkv_preserves_gradients_above_int32_head_expanded_csr_offset():
    """Relocating identical CSR entries must preserve dK/dV.

    Small attention tests never cross the entry-offset times G Int32 boundary.
    An unused CSR prefix reaches it with about 1.34 GiB, without a full 1M
    attention graph or its quadratic metadata. Only eight entries are read.
    """
    from relax.models.qwen3_8.ops.qsa_cute.cute_qsa_attn import DKV_TILE_BLOCKS, qsa_cute_dkv
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa import QSAKVTileCSR

    torch.manual_seed(89)
    heads, rows, dim = 12, 8, 256
    q = torch.randn(1, rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 4, 1, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    do = torch.randn_like(q)
    logits = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float().expand(-1, -1, heads, -1)) * dim**-0.5
    lse = (logits.logsumexp(-1) / torch.log(torch.tensor(2.0, device="cuda"))).contiguous()
    output = torch.einsum("bqhk,bkhd->bqhd", logits.softmax(-1), v.float().expand(-1, -1, heads, -1))
    delta = (output.bfloat16().float() * do.float()).sum(-1).contiguous()
    positions = torch.full((rows,), 3, device="cuda", dtype=torch.int64)
    bases = torch.tensor([0], device="cuda", dtype=torch.int32)
    ends = torch.tensor([4], device="cuda", dtype=torch.int32)

    def run(prefix: int) -> tuple[torch.Tensor, torch.Tensor]:
        queries = torch.empty(prefix + rows, device="cuda", dtype=torch.int32)
        bits = torch.empty_like(queries)
        queries[prefix:] = torch.arange(rows, device="cuda", dtype=torch.int32)
        bits[prefix:] = 1
        offsets = torch.tensor([prefix, prefix + rows], device="cuda", dtype=torch.int64)
        csr = QSAKVTileCSR(offsets, queries, bits, DKV_TILE_BLOCKS, 1)
        result = qsa_cute_dkv(q, k, v, lse, delta, do, positions, csr, bases, ends, dim**-0.5)
        torch.cuda.synchronize()
        return result

    expected = run(0)
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) > 0 for value in expected)
    actual = run((2**31 + heads - 1) // heads + 64)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MiniMax MSA requires CUDA")
def test_packed_minimax_msa_matches_dense_masked_attention_and_gradients() -> None:
    from relax.models.minimax_m31.attention import MiniMaxM3Indexer, MiniMaxM3SparseCoreAttention
    from relax.models.minimax_m31.packed_layout import PackedBlockLayout

    block_size = 128
    head_dim = 128
    bounds = [0, 389, 656]
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(321)
    layout = PackedBlockLayout(bounds, block_size, device)
    indexer = SimpleNamespace(block_size=block_size, topk_blocks=2, local_blocks=1, init_blocks=0)

    # With a mandatory local block, index head 0 prefers the latest previous
    # block and head 1 prefers the first. Give the reference those selections
    # directly, independent of the production score, selector and tile code.
    index_q = torch.zeros(bounds[-1], 2, 64, device=device, dtype=torch.bfloat16)
    index_q[:, 0, 0], index_q[:, 1, 0] = 1, -1
    index_k = torch.zeros(bounds[-1], 64, device=device, dtype=torch.bfloat16)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        index_k[start:end, 0] = torch.arange(end - start, device=device) // block_size
    flags = MiniMaxM3Indexer.select_flags(indexer, index_q, index_k, layout, bounds)
    token_start = layout.token_start.int()
    block_form = (
        flags,
        token_start,
        torch.arange(bounds[-1], device=device, dtype=torch.int32),
        layout.token_block_start.int(),
        token_start,
        layout.block_first_token.int(),
        layout.block_last_token.int(),
        block_size,
    )
    scale = head_dim**-0.5
    core = SimpleNamespace(_owner=SimpleNamespace(_msa_block_form=block_form), softmax_scale=scale)
    q, k, v = [
        torch.randn(bounds[-1], heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator)
        for heads in (4, 2, 2)
    ]
    q, k, v = (tensor.requires_grad_() for tensor in (q, k, v))
    actual = MiniMaxM3SparseCoreAttention.forward(core, q, k, v).view_as(q)

    ref_q, ref_k, ref_v = (tensor.detach().double().requires_grad_() for tensor in (q, k, v))
    expected = []
    group = q.size(1) // k.size(1)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        positions = torch.arange(end - start, device=device)
        blocks = positions // block_size
        same_block = blocks[:, None] == blocks[None, :]
        previous_block = blocks[:, None] - 1 == blocks[None, :]
        first_block = blocks[None, :] == 0
        selected = torch.stack([same_block | previous_block, same_block | first_block]).repeat_interleave(group, dim=0)
        allowed = selected & (positions[:, None] >= positions[None, :])
        queries = ref_q[start:end].transpose(0, 1)
        keys, values = (tensor[start:end].transpose(0, 1).repeat_interleave(group, dim=0) for tensor in (ref_k, ref_v))
        scores = (queries @ keys.transpose(-1, -2)) * scale
        expected.append((scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ values).transpose(0, 1))
    expected = torch.cat(expected)

    grad_out = torch.randn(actual.shape, device=device, dtype=actual.dtype, generator=generator)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad_out)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k, ref_v), grad_out.double())
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.005)
    # The kernel rounds softmax terms to BF16 for dot products. Cancellation
    # in the backward pass can leave a small absolute error near zero.
    for name, got, want in zip(("dQ", "dK", "dV"), actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got.float(), want.float(), rtol=0.02, atol=0.008, msg=lambda msg: f"{name}: {msg}")

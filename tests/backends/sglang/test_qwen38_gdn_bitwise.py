# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math

import pytest
import torch

from relax.models.qwen3_8.ops.gdn_bitwise import GDNBitwiseState, gates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_qwen38_gdn_decay_rounds_exp_to_bf16() -> None:
    # exp(0.69140625) rounds to 2 in BF16, before multiplication by softplus(0).
    a_log = torch.full((1,), 0.69140625, device="cuda", dtype=torch.float32)
    projection = torch.zeros((1, 1), device="cuda", dtype=torch.bfloat16)
    decay, beta = gates(a_log, projection[0], projection, projection)
    assert torch.equal(decay, torch.full_like(decay, -2 * math.log(2)))
    assert torch.equal(beta, torch.full_like(beta, 0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@torch.no_grad()
def test_qwen38_gdn_chunking_batch_order_and_slot_reuse() -> None:
    chunk_gated_delta_rule = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    l2norm = pytest.importorskip("fla.modules.l2norm").l2norm
    rng = torch.Generator(device="cuda").manual_seed(42)
    n_k, n_v, d_k, d_v = 2, 6, 128, 64

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=rng)

    def request(length: int) -> tuple[torch.Tensor, ...]:
        q, k = (tensor(1, length, n_k, d_k) for _ in range(2))
        v = tensor(1, length, n_v, d_v)
        a, b = (tensor(1, length, n_v) for _ in range(2))
        g, beta = gates(tensor(n_v).float(), tensor(n_v), a, b)
        return q, k, v, g, beta

    rows = [request(137), request(125), request(19)]
    states = torch.full((3, n_v, d_v, d_k), 99.0, device="cuda")
    cached = GDNBitwiseState()
    lengths = [0] * len(rows)

    def reference(data: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, g, beta = data
        q, k = (l2norm(a.contiguous()).repeat_interleave(n_v // n_k, dim=2) for a in (q, k))
        return chunk_gated_delta_rule(
            q,
            k,
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            state_v_first=False,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, q.shape[1]], device="cuda", dtype=torch.int64),
        )

    expected = [reference(data)[0] for data in rows]

    def same_bits(a: torch.Tensor, b: torch.Tensor) -> None:
        dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))

    def advance(batch: tuple[tuple[int, int, int], ...]) -> None:
        pieces = [tuple(x[:, lengths[r] : lengths[r] + n] for x in rows[r]) for r, _, n in batch]
        q, k, v, g, beta = [torch.cat(parts, dim=1) for parts in zip(*pieces, strict=True)]
        out = cached.forward(
            0,
            q,
            k,
            v,
            g[0],
            beta[0],
            states,
            torch.tensor([s for _, s, _ in batch], device="cuda"),
            [lengths[r] for r, _, _ in batch],
            [n for _, _, n in batch],
        )
        offset = 0
        for row, slot, n in batch:
            start = lengths[row]
            lengths[row] += n
            same_bits(out[:, offset : offset + n], expected[row][:, start : start + n])
            complete = lengths[row] // 64 * 64
            anchor = (
                reference(tuple(x[:, :complete] for x in rows[row]))[1][0].transpose(-1, -2)
                if complete
                else states[slot].new_zeros(states.shape[1:])
            )
            same_bits(states[slot], anchor)
            offset += n

    advance(((0, 1, 17), (1, 2, 64)))
    advance(((1, 2, 1), (0, 1, 92)))
    advance(((0, 1, 28), (1, 2, 60)))
    # A new request reuses a slot containing both an old committed state and a tail.
    advance(((2, 1, 1),))
    for _ in range(18):
        advance(((2, 1, 1),))

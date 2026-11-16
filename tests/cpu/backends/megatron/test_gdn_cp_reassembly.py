# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for the GatedDeltaNet context-parallel zig-zag reassembly
helpers.

These cover the highest-risk part of the GDN all-gather CP path (Strategy A):
that the per-sample zig-zag reconstruction (`gdn_reassemble_full`) exactly
inverts relax's own CP data sharding (`slice_with_cp`), and that `gdn_cp_slice`
round-trips against it. Pure-tensor, no distributed / GPU.
"""

from __future__ import annotations

import pytest
import torch

from relax.backends.megatron.cp_utils import gdn_cp_slice, gdn_reassemble_full, slice_with_cp


def _full_cu_seqlens(local_lens: list[int], cp_size: int) -> torch.Tensor:
    """Build the full (x cp) cu_seqlens from per-sample LOCAL lengths.

    Mirrors data.py: cu_seqlens are accumulated from local per-sample lengths
    then multiplied by cp_size.
    """
    cu = [0]
    for n in local_lens:
        cu.append(cu[-1] + n)
    return torch.tensor(cu, dtype=torch.int64) * cp_size


def _shard_full_like_relax(full_samples: list[torch.Tensor], cp_size: int, cp_rank: int) -> torch.Tensor:
    """Replicate relax's data path: per-sample slice_with_cp then concat."""
    shards = [
        slice_with_cp(s, pad_value=0.0, qkv_format="thd", dynamic_cp_size=cp_size, dynamic_cp_rank=cp_rank)
        for s in full_samples
    ]
    return torch.cat(shards, dim=0)


def test_gdn_reassemble_full_inverts_slice_with_cp():
    # Each sample length must be a multiple of 2*cp so slice_with_cp needs no pad
    # (the real data path pre-pads samples to 2*cp*chunk_size before splitting).
    for cp_size in (2, 4, 8):
        chunk = 3
        # Distinct sample lengths, each = (2*cp)*chunk so chunks are exact.
        local_lens = []
        full_samples = []
        n_samples = 3
        base = 2 * cp_size * chunk
        for i in range(n_samples):
            full_len = base * (i + 1)  # different lengths per sample
            # feature dim 2 so we also catch any accidental dim mixups
            sample = torch.arange(full_len, dtype=torch.float32).reshape(full_len, 1, 1).repeat(1, 1, 2)
            # tag the feature channel so token identity is unambiguous
            sample[..., 1] = sample[..., 0] + 1000.0
            full_samples.append(sample)
            local_lens.append(full_len // cp_size)

        cu_seqlens = _full_cu_seqlens(local_lens, cp_size)
        full_expected = torch.cat(full_samples, dim=0)

        # Gather = list of each rank's local shard (as the all-gather would produce).
        gathered = [_shard_full_like_relax(full_samples, cp_size, r) for r in range(cp_size)]

        full_got = gdn_reassemble_full(gathered, cu_seqlens, cp_size)
        assert full_got.shape == full_expected.shape
        assert torch.equal(full_got, full_expected), f"cp={cp_size}: reassembly != sequential order"


def test_gdn_cp_slice_matches_slice_with_cp():
    for cp_size in (2, 4):
        chunk = 5
        base = 2 * cp_size * chunk
        full_samples = []
        local_lens = []
        for i in range(2):
            full_len = base * (i + 1)
            sample = torch.arange(full_len, dtype=torch.float32).reshape(full_len, 1, 1)
            full_samples.append(sample)
            local_lens.append(full_len // cp_size)
        cu_seqlens = _full_cu_seqlens(local_lens, cp_size)
        full = torch.cat(full_samples, dim=0)

        for cp_rank in range(cp_size):
            got = gdn_cp_slice(full, cu_seqlens, cp_size, cp_rank)
            expected = _shard_full_like_relax(full_samples, cp_size, cp_rank)
            assert torch.equal(got, expected), f"cp={cp_size} rank={cp_rank}: slice mismatch"


def test_gather_then_slice_round_trip_is_identity():
    cp_size = 4
    chunk = 4
    base = 2 * cp_size * chunk
    full_samples = [torch.randn(base * (i + 1), 1, 3) for i in range(3)]
    local_lens = [s.shape[0] // cp_size for s in full_samples]
    cu_seqlens = _full_cu_seqlens(local_lens, cp_size)

    gathered = [_shard_full_like_relax(full_samples, cp_size, r) for r in range(cp_size)]
    full = gdn_reassemble_full(gathered, cu_seqlens, cp_size)

    for cp_rank in range(cp_size):
        shard = gdn_cp_slice(full, cu_seqlens, cp_size, cp_rank)
        assert torch.equal(shard, gathered[cp_rank]), f"round-trip failed for rank {cp_rank}"


def test_cu_seqlens_accepts_precomputed_cpu_list():
    """Hot path passes a Python list[int] (precomputed once) instead of a
    device tensor to avoid a per-layer .tolist() sync; both must give the same
    result."""
    cp_size = 2
    base = 2 * cp_size * 3
    full_samples = [torch.randn(base * (i + 1), 1, 2) for i in range(2)]
    local_lens = [s.shape[0] // cp_size for s in full_samples]
    cu_tensor = _full_cu_seqlens(local_lens, cp_size)
    cu_list = cu_tensor.tolist()

    gathered = [_shard_full_like_relax(full_samples, cp_size, r) for r in range(cp_size)]
    assert torch.equal(
        gdn_reassemble_full(gathered, cu_tensor, cp_size),
        gdn_reassemble_full(gathered, cu_list, cp_size),
    )
    full = torch.cat(full_samples, dim=0)
    for r in range(cp_size):
        assert torch.equal(
            gdn_cp_slice(full, cu_tensor, cp_size, r),
            gdn_cp_slice(full, cu_list, cp_size, r),
        )


# ---------------------------------------------------------------------------
# Distributed forward / backward parity: CP=1 vs CP=cp for the *whole* GDN CP
# all-gather path (all-gather -> duplicated causal scan -> re-slice), including
# gradients. Runs on CPU via gloo (all_gather + reduce_scatter are supported), so
# it needs no GPU. This is the test that pins down the reduce-scatter backward:
# with a plain grads[rank] backward the input-grad parity below fails, because
# the causal scan couples every rank's output onto every earlier rank's input.
# ---------------------------------------------------------------------------


def _toy_gdn_like(full_seq, w_in, w_scan, w_out):
    """A tiny differentiable stand-in for the GDN core that couples tokens
    causally across the *full* sequence (so cross-CP-rank gradients exist).

    ``full_seq`` is ``[S, 1, C]`` (sequential order). Mirrors the real path's
    shape flow: projection -> causal (cumulative) coupling -> output
    projection.
    """
    proj = full_seq @ w_in  # [S, 1, C]
    scanned = torch.cumsum(proj, dim=0) @ w_scan  # causal coupling over all tokens
    return scanned @ w_out  # [S, 1, C]


def _parity_worker(rank, world_size, port, tmp_path):
    import os

    import torch.distributed as dist

    from relax.backends.megatron.cp_utils import (
        _AllGatherFullSequence,
        gdn_cp_slice,
        gdn_reassemble_full,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)  # identical params + input on every rank (DDP replicas)
        c = 4
        cp_size = world_size
        # Two samples, each length a multiple of 2*cp so slice_with_cp needs no pad.
        chunk = 3
        base = 2 * cp_size * chunk
        full_lens = [base * (i + 1) for i in range(2)]
        full_samples = [torch.randn(n, 1, c, dtype=torch.float64) for n in full_lens]
        x_full = torch.cat(full_samples, dim=0)
        local_lens = [n // cp_size for n in full_lens]
        cu_seqlens = _full_cu_seqlens(local_lens, cp_size).tolist()

        # Shared weights (identical across ranks). Separate leaves for the CP path
        # and the single-rank reference so their .grad don't collide.
        w_in0 = torch.randn(c, c, dtype=torch.float64)
        w_scan0 = torch.randn(c, c, dtype=torch.float64)
        w_out0 = torch.randn(c, c, dtype=torch.float64)

        def leaves():
            return (
                w_in0.clone().requires_grad_(True),
                w_scan0.clone().requires_grad_(True),
                w_out0.clone().requires_grad_(True),
            )

        # --- Reference: full sequence on one rank, loss over ALL positions ---
        x_ref = x_full.clone().requires_grad_(True)
        w_in_r, w_scan_r, w_out_r = leaves()
        out_ref = _toy_gdn_like(x_ref, w_in_r, w_scan_r, w_out_r)
        loss_ref = out_ref.pow(2).sum()
        loss_ref.backward()

        # --- CP path on this rank: shard -> project -> all-gather -> reassemble
        #     -> scan -> re-slice -> output proj; loss over this rank's shard ---
        x_local = _shard_full_like_relax(full_samples, cp_size, rank).clone().requires_grad_(True)
        w_in_c, w_scan_c, w_out_c = leaves()

        qkvzba_local = x_local @ w_in_c
        gathered = _AllGatherFullSequence.apply(qkvzba_local, dist.group.WORLD)
        full = gdn_reassemble_full(gathered, cu_seqlens, cp_size)
        scanned = torch.cumsum(full, dim=0) @ w_scan_c
        shard = gdn_cp_slice(scanned, cu_seqlens, cp_size, rank)
        out_cp = shard @ w_out_c
        loss_cp = out_cp.pow(2).sum()
        loss_cp.backward()

        # (1) forward parity: this rank's output == the rank's slice of the ref output
        out_ref_shard = gdn_cp_slice(out_ref.detach(), cu_seqlens, cp_size, rank)
        assert torch.allclose(out_cp.detach(), out_ref_shard, atol=1e-9), f"rank{rank}: forward mismatch"

        # (2) input-grad parity: x_local.grad == the rank's slice of dL/dx_full.
        #     Requires the reduce-scatter backward to sum cross-rank contributions.
        x_grad_ref_shard = gdn_cp_slice(x_ref.grad, cu_seqlens, cp_size, rank)
        assert torch.allclose(x_local.grad, x_grad_ref_shard, atol=1e-9), (
            f"rank{rank}: input-grad mismatch — cross-CP-rank gradient dropped (reduce-scatter backward missing?)"
        )

        # (3) param-grad parity: DDP sums param grads across ranks (all-reduce),
        #     which must equal the full reference param grads.
        for name, g_cp, g_ref in (
            ("w_in", w_in_c.grad.clone(), w_in_r.grad),
            ("w_scan", w_scan_c.grad.clone(), w_scan_r.grad),
            ("w_out", w_out_c.grad.clone(), w_out_r.grad),
        ):
            dist.all_reduce(g_cp, op=dist.ReduceOp.SUM)
            assert torch.allclose(g_cp, g_ref, atol=1e-9), f"rank{rank}: {name} grad mismatch"
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gdn_cp_forward_backward_parity(world_size, tmp_path):
    """CP=world_size reproduces the CP=1 (full-sequence) forward *and*
    gradients:

    output, input grad, and (all-reduced) parameter grads.
    """
    import torch.multiprocessing as mp

    port = _free_port()
    mp.spawn(_parity_worker, args=(world_size, port, str(tmp_path)), nprocs=world_size, join=True)

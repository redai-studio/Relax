# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""NCCL + real sparse CuTe CP parity, two GPUs and a bounded timeout."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.cuda.get_device_capability()[0] != 10,
    reason="Two Blackwell GPUs required for real CuTe CP/NCCL validation",
)


def _worker(rank, rendezvous):
    from relax.models.qwen3_8.ops.qsa_cute.adapter import sparse_attention
    from relax.models.qwen3_8.ops.qsa_cute.collectives import gather_sequence
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import qsa_cp_layout_static

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(23)
        device = torch.device("cuda", rank)
        cu = torch.tensor([0, 16, 32], device=device)
        rows, reorder = qsa_cp_layout_static(cu, 16, 2, rank, "per_document")
        shapes = ((32, 12, 256), (32, 1, 256), (32, 1, 256))
        full = [torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in shapes]
        q, k, v = [tensor[rows].detach().requires_grad_() for tensor in full]
        kg = gather_sequence(k, dist.group.WORLD, reorder)
        vg = gather_sequence(v, dist.group.WORLD, reorder)
        ids = torch.arange(4, device=device)[None].expand(16, -1)
        ids = torch.where(ids < ((rows % 16 + 1) // 4)[:, None], ids, -1).int()
        out = sparse_attention(q[None], kg[None], vg[None], ids[None], (0, 16, 32), 256**-0.5, rows)[0]
        out.float().square().sum().backward()
        qr, kr, vr = [tensor.detach().float().requires_grad_() for tensor in full]
        positions = torch.arange(32, device=device)
        mask = (positions[:, None] >= positions[None]) & (positions[:, None] // 16 == positions[None] // 16)
        logits = torch.einsum("qhd,khd->hqk", qr, kr.repeat_interleave(12, dim=1)) * 256**-0.5
        reference = torch.einsum(
            "hqk,khd->qhd", logits.masked_fill(~mask, -torch.inf).softmax(-1), vr.repeat_interleave(12, dim=1)
        )
        reference.square().sum().backward()
        torch.testing.assert_close(out.float(), reference[rows], atol=0.035, rtol=0.035)
        for actual, expected in zip((q, k, v), (qr, kr, vr)):
            torch.testing.assert_close(actual.grad.float(), expected.grad[rows], atol=0.15, rtol=0.06)
    finally:
        dist.destroy_process_group()


def test_qsa_cute_cp_remote_kv_gradients(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "nccl").as_uri(),), nprocs=2, join=True)

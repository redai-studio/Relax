# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two-rank Gloo regression for remote K/V gradient ownership (no CUDA)."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    import sys
    from pathlib import Path
    from types import ModuleType

    # Only bypass unrelated optional bridge registration in each fresh process.
    package = ModuleType("relax.models")
    package.__path__ = [str(Path(__file__).resolve().parents[4] / "relax/models")]
    sys.modules["relax.models"] = package
    from relax.models.qwen3_8.ops.qsa_cute.collectives import gather_sequence
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import qsa_cp_layout_static

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        group = dist.group.WORLD
        rows, reorder = qsa_cp_layout_static(torch.tensor([0, 8, 16]), 8, 2, rank, "per_document")
        generator = torch.Generator().manual_seed(11)
        full_q, full_k, full_v = [
            torch.randn(16, 4, generator=generator, dtype=torch.float64, requires_grad=True) for _ in range(3)
        ]
        q, k, v = [tensor.detach()[rows].requires_grad_() for tensor in (full_q, full_k, full_v)]
        global_k = gather_sequence(k, group, reorder)
        global_v = gather_sequence(v, group, reorder)
        keys = torch.arange(16)
        mask = (keys[None] <= rows[:, None]) & (keys[None] // 8 == rows[:, None] // 8)
        actual = (q @ global_k.T).masked_fill(~mask, -torch.inf).softmax(-1) @ global_v
        actual.square().sum().backward()
        full_mask = (keys[None] <= keys[:, None]) & (keys[None] // 8 == keys[:, None] // 8)
        expected = (full_q @ full_k.T).masked_fill(~full_mask, -torch.inf).softmax(-1) @ full_v
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected[rows])
        for local, global_tensor in zip((q, k, v), (full_q, full_k, full_v)):
            torch.testing.assert_close(local.grad, global_tensor.grad[rows])
    finally:
        dist.destroy_process_group()


def test_qsa_cp_all_gather_backward_sums_remote_queries(tmp_path):
    # One bounded two-process test is necessary: mocked collectives cannot expose
    # lost remote gradients or an erroneous CP averaging factor.
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)

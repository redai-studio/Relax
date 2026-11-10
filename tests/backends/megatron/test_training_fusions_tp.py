# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two-rank CUDA contracts: vocabulary dgrad SUM and differentiable PLE
halo."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


gpu_pair = pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.cuda.get_device_capability()[0] != 10,
    reason="TP fusion parity requires two Blackwell GPUs and NCCL",
)


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    try:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        from relax.backends.megatron.fused_linear_ce import linear_cross_entropy

        torch.manual_seed(84)
        full_hidden = torch.randn(32, 1, 64, device="cuda", dtype=torch.bfloat16)
        full_weight = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        labels = torch.randint(512, (32, 1), device="cuda")
        mask = torch.rand(32, 1, device="cuda")
        mask[::4] = 0
        reference_hidden = full_hidden.clone().requires_grad_()
        reference_weight = full_weight.clone().requires_grad_()
        logits = F.linear(reference_hidden, reference_weight)
        reference_loss = F.cross_entropy(logits[:, 0].float(), labels[:, 0], reduction="none").unsqueeze(1)
        (reference_loss * mask).sum().backward()
        hidden = full_hidden.chunk(2)[rank].clone().requires_grad_()
        weight = full_weight.chunk(2)[rank].clone().requires_grad_()
        gathered = gather_from_sequence_parallel_region(hidden, tensor_parallel_output_grad=False, group=group)
        loss, _ = linear_cross_entropy(gathered, weight, labels, group)
        (loss * mask).sum().backward()
        torch.testing.assert_close(loss, reference_loss, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(
            hidden.grad.float(), reference_hidden.grad.chunk(2)[rank].float(), atol=0.004, rtol=0.04
        )
        torch.testing.assert_close(
            weight.grad.float(), reference_weight.grad.chunk(2)[rank].float(), atol=0.004, rtol=0.04
        )

        source = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/ple_fused_conv.py"
        spec = importlib.util.spec_from_file_location("ple_tp_test", source)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        from torch.distributed.nn.functional import all_gather

        torch.manual_seed(86)
        full = torch.randn(32, 1, 16, device="cuda", dtype=torch.bfloat16)
        gated = torch.randn_like(full)
        conv_weight = torch.randn(16, 1, 3, device="cuda", dtype=torch.bfloat16)
        positions = (torch.arange(32, device="cuda") % 23).unsqueeze(1)
        reference = full.clone().requires_grad_()
        padded = F.pad(reference, (0, 0, 0, 0, 4, 0))
        wanted = patch.convolution_values(padded, reference, gated, conv_weight, positions, 2)
        wanted.float().square().sum().backward()
        local = full.chunk(2)[rank].clone().requires_grad_()
        tails = all_gather(local[-4:].contiguous(), group=group)
        # BOTH ranks must reach the gather backward, including rank zero.
        previous = tails[rank - 1] if rank else tails[-1] * 0.0
        extended = torch.cat((previous, local), 0)
        actual = patch._compiled_conv()(
            extended, local, gated.chunk(2)[rank], conv_weight, positions.chunk(2)[rank], 2
        )
        actual.float().square().sum().backward()
        torch.testing.assert_close(actual, wanted.chunk(2)[rank], rtol=0, atol=0)
        torch.testing.assert_close(local.grad.float(), reference.grad.chunk(2)[rank].float(), atol=0.04, rtol=0.02)
    finally:
        dist.destroy_process_group()


@gpu_pair
def test_training_fusions_tp_sp_and_halo_backward(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)

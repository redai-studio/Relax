# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Loss/gradient checks, including real CPU distributed communication."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.training import embedding
from tests.cpu.backends.megatron.test_rloo_policy_loss_dispatch import loss_module


def test_default_pooling_uses_last_real_token_and_excludes_tail_padding():
    features = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3).requires_grad_()
    pooled = embedding.pool_packed_embeddings(features, [4, 2])
    torch.testing.assert_close(pooled, features[0, [3, 5]], rtol=0, atol=0)
    pooled.sum().backward()
    assert torch.count_nonzero(features.grad).item() == 6


def _distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        generator = torch.Generator().manual_seed(123)
        all_query = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        all_positive = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        query = all_query[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        positive = all_positive[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        loss = loss_module.infonce_loss(query, positive, process_group=dist.group.WORLD)
        adapted = embedding.scale_for_megatron_sample_mean(loss, global_pairs=4, dp_size=2)
        # The existing shared loss wrapper's DP/global_batch multiplier.
        (adapted * 2 / 4).backward()
        gradient = torch.cat([query.grad, positive.grad], dim=0).sum(dim=0)
        dist.all_reduce(gradient, group=dist.group.WORLD)
        gradient /= 2  # Existing backend's averaged parameter-gradient contract.
        torch.save({"gradient": gradient, "loss": loss.detach()}, Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo unavailable")
def test_two_rank_real_gather_matches_native_averaged_gradient(tmp_path):
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.start_processes(
        _distributed_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True, start_method="spawn"
    )
    generator = torch.Generator().manual_seed(123)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    query = torch.randn(4, 3, generator=generator, dtype=torch.float64) + bias
    positive = torch.randn(4, 3, generator=generator, dtype=torch.float64) + bias
    reference = loss_module.infonce_loss(query, positive)
    (reference / 2).backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["loss"], reference.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["gradient"], bias.grad, rtol=1e-12, atol=1e-12)

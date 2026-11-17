# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Loss/gradient checks, including real CPU distributed communication."""

from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from relax.utils.sft_utils import align_loss_mask_for_sft
from relax.utils.training import embedding
from tests.cpu.backends.megatron.test_rloo_policy_loss_dispatch import loss_module


def test_default_pooling_uses_last_real_token_and_excludes_tail_padding():
    features = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3).requires_grad_()
    pooled = embedding.pool_packed_embeddings(features, [4, 2])
    torch.testing.assert_close(pooled, features[0, [3, 5]], rtol=0, atol=0)
    pooled.sum().backward()
    assert torch.count_nonzero(features.grad).item() == 6


@pytest.mark.parametrize("tie_weights", [False, True])
def test_freezing_output_head_preserves_input_embedding(tie_weights):
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(4, 3)
    model.output_layer = torch.nn.Linear(3, 4)
    if tie_weights:
        model.output_layer.weight = model.embedding.weight
    embedding.freeze_embedding_unused_heads(model)
    model.embedding(torch.tensor([0])).sum().backward()
    assert model.embedding.weight.grad is not None
    assert model.output_layer.weight.requires_grad == tie_weights
    assert not model.output_layer.bias.requires_grad


def _pair_inputs():
    return torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(123), dtype=torch.float64)


def _distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        group = dist.group.WORLD
        for mpu in (loss_module.mpu, loss_module.get_sum_of_sample_mean.__globals__["mpu"]):
            mpu.get_context_parallel_world_size = lambda: 1
            mpu.get_data_parallel_group = lambda **_kw: group
            mpu.get_data_parallel_world_size = lambda **_kw: 2

        encoder = torch.nn.Linear(3, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            encoder.weight.copy_(torch.eye(3, dtype=torch.float64))
        ddp = torch.nn.parallel.DistributedDataParallel(encoder, process_group=group)
        hidden = ddp(_pair_inputs()[rank * 2 : (rank + 1) * 2]).reshape(1, 4, 3)
        args = Namespace(
            loss_type="sft",
            task_type="embedding",
            global_batch_size=4,
            contrastive_temperature=0.05,
            qkv_format="thd",
            calculate_per_token_loss=False,
            allgather_cp=False,
            recompute_loss_function=False,
        )
        batch = dict(
            total_lengths=[1] * 4,
            response_lengths=[1] * 4,
            loss_masks=[align_loss_mask_for_sft(torch.ones(1, dtype=torch.int))] * 4,
        )
        loss, _, metrics = loss_module.loss_function(args, batch, 1, hidden)
        loss.backward()
        torch.save(
            {
                "gradient": encoder.weight.grad,
                "loss": loss.detach(),
                "logged_loss": metrics["values"][1] / 2,
                "signal": metrics["num_tokens"],
            },
            Path(output) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo unavailable")
def test_two_rank_ddp_matches_full_batch_gradient_with_single_token_pairs(tmp_path):
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.start_processes(
        _distributed_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True, start_method="spawn"
    )
    weight = torch.eye(3, dtype=torch.float64, requires_grad=True)
    query, positive = (_pair_inputs() @ weight.T).unbind(dim=1)
    reference = F.cross_entropy(F.cosine_similarity(query[:, None], positive[None], dim=-1) / 0.05, torch.arange(4))
    reference.backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["loss"], reference.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["logged_loss"], reference.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["gradient"], weight.grad, rtol=1e-12, atol=1e-12)
        assert actual["signal"].item() == 2

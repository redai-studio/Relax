# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Loss/gradient checks, including real CPU distributed communication."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[3] / "relax/utils/training/retrieval_utils.py"


def load_primitives():
    spec = importlib.util.spec_from_file_location("retrieval_primitives_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_rank_loss_and_gradients_match_reference():
    module = load_primitives()
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    positive = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    loss = module.retrieval_infonce(query, positive)
    actual_gradients = torch.autograd.grad(loss, (query, positive))
    reference = F.cross_entropy(
        F.cosine_similarity(F.normalize(query, dim=-1)[:, None], F.normalize(positive, dim=-1)[None], dim=-1) / 0.05,
        torch.arange(4),
    )
    expected_gradients = torch.autograd.grad(reference, (query, positive))
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pooling_uses_first_emb_preceding_state_and_excludes_tail_padding():
    module = load_primitives()
    features = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3).requires_grad_()
    pooled = module.select_retrieval_features(features, [torch.tensor([1, 2, 9, 9]), torch.tensor([3, 9])], 9)
    torch.testing.assert_close(pooled, features[0, [1, 4]], rtol=0, atol=0)
    pooled.sum().backward()
    assert torch.count_nonzero(features.grad).item() == 6


@pytest.mark.parametrize("tokens", [[torch.tensor([9, 1]), torch.tensor([2, 9])], [torch.tensor([1, 2])] * 2])
def test_pooling_rejects_missing_or_initial_emb(tokens):
    with pytest.raises(RuntimeError, match="<emb>"):
        load_primitives().select_retrieval_features(torch.randn(1, 4, 3), tokens, 9)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        load_primitives().retrieval_infonce(torch.randn(2, 3), torch.randn(2, 3), temperature=temperature)


def _distributed_worker(rank, rendezvous, output):
    module = load_primitives()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        generator = torch.Generator().manual_seed(123)
        all_query = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        all_positive = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        query = all_query[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        positive = all_positive[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        loss = module.retrieval_infonce(query, positive, process_group=dist.group.WORLD)
        adapted = module.scale_for_megatron_sample_mean(loss, global_pairs=4, dp_size=2)
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
    reference = load_primitives().retrieval_infonce(query, positive)
    (reference / 2).backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual["loss"], reference.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["gradient"], bias.grad, rtol=1e-12, atol=1e-12)


def test_rejects_fractional_dp_batch_before_scaling():
    with pytest.raises(ValueError, match="divide evenly"):
        load_primitives().scale_for_megatron_sample_mean(torch.tensor(1.0), global_pairs=7, dp_size=2)


def test_pooling_accounts_for_padding_between_packed_sequences():
    features = torch.arange(36, dtype=torch.float32).reshape(1, 12, 3)
    pooled = load_primitives().select_retrieval_features(
        features,
        [torch.tensor([1, 9]), torch.tensor([2, 3, 9])],
        9,
        padded_total_lengths=[4, 4],
    )
    torch.testing.assert_close(pooled, features[0, [0, 5]], rtol=0, atol=0)


def test_rejects_splitting_the_global_negative_set_into_microbatches():
    with pytest.raises(ValueError, match="complete contrastive microbatch"):
        load_primitives().scale_for_megatron_sample_mean(
            torch.tensor(1.0), global_pairs=8, dp_size=2, num_microbatches=2
        )

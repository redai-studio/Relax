# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RLOO invariance through the production context-parallel reducer."""

from __future__ import annotations

from datetime import timedelta

import pytest


torch = pytest.importorskip("torch")


def _response_slices(total_length: int, response_length: int, rank: int, world_size: int) -> list[slice]:
    """Derive the two shifted-token CP response slices independently."""
    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * world_size - 1) // (2 * world_size)
    chunk_ids = (rank, 2 * world_size - rank - 1)
    slices = []
    for chunk_id in chunk_ids:
        chunk_start = chunk_id * chunk_size
        chunk_end = (chunk_id + 1) * chunk_size
        logit_start = max(chunk_start, prompt_length - 1)
        logit_end = min(chunk_end, total_length - 1)
        if logit_start < logit_end:
            slices.append(slice(logit_start + 1 - prompt_length, logit_end + 1 - prompt_length))
    return slices


def _rloo_cp_worker(rank, world_size, init_file):
    import torch.distributed as dist

    from relax.backends.megatron.cp_utils import get_cp_local_num_tokens, get_sum_of_sample_mean
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(0)
        base_length = 2 * world_size * 4
        response_lengths = [base_length, base_length * 2, base_length * 3]
        prompt_lengths = [2 * world_size * (sample_index + 1) for sample_index in range(3)]
        total_lengths = [
            prompt_length + response_length
            for prompt_length, response_length in zip(prompt_lengths, response_lengths, strict=True)
        ]
        rewards = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
        advantages = compute_rloo_leave_one_out_rewards(rewards)
        full_log_probs = [torch.randn(length, dtype=torch.float64) for length in response_lengths]
        full_masks = [
            ((torch.arange(length) + sample_index) % 3 != 0).to(torch.float64)
            for sample_index, length in enumerate(response_lengths)
        ]
        full_losses = [
            -(advantage * log_probs) for advantage, log_probs in zip(advantages, full_log_probs, strict=True)
        ]

        reference = sum((loss * mask).sum() for loss, mask in zip(full_losses, full_masks, strict=True))
        sample_mean_reference = sum(
            (loss * mask).sum() / torch.clamp_min(mask.sum(), 1)
            for loss, mask in zip(full_losses, full_masks, strict=True)
        )

        local_losses = []
        local_ownership = []
        for total_length, response_length, full_loss in zip(total_lengths, response_lengths, full_losses, strict=True):
            response_slices = _response_slices(total_length, response_length, rank, world_size)
            local_losses.append(torch.cat([full_loss[response_slice] for response_slice in response_slices]))
            ownership = torch.zeros(response_length, dtype=torch.int64)
            for response_slice in response_slices:
                ownership[response_slice] += 1
            local_ownership.append(ownership)

        local_reducer = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            full_masks,
            calculate_per_token_loss=True,
            dynamic_cp_size=world_size,
            dynamic_cp_rank=rank,
        )
        reduced = local_reducer(torch.cat(local_losses))
        sample_mean_reducer = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            full_masks,
            calculate_per_token_loss=False,
            dynamic_cp_size=world_size,
            dynamic_cp_rank=rank,
        )
        sample_mean_reduced = sample_mean_reducer(torch.cat(local_losses))
        local_num_tokens = get_cp_local_num_tokens(
            total_lengths,
            response_lengths,
            full_masks,
            dynamic_cp_size=world_size,
            dynamic_cp_rank=rank,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_mean_reduced, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_num_tokens, op=dist.ReduceOp.SUM)

        reference_num_tokens = sum(mask.sum() for mask in full_masks)
        assert torch.allclose(reduced, reference, atol=1e-10), (
            f"CP={world_size} rank={rank}: reduced={reduced.item()} != reference={reference.item()}"
        )
        assert torch.equal(local_num_tokens, reference_num_tokens), (
            f"CP={world_size} rank={rank}: num_tokens={local_num_tokens.item()} "
            f"!= reference_num_tokens={reference_num_tokens.item()}"
        )
        assert torch.allclose(sample_mean_reduced, sample_mean_reference, atol=1e-10), (
            f"CP={world_size} rank={rank}: sample_mean={sample_mean_reduced.item()} "
            f"!= sample_mean_reference={sample_mean_reference.item()}"
        )

        for ownership in local_ownership:
            dist.all_reduce(ownership, op=dist.ReduceOp.SUM)
            assert torch.equal(ownership, torch.ones_like(ownership))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo is required",
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_rloo_cp_production_reducer_matches_unsplit(tmp_path, world_size):
    import torch.multiprocessing as mp

    init_file = tmp_path / f"gloo-{world_size}"
    mp.spawn(
        _rloo_cp_worker,
        args=(world_size, str(init_file)),
        nprocs=world_size,
        join=True,
    )

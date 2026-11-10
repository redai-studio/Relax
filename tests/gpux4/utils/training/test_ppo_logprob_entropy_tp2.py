# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-rank CUDA parity for fused PPO log-probability and entropy."""

import os

import pytest
import torch

from tests.utils.training.test_ppo_logprob_entropy_gpu import (
    _assert_legacy_parity,
    _free_port,
    _full_logits,
    _keep_mask,
)


PARITY_SCENARIOS = [
    (-1, False, False, False),
    (-1, False, True, False),
    (-1, False, True, True),
    (-1, True, False, False),
    (-1, True, True, False),
    (-1, True, True, True),
    (2, False, False, False),
    (2, False, True, False),
    (2, False, True, True),
    (2, True, False, False),
    (2, True, True, False),
    (2, True, True, True),
]


def _tp2_worker(rank: int, world_size: int, master_port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        process_group = dist.group.WORLD
        device = torch.device("cuda", rank)
        full_logits = _full_logits().to(device=device)
        full_keep_mask = _keep_mask().to(device=device)
        tokens = torch.tensor([5, 0, 3, 2], dtype=torch.long, device=device)

        vocab_per_rank = full_logits.size(-1) // world_size
        vocab_start = rank * vocab_per_rank
        vocab_end = vocab_start + vocab_per_rank
        for chunk_size, with_mask, with_entropy, entropy_has_grad in PARITY_SCENARIOS:
            logits = full_logits[:, vocab_start:vocab_end].detach().clone().requires_grad_()
            keep_mask = full_keep_mask[:, vocab_start:vocab_end] if with_mask else None
            _assert_legacy_parity(
                process_group=process_group,
                device=device,
                logits=logits,
                tokens=tokens,
                keep_mask=keep_mask,
                chunk_size=chunk_size,
                with_entropy=with_entropy,
                entropy_has_grad=entropy_has_grad,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_calculate_log_probs_and_entropy_matches_legacy_megatron_cuda_tp2():
    pytest.importorskip("megatron.core.fusions.fused_cross_entropy")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 parity requires two CUDA devices")

    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_nccl_available():
        pytest.skip("NCCL is required")

    world_size = 2
    mp.spawn(
        _tp2_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )

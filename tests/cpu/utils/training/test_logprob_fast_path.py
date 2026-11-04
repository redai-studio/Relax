# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check opt-in logprob kernels against dense autograd over real CPU TP
groups."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.training import ppo_utils


def _worker(rank: int, world: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    group = dist.group.WORLD
    try:
        for entropy_enabled in (False, True):
            for entropy_grad in (False, True):
                for masked in (False, True):
                    torch.manual_seed(19)
                    full = torch.randn(7, 16)
                    labels = torch.tensor([0, 8, 15, 3, 9, 6, 12])
                    keep = torch.rand(7, 16) > 0.3 if masked else None
                    start, end = rank * (16 // world), (rank + 1) * (16 // world)
                    results = []
                    for fast in (False, True):
                        ppo_utils._LOGPROB_FAST_PATH = fast
                        logits = full[:, start:end].clone().requires_grad_()
                        lp, entropy = ppo_utils.calculate_log_probs_and_entropy(
                            logits,
                            labels,
                            group,
                            with_entropy=entropy_enabled,
                            with_entropy_grad=entropy_grad,
                            chunk_size=3,
                            log_prob_keep_mask=keep[:, start:end] if keep is not None else None,
                        )
                        weights = torch.linspace(-1, 1, 7).reshape(-1, 1)
                        objective = (lp * weights).sum()
                        if entropy_enabled and entropy_grad:
                            objective = objective + 0.3 * entropy.sum()
                        objective.backward()
                        results.append((lp.detach(), entropy.detach() if entropy is not None else None, logits.grad))
                    reference = full.clone().requires_grad_()
                    logit_view = reference
                    if keep is not None:
                        effective_keep = keep.clone()
                        effective_keep[torch.arange(7), labels] = True
                        logit_view = reference.masked_fill(~effective_keep, -torch.inf)
                    expected_lp = logit_view.log_softmax(-1).gather(-1, labels[:, None])
                    expected_entropy = -(reference.softmax(-1) * reference.log_softmax(-1)).sum(-1, keepdim=True)
                    expected_objective = (expected_lp * weights).sum()
                    if entropy_enabled and entropy_grad:
                        expected_objective = expected_objective + 0.3 * expected_entropy.sum()
                    expected_objective.backward()
                    torch.testing.assert_close(results[1][0], expected_lp, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(results[1][2], reference.grad[:, start:end], rtol=1e-5, atol=2e-7)
                    for actual, expected in zip(results[1], results[0], strict=True):
                        if expected is None:
                            assert actual is None
                        else:
                            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("world", [1, 2])
def test_logprob_fast_path_forward_backward_tp(tmp_path: Path, world: int) -> None:
    mp.spawn(_worker, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)


def test_single_rank_collectives_are_elided_without_fast_path(monkeypatch):
    monkeypatch.setattr(ppo_utils, "_LOGPROB_FAST_PATH", False)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ppo_utils, "_get_vocab_parallel_rank_size", lambda group: (0, 1))

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("single-rank SUM/MAX must not enqueue a collective")

    monkeypatch.setattr(dist, "all_reduce", unexpected_collective)
    value = torch.tensor([2.0, -3.0])
    for operation in (dist.ReduceOp.SUM, dist.ReduceOp.MAX):
        ppo_utils._maybe_all_reduce(value, operation, object())
        torch.testing.assert_close(value, torch.tensor([2.0, -3.0]))

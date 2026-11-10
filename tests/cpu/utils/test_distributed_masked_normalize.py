# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.utils.distributed_utils import distributed_masked_normalize
from relax.utils.training.ppo_utils import get_reinforce_plus_plus_returns


def _distributed_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        if rank == 0:
            values = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)
            mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
            expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64) / torch.sqrt(
                torch.tensor(5.0, dtype=torch.float64)
            )
        else:
            values = torch.tensor([-1.0, 5.0], dtype=torch.float64)
            mask = torch.ones(2, dtype=torch.float64)
            expected = torch.tensor([-3.0, 3.0], dtype=torch.float64) / torch.sqrt(
                torch.tensor(5.0, dtype=torch.float64)
            )

        normalized, mean, variance, count = distributed_masked_normalize(values, mask)

        torch.testing.assert_close(mean, torch.tensor(2.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(variance, torch.tensor(5.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(count, torch.tensor(4.0, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(normalized, expected, atol=1e-12, rtol=1e-12)

        # Rank 0 has no valid token in this second collective; it must still
        # participate and receive the moments produced by rank 1.
        values = (
            torch.tensor([float("inf")], dtype=torch.float64)
            if rank == 0
            else torch.tensor([2.0, 4.0], dtype=torch.float64)
        )
        mask = torch.zeros_like(values) if rank == 0 else torch.ones_like(values)
        normalized, mean, variance, count = distributed_masked_normalize(values, mask)
        torch.testing.assert_close(mean, torch.tensor(3.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(variance, torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(count, torch.tensor(2.0, dtype=torch.float64), atol=0, rtol=0)
        if rank == 0:
            torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=0, rtol=0)
        else:
            torch.testing.assert_close(normalized, torch.tensor([-1.0, 1.0], dtype=torch.float64), atol=1e-12, rtol=0)

        values = (
            torch.tensor([1.0, -1.0], dtype=torch.float64)
            if rank == 0
            else torch.tensor([3.0, 5.0], dtype=torch.float64)
        )
        mask = torch.ones_like(values)
        normalized, mean, variance, count = distributed_masked_normalize(values, mask)
        expected = (
            torch.tensor([-1.0, -3.0], dtype=torch.float64)
            if rank == 0
            else torch.tensor([1.0, 3.0], dtype=torch.float64)
        ) / torch.sqrt(torch.tensor(5.0, dtype=torch.float64))
        torch.testing.assert_close(mean, torch.tensor(2.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(variance, torch.tensor(5.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(count, torch.tensor(4.0, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(normalized, expected, atol=1e-12, rtol=1e-12)

        values = torch.tensor([7.0], dtype=torch.float64)
        mask = torch.tensor([1.0 if rank == 0 else 0.0], dtype=torch.float64)
        normalized, mean, variance, count = distributed_masked_normalize(values, mask)
        torch.testing.assert_close(mean, torch.tensor(7.0, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(variance, torch.tensor(0.0, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(count, torch.tensor(1.0, dtype=torch.float64), atol=0, rtol=0)
        torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=0, rtol=0)

        # A rank with no valid response tokens must reach the same collective
        # as ranks with valid tokens. Otherwise the valid rank hangs in all_reduce.
        megatron = ModuleType("megatron")
        megatron_core = ModuleType("megatron.core")
        megatron_core.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1)
        megatron.core = megatron_core
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = megatron_core

        local_mask = torch.zeros(2, dtype=torch.float64) if rank == 0 else torch.ones(2, dtype=torch.float64)
        local_kl = (
            torch.tensor([float("inf"), float("nan")], dtype=torch.float64)
            if rank == 0
            else torch.tensor([0.2, -0.1], dtype=torch.float64)
        )
        local_returns = get_reinforce_plus_plus_returns(
            rewards=torch.tensor([10.0 if rank == 0 else 1.0], dtype=torch.float64),
            kl=[local_kl],
            loss_masks=[local_mask],
            response_lengths=[2],
            total_lengths=[2],
            kl_coef=0.1,
            gamma=1.0,
        )[0]
        normalized, mean, variance, count = distributed_masked_normalize(local_returns, local_mask)

        if rank == 0:
            torch.testing.assert_close(local_returns, torch.zeros_like(local_returns), atol=0, rtol=0)
            torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=0, rtol=0)
        else:
            torch.testing.assert_close(local_returns, torch.tensor([0.99, 1.01], dtype=torch.float64))
            torch.testing.assert_close(normalized, torch.tensor([-1.0, 1.0], dtype=torch.float64))
        torch.testing.assert_close(mean, torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(variance, torch.tensor(1e-4, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(count, torch.tensor(2.0, dtype=torch.float64), atol=0, rtol=0)

        with pytest.raises(RuntimeError, match="global mask sum"):
            distributed_masked_normalize(torch.tensor([float(rank)]), torch.zeros(1))
    finally:
        dist.destroy_process_group()


def test_real_gloo_population_statistics_across_two_ranks(tmp_path):
    init_method = f"file://{tmp_path / 'gloo-init'}"
    mp.spawn(_distributed_worker, args=(2, init_method), nprocs=2, join=True)


def test_population_normalize_validates_inputs():
    with pytest.raises(ValueError, match="same shape"):
        distributed_masked_normalize(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="variance_floor must be positive"):
        distributed_masked_normalize(torch.ones(2), torch.ones(2), variance_floor=0.0)


def test_population_normalize_does_not_extract_host_scalar():
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(distributed_masked_normalize)
    tree = ast.parse(textwrap.dedent(source))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "item"
        for node in ast.walk(tree)
    )
    assert "torch._assert_async" in source

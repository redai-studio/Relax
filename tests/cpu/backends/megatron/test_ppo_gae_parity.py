# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType

import pytest


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


def test_chunked_gae_matches_vanilla_gae(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_megatron(monkeypatch)

    from relax.utils.training.ppo_utils import chunked_gae, vanilla_gae

    torch.manual_seed(0)
    rewards = torch.randn(3, 257, dtype=torch.float32)
    values = torch.randn(3, 257, dtype=torch.float32)

    expected_advantages, expected_returns = vanilla_gae(rewards, values, gamma=0.99, lambd=0.95)
    actual_advantages, actual_returns = chunked_gae(rewards, values, gamma=0.99, lambd=0.95, chunk_size=32)

    assert torch.allclose(actual_advantages, expected_advantages, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_returns, expected_returns, atol=1e-5, rtol=1e-5)


def test_get_advantages_and_returns_batch_shapes(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_megatron(monkeypatch)

    from relax.utils.training.ppo_utils import get_advantages_and_returns_batch

    values = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
    rewards = [torch.tensor([1.0, 0.0, 0.5]), torch.tensor([0.25, 0.75])]

    advantages, returns = get_advantages_and_returns_batch(
        total_lengths=[6, 5],
        response_lengths=[3, 2],
        values_list=values,
        rewards_list=rewards,
        gamma=0.99,
        lambd=0.95,
    )

    assert [item.shape for item in advantages] == [torch.Size([3]), torch.Size([2])]
    assert [item.shape for item in returns] == [torch.Size([3]), torch.Size([2])]

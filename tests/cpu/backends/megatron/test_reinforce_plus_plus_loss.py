# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch

from relax.backends.megatron import cp_utils
from relax.utils.training.ppo_utils import compute_approx_kl as _compiled_compute_approx_kl
from relax.utils.training.ppo_utils import compute_policy_loss as _compiled_compute_policy_loss


compute_approx_kl = torch.compiler.disable(_compiled_compute_approx_kl)
compute_policy_loss = torch.compiler.disable(_compiled_compute_policy_loss)


def _response_mean(values: torch.Tensor, response_lengths: list[int], masks: list[torch.Tensor]) -> torch.Tensor:
    chunks = values.split(response_lengths)
    per_response = [(chunk * mask).sum() / mask.sum() for chunk, mask in zip(chunks, masks, strict=True)]
    return torch.stack(per_response).mean()


def test_token_and_response_reduced_policy_and_k2_losses_match_reference(monkeypatch):
    monkeypatch.setattr(cp_utils, "mpu", SimpleNamespace(get_context_parallel_world_size=lambda: 1))

    current = torch.tensor([-0.1, -1.0, -0.4, -0.7, -1.5], dtype=torch.float64)
    old = torch.tensor([-0.2, -0.6, -0.6, -0.8, -1.1], dtype=torch.float64)
    reference = torch.tensor([-0.3, -0.9, -0.1, -1.0, -1.2], dtype=torch.float64)
    advantages = torch.tensor([1.0, -2.0, 0.25, -0.5, 3.0], dtype=torch.float64)
    masks = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])]
    response_lengths = [2, 3]

    ppo_kl = old - current
    token_pg, _ = compute_policy_loss(ppo_kl, advantages, 0.2, 0.2)
    token_k2 = compute_approx_kl(current, reference, "k2")
    reducer = cp_utils.get_sum_of_sample_mean(
        total_lengths=response_lengths,
        response_lengths=response_lengths,
        loss_masks=masks,
        calculate_per_token_loss=False,
    )

    actual_pg = reducer(token_pg) / len(response_lengths)
    actual_k2 = reducer(token_k2) / len(response_lengths)

    ratio = torch.exp(current - old)
    expected_token_pg = torch.maximum(
        -ratio * advantages,
        -torch.clamp(ratio, 0.8, 1.2) * advantages,
    )
    expected_token_k2 = 0.5 * (current.float() - reference.float()).square()
    expected_pg = _response_mean(expected_token_pg, response_lengths, masks)
    expected_k2 = _response_mean(expected_token_k2, response_lengths, masks)

    torch.testing.assert_close(token_pg, expected_token_pg, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(token_k2, expected_token_k2, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_pg, expected_pg, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_k2, expected_k2, atol=1e-6, rtol=1e-6)


def test_masked_tokens_cannot_change_reduced_loss(monkeypatch):
    monkeypatch.setattr(cp_utils, "mpu", SimpleNamespace(get_context_parallel_world_size=lambda: 1))
    masks = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])]
    reducer = cp_utils.get_sum_of_sample_mean([2, 3], [2, 3], masks)
    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    changed = torch.tensor([1.0, 20000.0, 3.0, 4.0, -50000.0])

    torch.testing.assert_close(reducer(base), reducer(changed), atol=0, rtol=0)


def test_nonfinite_masked_tokens_are_scoped_to_reinforce_plus_plus(monkeypatch):
    monkeypatch.setattr(cp_utils, "mpu", SimpleNamespace(get_context_parallel_world_size=lambda: 1))
    masks = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])]
    values = torch.tensor([1.0, float("nan"), 3.0, 4.0, float("inf")])

    response_mean = cp_utils.get_sum_of_sample_mean([2, 3], [2, 3], masks)
    token_sum = cp_utils.get_sum_of_sample_mean([2, 3], [2, 3], masks, calculate_per_token_loss=True)

    # The shared reducers retain their upstream behavior for GRPO/GSPO/SAPO.
    assert torch.isnan(response_mean(values))
    assert torch.isnan(token_sum(values))

    try:
        import megatron  # noqa: F401
    except ModuleNotFoundError:
        megatron = ModuleType("megatron")
        megatron_core = ModuleType("megatron.core")
        megatron_core.mpu = SimpleNamespace()
        megatron.core = megatron_core
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    loss_module = importlib.import_module("relax.backends.megatron.loss")

    safe_response_mean = loss_module._get_reinforce_plus_plus_mask_safe_reducer(response_mean, masks)
    safe_token_sum = loss_module._get_reinforce_plus_plus_mask_safe_reducer(token_sum, masks)
    torch.testing.assert_close(safe_response_mean(values), torch.tensor(4.5), atol=0, rtol=0)
    torch.testing.assert_close(safe_token_sum(values), torch.tensor(8.0), atol=0, rtol=0)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
import types

import pytest
import torch


def _load_ci_utils(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    distributed = types.ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = torch.nn.Module
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)
    sys.modules.pop("relax.backends.megatron.ci_utils", None)
    return importlib.import_module("relax.backends.megatron.ci_utils")


class _Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = torch.nn.Linear(4, 1)


def test_critic_value_head_ci_check_detects_parameter_update(monkeypatch):
    module = _load_ci_utils(monkeypatch)
    critic = _Critic()
    for param in critic.output_layer.parameters():
        param.main_grad = torch.ones_like(param)

    snapshot = module.capture_critic_value_head_update([critic])
    with torch.no_grad():
        critic.output_layer.weight.add_(0.1)

    module.assert_critic_value_head_updated(snapshot, update_successful=True, learning_rates=[1e-3])


def test_critic_value_head_ci_check_rejects_missing_update(monkeypatch):
    module = _load_ci_utils(monkeypatch)
    critic = _Critic()
    for param in critic.output_layer.parameters():
        param.main_grad = torch.ones_like(param)

    snapshot = module.capture_critic_value_head_update([critic])

    with pytest.raises(AssertionError, match="did not change"):
        module.assert_critic_value_head_updated(snapshot, update_successful=True, learning_rates=[1e-3])


def test_critic_value_head_ci_check_rejects_nonfinite_grad(monkeypatch):
    module = _load_ci_utils(monkeypatch)
    critic = _Critic()
    critic.output_layer.weight.main_grad = torch.full_like(critic.output_layer.weight, float("nan"))
    critic.output_layer.bias.main_grad = torch.ones_like(critic.output_layer.bias)

    with pytest.raises(AssertionError, match="non-finite"):
        module.capture_critic_value_head_update([critic])


def test_critic_value_head_ci_check_allows_skipped_or_zero_grad_step(monkeypatch):
    module = _load_ci_utils(monkeypatch)
    critic = _Critic()
    for param in critic.output_layer.parameters():
        param.main_grad = torch.zeros_like(param)

    snapshot = module.capture_critic_value_head_update([critic])
    module.assert_critic_value_head_updated(snapshot, update_successful=True, learning_rates=[1e-3])
    module.assert_critic_value_head_updated(snapshot, update_successful=False, learning_rates=[1e-3])

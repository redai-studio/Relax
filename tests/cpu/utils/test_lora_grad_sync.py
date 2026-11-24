# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapter gradient metadata checks without a Megatron runtime."""

import sys
from types import ModuleType

import pytest
import torch

from relax.utils.training.lora_grad_sync import inherit_lora_tp_grad_sync


class _LinearAdapter(torch.nn.Module):
    pass


@pytest.fixture(autouse=True)
def _bridge_linear_adapter(monkeypatch):
    module = ModuleType("megatron.bridge.peft.lora_layers")
    module.LinearAdapter = _LinearAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)


class _PlainLoRAWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.to_wrap = base
        self.adapter = _LinearAdapter()
        self.adapter.linear_in = torch.nn.Linear(4, 2, bias=False)
        self.adapter.linear_out = torch.nn.Linear(2, 4, bias=False)


def test_inherit_lora_tp_grad_sync_marks_replicated_hf_adapter() -> None:
    base = torch.nn.Linear(4, 4)
    base.requires_grad_(False)
    base.weight.average_gradients_across_tp_domain = True
    wrapper = _PlainLoRAWrapper(base)
    parameters = tuple(wrapper.adapter.parameters())

    assert inherit_lora_tp_grad_sync(wrapper) == 2

    assert all(actual is expected for actual, expected in zip(wrapper.adapter.parameters(), parameters, strict=True))
    assert all(parameter.average_gradients_across_tp_domain for parameter in parameters)
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(not getattr(parameter, "sequence_parallel", False) for parameter in parameters)
    assert not base.weight.requires_grad


def test_inherit_lora_tp_grad_sync_marks_sequence_parallel_adapter() -> None:
    base = torch.nn.Linear(4, 4)
    base.requires_grad_(False)
    base.weight.sequence_parallel = True
    wrapper = _PlainLoRAWrapper(base)
    # An already marked A must not prevent B from inheriting the same contract.
    wrapper.adapter.linear_in.weight.sequence_parallel = True

    assert inherit_lora_tp_grad_sync(wrapper) == 1
    assert inherit_lora_tp_grad_sync(wrapper) == 0

    assert all(parameter.sequence_parallel for parameter in wrapper.adapter.parameters())
    assert all(parameter.requires_grad for parameter in wrapper.adapter.parameters())
    assert all(
        not getattr(parameter, "average_gradients_across_tp_domain", False)
        for parameter in wrapper.adapter.parameters()
    )
    assert not base.weight.requires_grad


@pytest.mark.parametrize(
    ("base_flag", "adapter_flag"),
    [
        ("sequence_parallel", "average_gradients_across_tp_domain"),
        ("sequence_parallel", "tensor_model_parallel"),
        ("sequence_parallel", "sum_gradients_across_tp_domain"),
        ("average_gradients_across_tp_domain", "sequence_parallel"),
    ],
)
def test_inherit_lora_tp_grad_sync_preserves_explicit_adapter_contract(base_flag: str, adapter_flag: str) -> None:
    base = torch.nn.Linear(4, 4)
    setattr(base.weight, base_flag, True)
    wrapper = _PlainLoRAWrapper(base)
    setattr(wrapper.adapter.linear_in.weight, adapter_flag, True)

    assert inherit_lora_tp_grad_sync(wrapper) == 0

    assert getattr(wrapper.adapter.linear_in.weight, adapter_flag)
    assert all(not getattr(parameter, base_flag, False) for parameter in wrapper.adapter.parameters())


def test_inherit_lora_tp_grad_sync_preserves_unmarked_language_adapter() -> None:
    base = torch.nn.Linear(4, 4)
    base.weight.sum_gradients_across_tp_domain = True
    wrapper = _PlainLoRAWrapper(base)
    wrapper.adapter.linear_in.weight.sequence_parallel = True

    assert inherit_lora_tp_grad_sync(wrapper) == 0

    assert all(not hasattr(parameter, "average_gradients_across_tp_domain") for parameter in wrapper.parameters())
    assert wrapper.adapter.linear_in.weight.sequence_parallel
    assert base.weight.sum_gradients_across_tp_domain


def test_inherit_lora_tp_grad_sync_skips_non_plain_linear_base() -> None:
    base = torch.nn.Module()
    base.weight = torch.nn.Parameter(torch.ones(4, 4))
    base.weight.average_gradients_across_tp_domain = True
    wrapper = _PlainLoRAWrapper(base)

    assert inherit_lora_tp_grad_sync(wrapper) == 0

    assert all(
        not hasattr(parameter, "average_gradients_across_tp_domain") for parameter in wrapper.adapter.parameters()
    )


@pytest.mark.parametrize("sync_flag", ["average_gradients_across_tp_domain", "sequence_parallel"])
def test_inherit_lora_tp_grad_sync_is_idempotent_and_counts_shared_parameters_once(sync_flag: str) -> None:
    base = torch.nn.Linear(4, 4)
    setattr(base.weight, sync_flag, True)
    first = _PlainLoRAWrapper(base)
    second = _PlainLoRAWrapper(base)
    second.adapter = first.adapter
    model = torch.nn.ModuleList([first, second])

    assert inherit_lora_tp_grad_sync(model) == 2
    assert inherit_lora_tp_grad_sync(model) == 0


@pytest.mark.parametrize("adapter", [None, torch.nn.Identity()])
def test_inherit_lora_tp_grad_sync_accepts_empty_adapter(adapter: torch.nn.Module | None) -> None:
    model = torch.nn.Module()
    model.to_wrap = torch.nn.Linear(4, 4)
    model.to_wrap.weight.average_gradients_across_tp_domain = True
    model.adapter = adapter

    assert inherit_lora_tp_grad_sync(model) == 0
    assert inherit_lora_tp_grad_sync(torch.nn.Module()) == 0

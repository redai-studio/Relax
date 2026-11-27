# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contracts for QSA metadata and routing; kernels remain external
boundaries."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from relax.backends.fsdp_turbo import qwen4_attention as adapter
from relax.backends.fsdp_turbo.performance_config import PerformanceConfig


class Qwen4ExpModel(nn.Module):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return kwargs


def config(implementation="npu"):
    return SimpleNamespace(
        relax_performance=PerformanceConfig(attention_implementation=implementation),
        distributed=SimpleNamespace(ulysses_parallel_size=2),
    )


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_qsa_sequence_extent_survives_model_call(embedded, positional):
    model = Qwen4ExpModel()
    adapter.install(model, config())
    adapter.install(model, config())
    tensor = torch.zeros((1, 7, 3) if embedded else (1, 7))
    # Valid tokens differ from the unpadded sequence extent.
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])
    if positional:
        args = (None, mask, None, None, tensor) if embedded else (tensor, mask)
        result = model(*args, cp_sequence_length=999, sentinel="retained")
    else:
        inputs = {"inputs_embeds" if embedded else "input_ids": tensor}
        result = model(**inputs, attention_mask=mask, cp_sequence_length=999, sentinel="retained")
    assert result == {"cp_sequence_length": 7, "sentinel": "retained"}
    assert len(model._forward_pre_hooks) == 1


def test_qsa_native_leaves_model_kwargs_unchanged():
    model = Qwen4ExpModel()
    adapter.install(model, config("native"))
    assert model(torch.zeros((1, 7)), sentinel="retained") == {"sentinel": "retained"}


def backend_boundaries(monkeypatch, cp_size=2, ulysses_size=2):
    group = object()
    state = SimpleNamespace(is_group_enable=lambda name: True, get_cp_group_size=lambda: cp_size)
    modules = {
        "fsdp_turbo.distributed.parallel_state": {"get_parallel_state": lambda: state},
        "fsdp_turbo.distributed.context_parallel.ulysses.utils": {
            "get_ulysses_context": lambda: (group, 0, ulysses_size)
        },
        "fsdp_turbo.distributed.context_parallel.kv_allgather.utils": {
            "get_kv_allgather_context": lambda: (None, 0, 1)
        },
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    return group


@pytest.mark.parametrize("grouped", [False, True])
def test_qsa_dispatch_preserves_group_and_length(monkeypatch, grouped):
    group = backend_boundaries(monkeypatch)
    calls = []

    def kernel(name):
        def capture(self, hidden, positions, mask, **kwargs):
            calls.append((name, hidden, positions, mask, kwargs))
            return "kernel-result"

        return capture

    boundary = ModuleType("relax.backends.fsdp_turbo.performance_ops.qsa_context_parallel")
    boundary.qsa_forward = kernel("standard")
    boundary.qsa_grouped_forward = kernel("grouped")
    monkeypatch.setitem(sys.modules, boundary.__name__, boundary)
    layer = SimpleNamespace(_relax_grouped_qsa=grouped)
    result = adapter.qsa_context_parallel_forward(layer, "hidden", "positions", "mask", cp_sequence_length=7)
    assert result == "kernel-result"
    assert calls == [
        (
            "grouped" if grouped else "standard",
            "hidden",
            "positions",
            "mask",
            {"ulysses_group": group, "cp_sequence_length": 7},
        )
    ]


@pytest.mark.parametrize("topology,cache", [(True, None), (False, object())])
def test_qsa_rejects_unsupported_runtime_before_kernel(monkeypatch, topology, cache):
    backend_boundaries(monkeypatch, cp_size=4 if topology else 2)
    with pytest.raises(ValueError, match="Ulysses CP only" if topology else "without a KV cache"):
        adapter.qsa_context_parallel_forward(SimpleNamespace(), None, None, None, past_key_values=cache)


def test_qsa_cp1_uses_original_attention(monkeypatch):
    backend_boundaries(monkeypatch, cp_size=1, ulysses_size=1)
    module = ModuleType("fsdp_turbo.models.qwen.modeling_qwen4_exp")

    class OriginalAttention:
        def forward(self, hidden, positions, mask, cache, **kwargs):
            return hidden, positions, mask, cache, kwargs

    module.Qwen4ExpTextAttention = OriginalAttention
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert adapter.qsa_context_parallel_forward(SimpleNamespace(), "hidden", "positions", "mask", sentinel=7) == (
        "hidden",
        "positions",
        "mask",
        None,
        {"sentinel": 7},
    )

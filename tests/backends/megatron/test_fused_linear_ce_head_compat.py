# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native head eligibility without importing the optional CUDA training
runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture
def heads(monkeypatch):
    class Column(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = None
            self.gather_output = False
            self.config = SimpleNamespace(defer_embedding_wgrad_compute=False)

    class Native(Column):
        pass

    class Adapter(Column):
        pass

    class NativeAdapter(Native):
        pass

    for name in ("megatron", "megatron.core", "megatron.core.tensor_parallel", "megatron.core.transformer"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    layers = ModuleType("megatron.core.tensor_parallel.layers")
    layers.ColumnParallelLinear = Column
    monkeypatch.setitem(sys.modules, layers.__name__, layers)
    native = ModuleType("megatron.core.transformer.linear_cross_entropy")
    native.LinearCrossEntropyModule = Native
    monkeypatch.setitem(sys.modules, native.__name__, native)
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/fused_linear_ce.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "supported"
    )
    scope = {"Any": Any, "torch": torch, "Envs": SimpleNamespace(RELAX_SFT_FUSED_LINEAR_CE=True)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    device = torch.device("cuda:0")
    hidden = SimpleNamespace(is_cuda=True, device=device, shape=(8, 1, 64))
    weight = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=2, shape=(512, 64), is_contiguous=lambda: True)
    return SimpleNamespace(
        supported=lambda layer: scope["supported"](layer, hidden, weight),
        Column=Column,
        Native=Native,
        Adapter=Adapter,
        NativeAdapter=NativeAdapter,
    )


def test_native_logits_head_retains_fused_ce_eligibility(heads):
    assert heads.supported(heads.Column())
    assert heads.supported(heads.Native())
    assert not heads.supported(heads.Adapter())
    assert not heads.supported(heads.NativeAdapter())


@pytest.mark.parametrize("guard", ["bias", "hook", "gather_output", "disable_grad_reduce", "defer_wgrad"])
def test_native_head_preserves_fused_ce_safety_guards(heads, guard):
    layer = heads.Native()
    if guard == "bias":
        layer.bias = torch.nn.Parameter(torch.zeros(1))
    elif guard == "hook":
        layer.register_forward_pre_hook(lambda module, inputs: inputs)
    elif guard == "defer_wgrad":
        layer.config.defer_embedding_wgrad_compute = True
    else:
        setattr(layer, guard, True)
    assert not heads.supported(layer)


def test_old_runtime_without_native_head_remains_supported(heads, monkeypatch):
    monkeypatch.delitem(sys.modules, "megatron.core.transformer.linear_cross_entropy")
    assert heads.supported(heads.Column())
    assert not heads.supported(heads.Adapter())

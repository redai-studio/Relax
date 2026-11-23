# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native head eligibility without importing the optional CUDA training
runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/fused_linear_ce.py"
    spec = importlib.util.spec_from_file_location("fused_linear_ce_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    device = torch.device("cuda:0")
    hidden = SimpleNamespace(is_cuda=True, device=device, shape=(8, 1, 64))
    weight = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=2, shape=(512, 64), is_contiguous=lambda: True)
    return SimpleNamespace(
        supported=lambda layer: module.supported(layer, hidden, weight),
        Column=Column,
        Native=Native,
        Adapter=Adapter,
        NativeAdapter=NativeAdapter,
        module=module,
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


@pytest.mark.parametrize("runtime_available", [False, True])
def test_hopper_requires_quack_epilogue_runtime(heads, monkeypatch, runtime_available):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(heads.module, "_hopper_runtime_available", lambda: runtime_available)
    assert heads.supported(heads.Native()) is runtime_available


def test_other_architectures_retain_native_head_fallback(heads, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    assert not heads.supported(heads.Native())

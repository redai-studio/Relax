# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    parent = "_test_v2_config"
    package = ModuleType(parent)
    package.__path__ = []
    base = ModuleType(parent + ".qsa_fused_attention")
    base.build_plan = Mock()
    base.supports = Mock()
    base.QSAFusedPlan = object
    monkeypatch.setitem(sys.modules, parent, package)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    name = parent + ".qsa_triton_v2"
    monkeypatch.setitem(sys.modules, name + ".kernels", ModuleType(name + ".kernels"))
    path = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_triton_v2/__init__.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "capability,dim,dtype,bm",
    [
        ((10, 3), 256, torch.bfloat16, 64),
        ((10, 3), 128, torch.bfloat16, 128),
        ((10, 3), 64, torch.bfloat16, 128),
        ((10, 3), 256, torch.float16, 128),
        ((10, 0), 256, torch.bfloat16, 128),
        ((9, 0), 256, torch.bfloat16, 128),
        ((8, 0), 256, torch.bfloat16, 128),
    ],
)
def test_device_config_is_limited_to_validated_case(backend, capability, dim, dtype, bm):
    config = backend._select_config(dim, capability, dtype)
    assert config.dq == (bm, 32, 8, 1)
    assert config.fwd == backend.DEFAULT.fwd
    assert config.dkv == backend.DEFAULT.dkv


@pytest.mark.parametrize("explicit", [False, True])
def test_dispatch_uses_query_device_and_preserves_explicit_config(backend, monkeypatch, explicit):
    query = SimpleNamespace(shape=(3, 3, 256), device=torch.device("cuda:2"), dtype=torch.bfloat16, requires_grad=True)
    key = SimpleNamespace(shape=(3, 1, 256), requires_grad=True)
    query.contiguous = lambda: query
    key.contiguous = lambda: key
    capability = Mock(return_value=(10, 3))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    apply = Mock(return_value=object())
    monkeypatch.setattr(backend._Attention, "apply", apply)
    override = backend.Config(dq=(32, 16, 4, 1)) if explicit else None
    backend.qsa_fused_attention(query, key, key, object(), SimpleNamespace(q_len=3), None, 0.0625, override)
    selected = apply.call_args.args[7]
    if explicit:
        assert selected is override
        capability.assert_not_called()
    else:
        assert selected.dq == (64, 32, 8, 1)
        capability.assert_called_once_with(query.device)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise environment parsing and explicit backend dispatch without Megatron
or CUDA."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.utils.env import Envs, known_env_names, resolve_in


ENV = "RELAX_QWEN38_QSA_FUSED_ATTENTION"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("OFF", False),
        ("1", True),
        ("true", True),
        ("triton", True),
        (" TRITON_V2 ", "triton_v2"),
    ],
)
def test_qsa_fused_switch_preserves_booleans_and_accepts_backend(monkeypatch, raw, expected):
    monkeypatch.delenv(ENV, raising=False)
    if raw is not None:
        monkeypatch.setenv(ENV, raw)
    assert ENV in known_env_names()
    assert Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION == expected
    assert resolve_in(ENV, {} if raw is None else {ENV: raw}) == expected


def test_qsa_fused_switch_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv(ENV, "triton_v22")
    with pytest.raises(ValueError, match=ENV):
        _ = Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION


def _load_backend(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    package = ModuleType("relax.models.qwen3_8")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    original = Mock(return_value="fallback")
    attention_class = type("QSACoreAttention", (), {"_flex_forward": original})
    dispatch = SimpleNamespace(block_flex=original)
    monkeypatch.setitem(sys.modules, package.__name__ + ".qsa_dispatch", dispatch)
    package.qsa_dispatch = dispatch
    ops = ModuleType(package.__name__ + ".ops")
    ops.__path__ = []
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    upstream = ModuleType("megatron.core.transformer.experimental_attention_variant")
    upstream.qsa = SimpleNamespace(QSACoreAttention=attention_class)
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    backends = {}
    for name in ("qsa_fused_attention", "qsa_triton_v2"):
        mod = ModuleType(ops.__name__ + "." + name)
        mod.supports = Mock(return_value=True)
        mod.build_plan = Mock(return_value=object())
        mod.qsa_fused_attention = Mock(side_effect=lambda q, *args: torch.zeros_like(q))
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        setattr(ops, name, mod)
        backends[name] = mod
    cp = ModuleType(package.__name__ + ".qsa_cp")
    cache = {}

    def cached(key, owner, build):
        if (key, id(owner)) not in cache:
            cache[key, id(owner)] = build()
        return cache[key, id(owner)]

    cp._cached = cached
    cp._fused_backend = None
    monkeypatch.setitem(sys.modules, cp.__name__, cp)
    package.qsa_cp = cp
    path = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/qsa_fused.py"
    spec = importlib.util.spec_from_file_location(package.__name__ + ".qsa_fused_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return attention_class, original, module, backends


def test_disabled_fused_switch_does_not_install(monkeypatch):
    cls, original, module, _ = _load_backend(monkeypatch, "0")
    assert cls._flex_forward is original
    assert module.backend() is None


@pytest.mark.parametrize("value,name", [("1", "qsa_fused_attention"), ("triton_v2", "qsa_triton_v2")])
@pytest.mark.parametrize("is_thd", [True, False])
def test_backend_switch_routes_cp1_and_cp_and_reuses_plan(monkeypatch, value, name, is_thd):
    cls, original, module, backends = _load_backend(monkeypatch, value)
    owner = cls()
    owner.sparse_backend, owner.softmax_scale = "flex", 0.125
    backend = backends[name]
    assert module.backend() is backend
    q = torch.zeros((11, 3, 64) if is_thd else (11, 1, 3, 64))
    k = torch.zeros((11, 1, 64) if is_thd else (11, 1, 1, 64))
    selection = object()
    for _ in range(2):
        output = module._fused_flex_forward(owner, q, k, k, selection, is_thd)
        assert output.shape == ((11, 192) if is_thd else (11, 1, 192))
    backend.build_plan.assert_called_once_with(selection, 11)
    assert backend.qsa_fused_attention.call_args.args[-2:] == (None, 0.125)
    original.assert_not_called()
    backend.supports.return_value = False
    assert module._fused_flex_forward(owner, q, k, k, selection, is_thd) == "fallback"
    original.assert_called_once()


def test_multi_batch_keeps_original_fallback(monkeypatch):
    cls, original, module, backends = _load_backend(monkeypatch, "triton_v2")
    owner = cls()
    q = torch.zeros(11, 2, 3, 64)
    assert module._fused_flex_forward(owner, q, q, q, object(), False) == "fallback"
    original.assert_called_once()
    backends["qsa_triton_v2"].supports.assert_not_called()

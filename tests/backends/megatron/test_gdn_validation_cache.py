# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validation memoization must not hide changed boundaries or validation
errors."""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch


def _install(monkeypatch):
    calls = []

    class GDN:
        def _resolve_cu_seqlens(self, padded, actual, length, name, cp_size=1):
            cu = padded if padded is not None else actual
            calls.append((length, cp_size))
            if cu is None or int(cu[-1]) != length:
                raise ValueError("invalid boundaries")
            return cu

    module = types.ModuleType("megatron.core.ssm.gated_delta_net")
    module.GatedDeltaNet = GDN
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_patch_gdn_cu_seqlens_validation"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    scope[node.name]()
    scope[node.name]()  # installation is idempotent
    return GDN(), calls


def test_reuses_only_same_unchanged_boundaries_and_geometry(monkeypatch):
    gdn, calls = _install(monkeypatch)
    cu = torch.tensor([0, 4])
    for _ in range(3):
        assert gdn._resolve_cu_seqlens(None, cu, 4, "q") is cu
    assert len(calls) == 1
    gdn._resolve_cu_seqlens(None, cu.clone(), 4, "kv")
    gdn._resolve_cu_seqlens(None, cu, 4, "kv", cp_size=2)
    assert len(calls) == 3
    cu[-1] = 3
    with pytest.raises(ValueError):
        gdn._resolve_cu_seqlens(None, cu, 4, "q", cp_size=2)
    with pytest.raises(ValueError):
        gdn._resolve_cu_seqlens(None, cu, 4, "q", cp_size=2)
    assert len(calls) == 5


def test_inference_tensors_bypass_cache(monkeypatch):
    gdn, calls = _install(monkeypatch)
    with torch.inference_mode():
        cu = torch.tensor([0, 4])
        for _ in range(2):
            assert gdn._resolve_cu_seqlens(None, cu, 4, "q") is cu
    assert len(calls) == 2

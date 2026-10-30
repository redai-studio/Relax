# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise production dispatch with external GPU kernels replaced, not kernel
numerics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def mixed_dispatcher(monkeypatch):
    """Import the dispatcher on CPU-only hosts without importing Triton
    kernels."""
    from relax.utils.env import Envs

    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    prefix = "relax.models.qwen3_8.ops.qsa_mixed_ops"
    flex = ModuleType(f"{prefix}.flex")
    grouped = ModuleType(f"{prefix}.grouped")

    def no_kernel(*_args, **_kwargs):
        pytest.fail("A dispatch-only test must replace GPU execution")

    flex.attention = flex.attention_auto = grouped.attention_bf16 = no_kernel
    path = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py"
    spec = importlib.util.spec_from_file_location("qsa_mixed_dispatch_under_test", path)
    mixed = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, flex.__name__, flex)
        imports.setitem(sys.modules, grouped.__name__, grouped)
        spec.loader.exec_module(mixed)
    assert mixed.Envs is Envs
    return mixed


@pytest.mark.parametrize(
    "limit,length,expected",
    [
        (None, 2052, "auto"),
        (None, 16384, "auto"),
        (None, 16385, "grouped"),
        ("0", 2051, "auto"),
        ("0", 2052, "grouped"),
        ("4096", 4096, "auto"),
        ("4096", 4097, "grouped"),
    ],
)
def test_mixed_threshold_keeps_dense_short_and_routes_sparse(monkeypatch, mixed_dispatcher, limit, length, expected):
    mixed = mixed_dispatcher
    if limit is None:
        monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", raising=False)
    else:
        monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", limit)
    seen = []

    def call(kind):
        def forward(q, *_args):
            seen.append(kind)
            return q

        return forward

    monkeypatch.setattr(mixed, "attention_auto", call("auto"))
    monkeypatch.setattr(mixed, "long_attention", call("grouped"))
    q = torch.empty(length, 12, 256, device="meta")
    kv = torch.empty(length, 1, 256, device="meta")
    indices = torch.empty(length, 2052, dtype=torch.int32, device="meta")
    assert mixed.attention(q, kv, kv, indices, 256**-0.5) is q
    assert seen == [expected]


@pytest.mark.parametrize("limit", [-1, 16385])
def test_mixed_threshold_rejects_unsupported_flex_layout(monkeypatch, mixed_dispatcher, limit):
    mixed = mixed_dispatcher
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", str(limit))
    q = torch.empty(2052, 12, 256, device="meta")
    kv = torch.empty(2052, 1, 256, device="meta")
    indices = torch.empty(2052, 2052, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="maximum length"):
        mixed.attention(q, kv, kv, indices, 256**-0.5)


@pytest.mark.parametrize("limit", [2047, 2051])
@pytest.mark.parametrize("length", [2047, 2048, 2051, 2052])
def test_explicit_flash_boundary_controls_legacy_long_branch(monkeypatch, mixed_dispatcher, limit, length):
    mixed = mixed_dispatcher
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(limit))
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "0")
    seen = []

    def route(name):
        def call(q, *_args):
            seen.append(name)
            return q

        return call

    monkeypatch.setattr(mixed, "attention_auto", route("flash"))
    monkeypatch.setattr(mixed, "long_attention", route("grouped"))
    q = torch.empty(length, 12, 256, device="meta")
    kv = torch.empty(length, 1, 256, device="meta")
    ids = torch.empty(length, 2052, dtype=torch.int32, device="meta")
    assert mixed.attention(q, kv, kv, ids, 256**-0.5) is q
    assert seen == ["flash" if length <= limit else "grouped"]


@pytest.mark.parametrize("flash,flex", [(2052, 0), (-2, 0), (2051, 16384)])
def test_explicit_flash_boundary_rejects_incompatible_settings(monkeypatch, mixed_dispatcher, flash, flex):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(flash))
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", str(flex))
    q = torch.empty(2052, 12, 256, device="meta")
    kv = torch.empty(2052, 1, 256, device="meta")
    ids = torch.empty(2052, 2052, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="dense selector budget|Flex maximum length zero"):
        mixed_dispatcher.attention(q, kv, kv, ids, 256**-0.5)

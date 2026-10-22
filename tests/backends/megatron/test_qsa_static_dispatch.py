# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verify QSA composition without modifying Megatron methods at runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "relax.models.qwen3_8"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispatch(monkeypatch):
    env = load("qsa_static_test_env", ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)
    module = load(PACKAGE + ".qsa_dispatch", ROOT / "relax/models/qwen3_8/qsa_dispatch.py")
    for key in (
        "CPU_LAYOUT",
        "RECOMPUTE_REPLAY",
        "FUSED_SELECTOR",
        "FIXED_POOL",
        "SPARSE_BLOCKMASK",
        "FUSED_ATTENTION",
        "TP_SELECTOR",
        "COMMUNITY_BACKEND",
    ):
        monkeypatch.delenv("RELAX_QWEN38_QSA_" + key, raising=False)
    return module


def stub(monkeypatch, name, function_name, function):
    module = ModuleType(PACKAGE + "." + name)
    setattr(module, function_name, function)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_indexer_stages_preserve_order_and_arguments(dispatch, monkeypatch):
    events = []
    native = lambda *args, **kwargs: (events.append("native"), (args, kwargs))[1]
    owner = SimpleNamespace(_relax_native_forward=native)
    for flag in ("CPU_LAYOUT", "RECOMPUTE_REPLAY", "FUSED_SELECTOR"):
        monkeypatch.setenv("RELAX_QWEN38_QSA_" + flag, "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_COMMUNITY_BACKEND", "mixed")

    def stage(label, next_stage):
        def run(self, *args, **kwargs):
            events.append(label)
            return next_stage(self, *args, **kwargs)

        return run

    stub(
        monkeypatch,
        "qsa_cpu_layout",
        "_indexer_forward",
        stage("cpu", lambda self, *a, **k: self._relax_native_forward(*a, **k)),
    )
    stub(monkeypatch, "qsa_cp", "_indexer_forward", stage("replay", dispatch.tp_forward))
    stub(monkeypatch, "qsa_select", "indexer_forward", stage("selector", dispatch.replay_forward))
    stub(monkeypatch, "ops.qsa_mixed_ops", "forward", stage("mixed", dispatch.selector_forward))
    assert dispatch.indexer_forward(owner, "hidden", output_format="ids") == (("hidden",), {"output_format": "ids"})
    assert events == ["mixed", "selector", "replay", "cpu", "native"]
    assert owner._relax_native_forward is native

    events.clear()
    monkeypatch.setenv("RELAX_QWEN38_QSA_TP_SELECTOR", "1")
    stub(monkeypatch, "ops.qsa_mixed_ops.selector", "forward", stage("tp", lambda self, *a, **k: "tp-result"))
    assert dispatch.indexer_forward(owner, "hidden") == "tp-result"
    assert events == ["mixed", "selector", "replay", "tp"]


def test_disabled_paths_call_native_without_optional_modules(dispatch):
    owner = SimpleNamespace(
        _relax_native_forward=lambda value: ("forward", value),
        _relax_native_pool_keys=lambda value: ("pool", value),
        _relax_native_select_blocks=lambda value: ("select", value),
        _relax_native_flex_forward=lambda value: ("flex", value),
    )
    assert dispatch.indexer_forward(owner, 1) == ("forward", 1)
    assert dispatch.pool_keys(owner, 2) == ("pool", 2)
    assert dispatch.select_blocks(owner, 3) == ("select", 3)
    assert dispatch.flex_forward(owner, 4) == ("flex", 4)


def test_tp_selector_rejects_other_backend_before_import(dispatch, monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_QSA_TP_SELECTOR", "1")
    with pytest.raises(ValueError, match="mixed"):
        dispatch.indexer_forward(object(), "hidden")


def test_fused_fallback_runs_sparse_mask_before_native(dispatch, monkeypatch):
    events = []
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_ATTENTION", "triton")
    monkeypatch.setenv("RELAX_QWEN38_QSA_SPARSE_BLOCKMASK", "1")
    owner = SimpleNamespace(_relax_native_flex_forward=lambda value: value)

    def fused(self, value):
        events.append("fused")
        return dispatch.block_flex(self, value)

    def block(self, value):
        events.append("block")
        return self._relax_native_flex_forward(value)

    stub(monkeypatch, "qsa_fused", "_fused_flex_forward", fused)
    stub(monkeypatch, "qsa_block_mask", "_sparse_flex_forward", block)
    assert dispatch.flex_forward(owner, "output") == "output"
    assert events == ["fused", "block"]

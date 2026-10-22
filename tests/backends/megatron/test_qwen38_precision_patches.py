# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration tests; numerical kernel validation requires the GPU
suite."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch):
    module = _load("precision_env", _ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", module)
    return module


@pytest.fixture
def router(env, monkeypatch):
    torch = SimpleNamespace(float32=object(), is_grad_enabled=Mock(return_value=False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = _load("precision_router", _ROOT / "relax/models/qwen3_8/ops/router_sorted_replay.py")
    return module, torch


def test_router_flag_defaults_off(env, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", raising=False)
    assert env.Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY is False
    monkeypatch.setenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", "true")
    assert env.Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY is True


@pytest.mark.parametrize("case", ["supported", "grad", "cpu", "dtype", "shape", "topk", "replay", "fused", "aux"])
def test_router_sorted_selection_is_limited_to_audited_inputs(router, case):
    module, torch = router
    logits = SimpleNamespace(ndim=2, shape=(7, 512), dtype=torch.float32, is_cuda=True)
    args = dict(
        logits=logits,
        topk=10,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=None,
        dense_output=False,
    )
    if case == "grad":
        torch.is_grad_enabled.return_value = True
    elif case == "cpu":
        logits.is_cuda = False
    elif case == "dtype":
        logits.dtype = object()
    elif case == "shape":
        logits.shape = (7, 64)
    elif case == "topk":
        args["topk"] = 2
    elif case == "replay":
        args["router_replay"] = object()
    elif case == "fused":
        args["fused"] = True
    elif case == "aux":
        args["scaling_factor"] = 2.0
    assert module.use_sorted_topk(**args) is (case == "supported")


def test_gdn_flag_retains_default_off(env, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", raising=False)
    assert env.Envs.RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG is False


@pytest.fixture
def pool(env, monkeypatch):
    torch = SimpleNamespace(bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(jit=lambda fn: fn, language=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "triton.language", SimpleNamespace())
    ops = _load("relax.models.qwen3_8.ops.qsa_pool", _ROOT / "relax/models/qwen3_8/ops/qsa_pool.py")
    monkeypatch.setitem(sys.modules, "relax.models.qwen3_8.ops.qsa_pool", ops)
    module = _load("relax.models.qwen3_8.qsa_fixed_pool", _ROOT / "relax/models/qwen3_8/qsa_fixed_pool.py")
    layout = SimpleNamespace(get=lambda: (0, 4, 8))
    monkeypatch.setitem(sys.modules, "relax.models.qwen3_8.qsa_select", SimpleNamespace(_layout=layout))
    return module, torch, None, None


def test_pool_missing_selector_rejected_before_execution(pool, env, monkeypatch):
    module, _, _, _ = pool
    assert "RELAX_QWEN38_QSA_FIXED_POOL" in env.known_env_names()
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "0")
    with pytest.raises(RuntimeError, match="FUSED_SELECTOR"):
        module.pool_keys(None, None, None, None, 4)


def test_pool_explicit_call_preserves_metadata_contract(pool, monkeypatch):
    module, _, _, _ = pool
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "1")
    model = SimpleNamespace(
        compress_ratio=4,
        config=SimpleNamespace(qsa_indexer_loss_coeff=0, context_parallel_size=1),
        parameters=lambda: [],
    )
    raw = SimpleNamespace(shape=(8, 3), device="cuda")
    docs = SimpleNamespace(shape=(8,), device="cuda")
    consumer = Mock(return_value=object())
    monkeypatch.setattr(module, "fixed_pool", consumer)
    assert module.pool_keys(model, raw, docs, docs, 4) is consumer.return_value
    consumer.assert_called_once_with(raw, (0, 4, 8), 4)
    for field, bad, good in [("qsa_indexer_loss_coeff", 0.1, 0), ("context_parallel_size", 2, 1)]:
        setattr(model.config, field, bad)
        with pytest.raises(RuntimeError, match="frozen packed"):
            module.pool_keys(model, raw, docs, docs, 4)
        setattr(model.config, field, good)
    model.parameters = lambda: [SimpleNamespace(requires_grad=True)]
    with pytest.raises(RuntimeError, match="frozen packed"):
        module.pool_keys(model, raw, docs, docs, 4)
    assert consumer.call_count == 1
    assert not hasattr(model, "_pool_keys")


@pytest.mark.parametrize(
    "offsets,max_len", [(None, 4), ((0,), 4), ((1, 8), 7), ((0, 4, 7), 4), ((0, 5, 3, 8), 5), ((0, 4, 8), 5)]
)
def test_pool_rejects_missing_or_inconsistent_cpu_layout_before_launch(pool, offsets, max_len):
    module, torch, _, _ = pool
    raw = SimpleNamespace(
        ndim=2, dtype=torch.bfloat16, is_cuda=True, shape=(8, 3), stride=lambda: (3, 1), requires_grad=False
    )
    with pytest.raises(ValueError, match="packed QSA offsets"):
        module.fixed_pool(raw, offsets, max_len)


@pytest.mark.parametrize("argument", ["qb_histogram", "qb_bin_bounds", "topk_indices"])
def test_static_router_preserves_native_output_and_qb_paths(router, argument):
    module, torch = router
    args = dict(
        logits=SimpleNamespace(ndim=2, shape=(7, 512), dtype=torch.float32, is_cuda=True),
        topk=10,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=None,
        dense_output=False,
    )
    args[argument] = object()
    assert not module.use_sorted_topk(**args)


def test_static_router_preserves_explicit_routing_replay(router, monkeypatch):
    module, torch = router
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    assert not module.use_sorted_topk(
        SimpleNamespace(ndim=2, shape=(7, 512), dtype=torch.float32, is_cuda=True),
        10,
        False,
        None,
        None,
        None,
        "softmax",
        None,
        False,
        None,
        False,
    )


@pytest.fixture
def legacy_precision(env):
    return _load("static_legacy_precision", _ROOT / "relax/models/qwen3_8/legacy_precision.py")


def test_static_prep_flag_defaults_off(legacy_precision, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", raising=False)
    assert not legacy_precision.prep_compile()


def test_static_prep_rechecks_configuration_after_initialization(legacy_precision, monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", "1")
    config = SimpleNamespace(kwargs={"BK": 64, "BV": 64}, num_warps=4, num_stages=3)
    tuner = type("CachedAutotuner", (), {})()
    tuner.fn = type("JITFunction", (), {})()
    tuner.configs = [config]
    kernel = type("Heuristics", (), {})()
    kernel.fn = tuner
    monkeypatch.setattr(
        legacy_precision.importlib, "import_module", lambda name: SimpleNamespace(chunk_fwd_kernel_o=kernel)
    )
    assert legacy_precision.prep_compile()
    config.num_stages = 2
    with pytest.raises(RuntimeError, match="configuration was changed"):
        legacy_precision.prep_compile()

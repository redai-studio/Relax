# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for GDN CP configuration and runtime dispatch.

Covers the CLI, construction-time mode constraints, packing validation, dynamic
CP group selection, and Relax's all-gather fallback guards. Uses fakes and
spies for dispatch; real layout communication is covered in
``test_gdn_cp_gpu.py``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # noqa: E402


def _load_relax_functions(filename, names):
    """Execute the real functions without importing the Ray/optimizer stack."""
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron" / filename
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ModuleType("_gdn_cp_test_" + path.stem)
    module.__package__ = "relax.backends.megatron"
    module.torch = torch
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), vars(module))
    return module


megatron_arguments = _load_relax_functions("arguments.py", ["_validate_linear_cp_mode"])
gdn_model = _load_relax_functions(
    "model.py", ["_patch_gdn_for_dynamic_cp", "_resolve_gdn_cp", "_assert_gdn_full_recompute"]
)
_validate_linear_cp_mode = megatron_arguments._validate_linear_cp_mode


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------
def _parse_megatron_args(monkeypatch, *argv):
    pytest.importorskip("triton", reason="the full Megatron training CLI imports Triton kernels")
    training_arguments = pytest.importorskip("megatron.training.arguments")

    monkeypatch.setattr("sys.argv", ["test-linear-cp-mode", *argv])
    return training_arguments.parse_args(
        ignore_unknown_args=False,
    )


def test_linear_cp_mode_flag_defaults_to_chunkwise(monkeypatch):
    args = _parse_megatron_args(monkeypatch)
    assert args.linear_cp_mode == "chunkwise"


@pytest.mark.parametrize("mode", ["headwise", "chunkwise", "all_gather"])
def test_linear_cp_mode_flag_accepts_all_concrete_modes(monkeypatch, mode):
    args = _parse_megatron_args(monkeypatch, "--linear-cp-mode", mode)
    assert args.linear_cp_mode == mode


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        linear_cp_mode="chunkwise",
        experimental_attention_variant="gated_delta_net",
        allgather_cp=False,
        deterministic_mode=False,
        dynamic_context_parallel=False,
        context_parallel_size=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("bad", ["auto", "allgather"])
def test_validate_linear_cp_mode_rejects_unsupported_value(bad):
    with pytest.raises(ValueError, match="must be one of"):
        _validate_linear_cp_mode(_args(linear_cp_mode=bad))


def test_validate_linear_cp_mode_rejects_chunkwise_with_allgather_cp():
    with pytest.raises(ValueError, match="allgather-cp"):
        _validate_linear_cp_mode(_args(linear_cp_mode="chunkwise", allgather_cp=True, context_parallel_size=2))


@pytest.mark.parametrize("cp_kwargs", [{"context_parallel_size": 2}, {"dynamic_context_parallel": True}])
def test_validate_linear_cp_mode_rejects_chunkwise_deterministic_when_cp_may_exceed_one(cp_kwargs):
    with pytest.raises(ValueError, match="deterministic"):
        _validate_linear_cp_mode(_args(linear_cp_mode="chunkwise", deterministic_mode=True, **cp_kwargs))


# ---------------------------------------------------------------------------
# Runtime dispatcher
# ---------------------------------------------------------------------------
class _FakeGroup:
    """Minimal process-group stand-in exposing only .size()/.rank(): the
    dispatcher and resolve_cp_group() never issue a real collective."""

    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _fake_packed_seq_params(qkv_format="thd", cp_group=None, local_cp_size=None):
    params = PackedSeqParams(qkv_format=qkv_format)
    if cp_group is not None:
        params.cp_group = cp_group
    if local_cp_size is not None:
        params.local_cp_size = local_cp_size
    return params


def _fake_gdn_module(*, linear_cp_mode, static_cp_size=1, deterministic_mode=False):
    """Duck-typed GatedDeltaNet 'self': just enough attributes for the
    dispatcher guards to read -- no real nn.Module/CUDA/FLA state."""
    return SimpleNamespace(
        pg_collection=SimpleNamespace(cp=_FakeGroup(static_cp_size)),
        cp_size=static_cp_size,
        config=SimpleNamespace(linear_cp_mode=linear_cp_mode, deterministic_mode=deterministic_mode),
    )


@pytest.fixture(autouse=True)
def _isolate_gdn_forward_patch():
    """`_patch_gdn_for_dynamic_cp` idempotently monkey-patches the *shared*
    GatedDeltaNet class attribute; save/restore it around every test so it
    cannot leak into test_gdn_cp_gpu.py."""
    orig_forward = GatedDeltaNet.forward
    orig_patched_flag = getattr(GatedDeltaNet, "_dcp_patched", False)
    yield
    GatedDeltaNet.forward = orig_forward
    GatedDeltaNet._dcp_patched = orig_patched_flag


def _install_dispatcher_with_spies():
    """Install the dispatcher over a spy for the original MCore forward."""
    calls = {"orig": 0}

    def spy_orig(self, hidden_states, attention_mask, inference_context=None, packed_seq_params=None, *a, **kw):
        calls["orig"] += 1
        return "orig", hidden_states

    GatedDeltaNet.forward = spy_orig
    GatedDeltaNet._dcp_patched = False
    gdn_model._patch_gdn_for_dynamic_cp()
    return calls


def test_dispatcher_cp1_goes_to_original_forward_regardless_of_mode():
    for mode in ("headwise", "chunkwise", "all_gather"):
        calls = _install_dispatcher_with_spies()
        m = _fake_gdn_module(linear_cp_mode=mode, static_cp_size=1)
        out = GatedDeltaNet.forward(m, "hs", None, None, None)
        assert calls == {"orig": 1}, mode
        assert out == ("orig", "hs")


@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
def test_dispatcher_headwise_and_chunkwise_cp_gt_1_go_to_original_forward(mode):
    calls = _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode=mode, static_cp_size=4)
    psp = _fake_packed_seq_params(cp_group=_FakeGroup(4, rank=2), local_cp_size=4)
    GatedDeltaNet.forward(m, "hs", None, None, psp)
    assert calls == {"orig": 1}


def test_dispatcher_all_gather_cp_gt_1_goes_to_relax_fallback():
    calls = _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="all_gather", static_cp_size=4)
    psp = _fake_packed_seq_params(qkv_format="sbhd", cp_group=_FakeGroup(4, rank=3), local_cp_size=4)
    with pytest.raises(AssertionError, match=r"packed \(thd\) sequences"):
        GatedDeltaNet.forward(m, "hs", None, None, psp)
    assert calls == {"orig": 0}


def test_dispatcher_prefers_dynamic_group_over_static_group():
    """Runtime CP (from packed_seq_params) must win over the module's static
    max-CP group -- e.g. a static CP=8 model running a CP=1 micro-batch must
    not take the all_gather branch just because the static group has size 8."""
    calls = _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="all_gather", static_cp_size=8)
    psp = _fake_packed_seq_params(cp_group=_FakeGroup(1, rank=0), local_cp_size=1)
    GatedDeltaNet.forward(m, "hs", None, None, psp)
    assert calls == {"orig": 1}


@pytest.mark.parametrize(
    ("mode", "runtime_cp_size"),
    [("headwise", 4), ("chunkwise", 4), ("all_gather", 1)],
)
def test_dispatcher_never_mutates_shared_module_or_config_state(mode, runtime_cp_size):
    """Covers handoff §6.3: self.cp_size / self.pg_collection.cp /
    self.config.linear_cp_mode must be bit-identical before and after, across
    every mode and every runtime CP size (static max CP fixed at 8, so a
    smaller runtime CP can only come from the dynamic packed_seq_params)."""
    _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode=mode, static_cp_size=8)
    psp = _fake_packed_seq_params(cp_group=_FakeGroup(runtime_cp_size, rank=0), local_cp_size=runtime_cp_size)
    before_cp_size = m.cp_size
    before_pg_cp = m.pg_collection.cp
    before_mode = m.config.linear_cp_mode
    GatedDeltaNet.forward(m, "hs", None, None, psp)
    assert m.cp_size == before_cp_size
    assert m.pg_collection.cp is before_pg_cp
    assert m.config.linear_cp_mode == before_mode


# ---------------------------------------------------------------------------
# All-gather guard clauses (checked before any real tensor operation).
# ---------------------------------------------------------------------------
def test_all_gather_fallback_rejects_inference():
    _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="all_gather", static_cp_size=4)
    psp = _fake_packed_seq_params(cp_group=_FakeGroup(4), local_cp_size=4)
    with pytest.raises(AssertionError, match="inference"):
        GatedDeltaNet.forward(m, "hs", None, object(), psp)


def test_all_gather_fallback_rejects_deterministic_mode():
    _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="all_gather", static_cp_size=4, deterministic_mode=True)
    psp = _fake_packed_seq_params(cp_group=_FakeGroup(4), local_cp_size=4)
    with pytest.raises(AssertionError, match="deterministic mode"):
        GatedDeltaNet.forward(m, "hs", None, None, psp)


@pytest.mark.parametrize("mode", ["headwise", "chunkwise", "all_gather"])
def test_gdn_modes_reject_contiguous_attention_packing(mode):
    with pytest.raises(ValueError, match="allgather-cp"):
        _validate_linear_cp_mode(_args(linear_cp_mode=mode, context_parallel_size=2, allgather_cp=True))


def test_default_chunkwise_does_not_restrict_non_gdn_models():
    _validate_linear_cp_mode(
        _args(
            experimental_attention_variant="dsa", context_parallel_size=4, allgather_cp=True, deterministic_mode=True
        )
    )


def test_bridge_validation_uses_provider_attention_variant():
    args = _args(experimental_attention_variant=None, allgather_cp=True, context_parallel_size=2)
    _validate_linear_cp_mode(args)
    provider = SimpleNamespace(
        experimental_attention_variant="gated_delta_net", linear_cp_mode="chunkwise", context_parallel_size=2
    )
    with pytest.raises(ValueError, match="allgather-cp"):
        _validate_linear_cp_mode(args, provider)


@pytest.mark.parametrize("group,size", [(None, 2), (_FakeGroup(2), None), (_FakeGroup(2), 4)])
def test_dispatcher_rejects_inconsistent_dynamic_metadata(group, size):
    _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="chunkwise", static_cp_size=8)
    with pytest.raises(ValueError, match="PackedSeqParams"):
        GatedDeltaNet.forward(
            m, "hs", None, packed_seq_params=_fake_packed_seq_params(cp_group=group, local_cp_size=size)
        )


def test_dispatcher_respects_explicit_process_group_collection():
    calls = _install_dispatcher_with_spies()
    m = _fake_gdn_module(linear_cp_mode="all_gather", static_cp_size=8)
    GatedDeltaNet.forward(m, "hs", None, pg_collection=SimpleNamespace(cp=_FakeGroup(1)))
    assert calls == {"orig": 1}


def test_all_gather_still_requires_full_recompute(monkeypatch):
    monkeypatch.setattr(gdn_model, "get_args", lambda: _args(recompute_granularity="selective"), raising=False)
    monkeypatch.setattr(gdn_model._assert_gdn_full_recompute, "_checked", False, raising=False)
    with torch.enable_grad(), pytest.raises(ValueError, match="all_gather.*whole-layer"):
        gdn_model._assert_gdn_full_recompute()
    with torch.no_grad():
        gdn_model._assert_gdn_full_recompute()


# ---------------------------------------------------------------------------
# Dynamic CP group resolution
# ---------------------------------------------------------------------------
def test_resolve_cp_group_prefers_packed_seq_params():
    static = object()
    dynamic = object()
    assert resolve_cp_group(static, None) is static
    assert resolve_cp_group(static, PackedSeqParams(qkv_format="thd")) is static
    assert resolve_cp_group(static, PackedSeqParams(qkv_format="thd", cp_group=dynamic)) is dynamic


# ---------------------------------------------------------------------------
# Construction-time capability gate
# ---------------------------------------------------------------------------
def _gdn_config(**overrides):
    import torch.nn.functional as F
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        hidden_size=2048,
        num_layers=1,
        num_attention_heads=16,
        num_query_groups=2,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_config_default_mode_is_chunkwise():
    """Upgrading the image must not silently reroute an existing recipe."""
    assert _gdn_config().linear_cp_mode == "chunkwise"


def test_headwise_config_requires_heads_divisible_by_tp_times_cp():
    # 16 key heads, tp=2, cp=4 -> 16 % 8 == 0: fine.
    _gdn_config(tensor_model_parallel_size=2, context_parallel_size=4, linear_cp_mode="headwise")
    # tp=2, cp=16 -> 16 % 32 != 0: the geometry headwise cannot express.
    with pytest.raises(AssertionError, match="linear_num_key_heads"):
        _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16, linear_cp_mode="headwise")


def test_chunkwise_config_only_requires_heads_divisible_by_tp():
    """This is what replaces Relax's temporary head-count rewrite hack."""
    cfg = _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16, linear_cp_mode="chunkwise")
    assert cfg.linear_num_key_heads == 16 and cfg.linear_num_value_heads == 32
    # ... but TP divisibility is still enforced: GDN weights stay TP-sharded.
    with pytest.raises(AssertionError, match="linear_num_key_heads"):
        _gdn_config(
            tensor_model_parallel_size=8,
            context_parallel_size=2,
            linear_cp_mode="chunkwise",
            num_query_groups=8,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )


def test_all_gather_config_uses_the_tp_only_head_rule():
    """`--linear-cp-mode=all_gather` must be constructible on a non-divisible
    geometry.

    Relax's all-gather fallback keeps GDN weights TP-only, so declaring it
    should relax the head check exactly as chunkwise does.
    """
    cfg = _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16, linear_cp_mode="all_gather")
    assert cfg.linear_num_key_heads == 16 and cfg.linear_num_value_heads == 32


def test_config_rejects_unresolved_and_unknown_linear_cp_mode():
    """MCore only accepts the three concrete execution modes."""
    for bad in ("auto", "allgather", "chunk", ""):
        with pytest.raises(AssertionError, match="linear_cp_mode"):
            _gdn_config(context_parallel_size=2, linear_cp_mode=bad)
        with pytest.raises(AssertionError, match="linear_cp_mode"):
            _gdn_config(context_parallel_size=4, tensor_model_parallel_size=2, linear_cp_mode=bad)


def test_gdn_forward_has_no_per_call_mode_override():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    assert "linear_cp_mode" not in inspect.signature(GatedDeltaNet.forward).parameters

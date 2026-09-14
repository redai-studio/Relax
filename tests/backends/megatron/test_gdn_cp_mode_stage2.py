# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only tests for the Task 32 Stage 2 Relax-side GDN CP routing.

Covers the pieces added on top of the Stage 1 FLA/MCore backport
(``test_gdn_chunkwise_cp_layout.py``): the ``--linear-cp-mode`` CLI, invalid
Chunkwise combinations, and the thin runtime dispatcher installed on
``GatedDeltaNet.forward``.

The dispatcher tests drive ``GatedDeltaNet.forward`` through duck-typed fakes
and hook/counter spies instead of a real distributed process group or FLA
kernel call, per task32-stage2-handoff.md §6.2 ("use hooks/counters, don't
infer routing from numerics"). Real-kernel / real-collective coverage stays in
``test_gdn_chunkwise_cp_gpu.py``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # noqa: E402

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402
from relax.backends.megatron import model as gdn_model  # noqa: E402
from relax.backends.megatron.arguments import _validate_linear_cp_mode  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: CLI flag
# ---------------------------------------------------------------------------
def _parse_megatron_args(monkeypatch, *argv):
    pytest.importorskip("sglang.srt.server_args")
    from relax.utils.arguments import get_slime_extra_args_provider

    monkeypatch.setattr("sys.argv", ["test-linear-cp-mode", *argv])
    return megatron_arguments._megatron_parse_args(
        extra_args_provider=get_slime_extra_args_provider(),
        ignore_unknown_args=False,
    )


def test_linear_cp_mode_flag_defaults_to_headwise(monkeypatch):
    args = _parse_megatron_args(monkeypatch)
    assert args.linear_cp_mode == "headwise"


@pytest.mark.parametrize("mode", ["headwise", "chunkwise", "all_gather"])
def test_linear_cp_mode_flag_accepts_all_concrete_modes(monkeypatch, mode):
    args = _parse_megatron_args(monkeypatch, "--linear-cp-mode", mode)
    assert args.linear_cp_mode == mode


# ---------------------------------------------------------------------------
# Step 2: argument validation
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        linear_cp_mode="headwise",
        allgather_cp=False,
        deterministic_mode=False,
        dynamic_context_parallel=False,
        context_parallel_size=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("bad", ["auto", "allgather"])
def test_validate_linear_cp_mode_rejects_unsupported_value(bad):
    with pytest.raises(ValueError, match="does not support 'auto'|must be one of"):
        _validate_linear_cp_mode(_args(linear_cp_mode=bad))


def test_validate_linear_cp_mode_rejects_chunkwise_with_allgather_cp():
    with pytest.raises(ValueError, match="allgather-cp"):
        _validate_linear_cp_mode(_args(linear_cp_mode="chunkwise", allgather_cp=True))


@pytest.mark.parametrize("cp_kwargs", [{"context_parallel_size": 2}, {"dynamic_context_parallel": True}])
def test_validate_linear_cp_mode_rejects_chunkwise_deterministic_when_cp_may_exceed_one(cp_kwargs):
    with pytest.raises(ValueError, match="deterministic"):
        _validate_linear_cp_mode(_args(linear_cp_mode="chunkwise", deterministic_mode=True, **cp_kwargs))


# ---------------------------------------------------------------------------
# Steps 4-6: runtime dispatcher
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
    cannot leak into test_gdn_chunkwise_cp_gpu.py."""
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

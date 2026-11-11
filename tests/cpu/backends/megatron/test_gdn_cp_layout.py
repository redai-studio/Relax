# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU checks for the packed GDN boundary and layout plan caches."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.context_parallel import layout
from megatron.core.packed_seq_params import PackedSeqParams


def _cu(lengths, unit=1):
    lengths = torch.tensor([0, *lengths], dtype=torch.int32)
    return lengths.cumsum(0, dtype=torch.int32) * unit


def _packed_seq_params(cu):
    return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_both_directions_share_one_cached_plan(monkeypatch, cp_size):
    plans = []

    def inspect_layout(value, source, target, group, thd_plan):
        plans.append(thd_plan)
        return value

    monkeypatch.setattr(layout, "convert_cp_layout", inspect_layout)
    cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0)
    cu = _cu([3, 0, 1], unit=2 * cp_size)
    packed = _packed_seq_params(cu)
    x = torch.randn(8, 1, 2)
    layout.convert_relax_cp_layout(x, packed, "zigzag", "contiguous", cp_group)
    layout.convert_relax_cp_layout(x, packed, "contiguous", "zigzag", cp_group)
    assert plans[0] is plans[1]

    cu.copy_(_cu([2, 0, 2], unit=2 * cp_size))
    layout.convert_relax_cp_layout(x, packed, "zigzag", "contiguous", cp_group)
    assert plans[2] is not plans[0]


@pytest.mark.parametrize("boundaries", [[1, 8, 16], [0, 12, 8, 16], [0, 6, 16], [0, 12]])
def test_layout_rejects_invalid_boundaries(boundaries):
    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    packed = _packed_seq_params(torch.tensor(boundaries, dtype=torch.int32))
    with pytest.raises(ValueError, match="boundaries"):
        layout.convert_relax_cp_layout(torch.randn(8, 1, 2), packed, "zigzag", "contiguous", cp_group)


def _boundary_resolver():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    calls = []
    module = SimpleNamespace(cp_size=8)

    def resolve(*args, **kwargs):
        calls.append(kwargs["cp_size"])
        return GatedDeltaNet._resolve_cu_seqlens(module, *args, **kwargs)

    module._resolve_cu_seqlens = resolve
    return module, calls, GatedDeltaNet._resolve_thd_cu_seqlens


def test_boundary_validation_is_shared_across_layers_and_recompute():
    module, calls, resolve = _boundary_resolver()
    # Static max CP=8 would reject a length of 12; runtime CP=2 is legal.
    cu = _cu([3, 1], unit=4)
    psp = _packed_seq_params(cu)
    first = resolve(module, psp, 16, 2)
    assert calls == [2, 2]
    assert resolve(module, psp, 16, 2) is first
    assert calls == [2, 2]

    other_module, other_calls, _ = _boundary_resolver()
    assert resolve(other_module, psp, 16, 2) is first
    assert other_calls == []


@pytest.mark.parametrize("change", ["replace", "inplace", "view", "new_pack", "runtime_cp", "global_length", "padded"])
def test_boundary_cache_is_invalidated_when_inputs_change(change):
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    cp_size, total = 2, 32
    if change == "replace":
        psp.cu_seqlens_q = cu.clone()
        psp.cu_seqlens_kv = psp.cu_seqlens_q
    elif change == "inplace":
        cu.copy_(_cu([2, 2], unit=8))
    elif change == "view":
        cu[1:2].fill_(16)
    elif change == "new_pack":
        psp = _packed_seq_params(cu)
    elif change == "runtime_cp":
        cp_size = 4
    elif change == "global_length":
        total = 64
    elif change == "padded":
        psp.cu_seqlens_q_padded = _cu([2, 2], unit=8)
        psp.cu_seqlens_kv_padded = psp.cu_seqlens_q_padded
    if change == "global_length":
        with pytest.raises(ValueError, match="total_sequence_length"):
            resolve(module, psp, total, cp_size)
        assert len(calls) == 3
    else:
        resolve(module, psp, total, cp_size)
        assert len(calls) == 4


def test_boundary_validation_rechecks_inference_tensors():
    module, calls, resolve = _boundary_resolver()
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        psp = _packed_seq_params(cu)
        resolve(module, psp, 32, 2)
        cu[1] = 16
        resolve(module, psp, 32, 2)
        assert len(calls) == 4


def test_boundary_cache_does_not_hide_q_kv_mismatch():
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    psp.cu_seqlens_kv = _cu([2, 2], unit=8)
    with pytest.raises(AssertionError, match="cu_seqlens_q equals"):
        resolve(module, psp, 32, 2)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.backends.fsdp_turbo.qwen4_gdn import _triton_convolution, configure


def module(monkeypatch, name, **members):
    value = ModuleType(name)
    value.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, value)


@pytest.mark.parametrize("skip", [False, True])
def test_explicit_policy_reaches_kernel_from_cp1_and_cp(monkeypatch, skip):
    registered = {}

    def register(name, device):
        def save(fn):
            registered[name] = fn
            return fn

        return save

    def kernel(value, *, skip_recompute=False):
        return value, skip_recompute

    module(monkeypatch, "fsdp_turbo.ops.registry", register_op=register)
    module(monkeypatch, "fsdp_turbo.ops.fla_compat", stock_chunk_gated_delta_rule=lambda: kernel)
    options = SimpleNamespace(
        skip_gdn_recompute=skip, gdn_implementation="triton", causal_conv1d_implementation="auto"
    )
    configure(SimpleNamespace(relax_performance=options), SimpleNamespace(skip_gdn_recompute=False))
    call = registered["chunk_gated_delta_rule"]
    assert call(7, implementation="auto", head_first=False) == (7, skip)
    assert call(7) == (7, skip)


def test_auto_rejects_unverifiable_policy():
    options = SimpleNamespace(skip_gdn_recompute=True, gdn_implementation="auto")
    with pytest.raises(ValueError, match="specific GDN"):
        configure(SimpleNamespace(relax_performance=options), SimpleNamespace())


def test_triton_convolution_preserves_packed_metadata(monkeypatch):
    def kernel(x, weight, *, cu_seqlens, output_final_state):
        return x, weight, cu_seqlens, output_final_state

    module(monkeypatch, "fsdp_turbo.ops.triton.causal_conv1d", causal_conv1d_triton=kernel)
    assert _triton_convolution(1, 2, head_num=8, implementation="auto", cu_seqlens=3, output_final_state=True) == (
        1,
        2,
        3,
        True,
    )


@pytest.mark.parametrize("skip", [None, True, False])
def test_ascendc_fixed_recompute_policy(monkeypatch, skip):
    registered = {}

    def register(name, device):
        def save(fn):
            registered[name] = fn
            return fn

        return save

    module(monkeypatch, "fsdp_turbo.ops.registry", register_op=register)
    module(monkeypatch, "fsdp_turbo.ops.npu.chunk_gdn_ascendc", chunk_gdn_ascendc=lambda value: value)
    options = SimpleNamespace(
        skip_gdn_recompute=skip, gdn_implementation="ascendc", causal_conv1d_implementation="auto"
    )
    config = SimpleNamespace(relax_performance=options)
    if skip is False:
        with pytest.raises(ValueError, match="cannot honor"):
            configure(config, SimpleNamespace(skip_gdn_recompute=True))
    else:
        configure(config, SimpleNamespace(skip_gdn_recompute=True))
        assert registered["chunk_gated_delta_rule"](7, implementation="auto", head_first=False) == 7

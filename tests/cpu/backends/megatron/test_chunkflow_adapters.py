# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron.chunkflow import adapters
from relax.backends.megatron.chunkflow.adapters import (
    AttentionAdapter,
    DenseAttentionCall,
    dense_attention_adapter,
    megatron_native_attention,
    register_attention_adapter,
)
from relax.backends.megatron.chunkflow.integration import install, uninstall
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops
from tests.cpu.backends.megatron.test_chunkflow_attention_ops import SDPAttention, native_sdpa
from tests.cpu.backends.megatron.test_chunkflow_integration import restore_rope_patch as restore_rope_patch


class NativeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(context_parallel_size=1, attention_dropout=0, apply_rope_fusion=True)
        self.layer_number = 1
        self.projection = torch.nn.Linear(4, 12, bias=False, dtype=torch.float64)
        self.core_attention = SDPAttention(gain=True)
        self.core_attention.softmax_scale = 0.31


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(adapters, "_ADAPTERS", dict(adapters._ADAPTERS))
    register_attention_adapter(
        NativeLayer, SDPAttention, dense_attention_adapter(name="native-test", execute=native_sdpa)
    )


@pytest.mark.parametrize("keep_last_k", [1, 1000])
def test_native_module_packed_outputs_input_and_parameter_gradients(keep_last_k):
    torch.manual_seed(32)
    layer = NativeLayer()
    reference = copy.deepcopy(layer)
    lengths = [11, 7, 2, 1]
    x = torch.randn(sum(lengths), 4, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    q, k, v = reference.projection(xr).reshape(-1, 3, 1, 4).unbind(1)
    expected = []
    start = 0
    for length in lengths:
        sl = slice(start, start + length)
        output = (
            F.scaled_dot_product_attention(
                q[sl].transpose(0, 1),
                k[sl].transpose(0, 1),
                v[sl].transpose(0, 1),
                is_causal=True,
                scale=reference.core_attention.softmax_scale,
            )
            .transpose(0, 1)
            .reshape(length, -1)
        )
        expected.append(output * reference.core_attention.gain)
        start += length
    expected = torch.cat(expected)
    dout = torch.randn_like(expected)
    expected.backward(dout)

    core, original_forward = layer.core_attention, layer.core_attention.forward
    parameters, state_keys = tuple(layer.parameters()), tuple(layer.state_dict())
    assert install(layer, 4) == 1
    assert install(layer, 4) == 0
    assert layer.core_attention is core
    assert tuple(layer.parameters()) == parameters
    assert tuple(layer.state_dict()) == state_keys

    chunks = build_chunks(lengths, 4)
    units = build_micro_batches(chunks)
    pending = {}
    actual = torch.empty_like(expected)
    for kind, uid in gen_unit_ops(units, chunks, keep_last_k):
        unit = units[uid]
        indices = [
            sum(lengths[: s.seq_id]) + i
            for c in unit.chunks
            for s in c.segments
            for i in range(s.prefix, s.prefix + s.length)
        ]
        if kind == "B":
            pending.pop(uid).backward(dout[indices])
            for group in completed_groups(unit):
                layer.chunkflow_runtime.release_group(group)
            continue
        cu = [0]
        for chunk in unit.chunks:
            for segment in chunk.segments:
                cu.append(cu[-1] + segment.length)
        packed = SimpleNamespace(
            cu_seqlens_q_cpu=cu,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
            **{UNIT_ATTR: unit, RECOMPUTE_ATTR: kind == "Fr"},
        )
        with torch.set_grad_enabled(kind != "Fp"):
            q, k, v = layer.projection(x[indices]).reshape(-1, 3, 1, 4).unbind(1)
            output = core(q, k, v, packed_seq_params=packed)
        if kind != "Fp":
            pending[uid] = output
            actual[indices] = output.detach()

    layer.chunkflow_runtime.assert_empty("native test")
    for got, want in ((actual, expected), (x.grad, xr.grad)):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)
    for got, want in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-9, atol=1e-9)
    assert uninstall(layer) == 1
    assert uninstall(layer) == 0
    assert core.forward == original_forward
    assert layer.config.apply_rope_fusion is True
    assert tuple(layer.state_dict()) == state_keys


@pytest.mark.parametrize(
    "feature,value",
    [
        ("window_size", (4, 0)),
        ("attention_dropout", 0.1),
        ("softmax_type", "learnable"),
        ("ple_layer_ids", [1]),
        ("recompute_granularity", "full"),
        ("checkpoint_core_attention", True),
        ("_relax_qwen35_optimized", True),
    ],
)
def test_unsupported_features_do_not_partially_install(feature, value):
    model = torch.nn.ModuleList([NativeLayer(), NativeLayer()])
    setattr(model[1].config, feature, value)
    forwards = [layer.core_attention.forward for layer in model]
    with pytest.raises(NotImplementedError):
        install(model, 4)
    assert [layer.core_attention.forward for layer in model] == forwards
    assert all(layer.config.apply_rope_fusion for layer in model)
    assert not hasattr(model, "chunkflow_runtime")


def test_unknown_outer_attention_does_not_inherit_dense_adapter():
    class SparseLayer(NativeLayer):
        pass

    layer = SparseLayer()
    original = layer.core_attention.forward
    with pytest.raises(NotImplementedError, match="SparseLayer.*SDPAttention"):
        install(layer, 4)
    assert layer.core_attention.forward == original
    assert not hasattr(layer, "chunkflow_runtime")


def test_uninstalled_model_accepts_a_new_chunk_size():
    layer = NativeLayer()
    # TE represents unrestricted causal attention with this window pair.
    layer.core_attention.window_size = (-1, 0)
    assert install(layer, 4) == 1
    with pytest.raises(ValueError, match="different chunk size"):
        install(layer, 8)
    assert uninstall(layer) == 1
    assert not hasattr(layer, "chunkflow_runtime")
    assert install(layer, 8) == 1
    assert layer.chunkflow_runtime.chunk_size == 8
    assert uninstall(layer) == 1


def test_failed_adapter_install_restores_previously_installed_layers():
    class FailingLayer(NativeLayer):
        pass

    def fail(*args):
        raise RuntimeError("native setup failed")

    register_attention_adapter(FailingLayer, SDPAttention, AttentionAdapter("failure", lambda layer: None, fail))
    model = torch.nn.ModuleList([NativeLayer(), FailingLayer()])
    original = model[0].core_attention.forward
    with pytest.raises(RuntimeError, match="native setup failed"):
        install(model, 4)
    assert model[0].core_attention.forward == original
    assert not hasattr(model, "chunkflow_runtime")
    assert not hasattr(model[0], "_chunkflow_attention_adapter")


def test_native_bridge_uses_physical_boundaries_and_bottom_right_mask(monkeypatch):
    # Only metadata classes are stubbed. Numerical attention uses native SDPA;
    # the real MCore/TE forward is exercised by the separate CUDA integration test.
    @dataclass
    class PackedSeqParams:
        qkv_format: str
        cu_seqlens_q: torch.Tensor
        cu_seqlens_kv: torch.Tensor
        max_seqlen_q: int
        max_seqlen_kv: int
        cu_seqlens_q_padded: object = None
        cu_seqlens_kv_padded: object = None

    class AttnMaskType(Enum):
        causal = 1

    for name, attr, value in (
        ("megatron.core.packed_seq_params", "PackedSeqParams", PackedSeqParams),
        ("megatron.core.transformer.enums", "AttnMaskType", AttnMaskType),
    ):
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    torch.manual_seed(15)
    q = torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True)
    k, v = [torch.randn(5, 1, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    call = DenseAttentionCall(
        q,
        (k[:3], k[3:]),
        (v[:3], v[3:]),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 5], dtype=torch.int32),
        (2,),
        (5,),
        (2, 5),
        0.5,
    )

    def original(q, k, v, attention_mask, attn_mask_type, packed_seq_params):
        packed = packed_seq_params
        assert attention_mask is None
        assert packed.cu_seqlens_q_padded is packed.cu_seqlens_kv_padded is None
        assert packed.max_seqlen_q == 2 and packed.max_seqlen_kv == 5
        torch.testing.assert_close(packed.cu_seqlens_q, call.cu_q)
        torch.testing.assert_close(packed.cu_seqlens_kv, call.cu_kv)
        right_aligned = attn_mask_type.name == "padding_causal_bottom_right"
        positions = torch.arange(2) + (3 if right_aligned else 0)
        mask = torch.arange(5)[None] <= positions[:, None]
        return F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
        ).transpose(0, 1)

    result = megatron_native_attention(original, call)
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                q[i : i + 1].transpose(0, 1), k[: 4 + i].transpose(0, 1), v[: 4 + i].transpose(0, 1)
            ).transpose(0, 1)
            for i in range(2)
        ]
    )
    torch.testing.assert_close(result, expected)
    actual_grads = torch.autograd.grad(result.sum(), (q, k, v))
    reference_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    for got, want in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, want)

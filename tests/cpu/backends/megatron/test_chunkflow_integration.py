# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core attention execution, recomputation and installation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.backends.megatron.chunkflow.integration import (
    ChunkedCoreAttention,
    install,
    uninstall,
)
from relax.backends.megatron.chunkflow.runtime import ChunkflowRuntime
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.units import build_micro_batches


@pytest.fixture(autouse=True)
def restore_rope_patch():
    from relax.backends.megatron.chunkflow import rope

    original = rope._ORIGINAL_GET_ROTARY_SEQ_LEN
    if original is not None:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        patched_method = RotaryEmbedding.get_rotary_seq_len
    try:
        yield
    finally:
        if original is None:
            rope.uninstall()
        else:
            rope._ORIGINAL_GET_ROTARY_SEQ_LEN = original
            RotaryEmbedding.get_rotary_seq_len = patched_method


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = torch.nn.Identity()


def _sdpa_reference(q, k, v):
    repeat = q.shape[1] // k.shape[1]
    qh, kh, vh = (
        t.permute(1, 0, 2) for t in (q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1))
    )
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    return out.permute(1, 0, 2).reshape(q.shape[0], -1)


def _qkv(t, h, d, seed=0, kv_heads=None):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(t, h, d, generator=g, dtype=torch.float64)
    kv_h = kv_heads or h
    k = torch.randn(t, kv_h, d, generator=g, dtype=torch.float64)
    v = torch.randn(t, kv_h, d, generator=g, dtype=torch.float64)
    return q, k, v


def _attn(chunk_size, cp_size=1):
    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=cp_size), layer_number=1, chunk_size=chunk_size
    )


@pytest.mark.parametrize("chunk_size,heads,kv_heads", [(size, 2, 2) for size in (1, 3, 16)] + [(3, 4, 2)])
def test_chunkflow_output_and_gradients_match_sdpa(chunk_size, heads, kv_heads):
    q, k, v = _qkv(t=8, h=heads, d=4, kv_heads=kv_heads)
    reference = [t.clone().requires_grad_(True) for t in (q, k, v)]
    chunked = [t.clone().requires_grad_(True) for t in (q, k, v)]
    expected = _sdpa_reference(*reference)
    actual = _attn(chunk_size)(*chunked)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    expected.sum().backward()
    actual.sum().backward()
    for got, want in zip(chunked, reference):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-9, atol=1e-9)


def test_chunkflow_packed_sequences_are_attended_independently():
    q, k, v = _qkv(t=10, h=2, d=4)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 6, 10]), cu_seqlens_q_cpu=[0, 6, 10])
    got = _attn(chunk_size=4)(q, k, v, packed_seq_params=packed)
    want = torch.cat([_sdpa_reference(q[:6], k[:6], v[:6]), _sdpa_reference(q[6:], k[6:], v[6:])], dim=0)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_chunkflow_context_parallel_is_rejected():
    q, k, v = _qkv(t=4, h=2, d=4)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _attn(chunk_size=2, cp_size=2)(q, k, v)


def test_chunkflow_sbhd_is_rejected():
    with pytest.raises(NotImplementedError, match="THD"):
        _attn(chunk_size=2)(torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4))


def test_chunkflow_install_swaps_and_uninstall_restores():
    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    original = model[0].core_attention
    model[0].config.apply_rope_fusion = True

    assert install(model, chunk_size=4) == 2
    assert isinstance(model[0].core_attention, ChunkedCoreAttention)
    assert model[0].config.apply_rope_fusion is False
    assert install(model, chunk_size=4) == 0

    assert uninstall(model) == 2
    assert model[0].core_attention is original


def _unit_attn(chunk_size, layer=1):

    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=1),
        layer_number=layer,
        chunk_size=chunk_size,
        runtime=ChunkflowRuntime(chunk_size),
    )


def test_chunkflow_keeps_native_moe_parameters_and_methods():
    moe = torch.nn.Module()
    moe.router = torch.nn.Linear(4, 2)
    moe.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
    moe.token_dispatcher = SimpleNamespace(token_dispatch=lambda x: x, token_combine=lambda x: x)
    model = torch.nn.ModuleList([_FakeAttn(), moe])
    parameters = tuple(model.parameters())
    methods = (moe.router.forward, *(expert.forward for expert in moe.experts))
    dispatch = dict(vars(moe.token_dispatcher))

    assert install(model, 4) == 1
    assert tuple(model.parameters()) == parameters
    assert (moe.router.forward, *(expert.forward for expert in moe.experts)) == methods
    assert vars(moe.token_dispatcher) == dispatch
    assert uninstall(model) == 1


@pytest.mark.parametrize("original_cache_disabled", [False, True])
def test_reversed_backward_preserves_fused_weight_gradients(original_cache_disabled):
    class FusedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, first):
            ctx.save_for_backward(x, weight)
            ctx.first = first
            return x @ weight.T

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            wgrad = grad.T @ x
            # Transformer Engine overwrites main_grad when first=True.
            if ctx.first is True:
                weight.main_grad.copy_(wgrad)
            else:
                weight.main_grad.add_(wgrad)
            return grad @ weight, None, None

    model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    model.disable_parameter_transpose_cache = original_cache_disabled
    model.is_first_microbatch = True
    model.weight.main_grad = torch.zeros_like(model.weight)
    inputs = [torch.arange(12, dtype=torch.float64).reshape(4, 3) + offset for offset in (1, 3)]
    reference = model.weight.detach().clone().requires_grad_()
    sum((x @ reference.T).square().sum() for x in inputs).backward()

    install(model, 4)
    install(model, 4)
    outputs = []
    for x in inputs:
        first = None if model.disable_parameter_transpose_cache else model.is_first_microbatch
        outputs.append(FusedLinear.apply(x, model.weight, first))
        model.is_first_microbatch = False
    for out in reversed(outputs):
        out.square().sum().backward()
    torch.testing.assert_close(model.weight.main_grad, reference.grad)
    uninstall(model)
    assert model.disable_parameter_transpose_cache is original_cache_disabled


def test_unit_mode_passes_cpu_lengths_with_prefix_and_padding(monkeypatch):
    from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention
    from relax.backends.megatron.chunkflow.runtime import UNIT_ATTR

    recorded = []
    metadata = []
    original = UnitCacheAttention.apply

    def capture(*args):
        recorded.append((args[0].shape[0], args[-1]))
        metadata.append(args[3:5])
        fragments = args[5:-1]
        assert len(fragments) == (2 if args[0].shape[0] == 8 else 4)
        assert all(t.grad_fn is None or "CatBackward" not in t.grad_fn.name() for t in fragments)
        return original(*args)

    monkeypatch.setattr(UnitCacheAttention, "apply", capture)
    attn, next_layer = _unit_attn(8), _unit_attn(8, layer=2)
    units = build_micro_batches(build_chunks([13], chunk_size=8), "staggered")
    for unit in units:
        length = sum(seg.length for chunk in unit.chunks for seg in chunk.segments)
        cu_q = torch.tensor([0, length], dtype=torch.int32)
        packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], cu_seqlens_q=cu_q, **{UNIT_ATTR: unit})
        q, k, v = (t.requires_grad_() for t in _qkv(length + 3, 4, 8, kv_heads=2))
        output = attn(q, k, v, packed_seq_params=packed)
        other = next_layer(q, k, v, packed_seq_params=packed)
        torch.testing.assert_close(output, other)
        assert metadata[-1][0] is metadata[-2][0] is cu_q
        assert metadata[-1][1] is metadata[-2][1]
        assert output.shape[0] == length + 3
    assert recorded == [(8, (8, 8)), (8, (8, 8)), (5, (5, 13)), (5, (5, 13))]


@pytest.mark.parametrize("keep_last_k", [1 << 30, 1])
@pytest.mark.parametrize("fp32", [False, True])
def test_unit_schedule_outputs_gradients_and_release_match_sdpa(keep_last_k, fp32, monkeypatch):
    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", str(int(fp32)))
    from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
    from relax.backends.megatron.chunkflow.units import completed_groups, gen_unit_ops

    torch.manual_seed(0)
    total, chunk_size = 64, 16
    reference = [t.requires_grad_(True) for t in _qkv(total, 4, 8, kv_heads=2)]
    chunked = [t.detach().clone().requires_grad_(True) for t in reference]
    expected_output = _sdpa_reference(*reference)
    dout = torch.randn_like(expected_output)
    expected_output.backward(dout)
    attn = _unit_attn(chunk_size)
    chunks = build_chunks([total], chunk_size=chunk_size)
    units = build_micro_batches(chunks, "staggered")
    ops = gen_unit_ops(units, chunks, keep_last_k)
    expected_fr = set(range(len(units) - 1)) if keep_last_k == 1 else set()
    assert {uid for kind, uid in ops if kind == "Fp"} == {uid for kind, uid in ops if kind == "Fr"} == expected_fr

    pending, outputs = {}, {}
    for kind, uid in ops:
        unit = units[uid]
        start = unit.chunks[0].segments[0].prefix
        length = unit.num_tokens
        if kind == "B":
            pending.pop(uid).backward(dout[start : start + length])
            for group in completed_groups(unit):
                assert attn.runtime.release_group(group) == 1
            continue
        packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], **{UNIT_ATTR: unit, RECOMPUTE_ATTR: kind == "Fr"})
        inputs = [t[start : start + length] for t in chunked]
        if kind == "Fp":
            with torch.no_grad():
                attn(*inputs, packed_seq_params=packed)
        else:
            pending[uid] = attn(*inputs, packed_seq_params=packed)
            outputs[uid] = pending[uid].detach()
    assert not pending
    attn.runtime.assert_empty("unit schedule")
    torch.testing.assert_close(torch.cat([outputs[u.unit_id] for u in units]), expected_output, atol=1e-9, rtol=1e-9)
    for got, want in zip(chunked, reference):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_kv_cache_fp32_accumulates_across_backward_without_changing_values(monkeypatch, dtype):
    from relax.backends.megatron.chunkflow.integration import _cache_leaf, _fold_leaf_grad_into

    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", "1")
    live = torch.zeros(2, 1, 4, dtype=dtype, requires_grad=True)
    leaf = _cache_leaf(live)
    assert leaf.dtype == dtype and torch.equal(leaf, live)
    for coefficient in (256, 1, 1):
        (leaf * coefficient).sum().backward()
        assert leaf.grad is None
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert leaf._chunkflow_grad_sum.dtype == expected_dtype
    torch.testing.assert_close(leaf._chunkflow_grad_sum, torch.full_like(leaf._chunkflow_grad_sum, 258))
    _fold_leaf_grad_into(live, leaf)
    (live * 0).sum().backward()
    torch.testing.assert_close(live.grad, torch.full_like(live, 258), rtol=0, atol=0)
    monkeypatch.setenv("RELAX_CHUNKFLOW_KV_GRAD_FP32", "0")
    old_live = torch.zeros_like(live, requires_grad=True)
    old_leaf = _cache_leaf(old_live)
    for coefficient in (256, 1, 1):
        (old_leaf * coefficient).sum().backward()
    _fold_leaf_grad_into(old_live, old_leaf)
    (old_live * 0).sum().backward()
    torch.testing.assert_close(
        old_live.grad, torch.full_like(old_live, 256 if dtype == torch.bfloat16 else 258), rtol=0, atol=0
    )

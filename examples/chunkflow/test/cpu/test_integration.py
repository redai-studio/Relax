# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core attention execution, recomputation and installation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.chunkflow.adapters import DenseAttentionCall, dense_attention_adapter, register_attention_adapter
from examples.chunkflow.integration import (
    ChunkedCoreAttention,
    install,
    uninstall,
)
from examples.chunkflow.runtime import ChunkflowRuntime, require_runtime
from examples.chunkflow.scheduler import build_chunks
from examples.chunkflow.units import build_micro_batches


@pytest.fixture(autouse=True)
def restore_rope_patch():
    from examples.chunkflow import rope

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


class SDPAttention(torch.nn.Module):
    """Native PyTorch oracle shared by dense adapter and gradient tests."""

    def __init__(self, gain: bool = False):
        super().__init__()
        if gain:
            self.gain = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, query, key, value, *, q_lengths, kv_lengths, scale):
        outputs = []
        q_start = k_start = 0
        repeat = query.shape[1] // key.shape[1]
        for q_length, kv_length in zip(q_lengths, kv_lengths):
            q = query[q_start : q_start + q_length]
            k = key[k_start : k_start + kv_length].repeat_interleave(repeat, dim=1)
            v = value[k_start : k_start + kv_length].repeat_interleave(repeat, dim=1)
            q_pos = torch.arange(q_length, device=q.device) + kv_length - q_length
            k_pos = torch.arange(kv_length, device=q.device)
            mask = k_pos[None, :] <= q_pos[:, None]
            output = F.scaled_dot_product_attention(
                q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2), attn_mask=mask, scale=scale
            ).permute(1, 0, 2)
            outputs.append(output * self.gain if hasattr(self, "gain") else output)
            q_start += q_length
            k_start += kv_length
        return torch.cat(outputs)


def native_sdpa(original, call: DenseAttentionCall):
    key, value = call.joined_kv()
    return original(
        call.query,
        key,
        value,
        q_lengths=call.q_lengths,
        kv_lengths=call.kv_lengths,
        scale=call.softmax_scale,
    )


def test_schedule_reuses_runtime_installed_below_model_wrappers():
    class Wrapper:
        def __init__(self, module):
            self.module = module

    bare = Wrapper(None)
    runtime = ChunkflowRuntime(chunk_size=16)
    bare.chunkflow_runtime = runtime
    wrapped = Wrapper(Wrapper(bare))
    assert require_runtime(wrapped) is runtime
    assert not hasattr(wrapped, "chunkflow_runtime")


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = SDPAttention()


register_attention_adapter(_FakeAttn, SDPAttention, dense_attention_adapter(name="test-sdpa", execute=native_sdpa))


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


def _attn(cp_size=1):
    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=cp_size),
        layer_number=1,
        executor=lambda call: native_sdpa(SDPAttention().forward, call),
    )


def test_context_parallel_is_rejected():
    q, k, v = _qkv(t=4, h=2, d=4)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _attn(cp_size=2)(q, k, v)


def test_sbhd_is_rejected():
    with pytest.raises(NotImplementedError, match="THD"):
        _attn()(torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 2, 4))


@pytest.mark.parametrize(
    "kwargs",
    [{"attention_mask": torch.ones(4, 4)}, {"attention_bias": torch.ones(4, 4)}, {"attn_mask_type": "no_mask"}],
)
def test_dense_adapter_rejects_unhandled_attention_semantics(kwargs):
    with pytest.raises(NotImplementedError):
        _attn()(*_qkv(4, 2, 4), **kwargs)


def test_install_preserves_module_and_uninstall_restores_forward():
    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    original = model[0].core_attention
    original_forward = original.forward
    model[0].config.apply_rope_fusion = True

    assert install(model, chunk_size=4) == 2
    assert model[0].core_attention is original
    assert original.forward != original_forward
    assert model[0].config.apply_rope_fusion is False
    assert install(model, chunk_size=4) == 0

    assert uninstall(model) == 2
    assert model[0].core_attention is original
    assert original.forward == original_forward
    assert model[0].config.apply_rope_fusion is True


def _qwen3_vl_container():
    # CPU coverage of the explicit container contract; the GPU test constructs
    # the real Bridge model, including its native vision and GDN/MoE branches.
    model_type = type(
        "Qwen3VLModel",
        (torch.nn.Module,),
        {"__module__": "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"},
    )
    model = model_type()
    for name in ("language_model", "vision_model"):
        layer = _FakeAttn()
        layer.config.apply_rope_fusion = True
        layer.projection = torch.nn.Linear(4, 4)
        layer.projection.disable_parameter_transpose_cache = False
        layer.projection.is_first_microbatch = True
        setattr(model, name, layer)
    model.vision_model.attn_mask_type = "no_mask"
    return model


def test_qwen3_vl_install_only_changes_language_subtree():
    model = _qwen3_vl_container()
    text, vision = model.language_model, model.vision_model
    original = (text.core_attention.forward, vision.core_attention.forward)
    parameters, state_keys = tuple(model.parameters()), tuple(model.state_dict())
    assert install(model, 4) == 1
    assert install(model, 4) == 0
    assert require_runtime(model).chunk_size == 4
    assert not hasattr(text, "chunkflow_runtime")
    assert text.core_attention.forward != original[0]
    assert text.config.apply_rope_fusion is False
    assert text.projection.disable_parameter_transpose_cache is True
    assert vision.core_attention.forward == original[1]
    assert vision.config.apply_rope_fusion is True
    assert vision.projection.disable_parameter_transpose_cache is False
    assert tuple(model.parameters()) == parameters and tuple(model.state_dict()) == state_keys
    assert uninstall(model) == 1
    assert text.core_attention.forward == original[0]
    assert text.config.apply_rope_fusion is True
    assert text.projection.disable_parameter_transpose_cache is False
    assert vision.core_attention.forward == original[1]


@pytest.mark.parametrize("invalid", ["no_mask", "unknown_attention", "missing_language", "unknown_container"])
def test_qwen3_vl_subtree_selection_keeps_validation_strict(invalid):
    model = _qwen3_vl_container()
    if invalid == "no_mask":
        model.language_model.attn_mask_type = "no_mask"
        match = "mask type no_mask"
    elif invalid == "unknown_attention":
        model.language_model = type("UnknownAttention", (_FakeAttn,), {})()
        match = "no attention adapter"
    elif invalid == "missing_language":
        model.language_model = None
        match = "requires a language_model"
    else:
        unknown = torch.nn.Module()
        unknown.language_model, unknown.vision_model = model.language_model, model.vision_model
        model = unknown
        match = "mask type no_mask"
    forwards = [layer.core_attention.forward for layer in model.modules() if isinstance(layer, _FakeAttn)]
    with pytest.raises(NotImplementedError, match=match):
        install(model, 4)
    assert not hasattr(model, "chunkflow_runtime")
    assert [layer.core_attention.forward for layer in model.modules() if isinstance(layer, _FakeAttn)] == forwards


def _unit_attn(chunk_size, layer=1):

    return ChunkedCoreAttention(
        config=SimpleNamespace(context_parallel_size=1),
        layer_number=layer,
        runtime=ChunkflowRuntime(chunk_size),
        executor=lambda call: native_sdpa(SDPAttention().forward, call),
    )


def test_keeps_native_moe_parameters_and_methods():
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


def test_unit_mode_passes_cpu_lengths_with_prefix_and_padding():
    from examples.chunkflow.runtime import UNIT_ATTR

    recorded, metadata = [], []

    def capture(call):
        recorded.append((call.query.shape[0], call.max_seqlens))
        metadata.append((call.cu_q, call.cu_kv))
        assert len(call.keys) == (1 if call.query.shape[0] == 8 else 2)
        return native_sdpa(SDPAttention().forward, call)

    attn, next_layer = _unit_attn(8), _unit_attn(8, layer=2)
    attn.executor = next_layer.executor = capture
    units = build_micro_batches(build_chunks([13], chunk_size=8), "staggered")
    for unit in units:
        length = unit.num_tokens
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
    from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR
    from examples.chunkflow.units import completed_groups, gen_unit_ops

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
def test_kv_cache_fp32_accumulates_across_backward_without_changing_values(dtype):
    from examples.chunkflow.state_store import ChunkState

    live = torch.zeros(2, 1, 4, dtype=dtype, requires_grad=True)
    state = ChunkState({})
    state.link_tensor("k", live, accumulate_fp32=True)
    leaf = state.tensors["k"]
    assert leaf.dtype == dtype and torch.equal(leaf, live)
    for coefficient in (256, 1, 1):
        (leaf * coefficient).sum().backward()
        assert leaf.grad is None
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert leaf._chunkflow_grad_sum.dtype == expected_dtype
    torch.testing.assert_close(leaf._chunkflow_grad_sum, torch.full_like(leaf._chunkflow_grad_sum, 258))
    (live * 0).sum().backward()
    torch.testing.assert_close(live.grad, torch.full_like(live, 258), rtol=0, atol=0)
    old_live = torch.zeros_like(live, requires_grad=True)
    old_state = ChunkState({})
    old_state.link_tensor("k", old_live)
    old_leaf = old_state.tensors["k"]
    for coefficient in (256, 1, 1):
        (old_leaf * coefficient).sum().backward()
    (old_live * 0).sum().backward()
    torch.testing.assert_close(
        old_live.grad, torch.full_like(old_live, 256 if dtype == torch.bfloat16 else 258), rtol=0, atol=0
    )


@pytest.mark.parametrize("local_number,missing", [(2, False), (1, True)])
def test_install_checks_only_local_pipeline_ple_layers(local_number, missing):
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    layer = _FakeAttn()
    layer.layer_number = local_number
    layer.config.ple_layer_ids = [1]
    model.decoder.layers = torch.nn.ModuleList([layer])
    if missing:
        with pytest.raises(NotImplementedError, match="local layer 1"):
            install(model, 4)
        assert not hasattr(model, "chunkflow_runtime")
        assert "forward" not in layer.core_attention.__dict__
    else:
        assert install(model, 4) == 1
        assert uninstall(model) == 1

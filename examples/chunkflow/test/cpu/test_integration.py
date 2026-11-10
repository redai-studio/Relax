# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core attention execution, recomputation and installation."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

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
from examples.chunkflow.state_store import ChunkState, StateStore, StateStoreError


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


def test_state_store_preserves_history_order_and_rejects_missing_chunks():
    store = StateStore()
    assert store.lookup(-1) == []
    with pytest.raises(StateStoreError):
        store.lookup(0)
    first, second = ChunkState({}), ChunkState({})
    store.put(0, first)
    with pytest.raises(StateStoreError):
        store.put(2, second)
    store.put(1, second)
    assert store.lookup(0)[0] is first
    history = store.lookup(1)
    assert history[0] is first and history[1] is second
    with pytest.raises(StateStoreError, match=r"\[2, 3\]"):
        store.lookup(3)


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.config = SimpleNamespace(context_parallel_size=1)
        self.core_attention = SDPAttention()


register_attention_adapter(_FakeAttn, SDPAttention, dense_attention_adapter(name="test-sdpa", execute=native_sdpa))


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


def test_install_rejects_deferred_weight_gradients_before_mutation():
    model = torch.nn.Sequential(_FakeAttn(), _FakeAttn())
    model[1].config.defer_embedding_wgrad_compute = True
    model[1].config.wgrad_deferral_limit = 0
    originals = [layer.core_attention.forward for layer in model]
    with pytest.raises(NotImplementedError, match="defer_embedding_wgrad_compute"):
        install(model, 4)
    assert not hasattr(model, "chunkflow_runtime")
    assert [layer.core_attention.forward for layer in model] == originals


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


@pytest.mark.parametrize(
    "mode,coeff,hash_layer,rejected",
    [
        ("aux_loss", 0.01, False, True),
        ("seq_aux_loss", 0.01, False, True),
        ("global_aux_loss", 0.01, False, True),
        (["aux_loss", "seq_aux_loss"], [0.0, 0.01], False, True),
        ("aux_loss", 0.0, False, False),
        ("none", 0.01, False, False),
        ("aux_loss", 0.01, True, False),
    ],
)
def test_install_checks_native_router_objective_before_mutation(mode, coeff, hash_layer, rejected):
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.router import TopKRouter

    router = torch.nn.Module()
    router.routing_type, router.is_hash_layer = mode, hash_layer
    router.config = SimpleNamespace(moe_aux_loss_coeff=coeff, moe_z_loss_coeff=0.01)
    router.get_aux_loss_coeff = MethodType(TopKRouter.get_aux_loss_coeff, router)
    model = torch.nn.ModuleList([_FakeAttn(), router])
    original = model[0].core_attention.forward
    if rejected:
        with pytest.raises(NotImplementedError, match="load-balancing auxiliary loss"):
            install(model, 4)
        assert not hasattr(model, "chunkflow_runtime")
        assert model[0].core_attention.forward == original
    else:
        assert install(model, 4) == 1
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

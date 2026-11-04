# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.chunkflow import rope as cf_rope
from relax.backends.megatron.chunkflow.metadata import int32_metadata
from tests.cpu.backends.megatron.test_chunkflow_attention_ops import fa3_backend as fa3_backend
from tests.cpu.backends.megatron.test_chunkflow_gdn import _alignment
from tests.cpu.backends.megatron.test_chunkflow_gdn import kernels as kernels
from tests.cpu.backends.megatron.test_chunkflow_integration import _attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 numeric validation requires a CUDA GPU")
@pytest.mark.parametrize(
    "bounds_q,bounds_k,heads,kv_heads,dim",
    [
        ([0, 17, 26], [0, 40, 53], 16, 4, 128),
        ([0, 6144, 8192], [0, 6144, 8192], 8, 1, 256),
        ([0, 8192], [0, 8192], 8, 1, 256),
    ],
)
def test_unit_fa3_matches_cuda_reference(monkeypatch, fa3_backend, bounds_q, bounds_k, heads, kv_heads, dim):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires the Hopper FA3 build used by this experiment")
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("FA2 reference kernel is not installed")

    from relax.backends.megatron.chunkflow import flash_backends as fb

    monkeypatch.setenv("RELAX_CHUNKFLOW_FLASH_BACKEND", "fa3")
    try:
        fb.flash3_varlen_ops()
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ImportError):
            pytest.skip("optional FA3 interface is not installed")
        raise
    max_q = max(b - a for a, b in zip(bounds_q[:-1], bounds_q[1:]))
    max_k = max(b - a for a, b in zip(bounds_k[:-1], bounds_k[1:]))
    cu_q = torch.tensor(bounds_q, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(bounds_k, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v = (
        torch.randn(n, h, dim, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
        for n, h in ((bounds_q[-1], heads), (bounds_k[-1], kv_heads), (bounds_k[-1], kv_heads))
    )
    qr, kr, vr = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    parts = [k[a:b] for a, b in zip(bounds_k[:-1], bounds_k[1:])]
    values = [v[a:b] for a, b in zip(bounds_k[:-1], bounds_k[1:])]
    from relax.backends.megatron.chunkflow.attention_ops import UnitCacheAttention

    result, _ = UnitCacheAttention.apply(q, True, dim**-0.5, cu_q, cu_k, *parts, *values, (max_q, max_k))
    expected = flash_attn_varlen_func(qr, kr, vr, cu_q, cu_k, max_q, max_k, causal=True)
    dout = torch.randn_like(result)
    result.backward(dout)
    expected.backward(dout)
    for actual, wanted in ((result, expected), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(actual, wanted, atol=0.02, rtol=0.02)
        assert (actual.float() - wanted.float()).norm() / wanted.float().norm() < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and differentiable FLA state kernels")
@pytest.mark.parametrize("mode", ["fr", "last", "selective", "selective_last"])
def test_gdn_fla_state_outputs_and_gradients_match_full_sequence(kernels, mode):
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    kernels.causal_conv1d, kernels.rule = causal_conv1d, chunk_gated_delta_rule
    _alignment(kernels, mode, chunk_size=64, device="cuda", heads=2, dim=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunkflow_matches_flash_varlen_on_training_shapes():
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    h, kv_h, d = 16, 4, 128
    lengths = [3000, 1096]
    total = sum(lengths)
    bounds = [0, lengths[0], total]

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(total, h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, kv_h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, kv_h, d, generator=g, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(total, h * d, generator=g, device="cuda", dtype=torch.bfloat16)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(bounds, device="cuda", dtype=torch.int32))

    def run(fn):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        fn(*leaves).backward(dout)
        return [leaf.grad for leaf in leaves]

    def chunked(ql, kl, vl):
        return _attn(chunk_size=1024)(ql, kl, vl, packed_seq_params=packed)

    def reference(ql, kl, vl):
        cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        out = flash_attn_varlen_func(ql, kl, vl, cu, cu, max(lengths), max(lengths), causal=True)
        return out.reshape(total, -1)

    rels = {
        name: ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()
        for name, got, want in zip(("dq", "dk", "dv"), run(chunked), run(reference))
    }
    assert max(rels.values()) < 2e-2, f"chunked differs from flash_attn_varlen_func: {rels}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
def test_te_empty_expert_preserves_accumulated_gradient():
    te = pytest.importorskip("transformer_engine.pytorch")
    model = te.GroupedLinear(
        2, 16, 16, bias=False, params_dtype=torch.float32, fuse_wgrad_accumulation=True, device="cuda"
    )
    weights = [model.weight0, model.weight1]
    for weight in weights:
        weight.main_grad = torch.zeros_like(weight)
    inputs = [torch.arange(n * 16, device="cuda", dtype=torch.float32).reshape(n, 16) / 16 for n in (2, 3)]
    outputs = [model(inputs[0], [2, 0], is_first_microbatch=None), model(inputs[1], [0, 3], is_first_microbatch=None)]
    outputs[1].sum().backward()
    accumulated = weights[1].main_grad.clone()
    outputs[0].sum().backward()
    torch.testing.assert_close(weights[1].main_grad, accumulated, rtol=0, atol=0)
    for weight, x in zip(weights, inputs):
        expected = x.sum(dim=0).expand_as(weight)
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to verify asynchronous metadata copies")
def test_chunk_metadata_cuda_uses_current_stream_without_host_sync(monkeypatch):
    monkeypatch.setenv("RELAX_CHUNKFLOW_EXECUTION_OPT", "1")
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream()
    # Warm up allocator/runtime outside the measurement; keep outputs alive through completion.
    with torch.cuda.stream(stream):
        warmup = int32_metadata([0, 8192], device)
    stream.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.cuda.stream(stream):
            outputs = [int32_metadata([0, 8192, 8192 + i], device) + i for i in range(16)]
    stream.synchronize()
    names = {event.name for event in profile.events()}
    assert "cudaStreamSynchronize" not in names
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output.cpu(), torch.tensor([i, 8192 + i, 8192 + 2 * i], dtype=torch.int32))
    assert warmup.numel() == 2


HEAD_DIM = 64


TOTAL = 512


CHUNK = 128


PREFIX = 256


class _Cfg:
    rotary_base = 1_000_000
    rotary_percent = 1.0
    rotary_interleaved = False
    seq_len_interpolation_factor = None
    apply_rope_fusion = False
    mscale = 1.0
    mscale_all_dim = 1.0


@pytest.fixture(scope="module")
def rope_module():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Megatron model-parallel groups")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    own_dist = not dist.is_initialized()
    own_parallel = not parallel_state.is_initialized()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(cf_rope, "_ORIGINAL_GET_ROTARY_SEQ_LEN", cf_rope._ORIGINAL_GET_ROTARY_SEQ_LEN)
        patcher.setattr(RotaryEmbedding, "get_rotary_seq_len", RotaryEmbedding.get_rotary_seq_len)
        if own_dist:
            patcher.setenv("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
            patcher.setenv("MASTER_PORT", os.environ.get("MASTER_PORT", "29593"))
        try:
            if own_dist:
                dist.init_process_group("nccl", rank=0, world_size=1)
            if own_parallel:
                parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            cf_rope.install()
            yield RotaryEmbedding
        finally:
            if own_parallel and parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            if own_dist and dist.is_initialized():
                dist.destroy_process_group()


def _rotated(rope, tensor, packed, *, use_kv: bool):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    max_seqlen = packed.max_seqlen_kv if use_kv else packed.max_seqlen_q
    cu = packed.cu_seqlens_kv_padded if use_kv else packed.cu_seqlens_q_padded
    if cu is None:
        cu = packed.cu_seqlens_kv if use_kv else packed.cu_seqlens_q

    table_len = rope.get_rotary_seq_len(None, None, None, _Cfg(), packed)
    return apply_rotary_pos_emb(
        tensor,
        rope(table_len, packed_seq=True),
        config=_Cfg(),
        cu_seqlens=cu,
        mscale=1.0,
        cp_group=None,
        max_seqlen=max_seqlen,
    )


def _packed(device, lengths, max_q, max_kv):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.tensor(lengths, dtype=torch.int32, device=device)
    return PackedSeqParams(
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_q, max_seqlen_kv=max_kv, qkv_format="thd"
    )


@pytest.mark.parametrize("use_kv", [False, True])
@pytest.mark.parametrize("prefix", [0, PREFIX])
def test_chunk_rope_positions_match_the_full_sequence(rope_module, use_kv, prefix):
    device = "cuda"
    torch.manual_seed(0)
    rope = rope_module(kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_base=1_000_000).to(device)

    full = _packed(device, [0, TOTAL], TOTAL, TOTAL)
    tensor = torch.randn(TOTAL, 4, HEAD_DIM, device=device, dtype=torch.bfloat16)
    reference = _rotated(rope, tensor, full, use_kv=use_kv)[prefix : prefix + CHUNK]

    chunk = _packed(device, [0, CHUNK], CHUNK, CHUNK)
    cf_rope.apply_rope_offset(chunk, offsets=[prefix], lengths=[CHUNK])
    got = _rotated(rope, tensor[prefix : prefix + CHUNK].clone(), chunk, use_kv=use_kv)

    torch.testing.assert_close(got.float(), reference.float(), rtol=0, atol=0)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.chunkflow import rope as cf_rope
from relax.backends.megatron.chunkflow.adapters import DenseAttentionCall, megatron_native_attention
from relax.backends.megatron.chunkflow.integration import ChunkedCoreAttention
from relax.backends.megatron.chunkflow.metadata import int32_metadata
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from relax.backends.megatron.chunkflow.scheduler import build_chunks
from relax.backends.megatron.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops
from tests.cpu.backends.megatron.test_chunkflow_gdn import _alignment
from tests.cpu.backends.megatron.test_chunkflow_gdn import kernels as kernels


def _native_te(heads, kv_heads, dim):
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=heads * dim,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        kv_channels=dim,
        attention_dropout=0,
        params_dtype=torch.bfloat16,
    )
    return TEDotProductAttention(config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native MCore/TE validation requires a CUDA GPU")
@pytest.mark.parametrize(
    "bounds_q,bounds_k,heads,kv_heads,dim",
    [
        ([0, 17, 26], [0, 40, 53], 16, 4, 128),
        ([0, 6144, 8192], [0, 6144, 8192], 8, 1, 256),
        ([0, 8192], [0, 8192], 8, 1, 256),
    ],
)
def test_native_te_matches_cuda_reference(rope_module, bounds_q, bounds_k, heads, kv_heads, dim):
    flash_attn = pytest.importorskip("flash_attn")
    flash_attn_varlen_func = flash_attn.flash_attn_varlen_func
    core = _native_te(heads, kv_heads, dim)
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
    call = DenseAttentionCall(
        q,
        tuple(parts),
        tuple(values),
        cu_q,
        cu_k,
        tuple(b - a for a, b in zip(bounds_q[:-1], bounds_q[1:])),
        tuple(b - a for a, b in zip(bounds_k[:-1], bounds_k[1:])),
        (max_q, max_k),
        dim**-0.5,
    )
    result = megatron_native_attention(core.forward, call).reshape_as(q)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native MCore/TE CUDA forward and backward")
@pytest.mark.parametrize("keep_last_k", [1, 1000])
def test_native_te_history_gradients_and_recomputation(rope_module, keep_last_k):
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType

    torch.manual_seed(27)
    total, chunk_size, heads, kv_heads, dim = 317, 128, 4, 2, 64
    core = _native_te(heads, kv_heads, dim)
    leaves = [
        torch.randn(total, h, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (heads, kv_heads, kv_heads)
    ]
    reference = [t.detach().clone().requires_grad_() for t in leaves]
    full_cu = int32_metadata([0, total], torch.device("cuda"))
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=full_cu, cu_seqlens_kv=full_cu, max_seqlen_q=total, max_seqlen_kv=total
    )
    expected = core(
        *reference, attention_mask=None, attn_mask_type=AttnMaskType.causal, packed_seq_params=packed
    ).reshape(total, -1)
    dout = torch.randn_like(expected)
    expected.backward(dout)
    runtime = ChunkflowRuntime(chunk_size)
    adapter = ChunkedCoreAttention(
        config=core.config,
        layer_number=1,
        runtime=runtime,
        executor=lambda call: megatron_native_attention(core.forward, call),
    )
    chunks = build_chunks([total], chunk_size)
    units = build_micro_batches(chunks)
    actual, pending = {}, {}
    for kind, uid in gen_unit_ops(units, chunks, keep_last_k):
        unit = units[uid]
        start = unit.chunks[0].segments[0].prefix
        sl = slice(start, start + unit.num_tokens)
        if kind == "B":
            pending.pop(uid).backward(dout[sl])
            for group in completed_groups(unit):
                runtime.release_group(group)
            continue
        cu = int32_metadata([0, unit.num_tokens], torch.device("cuda"))
        packed = SimpleNamespace(
            cu_seqlens_q_cpu=[0, unit.num_tokens], cu_seqlens_q=cu, **{UNIT_ATTR: unit, RECOMPUTE_ATTR: kind == "Fr"}
        )
        with torch.set_grad_enabled(kind != "Fp"):
            output = adapter(*(t[sl] for t in leaves), packed_seq_params=packed)
        if kind != "Fp":
            pending[uid], actual[uid] = output, output.detach()
    runtime.assert_empty("native TE history")
    torch.testing.assert_close(torch.cat([actual[u.unit_id] for u in units]), expected, atol=0.02, rtol=0.02)
    for got, want in zip(leaves, reference):
        torch.testing.assert_close(got.grad, want.grad, atol=0.03, rtol=0.03)


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
    mrope_section = None


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

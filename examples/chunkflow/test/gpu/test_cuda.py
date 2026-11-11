# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from examples.chunkflow import rope as cf_rope
from examples.chunkflow.adapters import DenseAttentionCall, megatron_native_attention
from examples.chunkflow.integration import install, uninstall
from examples.chunkflow.metadata import int32_metadata
from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from examples.chunkflow.scheduler import build_chunks
from examples.chunkflow.test.cpu.test_gdn import _alignment
from examples.chunkflow.test.cpu.test_gdn import kernels as kernels
from examples.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops


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


def _run_native_te_check(check: str, *args) -> None:
    # Match the recipes' --attention-backend flash before TE imports or caches
    # a backend. A fresh process also isolates this setting from other tests.
    environment = {
        **os.environ,
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
    }
    code = """import json
import runpy
import sys

module = runpy.run_path(sys.argv[1])
with module["_megatron_context"]():
    module[sys.argv[2]](*json.loads(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, check, json.dumps(args)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native MCore/TE validation requires a CUDA GPU")
@pytest.mark.parametrize(
    "bounds_q,bounds_k,heads,kv_heads,dim",
    [
        ([0, 17, 26], [0, 40, 53], 16, 4, 128),
        ([0, 6144, 8192], [0, 6144, 8192], 8, 1, 256),
    ],
)
def test_native_te_matches_cuda_reference(bounds_q, bounds_k, heads, kv_heads, dim):
    if bounds_q != bounds_k:
        pytest.importorskip("flash_attn")
    _run_native_te_check("_assert_native_te_matches_cuda_reference", bounds_q, bounds_k, heads, kv_heads, dim)


def _assert_native_te_matches_cuda_reference(bounds_q, bounds_k, heads, kv_heads, dim):
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType

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
    if bounds_q == bounds_k:
        # Long BF16 gradients compare the adapter with unadapted native TE,
        # rather than requiring different FlashAttention versions to agree.
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, max_seqlen_q=max_q, max_seqlen_kv=max_k
        )
        expected = core(
            qr, kr, vr, attention_mask=None, attn_mask_type=AttnMaskType.causal, packed_seq_params=packed
        ).reshape_as(q)
    else:
        # Keep an independent oracle for unequal-length bottom-right masking.
        flash_attn = pytest.importorskip("flash_attn")
        expected = flash_attn.flash_attn_varlen_func(qr, kr, vr, cu_q, cu_k, max_q, max_k, causal=True)
    dout = torch.randn(result.shape, device=result.device, dtype=result.dtype, generator=generator)
    result.backward(dout)
    expected.backward(dout)
    for actual, wanted in ((result, expected), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(actual, wanted, atol=0.02, rtol=0.02)
        assert (actual.float() - wanted.float()).norm() / wanted.float().norm() < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native DDP gradient buffers require CUDA/NCCL")
def test_native_ddp_sync_after_chunk_backwards():
    _run_native_te_check("_assert_native_ddp_sync_after_chunk_backwards")


def _assert_native_ddp_sync_after_chunk_backwards():
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.transformer.transformer_config import TransformerConfig

    from examples.chunkflow.schedules import _start_grad_sync

    # Initialize native hook counts before testing the post-warmup no_sync path.
    for distributed_optimizer, overlap, aligned in (
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ):
        config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
        linear = torch.nn.Linear(4, 4, bias=False, device="cuda")
        for parameter in linear.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=distributed_optimizer, overlap_grad_reduce=overlap
            ),
            linear,
        )
        config.grad_sync_func = model.start_grad_sync if aligned else None
        model(torch.ones((2, 4), device="cuda")).sum().backward()
        model.finish_grad_sync()
        for step in (1, 2):
            model.zero_grad_buffer()
            with model.no_sync():
                for chunk in (1, 3):
                    model(torch.full((2, 4), float(step * chunk), device="cuda")).sum().backward()
            _start_grad_sync(model, config)
            model.finish_grad_sync()
            torch.testing.assert_close(linear.weight.main_grad, torch.full_like(linear.weight, 8 * step))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and differentiable FLA state kernels")
@pytest.mark.parametrize("mode", ["fr", "selective_last"])
def test_gdn_fla_state_outputs_and_gradients_match_full_sequence(kernels, mode):
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    kernels.causal_conv1d, kernels.rule = causal_conv1d, chunk_gated_delta_rule
    _alignment(kernels, mode, chunk_size=64, device="cuda", heads=2, dim=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Bridge Qwen3VL and TE CUDA execution")
def test_qwen35_native_model_installs_only_language_adapters():
    _run_native_te_check("_assert_qwen35_native_model_installs_only_language_adapters")


def _assert_qwen35_native_model_installs_only_language_adapters():
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider
    from megatron.core.transformer.enums import AttnBackend
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

    from examples.chunkflow.adapters import attention_modules
    from examples.chunkflow.runtime import require_runtime

    provider = Qwen35VLMoEModelProvider(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        rotary_percent=1.0,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        vocab_size=512,
        language_max_sequence_length=512,
        gdn_pre_gated_delta_rule_fusion=False,
        attention_backend=AttnBackend.flash,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        vision_config=Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            out_hidden_size=256,
            num_position_embeddings=16,
            patch_size=2,
            spatial_merge_size=2,
        ),
    )
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True)
    vision_attention = list(attention_modules(model.vision_model))
    assert len(vision_attention) == 1
    assert vision_attention[0].attn_mask_type.name == "no_mask"
    vision_modules = tuple(model.vision_model.modules())
    vision_forwards = tuple(module.forward for module in vision_modules)
    vision_configs = [module.config for module in vision_modules if hasattr(module, "config")]
    vision_rope = [config.apply_rope_fusion for config in vision_configs]
    vision_te = [module for module in vision_modules if hasattr(module, "disable_parameter_transpose_cache")]
    vision_cache = [module.disable_parameter_transpose_cache for module in vision_te]
    parameters, state_keys = tuple(model.parameters()), tuple(model.state_dict())
    # Three native GDN layers and one native dense layer; no vision adapter.
    assert install(model, 128) == 4
    assert install(model, 128) == 0
    assert require_runtime(model).chunk_size == 128
    assert tuple(model.parameters()) == parameters and tuple(model.state_dict()) == state_keys
    assert tuple(module.forward for module in vision_modules) == vision_forwards
    assert [config.apply_rope_fusion for config in vision_configs] == vision_rope
    assert [module.disable_parameter_transpose_cache for module in vision_te] == vision_cache
    assert uninstall(model) == 4
    assert not hasattr(model, "chunkflow_runtime")
    assert tuple(module.forward for module in vision_modules) == vision_forwards


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native Bridge Qwen3VL and TE CUDA execution")
@pytest.mark.parametrize("keep_last_k", [1, 1000])
def test_qwen3_vl_native_layer_outputs_and_gradients(keep_last_k):
    _run_native_te_check("_assert_qwen3_vl_native_layer_outputs_and_gradients", keep_last_k)


def _assert_qwen3_vl_native_layer_outputs_and_gradients(keep_last_k):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding, get_rope_index
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TERowParallelLinear,
    )
    from megatron.core.transformer.attention import SelfAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType

    from examples.chunkflow.test.cpu.test_unit_batch import _parent
    from examples.chunkflow.unit_batch import _rows_for_unit, slice_parent_batch

    torch.manual_seed(37)
    chunk_size, hidden, dim = 128, 256, 64
    config = Qwen3VLTransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=dim,
        attention_dropout=0,
        hidden_dropout=0,
        attention_output_gate=True,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        apply_rope_fusion=False,
        params_dtype=torch.bfloat16,
    )
    layer = Qwen3VLSelfAttention(
        config,
        SelfAttentionSubmodules(TEColumnParallelLinear, TEDotProductAttention, TERowParallelLinear),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    )
    rope = Qwen3VLMultimodalRotaryEmbedding(dim, cp_group=parallel_state.get_context_parallel_group())
    rope.is_thd_format = True
    # Exercise a dependent prefix, packed short documents and physical padding.
    lengths = [301, 11, 13]
    parent = _parent(lengths, device="cuda")
    full_positions, _ = get_rope_index(1, -1, -2, -3, parent["tokens"], packed_seq_params=parent["packed_seq_params"])
    x = torch.randn(sum(lengths), 1, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected, _ = layer(
        x, None, rotary_pos_emb=rope(full_positions, [11, 11, 10]), packed_seq_params=parent["packed_seq_params"]
    )
    dout = torch.randn_like(expected)
    expected.backward(dout)
    reference_input_grad = x.grad.clone()
    reference_grads = {name: parameter.grad.clone() for name, parameter in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    x.grad = None

    core, outer_forward = layer.core_attention, layer.forward
    parameters, state_keys = tuple(layer.parameters()), tuple(layer.state_dict())
    assert install(layer, chunk_size) == 1
    assert layer.core_attention is core and layer.forward == outer_forward
    assert tuple(layer.parameters()) == parameters and tuple(layer.state_dict()) == state_keys
    chunks = build_chunks(lengths, chunk_size)
    units = build_micro_batches(chunks)
    actual, pending = torch.empty_like(expected), {}
    for kind, uid in gen_unit_ops(units, chunks, keep_last_k):
        unit = units[uid]
        rows = _rows_for_unit(parent["packed_seq_params"].cu_seqlens_q_cpu, unit)
        indices = [index for start, end in rows for index in range(start, end)]
        if kind == "B":
            pending.pop(uid).backward(dout[indices])
            for group in completed_groups(unit):
                layer.chunkflow_runtime.release_group(group)
            continue
        batch = slice_parent_batch(parent, unit, pad_token_id=0, args=SimpleNamespace(uses_unsplit_forward=True))
        packed = batch["packed_seq_params"]
        setattr(packed, UNIT_ATTR, unit)
        setattr(packed, RECOMPUTE_ATTR, kind == "Fr")
        with torch.set_grad_enabled(kind != "Fp"):
            inputs = x[indices]
            inputs = torch.cat((inputs, inputs.new_zeros(batch["tokens"].numel() - len(indices), 1, hidden)))
            output, _ = layer(
                inputs,
                None,
                rotary_pos_emb=rope(batch["position_ids"], [11, 11, 10], packed),
                packed_seq_params=packed,
            )
        if kind != "Fp":
            pending[uid] = output[: len(indices)]
            actual[indices] = output[: len(indices)].detach()
    layer.chunkflow_runtime.assert_empty("Qwen3VL native layer")
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    torch.testing.assert_close(x.grad, reference_input_grad, atol=0.02, rtol=0.03)
    for name, parameter in layer.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_grads[name], atol=0.03, rtol=0.03)
    assert uninstall(layer) == 1
    assert layer.forward == outer_forward and tuple(layer.state_dict()) == state_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to verify asynchronous metadata copies")
def test_chunk_metadata_cuda_uses_current_stream_without_host_sync():
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the CUDA PP1 executor and native SFT loss")
def test_pp1_native_loss_callback_preserves_head_gradients_and_token_count(monkeypatch):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training import global_vars

    from examples.chunkflow.schedules import chunkflow_forward_backward_no_pipelining
    from examples.chunkflow.test.cpu.test_unit_batch import _parent
    from relax.backends.megatron.data import PrepackedBatch, get_batch
    from relax.backends.megatron.loss import loss_function

    args = SimpleNamespace(
        chunkflow_chunk_size=128,
        chunkflow_keep_last_k=1,
        data_pad_size_multiplier=128,
        qkv_format="thd",
        allgather_cp=False,
        loss_type="sft",
        sft_chunked_logits=True,
        sft_logits_chunk_size=32,
        recompute_loss_function=True,
        calculate_per_token_loss=True,
        rollout_temperature=0.7,
        global_batch_size=3,
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    with _megatron_context():
        parent = PrepackedBatch(_parent([301, 11, 13], device="cuda"))
        model = torch.nn.Sequential(torch.nn.Embedding(384, 16), torch.nn.Linear(16, 384, bias=False)).cuda()
        model.chunkflow_runtime = ChunkflowRuntime(128)
        model.set_input_tensor = lambda tensors: None
        finalized = []
        model.config = TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=1, calculate_per_token_loss=True
        )
        model.config.finalize_model_grads_func = lambda models, count, **kwargs: finalized.append(count.clone())
        targets = torch.cat([torch.cat((tokens[1:], tokens.new_zeros(1))) for tokens in parent["unconcat_tokens"]])
        logits = model(parent["tokens"]).squeeze(0) / args.rollout_temperature
        expected = (
            torch.nn.functional.cross_entropy(logits, targets, reduction="none") * parent["full_loss_masks"][0]
        ).sum()
        expected.backward()
        expected_grads = [parameter.grad.clone() for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)

        def forward_step(iterator, module):
            batch = get_batch(iterator, ["tokens"], 128, "thd", False)
            hidden = module[0](batch["tokens"])

            def head(inputs):
                return module[1](inputs), None

            head._relax_unfused_head_wgrad = True

            def loss_callback(output):
                # Fp builds history only; it must not count tokens or compute loss.
                assert torch.is_grad_enabled()
                return loss_function(args, batch, 1, output, lm_head_forward=head)

            return hidden, loss_callback

        logs = chunkflow_forward_backward_no_pipelining(
            forward_step_func=forward_step,
            data_iterator=SimpleNamespace(get_next=lambda keys: parent),
            model=model,
            num_microbatches=1,
            seq_length=325,
            micro_batch_size=1,
            pg_collection=SimpleNamespace(),
        )
        expected_count = parent["full_loss_masks"].sum()
        assert len(finalized) == 1
        torch.testing.assert_close(finalized[0], expected_count.to(torch.int))
        torch.testing.assert_close(sum(log["values"][0] for log in logs), expected_count.float())
        torch.testing.assert_close(sum(log["values"][1] for log in logs), expected.detach())
        for parameter, gradient in zip(model.parameters(), expected_grads, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=1e-5, rtol=1e-5)


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
    with _megatron_context() as rotary_embedding:
        yield rotary_embedding


@contextmanager
def _megatron_context():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and Megatron model-parallel groups")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

    own_dist = not dist.is_initialized()
    own_parallel = not parallel_state.is_initialized()
    rng_tracker = get_cuda_rng_tracker()
    rng_states = rng_tracker.get_states()
    with pytest.MonkeyPatch.context() as patcher, TemporaryDirectory() as directory:
        patcher.setattr(cf_rope, "_ORIGINAL_GET_ROTARY_SEQ_LEN", cf_rope._ORIGINAL_GET_ROTARY_SEQ_LEN)
        patcher.setattr(RotaryEmbedding, "get_rotary_seq_len", RotaryEmbedding.get_rotary_seq_len)
        try:
            if own_dist:
                rendezvous = (Path(directory) / "rendezvous").as_uri()
                dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1)
            if own_parallel:
                parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            # TE enters Megatron's named RNG stream even with dropout=0.
            # Process-group initialization alone does not create that stream.
            with torch.random.fork_rng():
                model_parallel_cuda_manual_seed(1234)
            cf_rope.install()
            yield RotaryEmbedding
        finally:
            rng_tracker.set_states(rng_states)
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

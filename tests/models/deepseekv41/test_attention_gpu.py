# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V4.1 GPU formula checks and real NCCL CP comparisons.

Single GPU: RELAX_TEST_V41_GPU=1 pytest -s tests/models/deepseekv41/test_attention_gpu.py
CP: torchrun --standalone --nproc-per-node=2 tests/models/deepseekv41/test_attention_gpu.py
TP/SP: append --tp to the torchrun command.
TP single-layer isolation: append --tp --isolated; --tp also checks the four-layer chain.
THD packing: torchrun --standalone --nproc-per-node=1, 2 or 4 <this file> --thd.
THD TP2/SP: append --thd --tp with two GPUs; four GPUs also exercise CP2.
Both require the Relax training image. No kernels or collectives are replaced.
"""

import json
import os
import sys
from datetime import timedelta

import pytest
import torch
from torch import Tensor


pytestmark = pytest.mark.skipif(
    os.environ.get("RELAX_TEST_V41_GPU") != "1" or not torch.cuda.is_available(),
    reason="Requires RELAX_TEST_V41_GPU=1 and an idle CUDA GPU in the Relax image.",
)


def _compare(actual: Tensor, expected: Tensor, name: str, tolerance: float) -> None:
    difference = actual.double() - expected.double()
    error, norm = difference.norm().item(), expected.double().norm().item()
    print(
        json.dumps(
            {
                "name": name,
                "dtype": str(actual.dtype),
                "shape": list(actual.shape),
                "rel_l2": error / max(norm, 1e-12),
                "max_abs": difference.abs().max().item(),
                "tolerance": tolerance,
            }
        ),
        flush=True,
    )
    assert torch.isfinite(actual).all(), name
    assert error <= tolerance * norm + 1e-7, name


def _dense_sparse_oracle(q: Tensor, kv: Tensor, indices: Tensor, sink: Tensor) -> Tensor:
    """Explicit selected-key softmax, including the zero-valued sink column."""
    batch, seq, heads, dim = q.shape
    keys = torch.stack([kv[b, indices[b, :, 0].clamp_min(0), 0] for b in range(batch)])
    logits = torch.einsum("bshd,bskd->bshk", q, keys) * dim**-0.5
    logits = logits.masked_fill(indices[:, :, 0].unsqueeze(2) < 0, -torch.inf)
    logits = torch.cat((logits, sink.view(1, 1, heads, 1).expand(batch, seq, -1, -1)), dim=-1)
    weights = logits.softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", weights, keys)


@pytest.mark.parametrize(
    "backend,seq,topk,batch,heads",
    [
        ("tilelang", 37, 65, 2, 8),
        ("flash_mla", 37, 65, 2, 8),
        ("tilelang", 160, 128, 1, 8),
        ("flash_mla", 160, 128, 1, 32),
        ("flash_mla", 37, 640, 1, 64),
    ],
)
def test_sparse_attention_matches_dense_formula(backend: str, seq: int, topk: int, batch: int, heads: int) -> None:
    from relax.models.deepseekv41.ops.sparse_attention import SparseAttentionConfig, sparse_attention

    torch.manual_seed(271)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = (torch.randn(batch, seq, heads, 512, device="cuda") * 0.5).bfloat16().requires_grad_()
    kv = (torch.randn(batch, seq + 7, 1, 512, device="cuda") * 0.5).bfloat16().requires_grad_()
    sink = torch.randn(heads, device="cuda", requires_grad=True)
    positions = torch.arange(seq, device="cuda")[:, None]
    selected = (positions - topk + 1).clamp_min(0) + torch.arange(topk, device="cuda")
    selected = torch.where(selected <= positions, selected, -1).int()
    indices = selected[None, :, None].expand(batch, -1, 1, -1).contiguous()
    dout = torch.randn_like(q) * 0.03
    leaves = [value.detach().float().requires_grad_() for value in (q, kv, sink)]
    reference = _dense_sparse_oracle(leaves[0], leaves[1], indices, leaves[2])
    output = sparse_attention(
        q,
        kv,
        indices,
        512**-0.5,
        attn_sink=sink,
        config=SparseAttentionConfig(
            forward_backend=backend,
            forward_threads=128 if heads < 32 else 256,
            backward_split_store=4 if heads >= 32 else 2,
        ),
    )
    output.backward(dout)
    reference.backward(dout.float())
    prefix = f"sparse/seed271/{backend}/b{batch}s{seq}h{heads}k{topk}"
    _compare(output, reference, prefix + "/output", 0.006)
    for name, actual, expected in zip(("q", "kv", "sink"), (q, kv, sink), leaves, strict=True):
        _compare(actual.grad, expected.grad, prefix + "/" + name, 0.015)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_tail_matches_complex_formula(dtype: torch.dtype, inverse: bool) -> None:
    from relax.models.deepseekv41.ops.position import apply_rotary_emb, precompute_freqs_cis

    torch.manual_seed(277)
    x = torch.randn(2, 37, 3, 512, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().float().requires_grad_()
    freqs = precompute_freqs_cis(64, 37, 0, 10000.0, 1.0, 32.0, 1.0, x.device)
    frequency = freqs.conj() if inverse else freqs
    pairs = torch.view_as_complex(reference[..., -64:].contiguous().reshape(2, 37, 3, 32, 2))
    expected = torch.cat((reference[..., :-64], torch.view_as_real(pairs * frequency[None, :, None]).flatten(-2)), -1)
    output = x.clone()
    apply_rotary_emb(output[..., -64:], freqs, inverse)
    dout = torch.randn_like(x)
    output.backward(dout)
    expected.backward(dout.float())
    tolerance = 3e-7 if dtype == torch.float32 else 0.004
    _compare(output, expected, f"rope/seed277/{dtype}/{inverse}/output", tolerance)
    _compare(x.grad, reference.grad, f"rope/seed277/{dtype}/{inverse}/grad", tolerance)


@pytest.mark.parametrize("heads", [8, 32])
def test_indexer_logits_matches_relu_weighted_formula(heads: int) -> None:
    from relax.models.deepseekv41.ops.indexer import indexer_logits

    torch.manual_seed(283)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn(2, 37, heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 53, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2, 37, heads, device="cuda") * 0.1
    lengths = torch.arange(37, device="cuda")
    actual = indexer_logits(q, k, w, lengths)
    expected = (torch.einsum("bshd,bkd->bshk", q.float(), k.float()).relu() * w[..., None]).sum(2)
    mask = torch.arange(53, device="cuda")[None, None] < lengths[None, :, None]
    assert torch.equal(torch.isneginf(actual), ~mask.expand_as(actual))
    _compare(
        actual[mask.expand_as(actual)], expected[mask.expand_as(expected)], f"indexer/seed283/h{heads}/logits", 2e-5
    )


@pytest.mark.parametrize("kind,block", [("fp4", 32), ("compressed", 16), ("fp8", 64)])
def test_quantization_matches_codebook_and_straight_through_gradient(kind: str, block: int) -> None:
    from relax.models.deepseekv41.ops.kernels.fake_quant import fake_quant_fp8_block
    from relax.models.deepseekv41.ops.quant import fake_quant_compressed_kv, fake_quant_fp4

    torch.manual_seed(287)
    x = (torch.randn(2, 37, 512, device="cuda") * 2).bfloat16().requires_grad_()
    with torch.no_grad():
        x[0, 0].zero_()
    rows = x.detach().float().reshape(-1, block)
    amax = rows.abs().amax(-1, keepdim=True)
    if kind == "compressed":
        scale = (amax / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
    else:
        maximum, floor = (6, 6 * 2**-126) if kind == "fp4" else (448, 1e-4)
        scale = 2 ** torch.ceil(torch.log2(amax.clamp_min(floor) / maximum))
    if kind == "fp8":
        expected = (rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
        actual = fake_quant_fp8_block(x, block)
    else:
        magnitudes = (rows / scale).abs().clamp_max(6)
        codebook = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
        distances = (magnitudes[..., None] - codebook).abs()
        # E2M1 round-to-nearest-even resolves ties toward an even code index.
        nearest = distances == distances.amin(-1, keepdim=True)
        odd = torch.arange(8, device="cuda") % 2
        priorities = odd * 8 + torch.arange(8, device="cuda")
        code = torch.where(nearest, priorities, 100).argmin(-1)
        expected = codebook[code] * rows.sign() * scale
        actual = fake_quant_fp4(x) if kind == "fp4" else fake_quant_compressed_kv(x)
    _compare(actual, expected.reshape_as(x).to(x.dtype), f"quant/seed287/{kind}/output", 0.0)
    gradient = torch.randn_like(x)
    actual.backward(gradient)
    _compare(x.grad, gradient, f"quant/seed287/{kind}/ste", 0.0)


def _config(ratios: list[int], cp_size: int, tp_size: int = 1, sequence_parallel: bool = False):
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=len(ratios),
        hidden_size=64,
        num_attention_heads=8,
        ffn_hidden_size=128,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        gradient_accumulation_fusion=False,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sequence_parallel,
        context_parallel_size=cp_size,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    for name, value in dict(
        q_lora_rank=32,
        o_lora_rank=32,
        kv_lora_rank=512,
        qk_pos_emb_head_dim=64,
        o_groups=2,
        csa_window_size=128,
        csa_compress_ratios=ratios,
        v41_kv_source_layer_ids=tuple(i for i, ratio in enumerate(ratios) if ratio and ratio not in ratios[:i]),
        v41_index_source_layer_ids=tuple(i for i, ratio in enumerate(ratios) if ratio),
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=32,
        v41_candidate_source_layer_id=2 if len(ratios) == 4 else -1,
        v41_candidate_topk_blocks=4,
        v41_candidate_block_size=8,
        original_max_position_embeddings=8192,
        rotary_base=10000.0,
        csa_compress_rotary_base=160000.0,
        rotary_scaling_factor=1.0,
        beta_fast=32.0,
        beta_slow=1.0,
    ).items():
        setattr(config, name, value)
    return config


def _cp_worker() -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from torch.utils.checkpoint import checkpoint

    from relax.models.deepseekv41.attention import DeepSeekV41Attention
    from relax.models.deepseekv41.runtime import V41Runtime

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    parallel_state.initialize_model_parallel(context_parallel_size=2)
    tp = parallel_state.get_tensor_model_parallel_group()
    group = dist.group.WORLD
    try:
        for case, ratios, seqlen, batch in (
            ("window", [0], 160, 2),
            ("ratio2", [2], 160, 1),
            ("reuse", [0, 2, 1, 1], 160, 1),
        ):
            torch.manual_seed(293)
            runtime = V41Runtime()
            config = _config(ratios, 1)
            reference = torch.nn.ModuleList(
                [
                    DeepSeekV41Attention(
                        config, runtime, layer_number=i + 1, pg_collection=ProcessGroupCollection(tp=tp, cp=None)
                    )
                    for i in range(len(ratios))
                ]
            ).cuda()
            with torch.no_grad():
                for layer in reference:
                    layer.core_attention.attn_sink.copy_(torch.linspace(-0.5, 0.5, 8, device="cuda"))
            candidate_runtime = V41Runtime()
            candidate_config = _config(ratios, 2)
            candidate = torch.nn.ModuleList(
                [
                    DeepSeekV41Attention(
                        candidate_config,
                        candidate_runtime,
                        layer_number=i + 1,
                        pg_collection=ProcessGroupCollection(tp=tp, cp=group),
                    )
                    for i in range(len(ratios))
                ]
            ).cuda()
            candidate.load_state_dict(reference.state_dict())
            source = torch.randn(seqlen, batch, 64, device="cuda", dtype=torch.bfloat16) * 0.1
            upstream = torch.randn_like(source) * 0.1

            def forward(module, rt, value):
                rt.reset()
                for layer in module:
                    value = value + layer(value)[0]
                return value

            full = source.detach().clone().requires_grad_()
            expected = forward(reference, runtime, full)
            expected.backward(upstream)
            baseline_output = baseline_input_grad = baseline_parameter_grads = None
            for mode in ("baseline", "repeat", "recompute"):
                recompute = mode == "recompute"
                candidate.zero_grad(set_to_none=True)
                local = source.chunk(2)[rank].detach().clone().requires_grad_()
                if recompute:
                    actual = checkpoint(forward, candidate, candidate_runtime, local, use_reentrant=True)
                else:
                    actual = forward(candidate, candidate_runtime, local)
                actual.backward(upstream.chunk(2)[rank].contiguous())
                prefix = f"cp/seed293/{case}/rank{rank}/{mode}"
                _compare(actual, expected.chunk(2)[rank], prefix + "/output", 0.008)
                _compare(local.grad, full.grad.chunk(2)[rank], prefix + "/input", 0.02)
                parameter_grads = {}
                for (name, parameter), (_, reference_parameter) in zip(
                    candidate.named_parameters(), reference.named_parameters(), strict=True
                ):
                    if not parameter.requires_grad:
                        assert parameter.grad is None
                        continue
                    assert parameter.grad is not None and reference_parameter.grad is not None, name
                    gradient = parameter.grad.float().clone()
                    dist.all_reduce(gradient, group=group)
                    _compare(gradient, reference_parameter.grad, prefix + "/" + name, 0.02)
                    parameter_grads[name] = gradient
                if mode != "baseline":
                    _compare(actual, baseline_output, prefix + "/replay_output", 0.0)
                    # Sparse dKV uses FP32 atomic adds before casting to BF16.
                    # Plain repeated backward has the same non-bitwise noise as checkpoint replay.
                    # Cross-layer cache/residual branches also accumulate BF16 gradients in varying order.
                    replay_tolerance = 0.001 if len(ratios) == 1 else 0.01
                    _compare(local.grad, baseline_input_grad, prefix + "/replay_input", replay_tolerance)
                    for name, gradient in parameter_grads.items():
                        _compare(
                            gradient, baseline_parameter_grads[name], prefix + "/replay_" + name, replay_tolerance
                        )
                else:
                    baseline_output = actual.detach().clone()
                    baseline_input_grad = local.grad.detach().clone()
                    baseline_parameter_grads = parameter_grads
                print(f"CASE_PASS {prefix}", flush=True)
            del candidate, reference, expected, actual
        dist.barrier(group=group)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _tp_worker() -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    from relax.models.deepseekv41.attention import DeepSeekV41Attention
    from relax.models.deepseekv41.runtime import V41Runtime

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    singleton = [dist.new_group([index]) for index in range(2)][rank]
    group = dist.group.WORLD
    try:
        for case, ratios in (("window", [0]), ("ratio1", [1]), ("ratio2", [2]), ("reuse", [0, 2, 1, 1])):
            if "--isolated" in sys.argv and case == "reuse":
                continue
            for sequence_parallel in (False, True):
                torch.manual_seed(307)
                reference_runtime, candidate_runtime = V41Runtime(), V41Runtime()
                reference_config, candidate_config = _config(ratios, 1), _config(ratios, 1, 2, sequence_parallel)
                reference = torch.nn.ModuleList(
                    [
                        DeepSeekV41Attention(
                            reference_config,
                            reference_runtime,
                            layer_number=i + 1,
                            pg_collection=ProcessGroupCollection(tp=singleton, cp=None),
                        )
                        for i in range(len(ratios))
                    ]
                ).cuda()
                candidate = torch.nn.ModuleList(
                    [
                        DeepSeekV41Attention(
                            candidate_config,
                            candidate_runtime,
                            layer_number=i + 1,
                            pg_collection=ProcessGroupCollection(tp=group, cp=None),
                        )
                        for i in range(len(ratios))
                    ]
                ).cuda()
                sharded_axes = {}
                with torch.no_grad():
                    for layer in reference:
                        layer.core_attention.attn_sink.copy_(torch.linspace(-0.5, 0.5, 8, device="cuda"))
                    for (name, parameter), (_, reference_parameter) in zip(
                        candidate.named_parameters(), reference.named_parameters(), strict=True
                    ):
                        if name.endswith(("linear_o_group_proj", "attn_sink", "linear_q_up_proj.weight")):
                            axis = 0
                        elif name.endswith("linear_proj.weight"):
                            axis = 1
                        else:
                            axis = None
                        sharded_axes[name] = axis
                        expected = reference_parameter if axis is None else reference_parameter.chunk(2, axis)[rank]
                        parameter.copy_(expected)
                source = torch.randn(160, 1, 64, device="cuda", dtype=torch.bfloat16) * 0.1
                upstream = torch.randn_like(source) * 0.1
                full = source.clone().requires_grad_()
                local = (source.chunk(2)[rank] if sequence_parallel else source).clone().requires_grad_()

                def forward(module, value):
                    for layer in module:
                        value = value + layer(value)[0]
                    return value

                expected = forward(reference, full)
                expected.backward(upstream)
                actual = forward(candidate, local)
                actual.backward(upstream.chunk(2)[rank].contiguous() if sequence_parallel else upstream)
                prefix = f"tp/seed307/{case}/rank{rank}/sp{sequence_parallel}"
                output_ref = expected.chunk(2)[rank] if sequence_parallel else expected
                input_ref = full.grad.chunk(2)[rank] if sequence_parallel else full.grad
                _compare(actual, output_ref, prefix + "/output", 0.008)
                _compare(local.grad, input_ref, prefix + "/input", 0.02)
                for (name, parameter), (_, reference_parameter) in zip(
                    candidate.named_parameters(), reference.named_parameters(), strict=True
                ):
                    if not parameter.requires_grad:
                        assert parameter.grad is None
                        continue
                    assert parameter.grad is not None and reference_parameter.grad is not None, name
                    axis = sharded_axes[name]
                    expected_grad = reference_parameter.grad
                    if axis is not None:
                        expected_grad = expected_grad.chunk(2, axis)[rank]
                    _compare(parameter.grad, expected_grad, prefix + "/" + name, 0.02)
                print(f"CASE_PASS {prefix}", flush=True)
                del candidate, reference, actual, expected
        dist.barrier(group=group)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _thd_worker() -> None:
    """Real packed attention vs separate BSHD samples, including empty CP ranks."""
    from types import SimpleNamespace

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from torch.utils.checkpoint import checkpoint

    from relax.models.deepseekv41.attention import DeepSeekV41Attention
    from relax.models.deepseekv41.runtime import V41Runtime
    from relax.models.deepseekv41.thd.layout import ThdLayout

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world not in (1, 2, 4):
        raise ValueError("The THD attention regression expects one, two or four GPUs.")
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    parallel_state.initialize_model_parallel(context_parallel_size=world)
    tp, group = parallel_state.get_tensor_model_parallel_group(), dist.group.WORLD
    try:
        # The second sample's ratio-2 groups straddle both CP2 and CP4 cuts.
        # The second case has ranks without valid queries or owned compressed keys.
        for case, lengths, total in (
            ("boundary", (37, 158), 256),
            ("padding_rank", (37, 41), 256),
            ("no_ratio2_groups", (1, 1), 16),
        ):
            torch.manual_seed(331)
            ratios = [2] if case == "no_ratio2_groups" else [0, 2, 1, 1, 1]
            configs = [_config(ratios, cp) for cp in (1, world)]
            for config in configs:
                if len(ratios) > 1:
                    config.v41_index_source_layer_ids = (1, 2, 3)
                    config.v41_candidate_source_layer_id = 2
            reference_runtime, candidate_runtime = V41Runtime(), V41Runtime()
            modules = []
            for config, runtime, cp in zip(
                configs, (reference_runtime, candidate_runtime), (None, group), strict=True
            ):
                modules.append(
                    torch.nn.ModuleList(
                        [
                            DeepSeekV41Attention(
                                config,
                                runtime,
                                layer_number=i + 1,
                                pg_collection=ProcessGroupCollection(tp=tp, cp=cp),
                            )
                            for i in range(len(ratios))
                        ]
                    ).cuda()
                )
            reference, candidate = modules
            with torch.no_grad():
                for layer in reference:
                    layer.core_attention.attn_sink.copy_(torch.linspace(-0.5, 0.5, 8, device="cuda"))
            candidate.load_state_dict(reference.state_dict())
            source = torch.randn(total, 1, 64, device="cuda", dtype=torch.bfloat16) * 0.1
            upstream = torch.randn_like(source) * 0.1
            valid_tokens = sum(lengths)
            upstream[valid_tokens:] = 0
            expected = torch.zeros_like(source)
            expected_grad = torch.zeros_like(source)

            def run(module, runtime, value, metadata=None):
                runtime.reset()
                if metadata is not None:
                    runtime.thd_layout = ThdLayout.from_packed_seq_params(
                        metadata, cp_rank=rank, cp_size=world, seqlen_local=value.shape[0]
                    )
                for layer in module:
                    value = value + layer(value, packed_seq_params=metadata)[0]
                return value

            begin = 0
            for length in lengths:
                # BSHD kernels require a nonempty compressor input. A one-token
                # oracle sample can have one trailing padding token: causality
                # keeps the extra compressed group invisible to its real token.
                sample = torch.nn.functional.pad(source[begin : begin + length], (0, 0, 0, 0, 0, max(0, 2 - length)))
                sample = sample.detach().requires_grad_()
                result = run(reference, reference_runtime, sample)
                sample_grad = torch.nn.functional.pad(
                    upstream[begin : begin + length], (0, 0, 0, 0, 0, max(0, 2 - length))
                )
                result.backward(sample_grad)
                expected[begin : begin + length] = result[:length].detach()
                expected_grad[begin : begin + length] = sample.grad[:length]
                begin += length
            metadata = SimpleNamespace(
                qkv_format="thd",
                cu_seqlens_q_cpu=(0, lengths[0], valid_tokens),
                valid_seqlens_q_cpu=lengths,
                cu_seqlens_q_padded_cpu=(0, lengths[0], total),
                num_valid_tokens=valid_tokens,
                cp_partition_mode="contiguous",
            )
            local_valid = torch.arange(total // world, device="cuda") + rank * (total // world) < valid_tokens
            for recompute in (False, True):
                candidate.zero_grad(set_to_none=True)
                local = source.chunk(world)[rank].clone().requires_grad_()
                if recompute:
                    actual = checkpoint(run, candidate, candidate_runtime, local, metadata, use_reentrant=True)
                else:
                    actual = run(candidate, candidate_runtime, local, metadata)
                actual.backward(upstream.chunk(world)[rank].contiguous())
                prefix = f"thd/seed331/{case}/cp{world}/rank{rank}/recompute{recompute}"
                failures = []

                def compare(value, target, name, tolerance):
                    try:
                        _compare(value, target, prefix + "/" + name, tolerance)
                    except AssertionError:
                        failures.append(name)

                compare(actual * local_valid[:, None, None], expected.chunk(world)[rank], "output", 0.008)
                compare(local.grad, expected_grad.chunk(world)[rank], "input", 0.02)
                try:
                    torch.testing.assert_close(actual[~local_valid], local[~local_valid], atol=0, rtol=0)
                except AssertionError:
                    failures.append("padding_output")
                for (name, parameter), (_, reference_parameter) in zip(
                    candidate.named_parameters(), reference.named_parameters(), strict=True
                ):
                    if not parameter.requires_grad:
                        if parameter.grad is not None:
                            failures.append(name + "/frozen_gradient")
                        continue
                    if parameter.grad is None or reference_parameter.grad is None:
                        failures.append(name + "/missing_gradient")
                    gradient = (
                        torch.zeros_like(parameter, dtype=torch.float32)
                        if parameter.grad is None
                        else parameter.grad.float().clone()
                    )
                    dist.all_reduce(gradient, group=group)
                    if reference_parameter.grad is not None:
                        compare(gradient, reference_parameter.grad, name, 0.02)
                failed = torch.tensor(bool(failures), device="cuda", dtype=torch.int32)
                dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
                assert not failed.item(), f"{prefix}: failed on this or another rank; local failures={failures}"
                print(f"CASE_PASS {prefix}", flush=True)
            del candidate, reference, actual, expected
        dist.barrier(group=group)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _thd_tp_worker() -> None:
    """Compare packed TP2/SP/CP against separate BSHD samples at the same TP.

    Keeping TP fixed isolates packing/CP from BF16 row-projection rounding that
    otherwise changes QAT values and discrete retrieval across a multi-layer chain.
    The existing non-packed TP1-to-TP2 worker remains the separate TP check.
    """
    from types import SimpleNamespace

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    from relax.models.deepseekv41.attention import DeepSeekV41Attention
    from relax.models.deepseekv41.runtime import V41Runtime

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (2, 4):
        raise ValueError("Packed TP2/SP attention expects two GPUs, or four to include CP2.")
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    cp_size = world // 2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=cp_size)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
    tp_rank = dist.get_rank(tp_group)
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0
    singleton = [dist.new_group([index]) for index in range(world)][rank]
    group = dist.group.WORLD
    try:
        torch.manual_seed(307)
        ratios = [0, 2, 1, 1]
        runtimes = [V41Runtime() for _ in range(3)]
        configs = [_config(ratios, 1), _config(ratios, 1, 2), _config(ratios, cp_size, 2, True)]
        groups = [
            ProcessGroupCollection(tp=singleton, cp=None),
            ProcessGroupCollection(tp=tp_group, cp=None),
            ProcessGroupCollection(tp=tp_group, cp=cp_group),
        ]
        modules = [
            torch.nn.ModuleList(
                [
                    DeepSeekV41Attention(config, runtime, layer_number=i + 1, pg_collection=pg)
                    for i in range(len(ratios))
                ]
            ).cuda()
            for config, runtime, pg in zip(configs, runtimes, groups, strict=True)
        ]
        master, reference, candidate = modules
        with torch.no_grad():
            for layer in master:
                layer.core_attention.attn_sink.copy_(torch.linspace(-0.5, 0.5, 8, device="cuda"))
            for (name, parameter), (_, ref), (_, actual) in zip(
                master.named_parameters(), reference.named_parameters(), candidate.named_parameters(), strict=True
            ):
                dist.broadcast(parameter, src=0, group=group)
                axis = None
                if name.endswith(("linear_o_group_proj", "attn_sink", "linear_q_up_proj.weight")):
                    axis = 0
                elif name.endswith("linear_proj.weight"):
                    axis = 1
                shard = parameter if axis is None else parameter.chunk(2, axis)[tp_rank]
                ref.copy_(shard)
                actual.copy_(shard)
        del master, modules
        source = torch.randn(256, 1, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        upstream = torch.randn_like(source) * 0.1
        dist.broadcast(source, src=0, group=group)
        dist.broadcast(upstream, src=0, group=group)
        upstream[195:] = 0
        metadata = SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q_cpu=(0, 37, 195),
            valid_seqlens_q_cpu=(37, 158),
            cu_seqlens_q_padded_cpu=(0, 37, 256),
            num_valid_tokens=195,
            cp_partition_mode="contiguous",
        )

        def forward(module, runtime, value, packed=None):
            runtime.reset()
            for layer in module:
                value = value + layer(value, packed_seq_params=packed)[0]
            return value

        def local_slice(value):
            return value.chunk(cp_size)[cp_rank].chunk(2)[tp_rank]

        full = source.clone().requires_grad_()
        expected = torch.cat(
            (
                forward(reference, runtimes[1], full[:37]),
                forward(reference, runtimes[1], full[37:195]),
                full[195:] * 0,
            )
        )
        expected.backward(upstream)
        local = local_slice(source).clone().requires_grad_()
        actual = forward(candidate, runtimes[2], local, metadata)
        actual.backward(local_slice(upstream).contiguous())
        prefix = f"thd_same_tp/seed307/rank{rank}/tp2_sp_cp{cp_size}"
        failures = []

        def compare(value, target, name, tolerance):
            try:
                _compare(value, target, prefix + "/" + name, tolerance)
            except AssertionError:
                failures.append(name)

        valid = local_slice((torch.arange(256, device="cuda") < 195).view(256, 1, 1))
        compare(actual * valid, local_slice(expected), "output", 0.008)
        compare(local.grad, local_slice(full.grad), "input", 0.02)
        for (name, parameter), (_, ref) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    failures.append(name + "/frozen_gradient")
                continue
            # Complete the same collectives on every rank before failing the test.
            if parameter.grad is None or ref.grad is None:
                failures.append(name + "/missing_gradient")
            gradient = (
                torch.zeros_like(parameter, dtype=torch.float32)
                if parameter.grad is None
                else parameter.grad.float().clone()
            )
            if cp_group is not None:
                dist.all_reduce(gradient, group=cp_group)
            if ref.grad is not None:
                compare(gradient, ref.grad, name, 0.02)
        failed = torch.tensor(bool(failures), device="cuda", dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        assert not failed.item(), f"{prefix}: failed on this or another rank; local failures={failures}"
        print(f"CASE_PASS {prefix}", flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--tp" in sys.argv:
        _thd_tp_worker() if "--thd" in sys.argv else _tp_worker()
    elif "--thd" in sys.argv:
        _thd_worker()
    else:
        _cp_worker()

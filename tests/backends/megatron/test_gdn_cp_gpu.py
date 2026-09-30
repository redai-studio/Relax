# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real layout and GDN forward/backward regression for context parallelism.

Checks exact THD/SBHD layout round trips and BF16 CP=1/CP=2 GDN parity in one
two-GPU spawn. FLA kernels use a fixed candidate configuration to avoid
autotune benchmarks. Run with: pytest
tests/backends/megatron/test_gdn_cp_gpu.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and the patched Megatron-LM image",
)


def _check_layout_round_trip(cp_group):
    from megatron.core.context_parallel_layout import (
        contiguous_to_zigzag_chunks,
        get_thd_context_parallel_rank_indices,
        zigzag_to_contiguous_chunks,
    )

    rank, world_size = cp_group.rank(), cp_group.size()
    device = torch.device("cuda", rank)

    # Packed THD: short, unequal samples exercise the original route checks.
    lengths = [2 * world_size * f for f in (5, 1, 3)]
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=device, dtype=torch.int32)
    total = int(cu[-1])
    # Row t is (t, t+1e6, t+2e6): a permuted token is impossible to miss.
    full = (
        torch.arange(total, dtype=torch.float64, device=device).unsqueeze(1)
        + torch.arange(3, dtype=torch.float64, device=device).unsqueeze(0) * 1e6
    )

    zig_idx = get_thd_context_parallel_rank_indices(cu, world_size, rank, "zigzag")
    con_idx = get_thd_context_parallel_rank_indices(cu, world_size, rank, "contiguous")
    local_zig = full[zig_idx]

    got_con = zigzag_to_contiguous_chunks(local_zig, cp_group, seq_dim=0, cu_seqlens=cu)
    assert torch.equal(got_con, full[con_idx]), f"rank {rank}: THD zigzag->contiguous is wrong"
    got_zig = contiguous_to_zigzag_chunks(got_con, cp_group=cp_group, seq_dim=0, cu_seqlens=cu)
    assert torch.equal(got_zig, local_zig), f"rank {rank}: THD round trip is not identity"

    # SBHD: chunk-level swap without packed sequence metadata.
    seq_local = 2 * world_size * 4
    sbhd = torch.arange(seq_local * 2 * 3, dtype=torch.float64, device=device).reshape(seq_local, 2, 3) + rank * 1e9
    swapped = zigzag_to_contiguous_chunks(sbhd, cp_group, seq_dim=0)
    back = contiguous_to_zigzag_chunks(swapped, cp_group=cp_group, seq_dim=0)
    assert torch.equal(back, sbhd), f"rank {rank}: SBHD round trip is not identity"


def _build_gdn(mode):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=512,
        num_layers=1,
        num_attention_heads=8,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        activation_func=torch.nn.functional.silu,
        bf16=True,
        context_parallel_size=2,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_conv_kernel_dim=4,
        # Qwen3.5 uses 128-wide heads with a 1:2 key/value head ratio.
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_cp_mode=mode,
        transformer_impl="transformer_engine",
    )
    return (
        GatedDeltaNet(
            config,
            submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        )
        .cuda()
        .to(torch.bfloat16)
    )


def _forward_backward(gdn, hidden, packed, grad_out, *, recompute=False):
    from torch.utils.checkpoint import checkpoint

    gdn.zero_grad(set_to_none=True)
    hidden = hidden.detach().clone().requires_grad_(True)

    def forward(x):
        return gdn(x, None, packed_seq_params=packed)[0]

    out = checkpoint(forward, hidden, use_reentrant=False) if recompute else forward(hidden)
    (out.float() * grad_out).sum().backward()
    grads = {name: param.grad.detach().float().clone() for name, param in gdn.named_parameters()}
    return out.detach(), hidden.grad.detach(), grads


def _assert_grad_close(got, expected, name):
    # BF16 parameter gradients sum over different token partitions. Bound both
    # magnitude and direction, without imposing sub-ULP elementwise agreement.
    got, expected = got.flatten().float(), expected.flatten().float()
    assert torch.isfinite(got).all() and torch.isfinite(expected).all(), name
    relative_rms = (got - expected).norm() / expected.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(got, expected, dim=0)
    assert relative_rms < 1e-2, f"{name}: relative RMS error {relative_rms.item():.3e}"
    assert cosine >= 0.9999, f"{name}: cosine {cosine.item():.8f}"


def _use_one_fla_config():
    """Skip FLA tuning in this spawned worker while retaining real JIT
    kernels."""
    import triton

    autotune = triton.autotune

    def single_config(configs, *args, **kwargs):
        def decorate(fn):
            selected = configs[:1] if fn.__module__.startswith("fla.") else configs
            return autotune(selected, *args, **kwargs)(fn)

        return decorate

    # The override covers lazy imports and ends with this isolated worker.
    triton.autotune = single_config


def _worker_gdn_cp(rank, init_method):
    torch.cuda.set_device(rank)
    _use_one_fla_config()  # Install before Megatron/FLA imports create kernels.
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.training.global_vars import set_args

    from relax.backends.megatron.model import _patch_gdn_for_dynamic_cp

    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2)
    parallel_state.initialize_model_parallel(context_parallel_size=2)
    try:
        set_args(SimpleNamespace(recompute_granularity="full"))
        _patch_gdn_for_dynamic_cp()
        cp_group = parallel_state.get_context_parallel_group()
        _check_layout_round_trip(cp_group)
        solo = [dist.new_group([r]) for r in range(2)][rank]
        models = {mode: _build_gdn(mode) for mode in ("chunkwise", "all_gather")}
        for param in models["chunkwise"].parameters():
            dist.broadcast(param.data, src=0)
        models["all_gather"].load_state_dict(models["chunkwise"].state_dict())

        # Unequal packed samples; the contiguous CP boundary at 512 splits the
        # first sample, exercising recurrent/convolution state across ranks.
        lengths = [768, 256]
        total = sum(lengths)
        hidden_size = models["chunkwise"].config.hidden_size
        cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
        generator = torch.Generator().manual_seed(7)
        hidden = torch.randn(total, 1, hidden_size, generator=generator).cuda().to(torch.bfloat16)
        grad_out = torch.randn(total, 1, hidden_size, generator=generator).cuda()
        # Build expected zigzag ownership independently of the production route.
        indices = torch.cat(
            [
                part.reshape(4, -1)[[rank, 3 - rank]].flatten()
                for part in torch.arange(total, device="cuda").split(lengths)
            ]
        )

        def packed(group, size):
            return PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=max(lengths),
                max_seqlen_kv=max(lengths),
                cp_group=group,
                local_cp_size=size,
            )

        ref_out, ref_dx, ref_grads = _forward_backward(models["chunkwise"], hidden, packed(solo, 1), grad_out)
        shard = hidden[indices]
        grad_shard = grad_out[indices]
        for mode, gdn in models.items():
            out, dx, grads = _forward_backward(
                gdn, shard, packed(cp_group, 2), grad_shard, recompute=mode == "all_gather"
            )
            for name, got, expected in (("output", out, ref_out), ("input gradient", dx, ref_dx)):
                expected = expected[indices]
                torch.testing.assert_close(
                    got,
                    expected,
                    atol=2e-3,
                    rtol=1e-2,
                    msg=lambda message: f"{mode} {name}: {message}",
                )
                cosine = torch.nn.functional.cosine_similarity(
                    got.flatten().float(), expected.flatten().float(), dim=0
                )
                assert cosine >= 0.9999, f"{mode} {name}: cosine {cosine.item():.8f}"
            for name, grad in grads.items():
                # CP owns partial token sums; DDP would sum these replicas.
                dist.all_reduce(grad, group=cp_group)
                _assert_grad_close(grad, ref_grads[name], f"{mode} {name}")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_gdn_cp_layout_and_gradients(tmp_path):
    """Check exact layouts and CP=2 output/input/parameter gradients against
    CP=1."""
    mp.spawn(_worker_gdn_cp, args=(f"file://{tmp_path / 'gdn_cp_init'}",), nprocs=2, join=True)

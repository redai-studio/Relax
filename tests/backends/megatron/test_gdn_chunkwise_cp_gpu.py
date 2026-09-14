# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real-kernel / real-collective tests for the GDN chunkwise-CP backport.

RFC Task 32 phase-1 acceptance items 3 and 4:

* a minimal CP=2 chunkwise case must drive the *actual* FLA kernels and match a
  CP=1 reference in forward and backward within tolerance;
* the GDN parameter keys and shard dimensions in ``state_dict`` /
  ``sharded_state_dict`` must not move, i.e. GDN weights stay TP-only and
  checkpoints are unaffected by the CP mode.

Three layers are covered:

1. the FLA kernels directly (``causal_conv1d`` / ``chunk_gated_delta_rule`` with
   a ``cp_context``);
2. the whole MCore ``GatedDeltaNet`` module in fp32 -- the *algebraic* check. In
   fp32 the only difference between CP=1 and CP=2 is float summation order, so
   the tolerances can be tight enough to catch a genuinely wrong permutation or
   a dropped boundary term;
3. the same module in bf16 -- the *production* check, at the dtype training
   actually uses, where the achievable agreement is bounded by the storage
   format rather than by the algorithm.

A headwise CP=2 run is included at every layer as a control: if headwise and
chunkwise both drift the same way, the cause is shared plumbing, not the new
code.

Most tests need 2 visible GPUs; the TP2/CP2 matrix test needs 4:
    pytest tests/backends/megatron/test_gdn_chunkwise_cp_gpu.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp


WORLD_SIZE = 2

# --- tolerances ------------------------------------------------------------
# bf16 module output / input gradient: RFC section 5, "MCore GDN, CP>1 vs CP=1".
ATOL_BF16 = 2e-3
RTOL_BF16 = 1e-2
MIN_COSINE = 0.9999
# FLA kernels: RFC section 5, normalised RMS error thresholds.
CONV_RMS_RATIO = 1e-3
GDN_RMS_RATIO = 2e-3
# fp32 module run: both CP algorithms must land far below any bf16 threshold.
# 1e-3 is 2x below the bf16 element-wise atol of the RFC gate, i.e. "fp32 must be
# comfortably better than the dtype we actually ship".
RMS_RATIO_FP32 = 1e-3
# fp32 kernel-level: measured ~1e-7, so 1e-5 is a real gate, not a rubber stamp.
KERNEL_RMS_RATIO_FP32 = 1e-5
# chunkwise vs headwise. headwise is the already-shipped CP algorithm, so whatever
# CP-vs-no-CP disagreement it shows is the floor this environment imposes (reduced
# precision inside the Triton dots, changed summation order, bf16 storage) rather
# than anything about the algorithm. Requiring chunkwise to be no worse than that
# floor is the assertion that actually means something; a fixed atol on a bf16
# token-sum gradient mostly measures rounding luck.
CHUNKWISE_VS_HEADWISE_RMS_FACTOR = 4.0
# ...with a floor, so a headwise value that happens to land at or near zero on a given
# run cannot turn into an impossible budget. 1e-6 is still ~1000x tighter than the fp32
# absolute gate, so the comparison keeps its teeth.
RMS_FLOOR_FP32 = 1e-6
# ...applied only where headwise is not bit-exact. headwise hands each rank the
# whole sequence and 1/cp of the heads, so none of its reductions are
# repartitioned and it can land exactly on the CP=1 result; "4x zero" would be a
# budget no correct implementation could meet. Those tensors are covered by the
# absolute gates instead.

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)


def _has_backport() -> bool:
    try:
        import megatron.core.context_parallel_layout  # noqa: F401
        from fla.ops.cp import build_cp_context  # noqa: F401
    except ImportError:
        return False
    return True


needs_backport = pytest.mark.skipif(not _has_backport(), reason="requires patched Megatron-LM + FLA >= 0.4.2")


# ---------------------------------------------------------------------------
# comparison helpers (run inside the workers)
# ---------------------------------------------------------------------------
def _prep(name, got, want):
    got32 = got.detach().float().flatten()
    want32 = want.detach().float().flatten()
    assert got32.shape == want32.shape, f"{name}: shape {got32.shape} vs {want32.shape}"
    assert torch.isfinite(got32).all(), f"{name}: non-finite values in candidate"
    assert torch.isfinite(want32).all(), f"{name}: non-finite values in reference"
    return got32, want32


def _stats(got32, want32):
    diff = (got32 - want32).abs()
    rms = (diff.square().mean().sqrt() / (want32.square().mean().sqrt() + 1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(got32, want32, dim=0).item()
    return diff, rms, cos


def _report_elementwise(name, got, want, atol, rtol):
    """Per-token tensors: every element within atol + rtol * |ref|."""
    got32, want32 = _prep(name, got, want)
    diff, rms, cos = _stats(got32, want32)
    worst = (diff - (atol + rtol * want32.abs())).max().item()
    assert worst <= 0, (
        f"{name}: max |diff| {diff.max().item():.3e} exceeds atol({atol:.0e})+rtol({rtol:.0e})*|ref| "
        f"by {worst:.3e} (rms {rms:.3e}, cosine {cos:.8f})"
    )
    assert cos >= MIN_COSINE, f"{name}: cosine {cos:.8f} < {MIN_COSINE}"


def _report_rms(name, got, want, ratio):
    """Whole-tensor normalised RMS error -- the metric FLA's own CP tests
    use."""
    got32, want32 = _prep(name, got, want)
    diff, rms, cos = _stats(got32, want32)
    assert rms < ratio, (
        f"{name}: normalised RMS error {rms:.3e} >= {ratio:.1e} (max |diff| {diff.max().item():.3e}, cosine {cos:.8f})"
    )
    assert cos >= MIN_COSINE, f"{name}: cosine {cos:.8f} < {MIN_COSINE} (rms {rms:.3e})"


def _zigzag_shard(full: torch.Tensor, cu, cp_size: int, cp_rank: int) -> torch.Tensor:
    from relax.backends.megatron.cp_utils import gdn_cp_slice

    return gdn_cp_slice(full, cu, cp_size, cp_rank)


def _init_dist(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# ---------------------------------------------------------------------------
# worker: FLA kernel level
# ---------------------------------------------------------------------------
def _worker_fla_kernels(rank, world_size, dtype_name, _unused):
    seq_lens = [256, 128]
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    device = torch.device("cuda", rank)
    cp_group = dist.new_group(list(range(world_size)))

    H, DK, DV, W = 2, 64, 64, 4
    total = sum(seq_lens)
    cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(), device=device, dtype=torch.int32)
    part = total // world_size
    lo, hi = rank * part, (rank + 1) * part

    gen = torch.Generator(device="cpu").manual_seed(1234)

    def _mk(*shape, dtype=dtype):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    conv_ratio = KERNEL_RMS_RATIO_FP32 if dtype is torch.float32 else CONV_RMS_RATIO
    gdn_ratio = KERNEL_RMS_RATIO_FP32 if dtype is torch.float32 else GDN_RMS_RATIO
    tag = f"[rank{rank}/{dtype_name}]"

    # ---- causal conv ----
    # weight/bias are leaves here on purpose: their gradients are sums over every
    # token, which is exactly the quantity chunkwise CP repartitions. Checking only
    # dx would leave that untested.
    x = _mk(1, total, H * DK)
    w0 = _mk(H * DK, W)
    b0 = _mk(H * DK)
    conv_grad = _mk(1, total, H * DK)

    x_ref = x.clone().requires_grad_(True)
    w_ref = w0.clone().requires_grad_(True)
    b_ref = b0.clone().requires_grad_(True)
    out_ref, _ = causal_conv1d(x=x_ref, weight=w_ref, bias=b_ref, activation="silu", cu_seqlens=cu)
    (out_ref.float() * conv_grad.float()).sum().backward()

    x_cp = x[:, lo:hi].clone().requires_grad_(True)
    w_cp = w0.clone().requires_grad_(True)
    b_cp = b0.clone().requires_grad_(True)
    ctx = build_cp_context(cu_seqlens=cu, group=cp_group, conv1d_kernel_size=W)
    out_cp, _ = causal_conv1d(x=x_cp, weight=w_cp, bias=b_cp, activation="silu", cu_seqlens=cu, cp_context=ctx)
    _report_rms(f"{tag} conv fwd", out_cp, out_ref[:, lo:hi], conv_ratio)
    (out_cp.float() * conv_grad[:, lo:hi].float()).sum().backward()
    _report_rms(f"{tag} conv dx", x_cp.grad, x_ref.grad[:, lo:hi], conv_ratio)

    for pname, cp_leaf, ref_leaf in (("dweight", w_cp, w_ref), ("dbias", b_cp, b_ref)):
        summed = cp_leaf.grad.detach().float().clone()
        dist.all_reduce(summed, group=cp_group)
        if dtype is torch.float32:
            _report_rms(f"{tag} conv {pname}", summed, ref_leaf.grad, conv_ratio)
        else:
            # These are 384-token sums landing in bf16. The fp32 parametrisation of
            # this very test pins the algebra at ~1e-7; in bf16 the achievable
            # agreement is set by the storage format, so assert direction and report
            # the size rather than pretend a sub-ULP threshold is meaningful.
            got32, want32 = _prep(f"{tag} conv {pname}", summed, ref_leaf.grad)
            _, rms, cos = _stats(got32, want32)
            assert cos >= MIN_COSINE, f"{tag} conv {pname}: cosine {cos:.8f} (rms {rms:.3e})"
            if rank == 0:
                print(f"    {tag} conv {pname}: rms {rms:.3e} cosine {cos:.10f}")

    # ---- gated delta rule ----
    # Inputs must look like what GatedDeltaNet actually feeds the kernel:
    #   * q/k are L2-normalised (the module sets use_qk_l2norm=True). Un-normalised
    #     q/k make the recurrent state diverge over hundreds of steps and the
    #     reference itself goes to NaN -- that would test nothing.
    #   * g is a log-domain decay built as -A.exp() * softplus(...), hence <= 0.
    q = l2norm(_mk(1, total, H, DK).contiguous())
    k = l2norm(_mk(1, total, H, DK).contiguous())
    v = _mk(1, total, H, DV)
    g0 = -_mk(1, total, H, dtype=torch.float32).abs() * 0.1
    beta0 = _mk(1, total, H, dtype=torch.float32).sigmoid()

    leaves_ref = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    g_ref = g0.detach().clone().requires_grad_(True)
    beta_ref = beta0.detach().clone().requires_grad_(True)
    o_ref, _ = chunk_gated_delta_rule(
        *leaves_ref,
        g=g_ref,
        beta=beta_ref,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu,
    )
    o_grad = _mk(1, total, H, DV)
    (o_ref.float() * o_grad.float()).sum().backward()

    leaves_cp = [t.detach()[:, lo:hi].clone().requires_grad_(True) for t in (q, k, v)]
    g_cp = g0.detach()[:, lo:hi].clone().requires_grad_(True)
    beta_cp = beta0.detach()[:, lo:hi].clone().requires_grad_(True)
    ctx2 = build_cp_context(cu_seqlens=cu, group=cp_group, conv1d_kernel_size=W)
    o_cp, _ = chunk_gated_delta_rule(
        *leaves_cp,
        g=g_cp,
        beta=beta_cp,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu,
        cp_context=ctx2,
    )
    _report_rms(f"{tag} gdr fwd", o_cp, o_ref[:, lo:hi], gdn_ratio)
    (o_cp.float() * o_grad[:, lo:hi].float()).sum().backward()
    for name, a, b in zip("qkv", leaves_cp, leaves_ref):
        _report_rms(f"{tag} gdr d{name}", a.grad, b.grad[:, lo:hi], gdn_ratio)
    _report_rms(f"{tag} gdr dg", g_cp.grad, g_ref.grad[:, lo:hi], gdn_ratio)
    _report_rms(f"{tag} gdr dbeta", beta_cp.grad, beta_ref.grad[:, lo:hi], gdn_ratio)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# worker: full MCore GatedDeltaNet
# ---------------------------------------------------------------------------
def _build_gdn(
    cp_size,
    linear_cp_mode,
    dtype,
    num_key_heads=4,
    num_value_heads=8,
    tp_size=1,
    deterministic_mode=False,
):
    import torch.nn.functional as F
    from megatron.core import parallel_state
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=512,
        num_layers=1,
        num_attention_heads=8,
        num_query_groups=2,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        bf16=dtype is torch.bfloat16,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        deterministic_mode=deterministic_mode,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        linear_cp_mode=linear_cp_mode,
        transformer_impl="transformer_engine",
    )
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    gdn = GatedDeltaNet(
        config,
        submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )
    return gdn.cuda().to(dtype), config


def _run_gdn_once(gdn, hidden, psp, grad_out, *, recompute=False, **forward_kwargs):
    """One forward+backward; returns (out, d_hidden, {param: grad})."""
    gdn.zero_grad(set_to_none=True)
    h = hidden.clone().requires_grad_(True)
    if recompute:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(
            lambda x: gdn(x, None, packed_seq_params=psp, **forward_kwargs)[0],
            h,
            use_reentrant=False,
        )
    else:
        out, _ = gdn(h, None, packed_seq_params=psp, **forward_kwargs)
    (out.float() * grad_out).sum().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in gdn.named_parameters()}
    return out.detach().clone(), h.grad.detach().clone(), grads


def _worker_gdn_module(rank, world_size, dtype_name, _unused):
    """CP=1 reference vs CP=N, for BOTH CP algorithms, in one process.

    Running headwise and chunkwise side by side is the point: it turns "is
    chunkwise close enough to CP=1" (which needs an absolute threshold, and in
    bf16 lands on the noise floor of the storage format) into "is chunkwise as
    close to CP=1 as the algorithm we already ship" -- a comparison with no
    free parameters to tune.
    """
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]

    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    device = torch.device("cuda", rank)
    # The CP algorithm is static config now, so each mode needs its own module. They
    # share weights, so the comparison is still like-for-like.
    modules = {}
    gdn, config = _build_gdn(world_size, "headwise", dtype)
    for p in gdn.parameters():
        # Same weights on every rank so the CP=1 reference is rank-independent.
        dist.broadcast(p.data, src=0)
    modules["headwise"] = gdn
    modules["chunkwise"], _ = _build_gdn(world_size, "chunkwise", dtype)
    modules["chunkwise"].load_state_dict(gdn.state_dict())

    cp_group = parallel_state.get_context_parallel_group()
    # A per-rank size-1 group gives us the CP=1 reference *inside* the same
    # process, driving the very same weights through the very same forward.
    solo = [dist.new_group([r]) for r in range(world_size)][rank]

    seq_lens = [256, 128]
    total = sum(seq_lens)
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int32)

    gen = torch.Generator(device="cpu").manual_seed(7)
    hidden_full = torch.randn(total, 1, config.hidden_size, generator=gen, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    grad_seed = torch.randn(total, 1, config.hidden_size, generator=gen, dtype=torch.float32).to(device)

    def _psp(group, local_cp_size):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max(seq_lens),
            max_seqlen_kv=max(seq_lens),
            cp_group=group,
            local_cp_size=local_cp_size,
        )

    # CP=1 reference. A size-1 group short-circuits before the mode is read, so either
    # module gives the same reference; use the headwise one.
    out_ref, in_grad_ref, param_grads_ref = _run_gdn_once(gdn, hidden_full, _psp(solo, 1), grad_seed)
    ref = {
        "out": _zigzag_shard(out_ref, cu, world_size, rank),
        "d_hidden": _zigzag_shard(in_grad_ref, cu, world_size, rank),
    }
    ref.update({f"grad {n}": g for n, g in param_grads_ref.items()})

    shard = _zigzag_shard(hidden_full, cu, world_size, rank)
    grad_shard = _zigzag_shard(grad_seed, cu, world_size, rank)

    metrics = {}
    for mode in ("headwise", "chunkwise"):
        out_cp, in_grad_cp, param_grads_cp = _run_gdn_once(
            modules[mode], shard, _psp(cp_group, world_size), grad_shard
        )
        got = {"out": out_cp, "d_hidden": in_grad_cp}
        # Each CP rank holds a partial parameter gradient; the total is the CP sum.
        for name, g in param_grads_cp.items():
            summed = g.clone()
            dist.all_reduce(summed, group=cp_group)
            got[f"grad {name}"] = summed
        metrics[mode] = {}
        for key, value in got.items():
            got32, want32 = _prep(f"[rank{rank}][{mode}/{dtype_name}] {key}", value, ref[key])
            diff, rms, cos = _stats(got32, want32)
            metrics[mode][key] = (rms, cos, diff.max().item())
            assert cos >= MIN_COSINE, (
                f"[rank{rank}][{mode}/{dtype_name}] {key}: cosine {cos:.8f} < {MIN_COSINE} (rms {rms:.3e})"
            )

        # RFC section 5 absolute gate, on the per-token tensors, in the dtype the
        # RFC specifies it for. Applied to headwise too, so a drift in the shared
        # plumbing cannot hide behind the comparative check below.
        if dtype is torch.bfloat16:
            for key in ("out", "d_hidden"):
                _report_elementwise(
                    f"[rank{rank}][{mode}/{dtype_name}] {key}", got[key], ref[key], ATOL_BF16, RTOL_BF16
                )
            # Parameter gradients are token-sum reductions stored in bf16. Bound them
            # by the RFC's own relative tolerance for this comparison row (rtol=1e-2)
            # applied to the whole tensor, plus the RFC's cosine floor. What actually
            # pins the algebra is the fp32 parametrisation of this same test.
            for key, (rms, cos, _) in metrics[mode].items():
                if key in ("out", "d_hidden"):
                    continue
                assert rms < RTOL_BF16, (
                    f"[rank{rank}][{mode}/bf16] {key}: relative RMS {rms:.3e} >= {RTOL_BF16:.0e} (cosine {cos:.8f})"
                )
        else:
            for key, (rms, _, _) in metrics[mode].items():
                assert rms < RMS_RATIO_FP32, f"[rank{rank}][{mode}/fp32] {key}: rms {rms:.3e} >= {RMS_RATIO_FP32:.0e}"

    # The comparative assertion -- fp32 only, on purpose.
    #
    # Its premise is "headwise's disagreement with CP=1 is the floor this environment
    # imposes". That holds only while both algorithms perform the *same* reductions.
    # They do not: headwise hands each rank the whole sequence and 1/cp of the heads, so
    # a gradient like conv1d.weight / dt_bias / A_log (a sum over every token) is summed
    # in one go exactly as at CP=1 and can come out bit-exact. Chunkwise splits the
    # tokens, so that same sum really is partitioned and re-added. In fp32 the mantissa
    # absorbs it and the two are directly comparable (observed 1.00x-1.10x). In bf16 the
    # repartitioned sum sits on the format's ULP floor while headwise sits near zero, so
    # their *ratio* measures the dtype, not the algorithm -- bf16 is covered by the
    # absolute gates above instead.
    worst = []
    for key, (rms_c, cos_c, max_c) in metrics["chunkwise"].items():
        rms_h = metrics["headwise"][key][0]
        worst.append((rms_c / max(rms_h, 1e-12), key, rms_c, rms_h))
        if dtype is not torch.float32:
            continue
        assert rms_c <= CHUNKWISE_VS_HEADWISE_RMS_FACTOR * max(rms_h, RMS_FLOOR_FP32), (
            f"[rank{rank}][{dtype_name}] {key}: chunkwise rms {rms_c:.3e} exceeds "
            f"{CHUNKWISE_VS_HEADWISE_RMS_FACTOR}x the shipped headwise rms {rms_h:.3e} "
            f"(cosine {cos_c:.8f}, max |diff| {max_c:.3e})"
        )
    worst.sort(reverse=True)
    if rank == 0:
        print(f"\n[{dtype_name}] chunkwise vs headwise, worst 6 by rms ratio:")
        for ratio, key, rms_c, rms_h in worst[:6]:
            shown = f"{ratio:6.2f}x" if rms_h > 0 else "   n/a"
            print(f"    {key:38s} {shown}   chunkwise {rms_c:.3e}  headwise {rms_h:.3e}")

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _worker_deterministic_reference(rank, world_size, _spec, _unused):
    """The torch-native deterministic rule must accept cp_context=None and
    preserve headwise CP correctness."""
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    device = torch.device("cuda", rank)
    cp_group = parallel_state.get_context_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    solo = [dist.new_group([r]) for r in range(world_size)][rank]

    gdn, config = _build_gdn(
        world_size,
        "headwise",
        torch.float32,
        deterministic_mode=True,
    )
    assert gdn.gated_delta_rule.__name__ == "torch_chunk_gated_delta_rule"
    for p in gdn.parameters():
        dist.broadcast(p.data, src=0, group=cp_group)

    total = 64
    cu = torch.tensor([0, total], device=device, dtype=torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(17)
    hidden_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)
    grad_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)

    solo_pg = ProcessGroupCollection(tp=tp_group, cp=solo)
    out_ref, in_grad_ref, param_grads_ref = _run_gdn_once(
        gdn,
        hidden_full,
        None,
        grad_full,
        pg_collection=solo_pg,
    )

    hidden_shard = _zigzag_shard(hidden_full, cu, world_size, rank)
    grad_shard = _zigzag_shard(grad_full, cu, world_size, rank)
    out_cp, in_grad_cp, param_grads_cp = _run_gdn_once(gdn, hidden_shard, None, grad_shard)

    _report_rms(
        f"[rank{rank}] deterministic out",
        out_cp,
        _zigzag_shard(out_ref, cu, world_size, rank),
        RMS_RATIO_FP32,
    )
    _report_rms(
        f"[rank{rank}] deterministic d_hidden",
        in_grad_cp,
        _zigzag_shard(in_grad_ref, cu, world_size, rank),
        RMS_RATIO_FP32,
    )
    for name, grad in param_grads_cp.items():
        summed = grad.clone()
        dist.all_reduce(summed, group=cp_group)
        _report_rms(
            f"[rank{rank}] deterministic grad {name}",
            summed,
            param_grads_ref[name],
            RMS_RATIO_FP32,
        )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _worker_recompute_parity(rank, world_size, _spec, _unused):
    """External activation checkpointing must replay chunkwise collectives
    without changing outputs or gradients."""
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    device = torch.device("cuda", rank)
    cp_group = parallel_state.get_context_parallel_group()

    gdn, config = _build_gdn(world_size, "chunkwise", torch.float32)
    for p in gdn.parameters():
        dist.broadcast(p.data, src=0, group=cp_group)

    seq_lens = [128, 64]
    total = sum(seq_lens)
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(seq_lens),
        max_seqlen_kv=max(seq_lens),
        cp_group=cp_group,
        local_cp_size=world_size,
    )

    gen = torch.Generator(device="cpu").manual_seed(29)
    hidden_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)
    grad_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)
    hidden = _zigzag_shard(hidden_full, cu, world_size, rank)
    grad = _zigzag_shard(grad_full, cu, world_size, rank)

    out_eager, in_grad_eager, param_grads_eager = _run_gdn_once(gdn, hidden, psp, grad)
    out_recompute, in_grad_recompute, param_grads_recompute = _run_gdn_once(
        gdn,
        hidden,
        psp,
        grad,
        recompute=True,
    )

    _report_rms(f"[rank{rank}] recompute out", out_recompute, out_eager, KERNEL_RMS_RATIO_FP32)
    _report_rms(
        f"[rank{rank}] recompute d_hidden",
        in_grad_recompute,
        in_grad_eager,
        KERNEL_RMS_RATIO_FP32,
    )
    assert set(param_grads_recompute) == set(param_grads_eager)
    for name in param_grads_eager:
        _report_rms(
            f"[rank{rank}] recompute grad {name}",
            param_grads_recompute[name],
            param_grads_eager[name],
            KERNEL_RMS_RATIO_FP32,
        )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _worker_tp2_cp2(rank, world_size, _spec, _unused):
    """Exercise TP head sharding and CP routing together."""
    assert world_size == 4
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )
    device = torch.device("cuda", rank)
    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = cp_group.rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_source = dist.get_process_group_ranks(cp_group)[0]
    solo = [dist.new_group([r]) for r in range(world_size)][rank]

    modules = {}
    headwise, config = _build_gdn(2, "headwise", torch.float32, tp_size=2)
    for p in headwise.parameters():
        dist.broadcast(p.data, src=cp_source, group=cp_group)
    modules["headwise"] = headwise
    modules["chunkwise"], _ = _build_gdn(2, "chunkwise", torch.float32, tp_size=2)
    modules["chunkwise"].load_state_dict(headwise.state_dict())
    sharded_signatures = {}
    for mode, module in modules.items():
        sharded_signatures[mode] = {
            key: (
                tuple(getattr(value, "global_shape", ())),
                tuple(getattr(value, "local_shape", ())),
                getattr(value, "axis_fragmentations", None),
            )
            for key, value in sorted(module.sharded_state_dict(prefix="mixer.").items())
        }
    assert sharded_signatures["headwise"] == sharded_signatures["chunkwise"]

    seq_lens = [128, 64]
    total = sum(seq_lens)
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(41)
    hidden_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)
    grad_full = torch.randn(total, 1, config.hidden_size, generator=gen).to(device)

    def _psp(group, cp_size):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max(seq_lens),
            max_seqlen_kv=max(seq_lens),
            cp_group=group,
            local_cp_size=cp_size,
        )

    solo_pg = ProcessGroupCollection(tp=tp_group, cp=solo)
    out_ref, in_grad_ref, param_grads_ref = _run_gdn_once(
        headwise,
        hidden_full,
        _psp(solo, 1),
        grad_full,
        pg_collection=solo_pg,
    )
    hidden = _zigzag_shard(hidden_full, cu, 2, cp_rank)
    grad = _zigzag_shard(grad_full, cu, 2, cp_rank)
    out_want = _zigzag_shard(out_ref, cu, 2, cp_rank)
    in_grad_want = _zigzag_shard(in_grad_ref, cu, 2, cp_rank)

    for mode, module in modules.items():
        out, in_grad, param_grads = _run_gdn_once(module, hidden, _psp(cp_group, 2), grad)
        _report_rms(f"[rank{rank}][{mode}] TP2/CP2 out", out, out_want, RMS_RATIO_FP32)
        _report_rms(
            f"[rank{rank}][{mode}] TP2/CP2 d_hidden",
            in_grad,
            in_grad_want,
            RMS_RATIO_FP32,
        )
        assert set(param_grads) == set(param_grads_ref)
        for name, param_grad in param_grads.items():
            summed = param_grad.clone()
            dist.all_reduce(summed, group=cp_group)
            _report_rms(
                f"[rank{rank}][{mode}] TP2/CP2 grad {name}",
                summed,
                param_grads_ref[name],
                RMS_RATIO_FP32,
            )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _worker_layout_round_trip(rank, world_size, _spec, _unused):
    """zigzag -> contiguous -> zigzag over a real CP group must be token-exact.

    This is the collective-level version of RFC 5.1: it drives the actual
    ``all_to_all`` in ``context_parallel_layout``, for packed THD (several
    unequal-length samples) and for SBHD.
    """
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core.context_parallel_layout import (
        contiguous_to_zigzag_chunks,
        get_thd_context_parallel_rank_indices,
        zigzag_to_contiguous_chunks,
    )

    device = torch.device("cuda", rank)
    cp_group = dist.new_group(list(range(world_size)))

    # --- packed THD, three samples of different lengths ---
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

    # --- SBHD (chunk-level swap, no cu_seqlens) ---
    seq_local = 2 * world_size * 4
    sbhd = torch.arange(seq_local * 2 * 3, dtype=torch.float64, device=device).reshape(seq_local, 2, 3) + rank * 1e9
    swapped = zigzag_to_contiguous_chunks(sbhd, cp_group, seq_dim=0)
    back = contiguous_to_zigzag_chunks(swapped, cp_group=cp_group, seq_dim=0)
    assert torch.equal(back, sbhd), f"rank {rank}: SBHD round trip is not identity"

    dist.barrier()
    dist.destroy_process_group()


def _worker_illegal_modes_fail_fast(rank, world_size, _spec, _unused):
    """Illegal / unresolved GDN CP modes must raise, not silently pick an
    algorithm."""
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    device = torch.device("cuda", rank)
    cp_group = parallel_state.get_context_parallel_group()
    solo = [dist.new_group([r]) for r in range(world_size)][rank]

    seq_lens = [256, 128]
    total = sum(seq_lens)
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int32)

    def _psp(group, local_cp_size):
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max(seq_lens),
            max_seqlen_kv=max(seq_lens),
            cp_group=group,
            local_cp_size=local_cp_size,
        )

    gdn, config = _build_gdn(world_size, "all_gather", torch.bfloat16)
    hidden = torch.randn(total // world_size, 1, config.hidden_size, device=device, dtype=torch.bfloat16)

    # 1. all_gather is a Relax-wrapper mode: reaching MCore's forward with cp>1 means the
    #    wrapper is missing, and that must be loud.
    with pytest.raises(RuntimeError, match="implemented by the Relax"):
        gdn(hidden, None, packed_seq_params=_psp(cp_group, world_size))

    # 2. ...but a CP=1 micro-batch is legal under any declared mode: it needs no CP
    #    communication at all, so the mode is never consulted.
    full = torch.randn(total, 1, config.hidden_size, device=device, dtype=torch.bfloat16)
    solo_psp = _psp(solo, 1)
    gdn(full, None, packed_seq_params=solo_psp)

    # 3. The final PackedSeqParams must describe the same runtime CP geometry
    #    through both fields.
    gdn_hw, _ = _build_gdn(world_size, "headwise", torch.bfloat16)
    with pytest.raises(ValueError, match="does not match cp_group.size"):
        gdn_hw(hidden, None, packed_seq_params=_psp(cp_group, world_size + 1))

    # 4. deterministic mode has no CP-context scan.
    gdn_cw, _ = _build_gdn(world_size, "chunkwise", torch.bfloat16)
    gdn_cw.config.deterministic_mode = True
    try:
        with pytest.raises((ValueError, AssertionError)):
            gdn_cw(hidden, None, packed_seq_params=_psp(cp_group, world_size))
    finally:
        gdn_cw.config.deterministic_mode = False

    # 5. inference is not supported.
    class _Ctx:
        def is_static_batching(self):
            return True

    with pytest.raises(NotImplementedError):
        gdn_cw(hidden, None, inference_context=_Ctx(), packed_seq_params=_psp(cp_group, world_size))

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _worker_state_dict_invariance(rank, world_size, _spec, _unused):
    """GDN weights must stay TP-only: same keys and shard dims in every CP
    mode."""
    _init_dist(rank, world_size)
    import io

    import torch.distributed as dist
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    device = torch.device("cuda", rank)

    signatures = {}
    for cp_size, mode in ((1, "headwise"), (world_size, "headwise"), (world_size, "chunkwise")):
        gdn, _ = _build_gdn(cp_size, mode, torch.bfloat16)
        sd = gdn.state_dict()
        sharded = gdn.sharded_state_dict(prefix="mixer.")
        signatures[(cp_size, mode)] = (
            {k: tuple(v.shape) for k, v in sd.items() if torch.is_tensor(v)},
            {
                k: (
                    tuple(getattr(v, "global_shape", ())),
                    tuple(getattr(v, "local_shape", ())),
                    getattr(v, "axis_fragmentations", None),
                )
                for k, v in sorted(sharded.items())
            },
        )

        # Exercise real serialization and loading, not just key comparison.
        checkpoint = io.BytesIO()
        torch.save(sd, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, map_location=device, weights_only=True)
        restored, _ = _build_gdn(cp_size, mode, torch.bfloat16)
        restored.load_state_dict(loaded, strict=True)
        for name, tensor in sd.items():
            if torch.is_tensor(tensor):
                assert torch.equal(restored.state_dict()[name], tensor), (
                    f"state_dict round trip changed {name} for {(cp_size, mode)}"
                )
        del restored
        del gdn

    baseline = signatures[(1, "headwise")]
    assert baseline[0], "state_dict is empty; the invariance check would be vacuous"
    assert baseline[1], "sharded_state_dict is empty; the invariance check would be vacuous"
    for key, sig in signatures.items():
        assert sig[0] == baseline[0], f"state_dict shapes changed for {key}"
        assert set(sig[1]) == set(baseline[1]), f"sharded_state_dict keys changed for {key}"
        for k in baseline[1]:
            assert sig[1][k] == baseline[1][k], f"sharded shard dims changed for {key} at {k}"

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def _spawn(fn, spec, port, extra=None):
    _spawn_world(fn, WORLD_SIZE, spec, port, extra=extra)


def _spawn_world(fn, world_size, spec, port, extra=None):
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(
        fn,
        args=(world_size, extra if extra is not None else spec, None),
        nprocs=world_size,
        join=True,
    )


@needs_backport
@pytest.mark.parametrize("dtype_name,port", [("fp32", 29540), ("bf16", 29541)])
def test_fla_cp_kernels_match_single_rank(dtype_name, port):
    """FLA causal_conv1d / chunk_gated_delta_rule under cp_context vs no CP."""
    _spawn(_worker_fla_kernels, dtype_name, port)


@needs_backport
@pytest.mark.parametrize("dtype_name,port", [("fp32", 29542), ("bf16", 29543)])
def test_gdn_cp_matches_cp1(dtype_name, port):
    """RFC 3.4 item 3: CP=2 vs CP=1, forward and backward, both CP
    algorithms."""
    _spawn(_worker_gdn_module, dtype_name, port)


@needs_backport
def test_deterministic_headwise_cp_matches_cp1():
    """The torch-native rule accepts cp_context=None and stays correct under
    headwise CP."""
    _spawn(_worker_deterministic_reference, "n/a", 29547)


@needs_backport
def test_chunkwise_recompute_matches_eager():
    """Replaying chunkwise forward during backward preserves every tested
    gradient."""
    _spawn(_worker_recompute_parity, "n/a", 29548)


@needs_backport
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 CUDA devices for TP2/CP2")
def test_gdn_tp2_cp2_matches_cp1():
    """TP2/CP2 exercises TP head shards and both CP algorithms together."""
    _spawn_world(_worker_tp2_cp2, 4, "n/a", 29549)


@needs_backport
def test_layout_round_trip_over_real_cp_group():
    """RFC 5.1 at the collective level: the layout swap is a pure
    permutation."""
    _spawn(_worker_layout_round_trip, "n/a", 29544)


@needs_backport
def test_illegal_gdn_cp_modes_fail_fast():
    """RFC emphasis 1: illegal combinations must fail fast, never pick
    silently."""
    _spawn(_worker_illegal_modes_fail_fast, "n/a", 29545)


@needs_backport
def test_gdn_state_dict_invariant_across_cp_modes():
    """RFC 3.4 item 4: checkpoint keys and shard dims do not depend on the CP
    mode."""
    _spawn(_worker_state_dict_invariance, "n/a", 29546)

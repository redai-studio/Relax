# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Packed GDN preprocessing with fused convolution, Q/K normalization and
layout gradients."""

import torch
import triton
import triton.language as tl
from fla.modules.conv.triton.ops import causal_conv1d_bwd as _conv_backward

from relax.models.qwen3_5.ops import gdn_preprocessing_kernels as _native


@triton.jit
def _qk_backward(
    Q,
    K,
    R,
    DQ,
    DK,
    DX,
    T,
    RT: tl.constexpr,
    RH: tl.constexpr,
    DQT: tl.constexpr,
    DQH: tl.constexpr,
    DQD: tl.constexpr,
    DKT: tl.constexpr,
    DKH: tl.constexpr,
    DKD: tl.constexpr,
    BT: tl.constexpr,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.arange(0, 128)
    token, head = (rows // 16, rows % 16)
    local = head % 8
    valid = token[:, None] < T
    offset = token[:, None] * 1024 + local[:, None] * 128 + cols[None, :]
    q = tl.load(Q + offset, valid & (head[:, None] < 8), 0).to(tl.float32)
    k = tl.load(K + offset, valid & (head[:, None] >= 8), 0).to(tl.float32)
    dq = tl.load(
        DQ + token[:, None] * DQT + local[:, None] * DQH + cols[None, :] * DQD, valid & (head[:, None] < 8), 0
    ).to(tl.float32)
    dk = tl.load(
        DK + token[:, None] * DKT + local[:, None] * DKH + cols[None, :] * DKD, valid & (head[:, None] >= 8), 0
    ).to(tl.float32)
    y, dy = (q + k, dq + dk)
    r = tl.load(R + token * RT + head * RH, token < T, 0)
    dx = dy * r[:, None] - tl.sum(dy * y, 1)[:, None] * y * r[:, None]
    tl.store(DX + token[:, None] * 5120 + head[:, None] * 128 + cols[None, :], dx, valid)


class FullPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, a_log, dt_bias, cu, cpu_cu, chunks):
        assert _native is not None and x.shape[1:] == (1, 8240) and x.is_contiguous()
        assert weight.shape == (5120, 1, 4) and a_log.shape == dt_bias.shape == (24,)
        assert cu is not None and cpu_cu.device.type == "cpu"
        result = _native._triton_pre_gated_delta_rule_forward(
            x,
            weight,
            a_log,
            dt_bias,
            num_key_heads=8,
            num_value_heads=24,
            key_head_dim=128,
            value_head_dim=128,
            cu_seqlens=cu,
            cp_group=None,
            cp_size=1,
            compute_g_beta=False,
        )
        q, k, value, gate, beta, g, r, boundary, offset, length = result
        assert boundary is None and offset == 0 and (length == x.shape[0])
        assert q.shape == k.shape == (1, x.shape[0], 8, 128)
        assert r.dtype == torch.float32 and r.shape == (1, 16, x.shape[0])
        ctx.save_for_backward(x, weight, a_log, dt_bias, q, k, r, cu, cpu_cu, chunks)
        beta_raw = x[:, :, 8192:8216].transpose(0, 1).contiguous()
        alpha_raw = x[:, :, 8216:8240].transpose(0, 1).contiguous()
        return (q, k, value, gate, beta_raw, alpha_raw)

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta_raw, dalpha_raw):
        x, weight, a_log, dt_bias, q, k, r, cu, cpu_cu, chunks = ctx.saved_tensors
        tokens = x.shape[0]
        main = torch.cuda.current_stream(x.device)
        qs = _native._get_side_stream(x.device, slot=_native._QK_STREAM_SLOT)
        vs = _native._get_side_stream(x.device, slot=_native._V_STREAM_SLOT)
        zs = _native._get_side_stream(x.device, slot=_native._Z_STREAM_SLOT)
        dconv = torch.empty((1, tokens, 5120), dtype=x.dtype, device=x.device)
        draw = torch.empty_like(x)
        qs.wait_stream(main)
        with torch.cuda.stream(qs):
            _qk_backward[triton.cdiv(tokens * 16, 64),](
                q,
                k,
                r,
                dq,
                dk,
                dconv,
                tokens,
                r.stride(2),
                r.stride(1),
                *dq.stride()[1:],
                *dk.stride()[1:],
                BT=64,
                num_warps=4,
            )
        _native._triton_v_layout_to_conv(
            dv, dconv.transpose(1, 2), v_channel_offset=2048, num_value_heads=24, value_head_dim=128, stream=vs
        )
        _native._triton_z_layout_to_qkvzba(
            dgate, draw, z_channel_offset=5120, num_value_heads=24, value_head_dim=128, stream=zs
        )
        main.wait_stream(qs)
        main.wait_stream(vs)
        dx, dw, _, _, _ = _conv_backward(
            x=x[:, :, :5120].transpose(0, 1),
            dy=dconv,
            dht=None,
            weight=weight.view(5120, 4),
            activation="silu",
            cu_seqlens=cu,
            cu_seqlens_cpu=cpu_cu,
            chunk_indices=chunks,
        )
        draw[:, :, :5120].copy_(dx.transpose(0, 1))
        # FLA has already accumulated then cast weight gradients to the weight dtype.
        # Join the original compiled gBeta gradients directly into their disjoint channels.
        assert dbeta_raw is not None and dalpha_raw is not None
        assert dbeta_raw.shape == dalpha_raw.shape == (1, tokens, 24)
        assert dbeta_raw.dtype == dalpha_raw.dtype == x.dtype
        draw[:, :, 8192:8216].copy_(dbeta_raw.transpose(0, 1))
        draw[:, :, 8216:8240].copy_(dalpha_raw.transpose(0, 1))
        dw = dw.view_as(weight).to(weight.dtype)
        main.wait_stream(zs)
        return (draw, dw, None, None, None, None, None)


def install():
    from functools import wraps

    from fla.ops.utils import prepare_chunk_indices
    from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet

    from relax.utils.logging_utils import get_logger

    original = GatedDeltaNet.pre_gated_delta_rule
    seen = set()

    @wraps(original)
    def pre(
        self,
        x,
        batch,
        seq_len,
        cp_size_headwise,
        cp_group_headwise,
        cu_seqlens_q=None,
        chunkwise_cp_context=None,
        packed_seq_params=None,
    ):
        if (
            batch != 1
            or cp_size_headwise != 1
            or chunkwise_cp_context is not None
            or packed_seq_params is None
            or packed_seq_params.qkv_format != "thd"
            or cu_seqlens_q is None
            or x.shape[1:] != (1, 8240)
            or self.conv_bias
            or not self.use_qk_l2norm
            or self.config.deterministic_mode
            or self.config.gdn_conv_pad_alignment is not None
            or not x.is_contiguous()
            or cp_group_headwise is not None
            and cp_group_headwise.size() != 1
        ):
            return original(
                self,
                x,
                batch,
                seq_len,
                cp_size_headwise,
                cp_group_headwise,
                cu_seqlens_q,
                chunkwise_cp_context,
                packed_seq_params=packed_seq_params,
            )
        cache = getattr(packed_seq_params, "_relax_q38_packed_pre_meta", None)
        if cache is None or cache[0] is not cu_seqlens_q:
            bounds = packed_seq_params.cu_seqlens_q_cpu
            if bounds[0] != 0 or bounds[-1] != seq_len:
                raise RuntimeError("CPU document boundaries do not span the physical packed GDN input")
            cpu = torch.as_tensor(bounds, dtype=torch.long, device="cpu")
            chunks = prepare_chunk_indices(cu_seqlens_q, 64, cu_seqlens_cpu=cpu)
            cache = (cu_seqlens_q, cpu, chunks)
            packed_seq_params._relax_q38_packed_pre_meta = cache
        q, k, v, z, beta, alpha = FullPre.apply(x, self.conv1d.weight, self.A_log, self.dt_bias, *cache)
        decay, gates = self._compute_gates(self.A_log, self.dt_bias, batch, seq_len, beta, alpha)
        if not seen:
            get_logger(__name__).info("Native-head packed GDN front end active for %d token rows", seq_len)
            seen.add(True)
        return q.repeat_interleave(3, dim=2), k.repeat_interleave(3, dim=2), v, z, gates["beta"], decay

    GatedDeltaNet.pre_gated_delta_rule = pre

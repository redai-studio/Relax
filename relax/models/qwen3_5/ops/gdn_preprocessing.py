# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Packed GDN preprocessing with fused convolution, Q/K normalization and
layout gradients."""

import torch
import triton
import triton.language as tl


_native = None
_conv_backward = None


def load_backends(root):
    global _native, _conv_backward
    if _native is not None:
        return
    from . import causal_conv_backward as adapter
    from . import gdn_preprocessing_kernels

    _native = gdn_preprocessing_kernels
    kernel = adapter.causal_conv1d_bwd_kernel
    while not hasattr(kernel, "arg_names"):
        assert hasattr(kernel, "fn"), "Cannot inspect FLA kernel signature"
        kernel = kernel.fn
    assert "NB" not in kernel.arg_names, "Expected the installed FLA adapter without the NB keyword"
    _conv_backward = adapter.causal_conv1d_bwd


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
    token, head = (rows // 8, rows % 8)
    local = head % 4
    valid = token[:, None] < T
    offset = token[:, None] * 512 + local[:, None] * 128 + cols[None, :]
    q = tl.load(Q + offset, valid & (head[:, None] < 4), 0).to(tl.float32)
    k = tl.load(K + offset, valid & (head[:, None] >= 4), 0).to(tl.float32)
    dq = tl.load(
        DQ + token[:, None] * DQT + local[:, None] * DQH + cols[None, :] * DQD, valid & (head[:, None] < 4), 0
    ).to(tl.float32)
    dk = tl.load(
        DK + token[:, None] * DKT + local[:, None] * DKH + cols[None, :] * DKD, valid & (head[:, None] >= 4), 0
    ).to(tl.float32)
    y, dy = (q + k, dq + dk)
    r = tl.load(R + token * RT + head * RH, token < T, 0)
    dx = dy * r[:, None] - tl.sum(dy * y, 1)[:, None] * y * r[:, None]
    tl.store(DX + token[:, None] * 2048 + head[:, None] * 128 + cols[None, :], dx, valid)


class FullPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, a_log, dt_bias, cu, cpu_cu, chunks):
        assert _native is not None and x.shape[1:] == (1, 3088) and x.is_contiguous()
        assert weight.shape == (2048, 1, 4) and a_log.shape == dt_bias.shape == (8,)
        assert cu is not None and cpu_cu.device.type == "cpu"
        result = _native._triton_pre_gated_delta_rule_forward(
            x,
            weight,
            a_log,
            dt_bias,
            num_key_heads=4,
            num_value_heads=8,
            key_head_dim=128,
            value_head_dim=128,
            cu_seqlens=cu,
            cp_group=None,
            cp_size=1,
            compute_g_beta=False,
        )
        q, k, value, gate, beta, g, r, boundary, offset, length = result
        assert boundary is None and offset == 0 and (length == x.shape[0])
        assert q.shape == k.shape == (1, x.shape[0], 4, 128)
        assert r.dtype == torch.float32 and r.shape == (1, 8, x.shape[0])
        ctx.save_for_backward(x, weight, a_log, dt_bias, q, k, r, cu, cpu_cu, chunks)
        beta_raw = x[:, :, 3072:3080].transpose(0, 1).contiguous()
        alpha_raw = x[:, :, 3080:3088].transpose(0, 1).contiguous()
        return (q, k, value, gate, beta_raw, alpha_raw)

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta_raw, dalpha_raw):
        x, weight, a_log, dt_bias, q, k, r, cu, cpu_cu, chunks = ctx.saved_tensors
        tokens = x.shape[0]
        main = torch.cuda.current_stream(x.device)
        qs = _native._get_side_stream(x.device, slot=_native._QK_STREAM_SLOT)
        vs = _native._get_side_stream(x.device, slot=_native._V_STREAM_SLOT)
        zs = _native._get_side_stream(x.device, slot=_native._Z_STREAM_SLOT)
        dconv = torch.empty((1, tokens, 2048), dtype=x.dtype, device=x.device)
        draw = torch.empty_like(x)
        qs.wait_stream(main)
        with torch.cuda.stream(qs):
            _qk_backward[triton.cdiv(tokens * 8, 64),](
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
            dv, dconv.transpose(1, 2), v_channel_offset=1024, num_value_heads=8, value_head_dim=128, stream=vs
        )
        _native._triton_z_layout_to_qkvzba(
            dgate, draw, z_channel_offset=2048, num_value_heads=8, value_head_dim=128, stream=zs
        )
        main.wait_stream(qs)
        main.wait_stream(vs)
        _, dw, _, _, _ = _conv_backward(
            x=x[:, :, :2048].transpose(0, 1),
            dy=dconv,
            dht=None,
            weight=weight.view(2048, 4),
            activation="silu",
            cu_seqlens=cu,
            cu_seqlens_cpu=cpu_cu,
            chunk_indices=chunks,
            dx_out=draw[:, :, :2048].transpose(0, 1),
        )
        # FLA has already accumulated then cast weight gradients to the weight dtype.
        # Join the original compiled gBeta gradients directly into their disjoint channels.
        assert dbeta_raw is not None and dalpha_raw is not None
        assert dbeta_raw.shape == dalpha_raw.shape == (1, tokens, 8)
        assert dbeta_raw.dtype == dalpha_raw.dtype == x.dtype
        draw[:, :, 3072:3080].copy_(dbeta_raw.transpose(0, 1))
        draw[:, :, 3080:3088].copy_(dalpha_raw.transpose(0, 1))
        dw = dw.view_as(weight).to(weight.dtype)
        main.wait_stream(zs)
        return (draw, dw, None, None, None, None, None)

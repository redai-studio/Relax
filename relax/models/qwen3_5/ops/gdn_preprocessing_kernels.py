# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Projection and layout kernels used by the Qwen GDN autograd implementation.

Forward projection keeps QK, V, and Z in separate streamed scopes. Packed THD
sequences use separate QK/V causal-convolution kernels. The custom autograd
entry point and backward orchestration live in ``gdn_preprocessing.py``.
"""

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


_L2NORM_EPS = 1e-06
_QK_STREAM_SLOT = 0
_V_STREAM_SLOT = 2
_Z_STREAM_SLOT = 4
_LAYOUT_BLOCK_S = 64


@triton.jit
def _thd_seq_bounds(cu_seqlens_ptr, token_offsets, total_tokens, num_packed_seqs):
    """Return lane-wise packed sequence bounds for flattened THD tokens."""
    safe_tokens = tl.minimum(token_offsets, total_tokens - 1)
    seq_start = token_offsets * 0
    seq_end = token_offsets * 0 + total_tokens
    seq_id = 0
    while seq_id < num_packed_seqs:
        start = tl.load(cu_seqlens_ptr + seq_id)
        end = tl.load(cu_seqlens_ptr + seq_id + 1)
        in_seq = (safe_tokens >= start) & (safe_tokens < end)
        seq_start = tl.where(in_seq, start, seq_start)
        seq_end = tl.where(in_seq, end, seq_end)
        seq_id += 1
    return (seq_start, seq_end)


@triton.jit
def _conv_silu_project_thd_kernel(
    qkvzba_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    silu_save_ptr,
    left_boundary_ptr,
    cu_seqlens_ptr,
    seq_len,
    global_token_offset,
    global_seq_len,
    num_packed_seqs,
    num_in_heads,
    in_channel_offset,
    in_group_stride,
    silu_save_chan_offset,
    silu_save_group_stride,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    bias_stride,
    out_group_dim_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    silu_save_b_stride,
    silu_save_c_stride,
    silu_save_s_stride,
    left_boundary_s_stride,
    left_boundary_b_stride,
    left_boundary_c_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LEFT_BOUNDARY: tl.constexpr,
    APPLY_L2: tl.constexpr,
    SAVE_SILU: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """THD depthwise conv1d + silu + optional l2norm/repeat.

    This is intentionally separate from ``_conv_silu_project_kernel`` so packed
    sequence boundary metadata never enters the dense BSHD hot path. Only the
    causal-conv loads use ``cu_seqlens``; the following per-token transforms
    and stores are identical to the dense path.
    """
    pid_bgh = tl.program_id(0)
    pid_s = tl.program_id(1)
    heads_per_batch = num_in_heads * NUM_GROUPS
    batch_id = pid_bgh // heads_per_batch
    local_bgh = pid_bgh - batch_id * heads_per_batch
    group_id = local_bgh // num_in_heads
    head_id = local_bgh - group_id * num_in_heads
    chan_off = tl.arange(0, HEAD_DIM)
    group_channel_offset = in_channel_offset + group_id * in_group_stride
    chan = group_channel_offset + head_id * HEAD_DIM + chan_off
    if HAS_BIAS:
        bias = tl.load(bias_ptr + chan * bias_stride).to(tl.float32)
    else:
        bias = tl.zeros([HEAD_DIM], dtype=tl.float32)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    global_s = global_token_offset + s_offs
    seq_start, seq_end = _thd_seq_bounds(cu_seqlens_ptr, global_s, global_seq_len, num_packed_seqs)
    acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for i in tl.static_range(K_W):
        x_s = s_offs - (K_W - 1) + i
        global_x_s = global_token_offset + x_s
        x_mask = s_mask & (x_s >= 0) & (global_x_s >= seq_start) & (global_x_s < seq_end)
        safe_x_s = tl.minimum(tl.maximum(x_s, 0), seq_len - 1)
        x_ptr = (
            qkvzba_ptr
            + safe_x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
        if HAS_LEFT_BOUNDARY:
            boundary_s = x_s + (K_W - 1)
            boundary_mask = s_mask & (x_s < 0) & (boundary_s >= 0) & (global_x_s >= seq_start) & (global_x_s < seq_end)
            safe_boundary_s = tl.maximum(boundary_s, 0)
            boundary_ptr = (
                left_boundary_ptr
                + safe_boundary_s[:, None] * left_boundary_s_stride
                + batch_id * left_boundary_b_stride
                + chan[None, :] * left_boundary_c_stride
            )
            boundary_val = tl.load(boundary_ptr, mask=boundary_mask[:, None], other=0.0).to(tl.float32)
            x_val = tl.where(x_mask[:, None], x_val, boundary_val)
        w_tap = tl.load(weight_ptr + chan * weight_c_stride + i * weight_w_stride).to(tl.float32)
        acc += w_tap[None, :] * x_val
    acc += bias[None, :]
    acc = acc
    silu_out = acc * tl.sigmoid(acc)
    if APPLY_L2:
        silu_out = silu_out.to(out_ptr.dtype.element_ty).to(tl.float32)
        norm_sq = tl.sum(silu_out * silu_out, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        out = silu_out * rstd[:, None]
        if SAVE_SILU:
            rstd_ptrs = (
                silu_save_ptr
                + batch_id * silu_save_b_stride
                + (group_id * num_in_heads + head_id) * silu_save_c_stride
                + s_offs * silu_save_s_stride
            )
            tl.store(rstd_ptrs, rstd, mask=s_mask)
    else:
        out = silu_out
    out_typed = out.to(out_ptr.dtype.element_ty)
    for r in tl.static_range(REPEAT):
        v_head = head_id * REPEAT + r
        write_ptr = (
            out_ptr
            + group_id * out_group_dim_stride
            + batch_id * out_b_stride
            + s_offs[:, None] * out_s_stride
            + v_head * out_h_stride
            + chan_off[None, :]
        )
        tl.store(write_ptr, out_typed, mask=s_mask[:, None])


@triton.jit
def _copy_z_kernel(
    qkvzba_ptr,
    gate_ptr,
    seq_len,
    num_v_heads,
    z_channel_offset,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    gate_b_stride,
    gate_s_stride,
    gate_h_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Copy the z slice from qkvzba into the final gate layout."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads
    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    z_chan = z_channel_offset + head_id * HEAD_DIM + chan_off
    z_src_ptr = (
        qkvzba_ptr + s_offs[:, None] * qkvzba_s_stride + batch_id * qkvzba_b_stride + z_chan[None, :] * qkvzba_c_stride
    )
    z_val = tl.load(z_src_ptr, mask=s_mask[:, None])
    z_write_ptr = (
        gate_ptr
        + batch_id * gate_b_stride
        + s_offs[:, None] * gate_s_stride
        + head_id * gate_h_stride
        + chan_off[None, :]
    )
    tl.store(z_write_ptr, z_val, mask=s_mask[:, None])


@triton.jit
def _v_layout_to_conv_kernel(
    dv_ptr,
    d_silu_conv_ptr,
    seq_len,
    num_v_heads,
    v_channel_offset,
    dv_b_stride,
    dv_s_stride,
    dv_h_stride,
    d_silu_b_stride,
    d_silu_c_stride,
    d_silu_s_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Write V-branch gradients into the conv-backward layout.

    ``dv`` is the gradient of ``value`` (forward layout ``(b, s, num_v_heads,
    value_head_dim)``). The conv backward needs ``d_silu_conv`` in layout ``(b,
    conv_dim, s)`` for the V channel slice.
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads
    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    dv_ptrs = (
        dv_ptr + batch_id * dv_b_stride + s_offs[:, None] * dv_s_stride + head_id * dv_h_stride + chan_off[None, :]
    )
    dv_val = tl.load(dv_ptrs, mask=s_mask[:, None], other=0.0)
    d_silu_chan = v_channel_offset + head_id * HEAD_DIM + chan_off
    d_silu_ptrs = (
        d_silu_conv_ptr
        + batch_id * d_silu_b_stride
        + d_silu_chan[None, :] * d_silu_c_stride
        + s_offs[:, None] * d_silu_s_stride
    )
    tl.store(d_silu_ptrs, dv_val, mask=s_mask[:, None])


@triton.jit
def _z_layout_to_qkvzba_kernel(
    dgate_ptr,
    d_qkvzba_ptr,
    seq_len,
    num_v_heads,
    z_channel_offset,
    dgate_b_stride,
    dgate_s_stride,
    dgate_h_stride,
    d_qkvzba_s_stride,
    d_qkvzba_b_stride,
    d_qkvzba_c_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Write gate gradients into the z slice of ``d_qkvzba``.

    ``dgate`` is the autograd-supplied gradient of ``gate`` (= the z slice of
    qkvzba in forward) with layout ``(b, s, num_v_heads, value_head_dim)``. We
    need to write it into ``d_qkvzba``'s z slice — layout ``(s, b,
    total_channels)`` with channels in ``[z_channel_offset, z_channel_offset +
    v_channels)``.
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads
    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    dgate_ptrs = (
        dgate_ptr
        + batch_id * dgate_b_stride
        + s_offs[:, None] * dgate_s_stride
        + head_id * dgate_h_stride
        + chan_off[None, :]
    )
    dgate_val = tl.load(dgate_ptrs, mask=s_mask[:, None], other=0.0)
    d_qkvzba_chan = z_channel_offset + head_id * HEAD_DIM + chan_off
    d_qkvzba_ptrs = (
        d_qkvzba_ptr
        + s_offs[:, None] * d_qkvzba_s_stride
        + batch_id * d_qkvzba_b_stride
        + d_qkvzba_chan[None, :] * d_qkvzba_c_stride
    )
    tl.store(d_qkvzba_ptrs, dgate_val, mask=s_mask[:, None])


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & value - 1 == 0


_SIDE_STREAMS: dict = {}


def _get_side_stream(device: torch.device, slot: int) -> "torch.cuda.Stream":
    """Lazily allocate and cache CUDA streams keyed by ``(device, slot)``.

    Reusing streams across calls keeps launches free of stream-creation
    overhead, which would otherwise dominate the small kernels.
    """
    key = (device.index if device.index is not None else torch.cuda.current_device(), slot)
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SIDE_STREAMS[key] = stream
    return stream


def _triton_v_layout_to_conv(
    dv: Tensor,
    d_silu_conv: Tensor,
    *,
    v_channel_offset: int,
    num_value_heads: int,
    value_head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Write ``dv`` into ``d_silu_conv``'s V channel slice."""
    batch, seq_len, _, _ = dv.shape
    device = dv.device
    BLOCK_S = _LAYOUT_BLOCK_S
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)
    with _launch_context(device, stream):
        _v_layout_to_conv_kernel[grid](
            dv,
            d_silu_conv,
            seq_len,
            num_value_heads,
            v_channel_offset,
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            d_silu_conv.stride(0),
            d_silu_conv.stride(1),
            d_silu_conv.stride(2),
            HEAD_DIM=value_head_dim,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )


def _triton_z_layout_to_qkvzba(
    dgate: Tensor,
    d_qkvzba: Tensor,
    *,
    z_channel_offset: int,
    num_value_heads: int,
    value_head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Write ``dgate`` into ``d_qkvzba``'s z channel slice."""
    batch, seq_len, _, _ = dgate.shape
    device = dgate.device
    BLOCK_S = _LAYOUT_BLOCK_S
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)
    with _launch_context(device, stream):
        _z_layout_to_qkvzba_kernel[grid](
            dgate,
            d_qkvzba,
            seq_len,
            num_value_heads,
            z_channel_offset,
            dgate.stride(0),
            dgate.stride(1),
            dgate.stride(2),
            d_qkvzba.stride(0),
            d_qkvzba.stride(1),
            d_qkvzba.stride(2),
            HEAD_DIM=value_head_dim,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )


def _launch_context(device: torch.device, stream: Optional["torch.cuda.Stream"]):
    """Return a CUDA launch context after wiring the optional side stream."""
    if stream is None:
        return nullcontext()
    stream.wait_stream(torch.cuda.current_stream(device))
    return torch.cuda.stream(stream)


def _wait_for_streams(dst_stream: "torch.cuda.Stream", *src_streams: "torch.cuda.Stream") -> None:
    for stream in src_streams:
        dst_stream.wait_stream(stream)


def _triton_pre_gated_delta_rule_forward(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    cu_seqlens: Optional[Tensor] = None,
    cp_group=None,
    cp_size: int = 1,
    compute_g_beta: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], int, int]:
    """Packed THD/CP1 projection; G/Beta activation is handled by GDN."""
    assert cu_seqlens is not None and cp_group is None and (cp_size == 1) and (not compute_g_beta)
    seq_len, batch, total_channels = qkvzba.shape
    assert batch == 1, f"Packed THD fused_pre_gated_delta_rule expects batch dimension 1; got batch={batch!r}."
    num_packed_seqs = cu_seqlens.shape[0] - 1
    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    k_w = conv1d_weight.shape[-1]
    left_boundary = None
    global_token_offset = 0
    global_seq_len = seq_len
    has_left_boundary = left_boundary is not None
    left_boundary = qkvzba
    assert _is_power_of_two(key_head_dim), (
        f"Triton kernel currently expects key_head_dim to be a power of two; got key_head_dim={key_head_dim!r}."
    )
    assert _is_power_of_two(value_head_dim), (
        f"Triton kernel currently expects value_head_dim to be a power of two; got value_head_dim={value_head_dim!r}."
    )
    expected_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
    assert total_channels == expected_channels, (
        f"qkvzba last-dim mismatch: got {total_channels}, expected {expected_channels}."
    )
    out_dtype = qkvzba.dtype
    device = qkvzba.device
    qk_out = torch.empty(2, batch, seq_len, num_key_heads, key_head_dim, dtype=out_dtype, device=device)
    query = qk_out[0]
    key = qk_out[1]
    value = torch.empty(batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device)
    g = None
    beta = None
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)
    bias_tensor = qkvzba
    bias_stride = 0
    gate = torch.empty(batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device)
    silu_qk_save = torch.empty((batch, seq_len, 2 * num_key_heads), dtype=torch.float32, device=device).permute(
        0, 2, 1
    )
    silu_save_b_stride = silu_qk_save.stride(0)
    silu_save_c_stride = silu_qk_save.stride(1)
    silu_save_s_stride = silu_qk_save.stride(2)
    main_stream = torch.cuda.current_stream(device=device)
    qk_stream = _get_side_stream(device, slot=_QK_STREAM_SLOT)
    v_stream = _get_side_stream(device, slot=_V_STREAM_SLOT)
    z_stream = _get_side_stream(device, slot=_Z_STREAM_SLOT)
    for stream in (qk_stream, v_stream, z_stream):
        stream.wait_stream(main_stream)
    v_channel_offset = 2 * qk_channels
    z_channel_offset = 2 * qk_channels + v_channels

    def _launch_z_copy() -> None:
        BLOCK_Z_S = _LAYOUT_BLOCK_S
        z_grid = (batch * num_value_heads, triton.cdiv(seq_len, BLOCK_Z_S))
        with torch.cuda.stream(z_stream):
            _copy_z_kernel[z_grid](
                qkvzba,
                gate,
                seq_len,
                num_value_heads,
                z_channel_offset,
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                gate.stride(0),
                gate.stride(1),
                gate.stride(2),
                HEAD_DIM=value_head_dim,
                BLOCK_S=BLOCK_Z_S,
                num_warps=4,
                num_stages=2,
            )

    qk_grid = lambda meta: (batch * 2 * num_key_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with torch.cuda.stream(qk_stream):
        _conv_silu_project_thd_kernel[qk_grid](
            qkvzba,
            weight_2d,
            bias_tensor,
            qk_out,
            silu_qk_save,
            left_boundary,
            cu_seqlens,
            seq_len,
            global_token_offset,
            global_seq_len,
            num_packed_seqs,
            num_key_heads,
            0,
            qk_channels,
            0,
            qk_channels,
            qkvzba.stride(0),
            qkvzba.stride(1),
            qkvzba.stride(2),
            weight_2d.stride(0),
            weight_2d.stride(1),
            bias_stride,
            qk_out.stride(0),
            qk_out.stride(1),
            qk_out.stride(2),
            qk_out.stride(3),
            silu_save_b_stride,
            silu_save_c_stride,
            silu_save_s_stride,
            left_boundary.stride(0),
            left_boundary.stride(1),
            left_boundary.stride(2),
            _L2NORM_EPS,
            HEAD_DIM=key_head_dim,
            K_W=k_w,
            REPEAT=1,
            NUM_GROUPS=2,
            HAS_BIAS=False,
            HAS_LEFT_BOUNDARY=has_left_boundary,
            SAVE_SILU=True,
            APPLY_L2=True,
            BLOCK_S=16,
            num_warps=2,
            num_stages=2,
        )
    v_grid = lambda meta: (batch * num_value_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with torch.cuda.stream(v_stream):
        _conv_silu_project_thd_kernel[v_grid](
            qkvzba,
            weight_2d,
            bias_tensor,
            value,
            qkvzba,
            left_boundary,
            cu_seqlens,
            seq_len,
            global_token_offset,
            global_seq_len,
            num_packed_seqs,
            num_value_heads,
            v_channel_offset,
            0,
            0,
            0,
            qkvzba.stride(0),
            qkvzba.stride(1),
            qkvzba.stride(2),
            weight_2d.stride(0),
            weight_2d.stride(1),
            bias_stride,
            0,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            0,
            0,
            0,
            left_boundary.stride(0),
            left_boundary.stride(1),
            left_boundary.stride(2),
            _L2NORM_EPS,
            HEAD_DIM=value_head_dim,
            K_W=k_w,
            REPEAT=1,
            NUM_GROUPS=1,
            HAS_BIAS=False,
            HAS_LEFT_BOUNDARY=has_left_boundary,
            SAVE_SILU=False,
            APPLY_L2=False,
            BLOCK_S=32,
            num_warps=2,
            num_stages=2,
        )
    _launch_z_copy()
    _wait_for_streams(main_stream, qk_stream, v_stream, z_stream)
    return (
        query,
        key,
        value,
        gate,
        beta,
        g,
        silu_qk_save,
        left_boundary if has_left_boundary else None,
        global_token_offset,
        global_seq_len,
    )

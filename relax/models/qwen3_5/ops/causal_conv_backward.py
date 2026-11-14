# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Generated local adapter: FLA backward with caller-owned strided dx.
# Adapt the native FLA convolution backward to write into caller-owned storage.
import torch
import triton
from einops import rearrange
from fla.modules.conv.triton.ops import (
    causal_conv1d_bwd_kernel,
    causal_conv1d_fwd,
    compute_dh0_triton,
    prepare_chunk_indices,
)


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    *,
    dx_out,
):
    shape = x.shape
    assert dx_out is not None and dx_out.shape == x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None
    stride_x_n, stride_x_t, stride_x_d = x.stride()
    BW = triton.next_power_of_2(W)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=False,
            chunk_indices=chunk_indices,
        )
    dx = dx_out
    dw = weight.new_empty(B * NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None
    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()
    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_triton(
            dy=dy, y=y, weight=weight, initial_state=initial_state, activation=activation, cu_seqlens=cu_seqlens
        )
    return (dx.view(shape), dw, db, dr, dh0)

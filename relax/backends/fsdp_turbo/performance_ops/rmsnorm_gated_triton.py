# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-MM; full third-party license notices are in __init__.py.
"""Fused Qwen4ExpTextRMSNormGated forward and backward for Ascend NPU.

A custom autograd function launches:

* one forward kernel that reads ``x``/``gate`` and writes ``y`` plus fp32 rstd;
* one backward kernel that reads saved ``x``/``gate``/rstd and writes
  ``dx``/``dgate`` while accumulating fp32 partial ``dweight`` values;
* one small kernel that reduces the per-program ``dweight`` partials.

The reference is Turbo's NPU weighted RMSNorm followed by activation of
an input-dtype gate. Normalization and weighting accumulate before casting.
"""

from __future__ import annotations

import torch


try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

try:
    import triton
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    triton = None
    tl = None
else:
    import triton.language as tl


_MAX_NUMEL_I32 = 2**31 - 1
_MAX_BLOCK_N = 4096
_MAX_FORWARD_TILE_ELEMENTS = 8192
_MAX_BACKWARD_TILE_ELEMENTS = 4096
_PREFETCH_ROW_THRESHOLD = 131072


if triton is not None:

    @triton.jit(do_not_specialize=["num_rows"])
    def _qwen4_gated_forward_kernel(
        y_ptr,
        rstd_ptr,
        x_ptr,
        gate_ptr,
        weight_ptr,
        num_rows,
        eps: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        IS_SILU: tl.constexpr,
        PREFETCH_LOADS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        if EVEN_N:
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        else:
            weight = tl.load(
                weight_ptr + cols,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)

        num_row_blocks = tl.cdiv(num_rows, BLOCK_M)
        blocks_per_program = tl.cdiv(num_row_blocks, num_programs)
        rows_per_program = blocks_per_program * BLOCK_M
        row_start = pid * rows_per_program
        row_end = tl.minimum(row_start + rows_per_program, num_rows)
        row_step = BLOCK_M

        for row0 in range(row_start, row_end, row_step):
            rows = row0 + tl.arange(0, BLOCK_M)
            row_mask = rows < num_rows
            if EVEN_N:
                mask = row_mask[:, None]
            else:
                mask = row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * N + cols[None, :]

            if PREFETCH_LOADS:
                x_raw = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                gate_raw = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
                gate = gate_raw.to(tl.float32)
            else:
                x_raw = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                gate = tl.load(
                    gate_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            x = x_raw.to(tl.float32)
            variance = tl.sum(x * x, axis=1) / N
            rstd = 1.0 / tl.sqrt(variance + eps)

            normalized = x * rstd[:, None]
            transformed = normalized * weight[None, :]
            # npu_rms_norm returns the input dtype even with FP32 weights.
            transformed = transformed.to(x_raw.dtype).to(tl.float32)

            sigmoid_gate = tl.sigmoid(gate)
            if IS_SILU:
                transformed *= (gate * sigmoid_gate).to(x_raw.dtype).to(tl.float32)
            else:
                transformed *= sigmoid_gate.to(x_raw.dtype).to(tl.float32)

            tl.store(y_ptr + offsets, transformed, mask=mask)
            tl.store(rstd_ptr + rows, rstd, mask=row_mask)

    @triton.jit(do_not_specialize=["num_rows"])
    def _qwen4_gated_backward_kernel(
        dx_ptr,
        dgate_ptr,
        partial_dweight_ptr,
        dy_ptr,
        x_ptr,
        gate_ptr,
        weight_ptr,
        rstd_ptr,
        num_rows,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        IS_SILU: tl.constexpr,
        COMPUTE_DWEIGHT: tl.constexpr,
        PREFETCH_LOADS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        if EVEN_N:
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        else:
            weight = tl.load(
                weight_ptr + cols,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
        dweight = tl.zeros((BLOCK_N,), dtype=tl.float32)

        num_row_blocks = tl.cdiv(num_rows, BLOCK_M)
        blocks_per_program = tl.cdiv(num_row_blocks, num_programs)
        rows_per_program = blocks_per_program * BLOCK_M
        row_start = pid * rows_per_program
        row_end = tl.minimum(row_start + rows_per_program, num_rows)
        row_step = BLOCK_M

        for row0 in range(row_start, row_end, row_step):
            rows = row0 + tl.arange(0, BLOCK_M)
            row_mask = rows < num_rows
            if EVEN_N:
                mask = row_mask[:, None]
            else:
                mask = row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * N + cols[None, :]

            if PREFETCH_LOADS:
                x_raw = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                gate_raw = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
                dy_raw = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
                rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
                x = x_raw.to(tl.float32)
                gate = gate_raw.to(tl.float32)
                dy = dy_raw.to(tl.float32)
            else:
                x_raw = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                gate = tl.load(
                    gate_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                dy = tl.load(
                    dy_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
                x = x_raw.to(tl.float32)
            normalized = x * rstd[:, None]
            transformed = normalized * weight[None, :]
            # npu_rms_norm returns the input dtype even with FP32 weights.
            transformed = transformed.to(x_raw.dtype).to(tl.float32)

            sigmoid_gate = tl.sigmoid(gate)
            if IS_SILU:
                activated_gate = (gate * sigmoid_gate).to(x_raw.dtype).to(tl.float32)
                grad_activation = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
            else:
                activated_gate = sigmoid_gate.to(x_raw.dtype).to(tl.float32)
                grad_activation = activated_gate * (1.0 - activated_gate)

            d_transformed = dy * activated_gate
            d_transformed = d_transformed.to(x_raw.dtype).to(tl.float32)
            d_activated_gate = (dy * transformed).to(x_raw.dtype).to(tl.float32)

            # Weighted RMSNorm accumulates its backward in FP32.
            d_normalized = d_transformed * weight[None, :]
            dot = tl.sum(d_normalized * x, axis=1)
            rstd3_over_n = rstd * rstd * rstd / N
            dx = rstd[:, None] * d_normalized - (x * dot[:, None] * rstd3_over_n[:, None])
            dgate = d_activated_gate * grad_activation

            tl.store(dx_ptr + offsets, dx, mask=mask)
            tl.store(dgate_ptr + offsets, dgate, mask=mask)
            if COMPUTE_DWEIGHT:
                dweight += tl.sum(d_transformed * normalized, axis=0)

        if COMPUTE_DWEIGHT:
            tl.store(
                partial_dweight_ptr + pid * N + cols,
                dweight,
                mask=col_mask,
            )

    @triton.jit(do_not_specialize=["num_programs"])
    def _qwen4_gated_reduce_dweight_kernel(
        partial_dweight_ptr,
        dweight_ptr,
        num_programs,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        value = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for program in range(num_programs):
            value += tl.load(
                partial_dweight_ptr + program * N + cols,
                mask=mask,
                other=0.0,
            )
        tl.store(dweight_ptr + cols, value, mask=mask)


def _fast_forward_available(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    if triton is None or torch_npu is None:
        return False
    if hidden_states.device.type not in ("npu", "privateuseone"):
        return False
    if hidden_states.shape != gate.shape or hidden_states.ndim == 0:
        return False
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.shape != (hidden_states.shape[-1],):
        return False
    if weight.device != hidden_states.device:
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    promoted_dtype = torch.promote_types(hidden_states.dtype, weight.dtype)
    if promoted_dtype != torch.float32 and (
        promoted_dtype != hidden_states.dtype or weight.dtype != hidden_states.dtype
    ):
        return False
    if hidden_states.numel() >= _MAX_NUMEL_I32:
        return False
    return 0 < hidden_states.shape[-1] <= _MAX_BLOCK_N


def _program_count(num_rows: int, block_rows: int, limit: int) -> int:
    return max(1, min(triton.cdiv(num_rows, block_rows), limit))


def _tile_rows(
    requested_rows: int,
    block_n: int,
    max_elements: int,
) -> int:
    return min(requested_rows, max_elements // block_n)


def _launch_forward(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
    block_rows: int,
    program_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = hidden_states.contiguous()
    g = gate.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()
    dim = x.shape[-1]
    num_rows = x.numel() // dim
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_n = triton.next_power_of_2(dim)
    effective_rows = _tile_rows(
        block_rows,
        block_n,
        _MAX_FORWARD_TILE_ELEMENTS,
    )
    rstd = torch.empty(
        num_rows,
        dtype=torch.float32,
        device=x.device,
    )
    grid = (_program_count(num_rows, effective_rows, program_limit),)
    prefetch_loads = num_rows >= _PREFETCH_ROW_THRESHOLD
    _qwen4_gated_forward_kernel[grid](
        y,
        rstd,
        x,
        g,
        w,
        num_rows,
        eps,
        N=dim,
        BLOCK_N=block_n,
        BLOCK_M=effective_rows,
        EVEN_N=block_n == dim,
        IS_SILU=activation == "silu",
        PREFETCH_LOADS=prefetch_loads,
    )
    return y, rstd


class _Qwen4RMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        activation: str,
        block_rows: int,
        program_limit: int,
        backward_block_rows: int,
    ):
        x = hidden_states.contiguous()
        g = gate.contiguous()
        w = weight if weight.is_contiguous() else weight.contiguous()
        y, rstd = _launch_forward(
            x,
            g,
            w,
            eps,
            activation,
            block_rows,
            program_limit,
        )
        ctx.save_for_backward(x, g, w, rstd)
        ctx.activation = activation
        ctx.program_limit = program_limit
        ctx.backward_block_rows = backward_block_rows
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate, weight, rstd = ctx.saved_tensors
        dy = grad_output.contiguous()
        dim = x.shape[-1]
        num_rows = x.numel() // dim
        if num_rows == 0:
            return (
                torch.empty_like(x) if ctx.needs_input_grad[0] else None,
                torch.empty_like(gate) if ctx.needs_input_grad[1] else None,
                (torch.zeros_like(weight) if ctx.needs_input_grad[2] else None),
                None,
                None,
                None,
                None,
                None,
            )
        block_n = triton.next_power_of_2(dim)
        effective_rows = _tile_rows(
            ctx.backward_block_rows,
            block_n,
            _MAX_BACKWARD_TILE_ELEMENTS,
        )
        num_programs = _program_count(
            num_rows,
            effective_rows,
            ctx.program_limit,
        )
        prefetch_loads = num_rows >= _PREFETCH_ROW_THRESHOLD

        dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        dgate = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
        compute_dweight = ctx.needs_input_grad[2]
        if compute_dweight:
            partial_dweight = torch.empty(
                (num_programs, dim),
                dtype=torch.float32,
                device=weight.device,
            )
            dweight = torch.empty_like(weight)
        else:
            partial_dweight = torch.empty(
                1,
                dtype=torch.float32,
                device=weight.device,
            )
            dweight = None

        _qwen4_gated_backward_kernel[(num_programs,)](
            dx,
            dgate,
            partial_dweight,
            dy,
            x,
            gate,
            weight,
            rstd,
            num_rows,
            N=dim,
            BLOCK_N=block_n,
            BLOCK_M=effective_rows,
            EVEN_N=block_n == dim,
            IS_SILU=ctx.activation == "silu",
            COMPUTE_DWEIGHT=compute_dweight,
            PREFETCH_LOADS=prefetch_loads,
        )
        if compute_dweight:
            _qwen4_gated_reduce_dweight_kernel[(1,)](
                partial_dweight,
                dweight,
                num_programs,
                N=dim,
                BLOCK_N=block_n,
            )

        return (
            dx if ctx.needs_input_grad[0] else None,
            dgate if ctx.needs_input_grad[1] else None,
            dweight,
            None,
            None,
            None,
            None,
            None,
        )

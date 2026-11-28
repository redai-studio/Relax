# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-MM; full third-party license notices are in __init__.py.
"""Single-operator Triton QHC implementation for Qwen4-Exp.

The public QHC module uses one custom autograd operator. The operator performs
grouped RMSNorm, both low-rank projections, the injection projection, and the
fused gated post-processing. Large projections remain on torch_npu GEMM because
Triton-Ascend `tl.dot` measured substantially slower than the native Linear
implementation on Ascend 910B3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_MAX_RMS_GROUP_SIZE = 4096
_CORE_CACHE: dict[tuple[str, int | None], int] = {}
_UB_CACHE: dict[tuple[str, int | None], int] = {}
_RMS_FWD_PLAN_CACHE: dict[tuple, tuple] = {}
_RMS_BWD_PLAN_CACHE: dict[tuple, tuple] = {}
_POST_FWD_PLAN_CACHE: dict[tuple, tuple] = {}
_POST_BWD_PLAN_CACHE: dict[tuple, tuple] = {}
_DUMMY_INJECTION: dict[tuple[str, int | None], torch.Tensor] = {}


@triton.jit(do_not_specialize=["num_rows"])
def _grouped_rmsnorm_forward_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    num_rows,
    row_stride,
    eps,
    n_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    STORE_RSTD: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row_step = nprog * BLOCK_ROWS
    col_offsets = tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < group_size

    for group in tl.static_range(n_groups):
        col0 = group * group_size
        gamma = 1.0 + tl.load(
            weight_ptr + col0 + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        group_rstd_ptr = rstd_ptr + group * num_rows
        for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
            rows = row0 + tl.arange(0, BLOCK_ROWS)
            row_mask = rows < num_rows
            mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * row_stride + (col0 + col_offsets)[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            variance = tl.sum(x * x, axis=1) / group_size
            rstd = 1.0 / tl.sqrt(variance + eps)
            y = x * rstd[:, None] * gamma[None, :]
            tl.store(y_ptr + offsets, y, mask=mask)
            if STORE_RSTD:
                tl.store(group_rstd_ptr + rows, rstd, mask=row_mask)


@triton.jit(do_not_specialize=["num_rows"])
def _grouped_rmsnorm_backward_kernel(
    grad_x_ptr,
    grad_y_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    partial_grad_weight_ptr,
    num_rows,
    row_stride,
    n_groups: tl.constexpr,
    group_size: tl.constexpr,
    total_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row_step = nprog * BLOCK_ROWS
    col_offsets = tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < group_size

    for group in tl.static_range(n_groups):
        col0 = group * group_size
        gamma = 1.0 + tl.load(
            weight_ptr + col0 + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        group_rstd_ptr = rstd_ptr + group * num_rows
        grad_weight_accum = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

        for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
            rows = row0 + tl.arange(0, BLOCK_ROWS)
            row_mask = rows < num_rows
            mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * row_stride + (col0 + col_offsets)[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(group_rstd_ptr + rows, mask=row_mask, other=0.0)
            grad_normalized = grad_y * gamma[None, :]
            dot = tl.sum(grad_normalized * x, axis=1)
            grad_x = rstd[:, None] * grad_normalized - x * dot[:, None] * (
                rstd[:, None] * rstd[:, None] * rstd[:, None] / group_size
            )
            grad_weight_accum += tl.sum(grad_y * x * rstd[:, None], axis=0)
            tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

        tl.store(
            partial_grad_weight_ptr + pid * total_size + col0 + col_offsets,
            grad_weight_accum,
            mask=col_mask,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qhc_post_forward_kernel(
    mixed_ptr,
    injection_ptr,
    up_ptr,
    normed_ptr,
    injection_logits_ptr,
    num_rows,
    row_stride,
    n_groups: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row_step = nprog * BLOCK_ROWS
    col_offsets = tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < hidden_size

    for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
        rows = row0 + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < num_rows
        mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
        mixed_accum = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

        for group in tl.static_range(n_groups):
            col0 = group * hidden_size
            offsets = rows[:, None] * row_stride + (col0 + col_offsets)[None, :]
            up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            probability = tl.sigmoid(up).to(up_ptr.dtype.element_ty).to(tl.float32)
            mixed_accum += (probability * normed).to(normed_ptr.dtype.element_ty).to(tl.float32)

            if HAS_INJECTION:
                logit = tl.load(
                    injection_logits_ptr + rows * n_groups + group,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                scaled = (logit / n_groups).to(injection_logits_ptr.dtype.element_ty).to(tl.float32)
                probability = tl.sigmoid(scaled).to(injection_logits_ptr.dtype.element_ty).to(tl.float32)
                injection = 2.0 * probability
                tl.store(
                    injection_ptr + rows * n_groups + group,
                    injection,
                    mask=row_mask,
                )

        mixed_offsets = rows[:, None] * hidden_size + col_offsets[None, :]
        tl.store(
            mixed_ptr + mixed_offsets,
            mixed_accum / n_groups,
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qhc_post_backward_kernel(
    grad_up_ptr,
    grad_normed_ptr,
    grad_injection_logits_ptr,
    grad_mixed_ptr,
    grad_injection_ptr,
    up_ptr,
    normed_ptr,
    injection_logits_ptr,
    num_rows,
    row_stride,
    n_groups: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row_step = nprog * BLOCK_ROWS
    col_offsets = tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < hidden_size

    for row0 in range(pid * BLOCK_ROWS, num_rows, row_step):
        rows = row0 + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < num_rows
        mask = row_mask[:, None] if EVEN_COLS else row_mask[:, None] & col_mask[None, :]
        mixed_offsets = rows[:, None] * hidden_size + col_offsets[None, :]
        grad_mixed = tl.load(grad_mixed_ptr + mixed_offsets, mask=mask, other=0.0).to(tl.float32)

        for group in tl.static_range(n_groups):
            col0 = group * hidden_size
            offsets = rows[:, None] * row_stride + (col0 + col_offsets)[None, :]
            up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            probability = tl.sigmoid(up).to(up_ptr.dtype.element_ty).to(tl.float32)
            grad_product = (grad_mixed / n_groups).to(up_ptr.dtype.element_ty).to(tl.float32)
            grad_probability = (grad_product * normed).to(up_ptr.dtype.element_ty).to(tl.float32)
            grad_up = grad_probability * probability * (1.0 - probability)
            grad_normed = grad_product * probability

            if HAS_INJECTION:
                injection_logit = tl.load(
                    injection_logits_ptr + rows * n_groups + group,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                scaled = (injection_logit / n_groups).to(injection_logits_ptr.dtype.element_ty).to(tl.float32)
                injection_probability = tl.sigmoid(scaled).to(injection_logits_ptr.dtype.element_ty).to(tl.float32)
                grad_injection = tl.load(
                    grad_injection_ptr + rows * n_groups + group,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                grad_probability = (grad_injection * 2.0).to(injection_logits_ptr.dtype.element_ty).to(tl.float32)
                grad_scaled = (
                    (grad_probability * injection_probability * (1.0 - injection_probability))
                    .to(injection_logits_ptr.dtype.element_ty)
                    .to(tl.float32)
                )
                grad_logit = grad_scaled / n_groups
                tl.store(
                    grad_injection_logits_ptr + rows * n_groups + group,
                    grad_logit,
                    mask=row_mask,
                )

            tl.store(grad_up_ptr + offsets, grad_up, mask=mask)
            tl.store(grad_normed_ptr + offsets, grad_normed, mask=mask)


def _previous_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value.bit_length() - 1)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return str(device.type), device.index


def _device_properties(device: torch.device):
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device)
    except Exception:
        return None


def _num_cores(device: torch.device) -> int:
    key = _device_key(device)
    if key in _CORE_CACHE:
        return _CORE_CACHE[key]
    cores = None
    properties = _device_properties(device)
    if properties is not None:
        for name in (
            "num_vectorcore",
            "num_aivectorcore",
            "num_aicore",
            "num_cube_core",
        ):
            value = properties.get(name) if isinstance(properties, dict) else getattr(properties, name, None)
            if value:
                cores = int(value)
                break
    if cores is None:
        cores = 40 if str(device.type) in ("npu", "privateuseone") else 1
    _CORE_CACHE[key] = cores
    return cores


def _ub_size(device: torch.device) -> int:
    key = _device_key(device)
    if key in _UB_CACHE:
        return _UB_CACHE[key]
    size = None
    properties = _device_properties(device)
    if properties is not None:
        for name in ("ubsize", "ub_size", "UB_SIZE"):
            value = properties.get(name) if isinstance(properties, dict) else getattr(properties, name, None)
            if value:
                size = int(value)
                break
    if size is None:
        size = 192 * 1024 if str(device.type) in ("npu", "privateuseone") else 64 * 1024
    _UB_CACHE[key] = size
    return size


def _pick_block_rows(
    block_cols: int,
    num_rows: int,
    element_size: int,
    device: torch.device,
    multiplier: int,
    max_rows: int,
) -> int:
    reserve = 6144
    rows_by_ub = max(
        1,
        (_ub_size(device) - reserve) // max(block_cols * element_size * multiplier, 1),
    )
    block_rows = _previous_power_of_two(min(rows_by_ub, max_rows))
    block_rows = min(block_rows, triton.next_power_of_2(max(num_rows, 1)))
    return max(block_rows, 1)


def _rms_launch_plan(
    num_rows: int,
    group_size: int,
    element_size: int,
    device: torch.device,
    backward: bool,
) -> tuple[int, int, int, bool]:
    cache = _RMS_BWD_PLAN_CACHE if backward else _RMS_FWD_PLAN_CACHE
    key = (num_rows, group_size, element_size, _device_key(device))
    plan = cache.get(key)
    if plan is None:
        block_cols = triton.next_power_of_2(group_size)
        # Backward keeps x, grad_y, grad_x, rstd-derived values and the dW
        # accumulator alive together. It needs a smaller row tile than forward.
        multiplier = 16 if backward else (5 if element_size == 4 else 7)
        block_rows = _pick_block_rows(
            block_cols,
            num_rows,
            element_size,
            device,
            multiplier,
            max_rows=8 if backward else 16,
        )
        nprog = max(1, min(triton.cdiv(num_rows, block_rows), _num_cores(device)))
        plan = (block_rows, block_cols, nprog, block_cols == group_size)
        if len(cache) > 512:
            cache.clear()
        cache[key] = plan
    return plan


def _post_launch_plan(
    num_rows: int,
    hidden_size: int,
    element_size: int,
    device: torch.device,
    backward: bool,
) -> tuple[int, int, int, bool]:
    cache = _POST_BWD_PLAN_CACHE if backward else _POST_FWD_PLAN_CACHE
    key = (num_rows, hidden_size, element_size, _device_key(device))
    plan = cache.get(key)
    if plan is None:
        block_cols = triton.next_power_of_2(hidden_size)
        block_rows = _pick_block_rows(
            block_cols,
            num_rows,
            element_size,
            device,
            multiplier=16,
            max_rows=8,
        )
        nprog = max(1, min(triton.cdiv(num_rows, block_rows), _num_cores(device)))
        plan = (block_rows, block_cols, nprog, block_cols == hidden_size)
        if len(cache) > 512:
            cache.clear()
        cache[key] = plan
    return plan


def _dummy_injection(device: torch.device) -> torch.Tensor:
    key = _device_key(device)
    value = _DUMMY_INJECTION.get(key)
    if value is None:
        value = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY_INJECTION[key] = value
    return value


def _linear_forward(x, weight):
    return F.linear(x, weight)


def _linear_backward(grad_output, input_tensor, weight):
    grad_input = grad_output @ weight
    grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).transpose(0, 1) @ input_tensor.reshape(
        -1, input_tensor.shape[-1]
    )
    return grad_input, grad_weight


def _scaled_silu_backward(grad_output, preactivation, groups):
    scaled = preactivation / groups
    sigmoid = torch.sigmoid(scaled)
    grad_scaled = sigmoid * (1.0 + scaled * (1.0 - sigmoid))
    return grad_output * grad_scaled / groups


class _QHCFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hyper_input,
        norm_weight,
        down_weight,
        up_weight,
        injection_weight,
        groups,
        hidden_size,
        eps,
    ):
        hyper_input = hyper_input.contiguous()
        num_rows = hyper_input.numel() // hyper_input.shape[-1]
        total_size = hyper_input.shape[-1]

        normed = torch.empty_like(hyper_input)
        rstd = torch.empty(
            (groups, num_rows),
            device=hyper_input.device,
            dtype=torch.float32,
        )
        if num_rows:
            block_rows, block_cols, nprog, even_cols = _rms_launch_plan(
                num_rows,
                hidden_size,
                hyper_input.element_size(),
                hyper_input.device,
                backward=False,
            )
            _grouped_rmsnorm_forward_kernel[(nprog,)](
                normed,
                hyper_input,
                norm_weight,
                rstd,
                num_rows,
                total_size,
                eps,
                n_groups=groups,
                group_size=hidden_size,
                BLOCK_ROWS=block_rows,
                BLOCK_COLS=block_cols,
                EVEN_COLS=even_cols,
                STORE_RSTD=True,
            )

        down = _linear_forward(normed, down_weight)
        activated = F.silu(down / groups)
        up = _linear_forward(activated, up_weight)

        has_injection = injection_weight is not None
        if has_injection:
            injection_logits = _linear_forward(normed, injection_weight)
        else:
            injection_logits = None

        mixed = up.new_empty((*up.shape[:-1], hidden_size))
        injection = up.new_empty((*up.shape[:-1], groups)) if has_injection else _dummy_injection(up.device)
        if num_rows:
            block_rows, block_cols, nprog, even_cols = _post_launch_plan(
                num_rows,
                hidden_size,
                up.element_size(),
                up.device,
                backward=False,
            )
            _qhc_post_forward_kernel[(nprog,)](
                mixed,
                injection,
                up,
                normed,
                injection_logits if injection_logits is not None else _dummy_injection(up.device),
                num_rows,
                total_size,
                n_groups=groups,
                hidden_size=hidden_size,
                BLOCK_ROWS=block_rows,
                BLOCK_COLS=block_cols,
                EVEN_COLS=even_cols,
                HAS_INJECTION=has_injection,
            )

        saved = [
            hyper_input,
            normed,
            rstd,
            down,
            activated,
            up,
            norm_weight,
            down_weight,
            up_weight,
        ]
        if injection_logits is not None:
            saved.append(injection_logits)
            saved.append(injection_weight)
        ctx.save_for_backward(*saved)
        ctx.has_injection = has_injection
        ctx.groups = groups
        ctx.hidden_size = hidden_size
        ctx.total_size = total_size
        ctx.num_rows = num_rows
        if has_injection:
            return mixed, injection
        return mixed

    @staticmethod
    def backward(ctx, grad_mixed, grad_injection=None):
        saved = ctx.saved_tensors
        (
            hyper_input,
            normed,
            rstd,
            down,
            activated,
            up,
            norm_weight,
            down_weight,
            up_weight,
        ) = saved[:9]
        if ctx.has_injection:
            injection_logits, injection_weight = saved[9:11]
        else:
            injection_logits = None
            injection_weight = None

        grad_up = torch.empty_like(up)
        grad_normed = torch.empty_like(normed)
        if ctx.num_rows:
            block_rows, block_cols, nprog, even_cols = _post_launch_plan(
                ctx.num_rows,
                ctx.hidden_size,
                up.element_size(),
                up.device,
                backward=True,
            )
            grad_injection_logits = (
                torch.empty_like(injection_logits) if injection_logits is not None else _dummy_injection(up.device)
            )
            _qhc_post_backward_kernel[(nprog,)](
                grad_up,
                grad_normed,
                grad_injection_logits,
                grad_mixed.contiguous(),
                grad_injection.contiguous() if grad_injection is not None else _dummy_injection(up.device),
                up,
                normed,
                injection_logits if injection_logits is not None else _dummy_injection(up.device),
                ctx.num_rows,
                ctx.total_size,
                n_groups=ctx.groups,
                hidden_size=ctx.hidden_size,
                BLOCK_ROWS=block_rows,
                BLOCK_COLS=block_cols,
                EVEN_COLS=even_cols,
                HAS_INJECTION=ctx.has_injection,
            )
        else:
            grad_injection_logits = None

        grad_activated, grad_up_weight = _linear_backward(grad_up, activated, up_weight)
        grad_down = _scaled_silu_backward(grad_activated, down, ctx.groups)
        grad_normed_down, grad_down_weight = _linear_backward(grad_down, normed, down_weight)
        grad_normed += grad_normed_down

        if ctx.has_injection:
            grad_normed_injection, grad_injection_weight = _linear_backward(
                grad_injection_logits, normed, injection_weight
            )
            grad_normed += grad_normed_injection
        else:
            grad_injection_weight = None

        grad_hyper_input = torch.empty_like(hyper_input)
        if ctx.num_rows:
            block_rows, block_cols, nprog, even_cols = _rms_launch_plan(
                ctx.num_rows,
                ctx.hidden_size,
                hyper_input.element_size(),
                hyper_input.device,
                backward=True,
            )
            partial_grad_norm = torch.empty(
                (nprog, ctx.total_size),
                device=hyper_input.device,
                dtype=torch.float32,
            )
            _grouped_rmsnorm_backward_kernel[(nprog,)](
                grad_hyper_input,
                grad_normed,
                hyper_input,
                norm_weight,
                rstd,
                partial_grad_norm,
                ctx.num_rows,
                ctx.total_size,
                n_groups=ctx.groups,
                group_size=ctx.hidden_size,
                total_size=ctx.total_size,
                BLOCK_ROWS=block_rows,
                BLOCK_COLS=block_cols,
                EVEN_COLS=even_cols,
            )
            grad_norm_weight = partial_grad_norm.sum(dim=0).to(norm_weight.dtype)
        else:
            grad_norm_weight = torch.zeros_like(norm_weight)

        return (
            grad_hyper_input,
            grad_norm_weight,
            grad_down_weight,
            grad_up_weight,
            grad_injection_weight,
            None,
            None,
            None,
        )

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-MM; full third-party license notices are in __init__.py.
"""Shared Triton grouped RMSNorm kernels and launch planning."""

import torch
import triton
import triton.language as tl


_CORE_CACHE = {}
_UB_CACHE = {}
_RMS_FWD_PLAN_CACHE = {}
_RMS_BWD_PLAN_CACHE = {}


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

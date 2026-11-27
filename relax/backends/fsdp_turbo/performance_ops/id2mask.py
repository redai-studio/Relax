# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-MM; full third-party license notices are in __init__.py.
"""Triton LightningIndexer ID-to-mask postprocessing for Ascend.

Sanitize complete-block IDs, scatter into a compact int8 block-membership map,
then fuse causal/tail handling and final mask materialization. The existing
eager path remains in the LightningIndexer adapter. The caller still combines
this selected mask with its causal/padding mask.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sanitize_block_ids(
    IDs,
    Safe,
    S: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SK: tl.constexpr,
    TOTAL: tl.constexpr,
    TILE: tl.constexpr,
):
    for start in range(tl.program_id(0) * TILE, TOTAL, tl.num_programs(0) * TILE):
        i = start.to(tl.int64) + tl.arange(0, TILE)
        if (SK == 1 and SR == K) and SB == S * K:
            x = tl.load(IDs + i, i < TOTAL, other=-1)
        else:
            row = i // K
            x = tl.load(IDs + row // S * SB + row % S * SR + i % K * SK, i < TOTAL, other=-1)
        tl.store(Safe + i, tl.where((x >= 0) & (x < N), x, N).to(tl.int32), i < TOTAL)


@triton.jit
def _materialize_mask(
    Table,
    Out,
    Q: tl.constexpr,
    KV: tl.constexpr,
    PREFIX: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    BINS: tl.constexpr,
    TILE: tl.constexpr,
    MIN: tl.constexpr,
    BOOL: tl.constexpr,
):
    keys = tl.arange(0, TILE)
    blocks = tl.arange(0, BINS)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        q = row % Q
        pos = q + KV - Q
        if q < PREFIX:
            flags = tl.full((BINS,), 1, tl.int8)
        else:
            flags = tl.load(Table + ((row.to(tl.int64) // Q) * S + q - PREFIX) * STRIDE + blocks, blocks < N, other=0)
        for start in range(tl.cdiv(KV, TILE)):
            key = start * TILE + keys
            selected = tl.gather(flags, tl.minimum(key // R, BINS - 1), 0) != 0
            tail = ((pos + 1) % R != 0) & (key // R == (pos + 1) // R)
            allow = (key <= pos) & (selected | tail)
            if BOOL:
                value = allow
            else:
                value = tl.where(allow, 0, MIN)
            tl.store(Out + row.to(tl.int64) * KV + key, value, key < KV)


@triton.jit
def _materialize_packed16(
    Table,
    Out,
    Q: tl.constexpr,
    KV: tl.constexpr,
    PREFIX: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    BINS: tl.constexpr,
    TILE: tl.constexpr,
    MASK16: tl.constexpr,
):
    # Even R places both packed 16-bit elements in the same selected block.
    # The final causal pair may have only its low half visible.
    pairs = tl.arange(0, TILE)
    blocks = tl.arange(0, BINS)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        q = row % Q
        pos = q + KV - Q
        if q < PREFIX:
            flags = tl.full((BINS,), 1, tl.int8)
        else:
            flags = tl.load(Table + ((row.to(tl.int64) // Q) * S + q - PREFIX) * STRIDE + blocks, blocks < N, other=0)
        for start in range(tl.cdiv(KV // 2, TILE)):
            pair = start * TILE + pairs
            key = pair * 2
            if q < PREFIX:
                active = tl.full((TILE,), True, tl.int1)
            else:
                active = tl.gather(flags, tl.minimum(key // R, BINS - 1), 0) != 0
                if (pos + 1) % R != 0:
                    active |= key // R == (pos + 1) // R
            word = tl.where(
                active & (key <= pos),
                tl.full((TILE,), 0, tl.uint32),
                tl.full((TILE,), MASK16 | (MASK16 << 16), tl.uint32),
            )
            if pos % 2 == 0:
                word |= tl.where(key == pos, tl.full((TILE,), MASK16 << 16, tl.uint32), tl.full((TILE,), 0, tl.uint32))
            tl.store(Out + row.to(tl.int64) * (KV // 2) + pair, word, pair < KV // 2)


@torch.no_grad()
def lightning_ids_to_mask(selected_blocks, *, query_length, kv_length, token_budget, compress_ratio, dtype):
    """Return [B,1,Q,KV], bool True=allowed or additive 0/finfo.min.

    Dense prefix, cache query offset, incomplete tail, invalid IDs and
    duplicate IDs follow the eager LightningIndexer adapter exactly. Input may
    be strided.
    """
    if selected_blocks.ndim != 3 or selected_blocks.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_blocks must be an int32/int64 [B,sparse_rows,topk] tensor")
    if selected_blocks.device.type != "npu":
        raise ValueError("Triton id2mask requires an NPU tensor")
    if not 0 <= query_length <= kv_length or compress_ratio <= 0 or token_budget <= 0:
        raise ValueError("Invalid QSA lengths, compression ratio or token budget")
    if token_budget % compress_ratio:
        raise ValueError("token_budget must be divisible by compress_ratio")
    if dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("id2mask supports bool, float16, bfloat16 and float32 masks")
    prefix = min(query_length, max(0, token_budget + compress_ratio - 1 - (kv_length - query_length)))
    batch, sparse_rows, topk = selected_blocks.shape
    if (sparse_rows, topk) != (query_length - prefix, token_budget // compress_ratio):
        raise ValueError("selected_blocks shape does not match the LightningIndexer complete-block output")
    if batch * query_length * kv_length == 0:
        return torch.empty((batch, 1, query_length, kv_length), device=selected_blocks.device, dtype=dtype)
    blocks = triton.cdiv(kv_length, compress_ratio)
    # One sink column for invalid IDs; pad rows so native scatter and Triton
    # loads use aligned addresses. Only 0/1 membership is needed, even for
    # duplicate IDs, so int8 is exact and minimizes intermediate storage.
    table_stride = triton.cdiv(blocks + 1, 32) * 32
    table = torch.zeros((batch, max(1, sparse_rows), table_stride), dtype=torch.int8, device=selected_blocks.device)
    if sparse_rows:
        safe_ids = torch.empty(selected_blocks.shape, dtype=torch.int32, device=selected_blocks.device)
        _sanitize_block_ids[(48,)](
            selected_blocks,
            safe_ids,
            sparse_rows,
            topk,
            blocks,
            *selected_blocks.stride(),
            selected_blocks.numel(),
            2048,
            multibuffer=False,
        )
        table.scatter_(-1, safe_ids, 1)

    output = torch.empty((batch, 1, query_length, kv_length), dtype=dtype, device=selected_blocks.device)
    # Store the exact bit patterns for 0 and finfo(dtype).min. This avoids a
    # BF16 constant-select lowering unsupported by the current Ascend compiler;
    # it does not change the output dtype or introduce a numerical approximation.
    integer_dtype, masked_bits = {
        torch.bool: (torch.bool, 0),
        torch.float16: (torch.int16, -1025),  # 0xfbff
        torch.bfloat16: (torch.int16, -129),  # 0xff7f
        torch.float32: (torch.int32, -8388609),  # 0xff7fffff
    }[dtype]
    storage = output if dtype == torch.bool else output.view(integer_dtype)
    if dtype in (torch.bfloat16, torch.float16) and kv_length % 2 == 0 and compress_ratio % 2 == 0:
        _materialize_packed16[(48,)](
            table,
            output.view(torch.int32),
            query_length,
            kv_length,
            prefix,
            sparse_rows,
            compress_ratio,
            blocks,
            table_stride,
            batch * query_length,
            triton.next_power_of_2(blocks),
            4096,
            masked_bits & 65535,
            multibuffer=False,
        )
    else:
        _materialize_mask[(48,)](
            table,
            storage,
            query_length,
            kv_length,
            prefix,
            sparse_rows,
            compress_ratio,
            blocks,
            table_stride,
            batch * query_length,
            triton.next_power_of_2(blocks),
            4096,
            masked_bits,
            dtype == torch.bool,
            multibuffer=False,
        )
    return output

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in exact split-key scoring and local top-k for the frozen K3.1 indexer.

Each key partition retains 2048 candidates on chip; a final exact merge avoids
materializing Q x K scores. Ties may select different, equally scored keys.
"""

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor


@lru_cache(maxsize=4)
def _build_kernel(partition: int = 8192, topk: int = 2048) -> Any:
    import tilelang
    import tilelang.language as T

    threads = 256
    items = partition // threads
    source = r"""
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
template<int THREADS, int ITEMS, int TOPK>
__device__ __forceinline__ void relax_select_candidates(
    const float* scores, float* values_out, int* indices_out,
    int row, int part, int nparts, int key_start) {
    using Sort = cub::BlockRadixSort<float, THREADS, ITEMS, int>;
    __shared__ typename Sort::TempStorage temp;
    float values[ITEMS]; int indices[ITEMS];
    const int tid = threadIdx.x;
    __syncthreads();
    #pragma unroll
    for (int i=0;i<ITEMS;i++) {
        const int offset=tid*ITEMS+i;
        values[i]=scores[offset]; indices[i]=key_start+offset;
    }
    Sort(temp).SortDescendingBlockedToStriped(values, indices);
    #pragma unroll
    for (int i=0;i<ITEMS;i++) {
        const int offset=i*THREADS+tid;
        if (offset<TOPK) {
            const long dst=((long)row*nparts+part)*TOPK+offset;
            values_out[dst]=values[i];
            indices_out[dst]=values[i] == -CUDART_INF_F ? -1 : indices[i];
        }
    }
    __syncthreads();
}
"""

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def factory():
        nq, nk = T.dynamic("nq"), T.dynamic("nk")
        nparts = T.ceildiv(nk, partition)
        bn, heads, dim = 128, 32, 128

        @T.prim_func
        def kernel(
            Q: T.Tensor([nq, heads, dim], T.bfloat16),
            K: T.Tensor([nk, dim], T.bfloat16),
            W: T.Tensor([nq, heads], T.float32),
            Starts: T.Tensor([nq], T.int32),
            Ends: T.Tensor([nq], T.int32),
            Values: T.Tensor([nq, nparts, topk], T.float32),
            Indices: T.Tensor([nq, nparts, topk], T.int32),
        ):
            with T.Kernel(nq, nparts, threads=threads) as (qi, pi):
                T.import_source(source)
                qs = T.alloc_shared([heads, dim], T.bfloat16)
                ks = T.alloc_shared([bn, dim], T.bfloat16)
                scores = T.alloc_shared([partition], T.float32)
                dots = T.alloc_fragment([bn, heads], T.float32)
                logits = T.alloc_fragment([bn], T.float32)
                weights = T.alloc_fragment([heads], T.float32)
                T.fill(scores, -T.infinity(T.float32))
                if pi * partition < Ends[qi] and (pi + 1) * partition > Starts[qi]:
                    T.copy(Q[qi, 0:heads, 0:dim], qs)
                    T.copy(W[qi, 0:heads], weights)
                    for block in T.serial(partition // bn):
                        if pi * partition + block * bn < Ends[qi] and pi * partition + (block + 1) * bn > Starts[qi]:
                            for n, d in T.Parallel(bn, dim):
                                ki = pi * partition + block * bn + n
                                ks[n, d] = T.if_then_else(ki < nk, K[ki, d], 0)
                            T.gemm(ks, qs, dots, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                            for n, h in T.Parallel(bn, heads):
                                dots[n, h] = T.max(dots[n, h], 0) * weights[h]
                            T.reduce_sum(dots, logits, dim=1)
                            for n in T.Parallel(bn):
                                ki = pi * partition + block * bn + n
                                scores[block * bn + n] = T.if_then_else(
                                    ki < nk and ki >= Starts[qi] and ki < Ends[qi], logits[n], -T.infinity(T.float32)
                                )
                T.evaluate(
                    T.call_extern(
                        "handle",
                        f"relax_select_candidates<{threads},{items},{topk}>",
                        T.access_ptr(scores, "r"),
                        T.access_ptr(Values, "w"),
                        T.access_ptr(Indices, "w"),
                        qi,
                        pi,
                        nparts,
                        pi * partition,
                    )
                )

        return kernel

    return factory()


@torch.no_grad()
def fused_indexer(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    topk: int = 2048,
    partition: int = 8192,
) -> tuple[Tensor, Tensor]:
    assert q.ndim == 3 and q.shape[1:] == (32, 128) and k.ndim == 2 and k.shape[1] == 128
    assert q.dtype == k.dtype == torch.bfloat16 and weights.dtype == torch.float32
    assert partition in (8192, 16384) and topk == 2048
    assert all(x.is_contiguous() for x in (q, k, weights, starts, ends))
    n, nk = q.shape[0], k.shape[0]
    nparts = (nk + partition - 1) // partition
    values = torch.empty((n, nparts, topk), device=q.device, dtype=torch.float32)
    indices = torch.empty_like(values, dtype=torch.int32)
    _build_kernel(partition, topk)(q, k, weights, starts, ends, values, indices)
    selected, offsets = torch.topk(values.flatten(1), min(topk, nk), dim=-1, sorted=False)
    output = indices.flatten(1).gather(1, offsets)
    return selected, output


def configure_fused_indexer() -> None:
    """Install the indices-only hook in this explicitly opted-in worker
    process.

    Core retains CP packing/remapping and chunking. The differentiable indexer
    loss path uses a different entrypoint and is not replaced. Unsupported
    shapes or precisions use the original indices-only implementation.
    """
    from megatron.core.transformer.experimental_attention_variant.ops import tilelang_dsa

    from relax.utils.logging_utils import get_logger

    original = tilelang_dsa.lighting_indexer_indices
    if original is None:
        raise RuntimeError("Native TileLang indexer is unavailable")
    if getattr(original, "_relax_fused_indexer", False):
        return
    if torch.cuda.get_device_capability()[0] != 10:
        raise ValueError("Fused K3.1 indexer is currently validated only on Blackwell")
    logged_fast_path = False

    def select(
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        starts: Tensor,
        ends: Tensor,
        topk: int,
        use_relu: bool = True,
    ) -> Tensor:
        nonlocal logged_fast_path
        eligible = (
            use_relu
            and topk == 2048
            and q.is_cuda
            and q.ndim == 3
            and q.size(0) > 0
            and tuple(q.shape[1:]) == (32, 128)
            and k.ndim == 2
            and k.size(1) == 128
            and k.size(0) >= topk
            and q.dtype == k.dtype == torch.bfloat16
            and weights.dtype == torch.float32
            and weights.shape == q.shape[:2]
            and starts.shape == ends.shape == q.shape[:1]
            and starts.dtype == ends.dtype == torch.int32
            and all(t.is_contiguous() and t.device == q.device for t in (q, k, weights, starts, ends))
        )
        if not eligible:
            return original(q, k, weights, starts, ends, topk, use_relu=use_relu)
        result = fused_indexer(q, k, weights, starts, ends, topk)[1]
        if not logged_fast_path:
            get_logger(__name__).info(
                "K3.1 fused indexer first execution: queries=%s, keys=%s, topk=%s.", q.size(0), k.size(0), topk
            )
            logged_fast_path = True
        return result

    select._relax_fused_indexer = True
    tilelang_dsa.lighting_indexer_indices = select
    get_logger(__name__).info("Enabled K3.1 fused indexer scoring and local top-k (8192 keys, exact merge).")

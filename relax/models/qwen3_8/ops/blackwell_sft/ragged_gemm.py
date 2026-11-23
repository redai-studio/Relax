# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use the installed BF16 ragged GEMM for the existing TE expert forward and
dgrad calls."""

from functools import lru_cache, wraps
from itertools import accumulate

import torch


@lru_cache(maxsize=256)
def offsets(device, counts):
    bounds = torch.tensor([0, *accumulate(counts)], dtype=torch.int32, pin_memory=True)
    return bounds.to(device=device, non_blocking=True)


def contiguous_rows(parts, counts):
    head = parts[0]
    base = head._base
    if base is None or head.ndim != 2:
        return None
    start = head.storage_offset()
    offset = start
    width = head.shape[1]
    for part, rows in zip(parts, counts, strict=True):
        if (
            part._base is not base
            or not part.is_contiguous()
            or part.shape != (rows, width)
            or part.storage_offset() != offset
        ):
            return None
        offset += rows * width
    return head.as_strided((sum(counts), width), (width, 1), start)


def adapt(native):
    from quack.gemm_interface import gemm

    from relax.utils.logging_utils import get_logger

    @lru_cache(maxsize=6)
    def report(shape, layout):
        get_logger(__name__).info("Ragged BF16 expert GEMM active for %s weights, %s layout", shape, layout)

    @wraps(native)
    def run(A, B, out, quantization_params, out_dtype, *args, **kw):
        weights, inputs, quantizers = A, B, quantization_params
        layout = kw.get("layout", "TN")
        shape = tuple(weights[0].shape)
        if (
            args
            or not kw.get("single_output")
            or len(out) != 1
            or out_dtype != torch.bfloat16
            or shape not in ((1280, 2560), (2560, 640))
            or layout not in ("TN", "NN")
            or (shape == (1280, 2560) and layout == "TN")
            or any(kw.get(k) for k in ("gelu", "accumulate", "use_bias", "D_dtype"))
            or not kw.get("m_splits")
            or any(q is not None for q in quantizers)
            or len(weights) != len(inputs)
            or not all(
                type(v) in (torch.Tensor, torch.nn.Parameter) and v.dtype == torch.bfloat16 and v.is_cuda
                for v in weights
            )
        ):
            return native(weights, inputs, out, quantizers, out_dtype, *args, **kw)
        sizes = tuple(kw["m_splits"])
        x = contiguous_rows(inputs, sizes)
        if x is None or x.numel() == 0 or not out[0].is_contiguous():
            return native(weights, inputs, out, quantizers, out_dtype, *args, **kw)
        packed_weight = torch.stack(tuple(weights))
        gemm(
            x,
            packed_weight.mT if layout == "TN" else packed_weight,
            out=out[0],
            cu_seqlens_m=offsets(x.device, sizes),
            tuned=False,
        )
        report(shape, layout)
        return out, kw.get("bias"), [None] * len(weights)

    return run


def install():
    import transformer_engine.pytorch.module.grouped_linear as te

    te.general_grouped_gemm = adapt(te.general_grouped_gemm)

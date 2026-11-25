# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Masked writes into contiguous native SGLang weight slices."""

import torch
import triton
import triton.language as tl


_WORD_FORMATS = {
    torch.float16: (tl.uint16, 0x7FFF, 0x7C00),
    torch.bfloat16: (tl.uint16, 0x7FFF, 0x7F80),
    torch.float32: (tl.uint32, 0x7FFFFFFF, 0x7F800000),
}
_has_names = getattr(torch.Tensor, "has_names", None)


@triton.jit(do_not_specialize=["size"])
def _copy_present(
    dst, src, size, WORD: tl.constexpr, MAGNITUDE: tl.constexpr, INFINITY: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    present = offset < size
    bits = tl.load(src.to(tl.pointer_type(WORD)) + offset, mask=present, other=0)
    # Mask the store itself so a placeholder never reads or rewrites the installed word.
    present &= (bits & MAGNITUDE) <= INFINITY
    tl.store(dst.to(tl.pointer_type(WORD)) + offset, bits, mask=present)


def copy_present(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Apply the common native copy in place.

    Return False for general PyTorch copy semantics.
    """
    word_format = _WORD_FORMATS.get(src.dtype)
    if (
        word_format is None
        or not dst.is_cuda
        or src.device != dst.device
        or src.dtype != dst.dtype
        or type(src) not in (torch.Tensor, torch.nn.Parameter)
        or type(dst) not in (torch.Tensor, torch.nn.Parameter)
        or src.layout != torch.strided
        or src.shape != dst.shape
        or not src.is_contiguous()
        or not dst.is_contiguous()
        or src.is_neg()
        or dst.is_neg()
        or (_has_names is not None and (_has_names(src) or _has_names(dst)))
        or torch.autograd.forward_ad._current_level >= 0
        or torch._C._are_functorch_transforms_active()
        or torch._is_functional_tensor(src)
        or torch._is_functional_tensor(dst)
        or (torch.is_grad_enabled() and (src.requires_grad or dst.requires_grad))
        or (dst.is_inference() and not torch.is_inference_mode_enabled())
    ):
        return False
    size = src.numel()
    dst_ptr, src_ptr = dst.data_ptr(), src.data_ptr()
    if not size or not dst_ptr or not src_ptr or abs(dst_ptr - src_ptr) < src.nbytes:
        return False
    block = 256
    with torch.cuda.device(dst.device):
        _copy_present[(triton.cdiv(size, block),)](dst, src, size, *word_format, BLOCK=block)
    torch.autograd.graph.increment_version(dst)
    return True

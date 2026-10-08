# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""H800 numerical simulation of indexer MXFP4 Q/K and BF16 final scores.

QDQ follows Hadamard and precedes CP gather, so top-k and auxiliary backward
share the same activation values. The existing SM90 kernel still consumes BF16.
Rounding final scores does not simulate BF16 intermediate accumulation and
provides no native FP4/BF16-score storage or throughput improvement.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache, wraps
from typing import Any


_PROCESS_MODE: tuple[bool, bool] | None = None
# None: unselected CSA; False: Q/K QDQ only; True: QDQ and score rounding.
_ACTIVE_INDEXER: ContextVar[bool | None] = ContextVar("relax_dsv4_fp4_indexer", default=None)


def select_process_mode(*, enabled: bool, bf16_scores: bool) -> None:
    """CSA uses module-level kernel bindings; reject mixed-model indexer
    configurations."""
    global _PROCESS_MODE
    mode = (enabled, enabled and bf16_scores)
    if _PROCESS_MODE is not None and mode != _PROCESS_MODE:
        raise ValueError("Run different DeepSeek-V4 MXFP4 QAT indexer configurations in separate worker processes")
    _PROCESS_MODE = mode


@lru_cache(maxsize=1)
def _ste() -> Any:
    import torch

    from relax.models.deepseek_v4.quantization import mxfp4_qdq

    class ActivationQDQ(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any) -> Any:
            if value.dtype != torch.bfloat16 or value.shape[-1] != 128:
                raise ValueError("DeepSeek-V4 MXFP4 QAT indexer expects BF16 activations with head_dim=128")
            return mxfp4_qdq(value.contiguous().view(-1, 128)).view(value.shape)

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> Any:
            return gradient

    return ActivationQDQ


def activation_qdq(value: Any) -> Any:
    return _ste().apply(value)


@lru_cache(maxsize=1)
def _round_score_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(scores, size, rows, cols, batch_stride, row_stride, col_stride, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % cols
        row = (i // cols) % rows
        batch = i // (rows * cols)
        offset = batch * batch_stride + row * row_stride + col * col_stride
        value = tl.load(scores + offset, mask=i < size, other=0)
        rounded = value.to(tl.bfloat16).to(tl.float32)
        tl.store(scores + offset, rounded, mask=i < size)

    return kernel


def round_scores_(scores: Any) -> Any:
    """Round in-place, avoiding multiple sequence-squared temporary buffers."""
    import torch

    if scores.ndim not in (2, 3) or scores.dtype != torch.float32 or not scores.is_cuda or scores.requires_grad:
        raise ValueError("Score rounding expects detached CUDA FP32 2D/3D indexer scores")
    rows, cols = scores.shape[-2:]
    size = scores.numel()
    if size:
        _round_score_kernel()[((size + 255) // 256,)](
            scores,
            size,
            rows,
            cols,
            scores.stride(0) if scores.ndim == 3 else 0,
            scores.stride(-2),
            scores.stride(-1),
            BLOCK=256,
        )
    return scores


class _ScoreNamespace:
    def __init__(self, original: Any) -> None:
        self.original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def indexer_forward_wrapper(self, q: Any, k: Any, w: Any, **kwargs: Any) -> Any:
        if _ACTIVE_INDEXER.get() is not True:
            return self.original.indexer_forward_wrapper(q, k, w, **kwargs)

        import torch

        if any(kwargs.get(key) is not None for key in ("out", "lse_out")) or kwargs.get("return_lse", False):
            raise ValueError("BF16 index scores with output buffers/LSE need a separate validated implementation")
        if kwargs.get("precision", "bf16") != "bf16":
            raise ValueError("DeepSeek-V4 MXFP4 QAT score simulation requires the SM90 BF16 indexer kernel")
        result = self.original.indexer_forward_wrapper(q, k, w, **kwargs)
        if result["scores"].dtype != torch.float32:
            raise TypeError("Unexpected indexer score dtype")
        # Keep the FP32 top-k ABI, including -inf masks, but round final I to
        # BF16 values. Preserve cuDNN's TupleDict object rather than copying it.
        result["scores"] = round_scores_(result["scores"])
        return result


def _install_forward_scope(module_type: type) -> None:
    """Scope shared CSA kernels to explicitly selected module instances.

    The CP path rotates queries directly inside CSA.forward, bypassing the
    indexer submodule. Unselected instances must clear an enclosing scope too.
    Full-layer recomputation re-enters this wrapper just like the first
    forward.
    """
    original = module_type.forward
    if getattr(original, "_relax_dsv4_fp4_scope", False):
        return

    @wraps(original)
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        token = _ACTIVE_INDEXER.set(getattr(self, "_relax_dsv4_fp4_indexer_scores", None))
        try:
            return original(self, *args, **kwargs)
        finally:
            _ACTIVE_INDEXER.reset(token)

    forward._relax_dsv4_fp4_scope = True
    module_type.forward = forward


def install_indexer_qat(model: Any, *, bf16_scores: bool = True) -> tuple[str, ...]:
    from megatron.core.transformer.experimental_attention_variant import csa, dsa_kernels

    select_process_mode(enabled=True, bf16_scores=bf16_scores)
    selected = []
    for name, module in model.named_modules():
        if module.__class__.__module__ != csa.__name__ or getattr(module, "indexer", None) is None:
            continue
        if not hasattr(module, "apply_dsa_kernel_fusion") and not hasattr(module, "use_fused_kernels"):
            continue
        if getattr(module.config, "relax_use_indexer_replay", False):
            raise ValueError("DeepSeek-V4 MXFP4 QAT indexer simulation cannot be combined with replayed top-k")
        if bf16_scores and (
            not getattr(module, "use_fused_kernels", getattr(module, "apply_dsa_kernel_fusion", False))
            or (module.config.dsa_indexer_loss_coeff or 0.0) != 0.0
            or getattr(module.config, "dsa_kernel_backend", "cudnn") != "cudnn"
        ):
            raise ValueError(
                "DeepSeek-V4 MXFP4 QAT BF16 scores require fused cuDNN DSA and indexer auxiliary loss coefficient=0"
            )
        if not module.indexer.compressor.rotate or (module.compressor is not None and module.compressor.rotate):
            raise ValueError("Unexpected CSA Hadamard placement: cannot restrict QDQ to indexer Q/K")
        selected.append((name, module))
    if not selected:
        return ()
    original = csa.rotate_activation
    if not getattr(original, "_relax_dsv4_fp4", False):

        @wraps(original)
        def rotated(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            return activation_qdq(value) if _ACTIVE_INDEXER.get() is not None else value

        rotated._relax_dsv4_fp4 = True
        csa.rotate_activation = rotated
    score_kernels = dsa_kernels
    namespace_attr = "_DSA"
    if not hasattr(score_kernels, "_ensure_dsa_namespace"):
        from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

        score_kernels = dsa_cudnn_kernels
        namespace_attr = "_cudnn_dsa"
    score_kernels._ensure_dsa_namespace()
    namespace = getattr(score_kernels, namespace_attr)
    if bf16_scores and not isinstance(namespace, _ScoreNamespace):
        setattr(score_kernels, namespace_attr, _ScoreNamespace(namespace))
    if not bf16_scores and isinstance(namespace, _ScoreNamespace):
        raise ValueError("Cannot mix different DeepSeek-V4 MXFP4 indexer score modes within one process")
    for _, module in selected:
        _install_forward_scope(type(module))
        module._relax_dsv4_fp4_indexer_scores = bf16_scores
    return tuple(name for name, _ in selected)

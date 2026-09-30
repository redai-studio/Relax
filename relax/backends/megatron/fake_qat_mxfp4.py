# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MXFP4 fake-QAT STE for Kimi K3 RL training.

The rollout engine serves Kimi K3 routed experts as native MXFP4: every weight
push requantizes the BF16 master with Bridge's ``quantize_mxfp4_e2m1_like_scale``
(see ``weight_update/bridge_converter.py:_Mxfp4OnlineQuantizer``). Training,
however, runs the forward on raw BF16 masters, so train/rollout logprobs differ
by the full MXFP4 quantization error — including the *routing* distribution,
which token-level TIS cannot correct.

This module closes the gap with a straight-through estimator (STE): the forward
sees ``dequantize(quantize(w))`` computed with arithmetic bitwise-identical to
the push path, while gradients flow to the BF16 master unchanged. Two facts
make bitwise identity cheap to keep:

* per-row quantization (block ``(1, 32)`` along the input dim) is invariant to
  the Megatron fc1 gate/up concatenation along dim 0, to EP whole-expert
  sharding, and to ETP column shards whose input dim stays a multiple of 32;
* every dequantized value is ``{0, .5, ..., 6} x 2^k`` — at most 3 mantissa
  bits — so the BF16 cast is exact and "training forward value" ==
  "engine dequantized value" bit for bit.

Only routed experts flow through ``TEGroupedLinear`` in K3 (dense MLP, shared
experts, latent projections and attention use plain TE linears), matching the
release checkpoint's quantization ignore list exactly.

Enable with env ``OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG=1`` (via
``--train-env-vars``); the K3 branch of ``model_provider`` installs the hook.
"""

from __future__ import annotations

import os

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

MXFP4_FAKE_QAT_ENV_FLAG = "OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG"

_FP4_E2M1_MAX = 6.0
_MXFP4_BLOCK_SIZE = 32
_E2M1_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def mxfp4_fake_quantize_tensor(weight: torch.Tensor, *, block_size: int = _MXFP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize-dequantize a 2-D weight onto the MXFP4 E2M1 grid.

    Mirrors Bridge's ``quantize_mxfp4_e2m1_like_scale`` op for op (fp32 math,
    ceil-log2 E8M0 group scale, midpoint bucketize onto the E2M1 grid), fused
    with the dequantization so no packed/scale tensors are materialized. The
    scale's uint8 round-trip is skipped: it is the identity for every reachable
    exponent (scale is exactly ``2^k`` with ``k`` in [-127, 127] after
    clamping).
    """
    if weight.ndim != 2:
        raise ValueError(f"MXFP4 fake-QAT expects a 2-D weight, got {weight.ndim}D shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % block_size != 0:
        raise ValueError(f"MXFP4 fake-QAT expects shape[1] % {block_size} == 0, got {tuple(weight.shape)}")

    groups = weight.to(torch.float32).reshape(rows, cols // block_size, block_size)
    amax = groups.abs().amax(dim=-1)
    unrounded_scale = torch.where(amax > 0, amax / _FP4_E2M1_MAX, torch.ones_like(amax))
    scale = torch.exp2(torch.ceil(torch.log2(unrounded_scale)).clamp(min=-127, max=127))
    normalized = groups / scale.unsqueeze(-1)
    boundaries = torch.tensor(_E2M1_BOUNDARIES, dtype=torch.float32, device=weight.device)
    codes = torch.bucketize(normalized.abs(), boundaries).to(torch.int64)
    codes = codes | ((normalized < 0).to(torch.int64) * 8)
    table = torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=weight.device)
    dequant = table[codes] * scale.unsqueeze(-1)
    return dequant.reshape(rows, cols).to(weight.dtype)


class _Mxfp4FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return mxfp4_fake_quantize_tensor(weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def mxfp4_fake_quant_ste(weight: torch.Tensor) -> torch.Tensor:
    """Apply MXFP4 fake quantization with straight-through gradients."""
    out = _Mxfp4FakeQuantSTE.apply(weight)
    # TE writes wgrad into the parameter's main_grad buffer; keep the reference
    # on the quantized stand-in so the fused path keeps working (same contract
    # as the K2.6 INT4 fake-QAT patch).
    main_grad = getattr(weight, "main_grad", None)
    if main_grad is not None:
        out.main_grad = main_grad
    return out


def install_mxfp4_fake_qat(grouped_linear_cls) -> bool:
    """Wrap ``grouped_linear_cls._get_weight_tensors`` with the MXFP4 STE.

    Idempotent; returns whether this call installed the wrap. Raises if the
    hook point is gone (Megatron/TE upgrade).
    """
    if getattr(grouped_linear_cls, "_mxfp4_fake_qat_installed", False):
        return False
    orig = getattr(grouped_linear_cls, "_get_weight_tensors", None)
    if orig is None:
        raise RuntimeError(
            f"{grouped_linear_cls.__name__} has no _get_weight_tensors to hook; "
            "the MXFP4 fake-QAT patch needs to be rebased onto this Megatron/TE version"
        )

    def _get_weight_tensors_with_fake_qat(self):
        return [mxfp4_fake_quant_ste(w) for w in orig(self)]

    grouped_linear_cls._mxfp4_fake_qat_orig_get_weight_tensors = orig
    grouped_linear_cls._get_weight_tensors = _get_weight_tensors_with_fake_qat
    grouped_linear_cls._mxfp4_fake_qat_installed = True
    logger.info(f"MXFP4 fake-QAT STE installed on {grouped_linear_cls.__name__}._get_weight_tensors")
    return True


def _import_te_grouped_linear():
    from megatron.core.extensions.transformer_engine import TEGroupedLinear

    return TEGroupedLinear


def maybe_install_mxfp4_fake_qat() -> bool:
    """Install the STE on Megatron's TEGroupedLinear when the env flag is
    set."""
    if os.getenv(MXFP4_FAKE_QAT_ENV_FLAG, "0") != "1":
        return False
    return install_mxfp4_fake_qat(_import_te_grouped_linear())

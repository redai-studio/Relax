# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright (c) 2026, Han Guo, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Quack epilogues for native BF16 rounding and per-token gradients.
"""CuTe GEMM epilogues for BF16 linear CE, using the image's existing quack
runtime."""

from cutlass import BFloat16, Float32, select_
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import F2, pexp
from quack.epilogue.ops import ColVecLoad, ColVecSelect, OnlineLSEReduce, RowVecLoad, Scalar


def _rounded(value):
    if isinstance(value, F2):
        return F2(value.lo.to(BFloat16).to(Float32), value.hi.to(BFloat16).to(Float32))
    return value.to(BFloat16).to(Float32)


_target = ColVecLoad("target")


@gemm_epilogue(
    ops={"inverse_temperature": Scalar("inverse_temperature")},
    outs={"lse": OnlineLSEReduce("lse"), "target_logit": ColVecSelect("target_logit", idx_op=_target)},
    extra_ops=(_target,),
)
def linear_ce_forward(acc, inverse_temperature):
    # Match the native BF16 head output BEFORE the FP32 temperature / CE.
    value = _rounded(acc) * inverse_temperature
    return {"D": value, "lse": value, "target_logit": value}


def _lane(value, index):
    return value[index] if isinstance(value, tuple) else value


@gemm_epilogue(
    ops={
        "lse": ColVecLoad("lse"),
        "grad": ColVecLoad("grad"),
        "target": ColVecLoad("target"),
        "vocab": RowVecLoad("vocab"),
        "inverse_temperature": Scalar("inverse_temperature"),
    }
)
def linear_ce_backward(acc, lse, grad, target, vocab, inverse_temperature):
    probability = pexp(_rounded(acc) * inverse_temperature - lse)
    if isinstance(probability, F2):
        delta = F2(
            probability.lo - select_(_lane(target, 0) == _lane(vocab, 0), Float32(1.0), Float32(0.0)),
            probability.hi - select_(_lane(target, 1) == _lane(vocab, 1), Float32(1.0), Float32(0.0)),
        )
    else:
        delta = probability - select_(target == vocab, Float32(1.0), Float32(0.0))
    # Subtract the target and scale in FP32, then write one BF16 dlogits tile.
    return {"D": delta * grad * inverse_temperature}

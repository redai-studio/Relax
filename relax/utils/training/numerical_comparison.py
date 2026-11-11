# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit tensor comparisons for offline training-correctness reports."""

import math
from typing import Any

import torch


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    zero_floor: float = 1e-12,
) -> dict[str, Any]:
    """Apply finite/shape/elementwise tolerance checks and report L2 error.

    Relative L2 is descriptive; pass/fail uses |y-x| <= atol + rtol*|x|. Call
    only in an explicit validation/export path, never a training hot loop.
    """
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and non-negative")
    if not math.isfinite(zero_floor) or zero_floor <= 0:
        raise ValueError("zero_floor must be finite and positive")
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref = reference.detach().to(device="cpu", dtype=torch.float64)
    value = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(value).all()):
        return {"passed": False, "reason": "non-finite value"}
    diff = value - ref
    norm = float(torch.linalg.vector_norm(ref))
    error = float(torch.linalg.vector_norm(diff))
    maximum = float(diff.abs().max()) if diff.numel() else 0.0
    passed = bool((diff.abs() <= atol + rtol * ref.abs()).all())
    return {
        "passed": passed,
        "max_abs": maximum,
        "relative_l2": error / max(norm, zero_floor),
        "reference_l2": norm,
        "difference_l2": error,
        "elements": ref.numel(),
    }


def compare_named_tensors(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], *, atol: float, rtol: float
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        return {
            "passed": False,
            "reason": "tensor keys differ",
            "missing": sorted(reference.keys() - candidate.keys()),
            "extra": sorted(candidate.keys() - reference.keys()),
        }
    results = {key: compare_tensors(value, candidate[key], atol=atol, rtol=rtol) for key, value in reference.items()}
    return {"passed": all(result["passed"] for result in results.values()), "tensors": results}

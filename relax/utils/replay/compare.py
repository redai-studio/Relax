# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical comparison helpers for replay adapters."""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.replay.report import FieldDivergence, StageResult, StageStatus
from relax.utils.replay.schema import Manifest, StageId


def tolerance_for(
    manifest: Manifest, stage: StageId, *, default_atol: float = 1e-6, default_rtol: float = 1e-5
) -> tuple[float, float]:
    """Return (atol, rtol) for stage from the manifest, with defaults."""
    entry = manifest.comparison_policy.tolerances.get(stage.value, {})
    return entry.get("atol", default_atol), entry.get("rtol", default_rtol)


def compare_scalar(expected: float, actual: float, *, atol: float, rtol: float) -> bool:
    """True when |actual - expected| <= atol + rtol * |expected|."""
    return abs(actual - expected) <= atol + rtol * abs(expected)


def scalar_error(expected: float, actual: float) -> float:
    return abs(actual - expected)


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    field: str,
    atol: float,
    rtol: float,
    sample_ids: list[str] | None = None,
    response_lengths: list[int] | None = None,
) -> tuple[list[FieldDivergence], float, int, int]:
    """Elementwise-compare two 1-D tensors and localize divergences.

    Returns (divergences, max_abs_error, mismatch_count, non_finite_count).
    Flat token index is mapped back to (sample_id, token_offset) when
    sample_ids/response_lengths are provided.
    """
    expected = expected.float().reshape(-1)
    actual = actual.float().reshape(-1)
    if expected.numel() != actual.numel():
        raise ValueError(f"shape mismatch for {field!r}: {tuple(expected.shape)} vs {tuple(actual.shape)}")

    diff = (actual - expected).abs()
    threshold = atol + rtol * expected.abs()
    mismatched = (diff > threshold).nonzero(as_tuple=False).flatten().tolist()

    non_finite = int((~torch.isfinite(actual)).sum().item())
    max_abs_error = float(diff.max().item()) if diff.numel() else 0.0

    divergences: list[FieldDivergence] = []
    for token_index in mismatched[:10]:  # cap the report at 10 concrete examples
        sample_id, token_offset = _locate(token_index, sample_ids, response_lengths)
        divergences.append(
            FieldDivergence(
                field=field,
                sample_id=sample_id,
                token_offset=token_offset,
                expected=float(expected[token_index].item()),
                actual=float(actual[token_index].item()),
                abs_error=float(diff[token_index].item()),
            )
        )
    return divergences, max_abs_error, len(mismatched), non_finite


def report_scalar_list(
    result: StageResult,
    *,
    expected: list[Any] | None,
    actual: list[float],
    sample_ids: list[str],
    field: str,
    atol: float,
    rtol: float,
    missing: str,
    mismatch: str,
) -> StageResult:
    """Compare per-sample scalars.

    Missing expected values fail: a recompute stage without recorded outputs is
    an incomplete bundle, not a silent pass.
    """
    if expected is None:
        result.status = StageStatus.FAIL
        result.message = missing
        return result
    for index, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        if not compare_scalar(float(exp), float(act), atol=atol, rtol=rtol):
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(
                    field=field,
                    sample_id=sample_ids[index],
                    expected=float(exp),
                    actual=float(act),
                    abs_error=scalar_error(float(exp), float(act)),
                )
            )
    if result.status == StageStatus.FAIL:
        result.message = mismatch.format(n=len(result.divergences))
    return result


def report_scalar_fields(
    result: StageResult,
    *,
    expected: dict[str, Any] | None,
    actual: dict[str, float],
    atol: float,
    rtol: float,
    missing: str,
    mismatch: str,
) -> StageResult:
    """Compare a map of named scalars.

    A missing expected map or key fails the stage.
    """
    if expected is None:
        result.status = StageStatus.FAIL
        result.message = missing
        return result
    for field, actual_value in actual.items():
        expected_value = expected.get(field)
        if expected_value is None:
            result.status = StageStatus.FAIL
            result.divergences.append(FieldDivergence(field=field, actual=float(actual_value)))
            continue
        if not compare_scalar(float(expected_value), float(actual_value), atol=atol, rtol=rtol):
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(
                    field=field,
                    expected=float(expected_value),
                    actual=float(actual_value),
                    abs_error=scalar_error(float(expected_value), float(actual_value)),
                )
            )
    if result.status == StageStatus.FAIL:
        result.message = mismatch.format(fields=[item.field for item in result.divergences])
    return result


def report_tensor(
    result: StageResult,
    *,
    expected: torch.Tensor | None,
    actual: torch.Tensor,
    field: str,
    atol: float,
    rtol: float,
    sample_ids: list[str],
    response_lengths: list[int],
    missing: str,
    mismatch: str,
) -> StageResult:
    """Compare a flat per-token tensor.

    Missing expected payload fails the stage.
    """
    if expected is None:
        result.status = StageStatus.FAIL
        result.message = missing
        return result
    divergences, max_err, mismatches, non_finite = compare_tensors(
        expected,
        actual,
        field=field,
        atol=atol,
        rtol=rtol,
        sample_ids=sample_ids,
        response_lengths=response_lengths,
    )
    result.divergences = divergences
    result.max_abs_error = max_err
    result.mismatch_count = mismatches
    result.non_finite_count = non_finite
    if mismatches or non_finite:
        result.status = StageStatus.FAIL
        result.message = mismatch.format(n=mismatches)
    return result


def _locate(
    token_index: int,
    sample_ids: list[str] | None,
    response_lengths: list[int] | None,
) -> tuple[str | None, int | None]:
    if not sample_ids or not response_lengths:
        return None, None
    offset = token_index
    for sample_id, length in zip(sample_ids, response_lengths, strict=False):
        if offset < length:
            return sample_id, offset
        offset -= length
    return None, None

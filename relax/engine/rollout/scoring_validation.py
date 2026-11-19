# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate scored Sample payloads before crossing the publication boundary."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from relax.utils.types import Sample


def validate_scoring_fields(
    sample: Sample,
    *,
    batch_id: str,
    sample_id: Any = None,
    required_fields: Iterable[str] = (),
    vocab_size: int | None = None,
) -> None:
    """Reject malformed targets without coercing token indices to integers.

    Non-finite log probabilities are allowed only on masked response rows or
    union padding. Union padding uses token ID -1 outside ``opd_topk_ksz``.
    Empty responses need no scoring fields.
    """
    identity = sample.index if sample_id is None else sample_id

    def fail(field: str, reason: str) -> None:
        raise ValueError(f"batch={batch_id} sample={identity} field={field}: {reason}")

    def array(value: Any, field: str) -> np.ndarray:
        try:
            return np.asarray(value)
        except (TypeError, ValueError) as error:
            fail(field, f"invalid array: {error}")
            raise AssertionError("unreachable") from error

    length = sample.response_length
    if length < 0:
        fail("response_length", "must be non-negative")
    if length == 0:
        return
    mask = np.ones(length, dtype=bool)
    if sample.loss_mask is not None:
        raw_mask = array(sample.loss_mask, "loss_mask")
        if raw_mask.shape != (length,) or not np.isin(raw_mask, [0, 1]).all():
            fail("loss_mask", "expected one binary value per response token")
        mask = raw_mask.astype(bool)
    for field in required_fields:
        if getattr(sample, field, None) is None:
            fail(field, "required scoring field is absent")
    if sample.teacher_log_probs is not None:
        values = array(sample.teacher_log_probs, "teacher_log_probs")
        if values.shape != (length,):
            fail("teacher_log_probs", f"expected shape ({length},)")
        if values.dtype.kind not in "fi" or not np.isfinite(values[mask]).all():
            fail("teacher_log_probs", "effective response positions must be finite")

    ids_value = sample.opd_topk_token_ids
    lp_fields = ("opd_topk_teacher_log_probs", "opd_topk_student_log_probs")
    if ids_value is None:
        for field in (*lp_fields, "opd_topk_ksz"):
            if getattr(sample, field) is not None:
                fail(field, "top-k token IDs are absent")
        return
    ids = array(ids_value, "opd_topk_token_ids")
    if ids.ndim != 2 or ids.shape[0] != length or ids.shape[1] == 0:
        fail("opd_topk_token_ids", "expected nonempty [response_length, K] matrix")
    if ids.dtype.kind not in "iu":
        fail("opd_topk_token_ids", "token IDs must have integer dtype")
    valid = np.ones(ids.shape, dtype=bool)
    if sample.opd_topk_ksz is not None:
        sizes = array(sample.opd_topk_ksz, "opd_topk_ksz")
        if sizes.shape != (length,) or sizes.dtype.kind not in "iu":
            fail("opd_topk_ksz", "expected integer [response_length] vector")
        if ((sizes < 0) | (sizes > ids.shape[1])).any() or (sizes[mask] == 0).any():
            fail("opd_topk_ksz", "invalid union row lengths")
        valid = np.arange(ids.shape[1])[None, :] < sizes[:, None]
        if (ids[~valid] != -1).any():
            fail("opd_topk_token_ids", "union padding must use -1")
    effective = valid & mask[:, None]
    if (ids[effective] < 0).any() or (vocab_size is not None and (ids[effective] >= vocab_size).any()):
        fail("opd_topk_token_ids", "effective token ID is outside vocabulary")
    if sample.opd_topk_teacher_log_probs is None:
        fail("opd_topk_teacher_log_probs", "top-k IDs require paired teacher probabilities")
    for field in lp_fields:
        value = getattr(sample, field)
        if value is None:
            continue
        values = array(value, field)
        if values.shape != ids.shape:
            fail(field, "log probabilities must match token ID matrix shape")
        if values.dtype.kind not in "fi" or not np.isfinite(values[effective]).all():
            fail(field, "effective top-k positions must be finite")


def normalize_masked_scoring_fields(sample: Sample) -> None:
    """Replace ignored probability sentinels after validating effective rows.

    Keep sample identity, token IDs and union row lengths unchanged. Downstream
    loss reducers multiply by the response mask, so ignored values must still
    be finite before arithmetic (NaN multiplied by zero remains NaN).
    """
    length = sample.response_length
    if not length:
        return
    rows = np.ones(length, dtype=bool) if sample.loss_mask is None else np.asarray(sample.loss_mask, dtype=bool)
    if sample.teacher_log_probs is not None:
        values = np.asarray(sample.teacher_log_probs).copy()
        values[~rows] = 0.0
        sample.teacher_log_probs = values.tolist()
    if sample.opd_topk_token_ids is None:
        return
    shape = np.asarray(sample.opd_topk_token_ids).shape
    effective = np.broadcast_to(rows[:, None], shape).copy()
    if sample.opd_topk_ksz is not None:
        effective &= np.arange(shape[1])[None, :] < np.asarray(sample.opd_topk_ksz)[:, None]
    for field in ("opd_topk_teacher_log_probs", "opd_topk_student_log_probs"):
        value = getattr(sample, field)
        if value is not None:
            values = np.asarray(value).copy()
            values[~effective] = 0.0
            setattr(sample, field, values)

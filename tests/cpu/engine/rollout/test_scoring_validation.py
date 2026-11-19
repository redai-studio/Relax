# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import numpy as np
import pytest

from relax.engine.rollout.scoring_validation import validate_scoring_fields
from relax.utils.types import Sample


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scoring_rejects_nonfinite_effective_logprobs(value):
    sample = Sample(index=7, response_length=2, teacher_log_probs=[-1, value])
    with pytest.raises(ValueError, match="batch=b sample=7 field=teacher_log_probs"):
        validate_scoring_fields(sample, batch_id="b")


@pytest.mark.parametrize(
    "ids,logps,field",
    [
        (np.array([[1.5]]), np.array([[-1.0]]), "opd_topk_token_ids"),
        (np.array([[9]]), np.array([[-1.0]]), "opd_topk_token_ids"),
        (np.array([[-1]]), np.array([[-1.0]]), "opd_topk_token_ids"),
        (np.array([1]), np.array([-1.0]), "opd_topk_token_ids"),
        (np.array([[1]]), np.array([[-1.0, -2.0]]), "opd_topk_teacher_log_probs"),
        (np.array([[1]]), np.array([[float("inf")]]), "opd_topk_teacher_log_probs"),
    ],
)
def test_scoring_rejects_invalid_topk(ids, logps, field):
    sample = Sample(index=7, response_length=1, opd_topk_token_ids=ids, opd_topk_teacher_log_probs=logps)
    with pytest.raises(ValueError, match=f"batch=b sample=7 field={field}"):
        validate_scoring_fields(sample, batch_id="b", vocab_size=9)


def test_scoring_accepts_masked_values_and_union_padding():
    sample = Sample(
        response_length=2,
        loss_mask=[1, 0],
        teacher_log_probs=[-1, float("nan")],
        opd_topk_token_ids=np.array([[2, -1], [-1, -1]], dtype=np.int32),
        opd_topk_teacher_log_probs=np.array([[-1, float("inf")], [float("nan"), float("nan")]]),
        opd_topk_ksz=np.array([1, 0], dtype=np.int32),
    )
    validate_scoring_fields(sample, batch_id="b", vocab_size=3)
    validate_scoring_fields(Sample(), batch_id="empty", required_fields=["teacher_log_probs"])


def test_scoring_rejects_missing_required_and_wrong_dimension():
    with pytest.raises(ValueError, match="field=teacher_log_probs"):
        validate_scoring_fields(Sample(response_length=1), batch_id="b", required_fields=["teacher_log_probs"])
    with pytest.raises(ValueError, match="field=teacher_log_probs"):
        validate_scoring_fields(Sample(response_length=1, teacher_log_probs=[[-1]]), batch_id="b")

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.training.numerical_comparison import compare_named_tensors, compare_tensors


def test_comparison_uses_elementwise_tolerance_including_near_zero():
    reference = torch.tensor([0.0, 100.0], dtype=torch.float64)
    assert compare_tensors(reference, torch.tensor([1e-7, 100.01]), atol=1e-6, rtol=1e-3)["passed"]
    report = compare_tensors(reference, torch.tensor([1e-4, 100.0]), atol=1e-6, rtol=1e-3)
    assert not report["passed"]
    assert report["relative_l2"] < 1e-5


@pytest.mark.parametrize("candidate", [torch.zeros(3), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_comparison_rejects_shape_changes_and_nonfinite(candidate):
    assert not compare_tensors(torch.zeros(1), candidate, atol=1e-6, rtol=1e-3)["passed"]


def test_comparison_requires_full_matching_parameter_keys():
    result = compare_named_tensors({"expert0": torch.zeros(1)}, {"expert1": torch.zeros(1)}, atol=0, rtol=0)
    assert not result["passed"] and result["missing"] == ["expert0"] and result["extra"] == ["expert1"]
    assert compare_tensors(torch.empty(0), torch.empty(0), atol=0, rtol=0)["passed"]


@pytest.mark.parametrize("kwargs", [{"atol": -1}, {"rtol": float("nan")}, {"zero_floor": 0}])
def test_comparison_rejects_invalid_tolerance(kwargs):
    options = dict(atol=0, rtol=0)
    options.update(kwargs)
    with pytest.raises(ValueError):
        compare_tensors(torch.zeros(1), torch.zeros(1), **options)

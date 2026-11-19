# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MOPD route budgets retain validation independently of task placement."""

import json

import pytest


@pytest.mark.parametrize("teacher_tp", [4, None])
def test_explicit_teacher_budgets_preserve_source_model_identity(teacher_tp):
    from relax.utils.opd.opd_utils import _parse_teacher_routes

    routes = {
        source: {"checkpoint_path": f"/ckpt/{source}", "num_gpus": size}
        for source, size in (("audit", 8), ("evidence", 4), ("evaluation", 4))
    }
    assert _parse_teacher_routes(json.dumps(routes), 16, teacher_tp) == routes


def test_legacy_teacher_paths_split_gpu_budget_equally():
    from relax.utils.opd.opd_utils import _parse_teacher_routes

    routes = _parse_teacher_routes(json.dumps({"math": "/math", "code": "/code"}), 8)
    assert routes == {
        "math": {"checkpoint_path": "/math", "num_gpus": 4},
        "code": {"checkpoint_path": "/code", "num_gpus": 4},
    }
    with pytest.raises(ValueError, match="evenly divisible"):
        _parse_teacher_routes(json.dumps({"math": "/math", "code": "/code"}), 7)


@pytest.mark.parametrize(
    "routes,total,tp,message",
    [
        ({}, 16, 4, "non-empty JSON object"),
        ([], 16, 4, "non-empty JSON object"),
        (
            {"a": "/a", "b": {"checkpoint_path": "/b", "num_gpus": 8}},
            16,
            4,
            "must be a checkpoint path string or an object",
        ),
        ({"a": {"checkpoint_path": "/a"}}, 16, 4, "must be a checkpoint path string or an object"),
        ({"a": {"checkpoint_path": "/a", "num_gpus": 8.0}}, 8, 4, "positive integer"),
        (
            {"a": {"checkpoint_path": "/a", "num_gpus": 0}, "b": {"checkpoint_path": "/b", "num_gpus": 16}},
            16,
            4,
            "positive integer",
        ),
        (
            {"a": {"checkpoint_path": "/a", "num_gpus": -4}, "b": {"checkpoint_path": "/b", "num_gpus": 20}},
            16,
            4,
            "positive integer",
        ),
        ({"a": {"checkpoint_path": "/a", "num_gpus": 6}}, 6, 4, "must be divisible"),
        ({"a": {"checkpoint_path": "/a", "num_gpus": 8}}, 16, 4, "must sum"),
    ],
)
def test_multi_teacher_rejects_invalid_route_budgets(routes: object, total: int, tp: int, message: str) -> None:
    from relax.utils.opd import opd_utils

    with pytest.raises(ValueError, match=message):
        opd_utils._parse_teacher_routes(json.dumps(routes), total, tp)

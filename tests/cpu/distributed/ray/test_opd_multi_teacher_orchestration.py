# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``_start_managed_multi_teacher`` was rewritten to delegate the GPU-budget
carve-up to the domain-agnostic ``start_multi_instance_managers`` helper (also
used by GenRM's multi-instance path).

This must not change MOPD's observable
contract: default equal-split validation, one TeacherManager per data_source at a
non-overlapping bundle_offset, and a ``(pg, list[manager])`` return shape.
"""

import json
import sys
from argparse import Namespace
from types import ModuleType

import pytest


def _install_fake_teacher_manager(monkeypatch, captured):
    teacher_manager_module = ModuleType("relax.distributed.ray.teacher_manager")

    class _RemoteMethod:
        def __init__(self, name, owner):
            self.name = name
            self.owner = owner

        def remote(self, **kwargs):
            captured["calls"].append((self.owner, self.name))
            return (self.owner, self.name)

    class _TeacherManagerHandle:
        def __init__(self, key):
            self.key = key
            self.get_urls = _RemoteMethod("get_urls", key)
            self.offload = _RemoteMethod("offload", key)

    class _TeacherManagerActor:
        @classmethod
        def options(cls, **options):
            return cls

        @classmethod
        def remote(cls, args, num_replicas, gpus_per_replica, *, pg, shared_pg, bundle_offset):
            key = args.teacher_hf_checkpoint
            captured["ctor_calls"][key] = {
                "num_replicas": num_replicas,
                "gpus_per_replica": gpus_per_replica,
                "pg": pg,
                "shared_pg": shared_pg,
                "bundle_offset": bundle_offset,
            }
            return _TeacherManagerHandle(key)

    teacher_manager_module.TeacherManager = _TeacherManagerActor
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.teacher_manager", teacher_manager_module)


def _base_args(**overrides):
    defaults = dict(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        offload_rollout=False,
        enable_affinity=True,
        rollout_num_gpus=8,
        teacher_num_gpus_per_engine=None,
        opd_teacher_key=None,
        resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 8]},
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_multi_teacher_bundle_offsets_are_prefix_sums_not_index_times_size(monkeypatch):
    """Legacy checkpoint paths still receive equal, non-overlapping GPU
    shares."""
    import ray

    from relax.core.service import create_placement_group as _real_create_pg  # noqa: F401
    from relax.utils.opd import opd_utils

    captured = {"calls": [], "ctor_calls": {}}
    _install_fake_teacher_manager(monkeypatch, captured)
    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    full_pg = ("pg", list(range(16)), list(range(16)))
    monkeypatch.setattr(
        "relax.core.service.create_placement_group",
        lambda **kwargs: full_pg,
    )
    checkpoint_to_source = {"/ckpt/math": "math", "/ckpt/code": "code"}
    monkeypatch.setattr(ray, "get", lambda ref: [f"http://{checkpoint_to_source[ref[0]]}/generate"])

    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    shared_pg, managers = opd_utils._start_managed_multi_teacher(args, routes_json)

    assert shared_pg == full_pg
    assert isinstance(managers, list) and len(managers) == 2

    # TeacherManager adds rollout_num_gpus itself, so these offsets are
    # relative to the teacher region: math at 0, code at 0+4=4.
    assert captured["ctor_calls"]["/ckpt/math"]["bundle_offset"] == 0
    assert captured["ctor_calls"]["/ckpt/code"]["bundle_offset"] == 4
    assert captured["ctor_calls"]["/ckpt/math"]["num_replicas"] == 1
    assert captured["ctor_calls"]["/ckpt/math"]["gpus_per_replica"] == 4
    assert captured["ctor_calls"]["/ckpt/math"]["shared_pg"] is True
    assert captured["ctor_calls"]["/ckpt/math"]["pg"] == full_pg

    assert args.opd_teacher_routes_map == {
        "math": ["http://math/generate"],
        "code": ["http://code/generate"],
    }


def test_multi_teacher_requires_colocate(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: False)
    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math"})

    with pytest.raises(ValueError, match="requires colocate mode"):
        opd_utils._start_managed_multi_teacher(args, routes_json)


def test_multi_teacher_rejects_uneven_gpu_split(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    args = _base_args(resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 7]})
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    with pytest.raises(ValueError, match="evenly divisible"):
        opd_utils._start_managed_multi_teacher(args, routes_json)


@pytest.mark.parametrize("teacher_tp,expected_replicas", [(4, (2, 1, 1)), (None, (1, 1, 1))])
def test_multi_teacher_explicit_budgets_preserve_routes_and_offsets(
    monkeypatch: pytest.MonkeyPatch, teacher_tp: int | None, expected_replicas: tuple[int, int, int]
) -> None:
    import ray

    from relax.utils.opd import opd_utils

    captured = {"calls": [], "ctor_calls": {}}
    _install_fake_teacher_manager(monkeypatch, captured)
    full_pg = ("pg", list(range(64)), list(range(64)))
    monkeypatch.setattr("relax.core.service.create_placement_group", lambda **kwargs: full_pg)
    budgets = {"audit": 8, "evidence": 4, "evaluation": 4}
    routes = {source: {"checkpoint_path": f"/ckpt/{source}", "num_gpus": size} for source, size in budgets.items()}
    expected_urls = {
        f"/ckpt/{source}": [f"http://{source}-{index}/generate" for index in range(count)]
        for source, count in zip(budgets, expected_replicas, strict=True)
    }
    monkeypatch.setattr(ray, "get", lambda ref: expected_urls[ref[0]])
    args = _base_args(
        opd_teacher_routes=json.dumps(routes),
        rollout_num_gpus=48,
        teacher_num_gpus_per_engine=teacher_tp,
        resource={"actor": [1, 64], "rollout": [1, 48], "teacher": [1, 16]},
    )
    shared_pg, managers = opd_utils._start_managed_multi_teacher(args, args.opd_teacher_routes)
    assert shared_pg == full_pg
    assert isinstance(managers, list) and len(managers) == 3
    for source, offset, count in zip(budgets, (0, 8, 12), expected_replicas, strict=True):
        call = captured["ctor_calls"][f"/ckpt/{source}"]
        assert call == {
            "num_replicas": count,
            "gpus_per_replica": teacher_tp or budgets[source],
            "pg": full_pg,
            "shared_pg": True,
            "bundle_offset": offset,
        }
        assert args.opd_teacher_routes_map[source] == expected_urls[f"/ckpt/{source}"]
    assert args.opd_teacher_routes == json.dumps(routes)


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

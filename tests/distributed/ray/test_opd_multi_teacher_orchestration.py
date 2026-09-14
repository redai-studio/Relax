# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``_start_managed_multi_teacher`` was rewritten to delegate the GPU-budget
carve-up to the domain-agnostic ``start_multi_instance_managers`` helper (also
used by GenRM's multi-instance path).

This must not change MOPD's observable
contract: equal-split validation, one TeacherManager per data_source at a
non-overlapping bundle_offset, and a ``(pg, list[manager])`` return shape.
"""

import json
import sys
from argparse import Namespace
from types import ModuleType


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
    """Two teachers with equal GPU shares (the only shape MOPD currently
    supports, since it enforces an even split) must land at non-overlapping,
    monotonically increasing bundle offsets starting after the rollout
    region."""
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

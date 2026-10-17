# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MOPD teachers are created on the task owner with prefix-sum offsets."""

import json
from argparse import Namespace

from conftest import FakeOwnerHandle


def _install_fake_gateway(monkeypatch):
    from relax.components import inference_gateway

    monkeypatch.setattr(inference_gateway, "deploy_gateway", lambda role, manager_handle: "http://gateway/teacher")


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
    from relax.engine.inference.types import Role
    from relax.utils.opd import opd_utils

    _install_fake_gateway(monkeypatch)
    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    full_pg = ("pg", list(range(16)), list(range(16)))
    monkeypatch.setattr(
        "relax.core.service.create_placement_group",
        lambda **kwargs: full_pg,
    )
    owner = FakeOwnerHandle(urls={"math": ["http://math"], "code": ["http://code/generate"]})
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    from relax.engine.inference import config_adapters

    monkeypatch.setattr(
        config_adapters,
        "teacher_role_model",
        lambda args, **kwargs: (Namespace(name=kwargs["model_id"], path=args.teacher_hf_checkpoint), None, kwargs),
    )

    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    shared_pg, models = opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=owner)

    assert shared_pg == full_pg
    assert [(handle.role, handle.model_id) for handle in models] == [(Role.TEACHER, "math"), (Role.TEACHER, "code")]

    ((create_args, _),) = owner.named("create_role")
    assert create_args[0] == Role.TEACHER
    configs = {config.name: (config, kwargs) for config, _, kwargs in create_args[1]}
    # Offsets are a prefix sum within the teacher region, which starts after
    # the rollout bundles.
    assert configs["math"][1]["bundle_offset"] == 0
    assert configs["code"][1]["bundle_offset"] == 4
    assert configs["math"][0].path == "/ckpt/math"
    assert configs["math"][1]["pg"] == full_pg

    assert args.opd_teacher_gateway_url == "http://gateway/teacher"
    assert args.opd_teacher_route_keys == ("math", "code")


def test_multi_teacher_requires_colocate(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: False)
    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math"})

    with pytest.raises(ValueError, match="requires colocate mode"):
        opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=FakeOwnerHandle())


def test_multi_teacher_rejects_uneven_gpu_split(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    args = _base_args(resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 7]})
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    with pytest.raises(ValueError, match="evenly divisible"):
        opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=FakeOwnerHandle())

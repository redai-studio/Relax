# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``_start_managed_multi_teacher`` was rewritten to delegate the GPU-budget
carve-up to the domain-agnostic ``start_multi_instance_managers`` helper (also
used by GenRM's multi-instance path).

This must not change MOPD's observable
contract: default equal-split validation, one TeacherManager per data_source at a
non-overlapping bundle_offset, and a ``(pg, list[manager])`` return shape.
"""

import argparse
import json
import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock

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


@pytest.mark.parametrize(("total", "tp"), [(3, 2), (2, 0)])
def test_shared_teacher_rejects_invalid_engine_budget(total: int, tp: int) -> None:
    from relax.utils.opd import opd_utils

    with pytest.raises(ValueError, match="must be divisible"):
        opd_utils._parse_teacher_routes('{"a":"/a"}', total, tp, shared=True)


def test_shared_teacher_cli_validates_physical_budget_independently_of_routes():
    from relax.utils.opd import opd_utils

    parser = opd_utils.add_opd_arguments(argparse.ArgumentParser())
    assert not parser.parse_args([]).opd_shared_teacher
    args = parser.parse_args(
        ["--use-opd", "--opd-type", "sglang", "--opd-teacher-routes", '{"a":"/a","b":"/b"}', "--opd-shared-teacher"]
    )
    args.colocate = True
    args.resource = {"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]}
    opd_utils.validate_opd_args(args, is_offline=False)
    args.teacher_num_gpus_per_engine = 2
    opd_utils.validate_opd_args(args, is_offline=False)
    args.teacher_num_gpus_per_engine = 3
    with pytest.raises(ValueError, match="must be divisible"):
        opd_utils.validate_opd_args(args, is_offline=False)


def test_shared_teacher_rejects_separate_draft_checkpoint():
    from relax.utils.opd import opd_utils

    args = _base_args(
        opd_shared_teacher=True,
        opd_teacher_routes='{"a":"/a","b":"/b"}',
        teacher_sglang_speculative_draft_model_path="/draft",
    )
    with pytest.raises(ValueError, match="reloads both target and draft from the same checkpoint path"):
        opd_utils.maybe_start_managed_opd_teacher(args)


@pytest.mark.parametrize(
    ("budget", "teacher_tp", "replicas", "failure"),
    [
        (1, None, 1, None),
        (2, 2, 1, None),
        (4, None, 1, None),
        (2, 1, 2, None),
        (2, 1, 2, "initialization failed"),
        (2, 1, 2, "missing replica"),
    ],
)
def test_shared_teacher_allocates_replicas_and_owns_their_lifecycle(
    monkeypatch, budget, teacher_tp, replicas, failure
):
    import ray

    from relax.utils.opd import opd_utils

    module = ModuleType("relax.distributed.ray.teacher_manager")
    physical = MagicMock()
    owner = MagicMock()
    module.TeacherManager = MagicMock()
    module.TeacherManager.options.return_value.remote.return_value = physical
    module.SharedTeacherManager = MagicMock()
    module.SharedTeacherManager.options.return_value.remote.return_value = owner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rollout_gpus = 6
    gpu_ids = list(range(rollout_gpus + budget))
    pg = ("pg", gpu_ids, gpu_ids)
    create = MagicMock(return_value=pg)
    remove = MagicMock()
    kill = MagicMock()
    monkeypatch.setattr("relax.core.service.create_placement_group", create)
    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", remove)
    monkeypatch.setattr(ray, "kill", kill)
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    urls = [f"http://engine-{i}/generate" for i in range(replicas)]
    physical.get_urls.remote.return_value = urls[:-1] if failure == "missing replica" else urls
    routes = {key: f"/checkpoints/{key}" for key in ("a", "b", "c")}
    monkeypatch.setattr(opd_utils, "_shared_checkpoints", lambda specs: routes)
    args = _base_args(
        opd_shared_teacher=True,
        offload_rollout=True,
        opd_teacher_routes=json.dumps(routes),
        rollout_num_gpus=rollout_gpus,
        teacher_num_gpus_per_engine=teacher_tp,
        teacher_sglang_dtype="float16",
        resource={"actor": [1, rollout_gpus + budget], "rollout": [1, rollout_gpus], "teacher": [1, budget]},
    )
    if failure:
        if failure == "initialization failed":
            owner.initialize.remote.side_effect = RuntimeError(failure)
        expected = "got 1 endpoints" if failure == "missing replica" else failure
        with pytest.raises(RuntimeError, match=expected):
            opd_utils.maybe_start_managed_opd_teacher(args)
        physical.shutdown.remote.assert_called_once()
        if failure == "missing replica":
            module.SharedTeacherManager.options.assert_not_called()
            kill.assert_called_once_with(physical)
        else:
            assert [call.args for call in kill.call_args_list] == [(owner,), (physical,)]
        remove.assert_called_once_with(pg[0])
        return

    assert opd_utils.maybe_start_managed_opd_teacher(args) == (pg, [owner])
    create.assert_called_once()
    spawn = module.TeacherManager.options.return_value.remote
    spawn.assert_called_once()
    engine_args, count, tp = spawn.call_args.args
    assert (engine_args.teacher_hf_checkpoint, count, tp) == (routes["a"], replicas, teacher_tp or budget)
    assert engine_args.teacher_sglang_dtype == "float16"
    assert spawn.call_args.kwargs == {"pg": pg, "shared_pg": True}
    module.SharedTeacherManager.options.return_value.remote.assert_called_once_with(args, physical, urls, routes)
    owner.initialize.remote.assert_called_once()
    owner.offload.remote.assert_called_once()
    assert args.opd_teacher_routes_map == {key: urls for key in routes}


def test_only_rollout_configuration_receives_shared_teacher_handle(monkeypatch):
    from relax.core import controller

    ctrl = controller.Controller.__new__(controller.Controller)
    ctrl.config = Namespace(opd_shared_teacher=True)
    ctrl.runtime_env = None
    ctrl._health_manager = Namespace(status=None)
    owner = object()
    ctrl._teacher_manager = [owner]
    service = MagicMock()
    monkeypatch.setattr(controller, "Service", service)
    for role in ("rollout", "actor"):
        _, _, error = ctrl._create_service_task(role, object(), 1, None, None, defer_deploy=True)
        assert error is None
        config = service.call_args.kwargs["config"]
        if role == "rollout":
            assert config is not ctrl.config
            assert config._opd_shared_teacher_manager is owner
        else:
            assert config is ctrl.config
    assert not hasattr(ctrl.config, "_opd_shared_teacher_manager")


def test_shared_checkpoints_match_architectures_and_text_vocab_size(tmp_path) -> None:
    from relax.utils.opd import opd_utils

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    routes = {"a": {"checkpoint_path": str(a)}, "b": {"checkpoint_path": str(b)}}
    config = {"architectures": ["Teacher"], "vocab_size": 32}
    (a / "config.json").write_text(json.dumps(config))
    config_b = b / "config.json"
    config_b.write_text(json.dumps({"architectures": ["Teacher"], "text_config": {"vocab_size": 32}}))
    assert opd_utils._shared_checkpoints(routes) == {"a": str(a), "b": str(b)}

    for mismatch in ({"architectures": ["OtherTeacher"]}, {"text_config": {"vocab_size": 64}}):
        config_b.write_text(json.dumps({**config, **mismatch}))
        with pytest.raises(ValueError, match="architectures, vocab_size"):
            opd_utils._shared_checkpoints(routes)

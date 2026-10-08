# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM engines start on the bundles the placement plan assigned.

``GenRMManager`` no longer derives its own region from ``rollout_num_gpus``:
``create_genrm_managers`` asks ``plan_placement`` for each instance's absolute
start and the manager only adds the per-engine stride.
"""

import importlib
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.distributed.ray import placement_group as placement_group_module


@pytest.fixture(autouse=True)
def _cleanup_genrm_module():
    yield
    sys.modules.pop("relax.distributed.ray.genrm", None)


def _import_genrm(monkeypatch):
    sglang_engine = ModuleType("relax.backends.sglang.sglang_engine")
    sglang_engine.SGLangEngine = object

    ray_utils = ModuleType("relax.distributed.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.Lock = MagicMock()

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.init_http_client = MagicMock()

    monkeypatch.setitem(sys.modules, "relax.backends.sglang.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.utils", ray_utils)
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)
    sys.modules.pop("relax.distributed.ray.genrm", None)
    return importlib.import_module("relax.distributed.ray.genrm")


def _bare_manager(genrm, *, bundle_offset, num_gpu_per_engine, args):
    manager_cls = genrm.GenRMManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = args
    manager.pg = "pg"
    manager.bundle_offset = bundle_offset
    manager.num_gpu_per_engine = num_gpu_per_engine
    return manager


def test_genrm_manager_resolve_placement_uses_absolute_bundle_offset(monkeypatch):
    genrm = _import_genrm(monkeypatch)
    manager = _bare_manager(genrm, bundle_offset=4, num_gpu_per_engine=2, args=SimpleNamespace())

    assert [manager._resolve_placement(rank) for rank in (0, 1)] == [("pg", False, 4), ("pg", False, 6)]


def test_genrm_manager_resolve_placement_ignores_rollout_num_gpus(monkeypatch):
    genrm = _import_genrm(monkeypatch)
    # Split colocate: the plan already put the rollout region into
    # bundle_offset, so adding rollout_num_gpus again would double-count.
    args = SimpleNamespace(fully_async=False, rollout_num_gpus=4, _genrm_colocate_with_rollout=False)
    manager = _bare_manager(genrm, bundle_offset=4, num_gpu_per_engine=2, args=args)

    assert manager._resolve_placement(0) == ("pg", False, 4)


def test_genrm_manager_starts_the_shared_engine_with_genrm_profile(monkeypatch):
    genrm = _import_genrm(monkeypatch)
    manager = _bare_manager(genrm, bundle_offset=0, num_gpu_per_engine=1, args=SimpleNamespace())

    assert manager._build_engine_ctor_kwargs(0) == {"profile": "genrm"}


def _install_fake_genrm_manager(monkeypatch):
    ctor_kwargs = []
    genrm_module = ModuleType("relax.distributed.ray.genrm")

    class FakeGenRMManager:
        @classmethod
        def options(cls, **options):
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            ctor_kwargs.append(kwargs)
            return MagicMock()

    genrm_module.GenRMManager = FakeGenRMManager
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.genrm", genrm_module)
    return ctor_kwargs


def _genrm_spec(num_gpus):
    return {
        "model_path": "/judge",
        "num_gpus": num_gpus,
        "num_gpus_per_engine": 1,
        "engine_config": {},
        "sampling_config": {},
    }


def _colocate_args(**overrides):
    values = dict(
        colocate=True,
        hybrid=False,
        fully_async=False,
        offload_rollout=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
    )
    values.update(overrides)
    return Namespace(**values)


def test_genrm_managers_receive_planned_offsets_in_split_layout(monkeypatch):
    ctor_kwargs = _install_fake_genrm_manager(monkeypatch)
    args = _colocate_args(_genrm_instances_resolved={"quality": _genrm_spec(2), "safety": _genrm_spec(2)})

    managers = placement_group_module.create_genrm_managers(args, "pg")

    assert list(managers) == ["quality", "safety"]
    assert [kwargs["bundle_offset"] for kwargs in ctor_kwargs] == [4, 6]


def test_genrm_managers_receive_planned_offset_in_shared_layout(monkeypatch):
    ctor_kwargs = _install_fake_genrm_manager(monkeypatch)
    args = _colocate_args(
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        _genrm_instances_resolved={"quality": _genrm_spec(4), "safety": _genrm_spec(4)},
        _genrm_colocate_with_rollout=True,
    )

    placement_group_module.create_genrm_managers(args, "pg")

    assert [kwargs["bundle_offset"] for kwargs in ctor_kwargs] == [0, 4]


def test_genrm_single_instance_receives_planned_offset(monkeypatch):
    ctor_kwargs = _install_fake_genrm_manager(monkeypatch)
    args = _colocate_args(_genrm_instances_resolved={"__default__": _genrm_spec(4)})

    managers = placement_group_module.create_genrm_managers(args, "pg")

    assert list(managers) == ["__default__"]
    assert ctor_kwargs == [{"bundle_offset": 4}]

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace


def _install_fake_teacher_manager(monkeypatch, captured):
    teacher_manager_module = ModuleType("relax.distributed.ray.teacher_manager")

    class _RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self):
            captured["calls"].append(self.name)
            return f"{self.name}-ref"

    class _TeacherManagerHandle:
        get_urls = _RemoteMethod("get_urls")
        offload = _RemoteMethod("offload")

    class _TeacherManagerActor:
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            captured["remote_args"] = args
            captured["remote_kwargs"] = kwargs
            captured["handle"] = _TeacherManagerHandle()
            return captured["handle"]

    teacher_manager_module.TeacherManager = _TeacherManagerActor
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.teacher_manager", teacher_manager_module)


def test_create_managed_opd_teacher_manager_offloads_shared_pg_teacher(monkeypatch, tmp_path):
    import ray

    from relax.utils.opd.opd_utils import create_managed_opd_teacher_manager

    captured = {"calls": []}
    _install_fake_teacher_manager(monkeypatch, captured)
    monkeypatch.setattr(
        ray,
        "get",
        lambda ref: ["http://teacher/generate"] if ref == "get_urls-ref" else None,
    )

    autoscaler_config = tmp_path / "autoscaler.yaml"
    autoscaler_config.write_text("enabled: true\n")
    args = SimpleNamespace(
        offload_rollout=True,
        autoscaler_config=str(autoscaler_config),
        enable_affinity=True,
    )
    manager, urls = create_managed_opd_teacher_manager(
        args,
        num_replicas=1,
        gpus_per_replica=4,
        pg=("pg", list(range(8)), list(range(8))),
        shared_pg=True,
        runtime_env={"env_vars": {"A": "B"}},
    )

    assert manager is captured["handle"]
    assert urls == ["http://teacher/generate"]
    assert captured["calls"] == ["get_urls", "offload"]
    assert captured["options"] == {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": {"A": "B"}},
        "resources": {"stable_cpu": 1},
    }
    assert captured["remote_args"] == (args, 1, 4)
    assert captured["remote_kwargs"] == {
        "pg": ("pg", list(range(8)), list(range(8))),
        "shared_pg": True,
    }

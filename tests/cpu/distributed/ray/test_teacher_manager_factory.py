# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock


def test_teacher_factory_uses_registered_task_owner_snapshot(monkeypatch):
    import ray

    from relax.distributed.ray import inference_manager
    from relax.utils.opd.opd_utils import create_managed_opd_teacher_manager

    snapshot = SimpleNamespace(models=[SimpleNamespace(replicas=[SimpleNamespace(base_url="http://teacher")])])
    route = SimpleNamespace(snapshot=SimpleNamespace(remote=Mock(return_value=snapshot)))
    create = Mock(return_value=route)
    monkeypatch.setattr(inference_manager, "create_static_inference_manager", create)
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    task = object()
    args = SimpleNamespace(_rollout_engine_manager=task)
    binding = object()
    owner, urls = create_managed_opd_teacher_manager(args, num_replicas=2, gpus_per_replica=8, pg=binding)
    create.assert_called_once_with(args, "teacher", binding, runtime_env=None, num_replicas=2, gpus_per_replica=8)
    route.snapshot.remote.assert_called_once_with("teacher")
    assert owner is route is args._teacher_inference_manager_handle
    assert urls == ["http://teacher/generate"]

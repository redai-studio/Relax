# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
from conftest import FakeOwnerHandle


@pytest.fixture
def owner(monkeypatch):
    import ray

    from relax.engine.inference import config_adapters

    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(config_adapters, "teacher_role_model", lambda args, **kwargs: ("teacher", None, kwargs))
    return FakeOwnerHandle(urls={"default": ["http://teacher"]})


def test_create_managed_opd_teacher_offloads_shared_pg_teacher(owner):
    from relax.engine.inference.types import Role
    from relax.utils.opd.opd_utils import create_managed_opd_teacher

    args = SimpleNamespace(offload_rollout=True)
    pg = ("pg", list(range(8)), list(range(8)))

    handle, urls = create_managed_opd_teacher(
        args, num_replicas=1, gpus_per_replica=4, inference_manager_handle=owner, pg=pg, shared_pg=True
    )

    assert (handle.role, handle.model_id) == (Role.TEACHER, "default")
    assert urls == ["http://teacher/generate"]
    ((create_args, _),) = owner.named("create_role")
    assert create_args[0] == Role.TEACHER
    assert create_args[1][0][2]["pg"] is pg
    assert [args[2] for args, _ in owner.named("call")] == ["get_urls", "deactivate"]


def test_create_managed_opd_teacher_keeps_a_dedicated_teacher_loaded(owner):
    from relax.utils.opd.opd_utils import create_managed_opd_teacher

    create_managed_opd_teacher(
        SimpleNamespace(offload_rollout=True), num_replicas=2, gpus_per_replica=4, inference_manager_handle=owner
    )

    ((create_args, _),) = owner.named("create_role")
    assert create_args[1][0][2]["pg"] is None
    assert [args[2] for args, _ in owner.named("call")] == ["get_urls"]

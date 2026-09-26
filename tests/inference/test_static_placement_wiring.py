# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("sglang")

from relax.distributed.ray import teacher_manager


@pytest.fixture
def teacher_module(monkeypatch):
    service = SimpleNamespace(
        create_placement_group=MagicMock(return_value=("dedicated-pg", list(range(4)), [0, 1, 0, 1]))
    )
    rollout = SimpleNamespace(
        _allocate_rollout_engine_addr_and_ports_normal=MagicMock(return_value=({0: {}, 1: {}}, {}))
    )
    monkeypatch.setattr(teacher_manager, "create_placement_group", service.create_placement_group)
    monkeypatch.setattr(
        teacher_manager,
        "_allocate_rollout_engine_addr_and_ports_normal",
        rollout._allocate_rollout_engine_addr_and_ports_normal,
    )
    return teacher_manager, service, rollout


def _manager(module):
    manager = object.__new__(module.TeacherManager.__ray_metadata__.modified_class)
    manager.args = SimpleNamespace(rollout_num_gpus=4, enable_affinity=False)
    manager._shared_pg = True
    manager._shared_pg_tuple = ("shared-pg", list(range(8)), list(range(8)))
    manager._bundle_offset = 0
    manager.gpus_per_replica = 4
    manager.nodes_per_engine = 2
    manager._local_gpus_per_worker = 2
    manager._planned_placement = SimpleNamespace(bundle_start=4)
    manager._dedicated_replica_pgs = {}
    return manager


def test_teacher_shared_workers_follow_global_offset(teacher_module):
    module, _service, _rollout = teacher_module
    manager = _manager(module)
    assert manager._resolve_placement(0) == (manager._shared_pg_tuple, False, 4)
    assert manager._resolve_placement(1) == (manager._shared_pg_tuple, False, 6)


def test_teacher_dedicated_workers_share_one_owned_pg_per_replica(teacher_module):
    module, service, _rollout = teacher_module
    manager = _manager(module)
    manager._shared_pg = False
    manager._planned_placement = None
    first = manager._resolve_placement(0)
    second = manager._resolve_placement(1)
    assert first[0] is second[0]
    assert first[1:] == (True, 0)
    assert second[1:] == (True, 2)
    service.create_placement_group.assert_called_once_with(num_gpus=4, node_group_affinity=False)


def test_teacher_multinode_ports_are_allocated_for_all_workers_together(teacher_module):
    module, _service, rollout = teacher_module
    manager = _manager(module)
    manager._shared_pg = False
    manager._teacher_args = manager.args
    manager._engine_addr_and_ports = {}
    workers = [(0, object()), (1, object())]
    manager._allocate_engine_addr_and_ports(new_engines=workers)
    kwargs = rollout._allocate_rollout_engine_addr_and_ports_normal.call_args.kwargs
    assert kwargs["rollout_engines"] == workers
    assert kwargs["num_gpus_per_engine"] == 4


@pytest.mark.parametrize("confirmed", [False, True])
def test_teacher_startup_owner_deletes_pg_only_after_confirmed_shutdown(monkeypatch, confirmed):
    import ray

    from relax.utils.opd.opd_utils import _rollback_teacher_startup

    events = []
    manager = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: "shutdown"))

    def get(ref, timeout=None):
        events.append(ref)
        if not confirmed:
            raise TimeoutError("worker still alive")

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", lambda handle: events.append("kill"))
    monkeypatch.setattr(ray.util, "remove_placement_group", lambda pg: events.append("remove"))
    if confirmed:
        _rollback_teacher_startup([manager], ("pg", [], []), RuntimeError("startup failed"))
        assert events == ["shutdown", "kill", "remove"]
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            _rollback_teacher_startup([manager], ("pg", [], []), RuntimeError("startup failed"))
        assert events == ["shutdown"]

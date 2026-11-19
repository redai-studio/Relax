# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock


def test_managed_teacher_colocate_registers_in_task_on_full_shared_pg(monkeypatch):
    from relax.distributed.ray import shared_task_startup

    full_pg = ("pg", list(range(8)), list(range(8)))
    teacher = SimpleNamespace(
        snapshot=SimpleNamespace(
            remote=lambda role: SimpleNamespace(
                models=[SimpleNamespace(replicas=[SimpleNamespace(base_url="http://teacher")])]
            )
        )
    )
    task = object()
    captured = []

    def start(args, binding, static, create, **kwargs):
        assert binding is full_pg
        captured.extend(static)
        return task, (teacher,), task

    monkeypatch.setattr(shared_task_startup, "start_shared_task", start)
    monkeypatch.setattr(shared_task_startup.ray, "get", lambda value: value)
    config = Namespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        teacher_hf_checkpoint="/teacher",
        rollout_num_gpus=4,
        num_gpus_per_node=8,
        use_critic=False,
    )
    result = shared_task_startup.prepare_shared_task(config, full_pg)
    assert result is teacher
    assert config._rollout_engine_manager is task
    assert config._inference_phase_coordinator is task
    assert captured[0].placement.binding is full_pg
    assert captured[0].placement.bundle_offset == 4
    assert captured[0].options["num_replicas"] == 1
    assert captured[0].options["gpus_per_replica"] == 4
    assert config.opd_teacher_url == "http://teacher/generate"


def test_rollout_only_reuses_teacher_placement_group(monkeypatch) -> None:
    from tests.cpu.core.controller_test_utils import load_controller_with_stubbed_dependencies

    controller = load_controller_with_stubbed_dependencies("_test_opd_rollout_only")
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = Namespace(
        debug_rollout_only=True,
        colocate=True,
        hybrid=False,
        fully_async=False,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        enable_affinity=False,
        num_rollout=1,
        advantage_estimator="grpo",
        data_source_path="relax.engine.rollout.data_source.RolloutDataSource",
        resource={"actor": (1, 8), "rollout": (1, 4), "teacher": (1, 4)},
    )
    instance.serve_dict = {}
    instance.runtime_env = None
    instance._health_manager = Mock()
    instance._validate_gpu_resources = Mock()
    teacher_pg = object()
    allocate_pg = Mock(return_value=teacher_pg)
    monkeypatch.setattr(controller, "create_placement_group", allocate_pg)
    registry = SimpleNamespace(resolve_algorithm=lambda _: controller.ALGOS["grpo"].copy())
    monkeypatch.setitem(sys.modules, "relax.core.registry", registry)
    monkeypatch.setattr(controller, "create_data_source_actor", Mock())
    prepared = []
    instance._start_services = lambda service_args: prepared.extend(service_args) or []

    instance.register_all_serve()

    allocate_pg.assert_called_once_with(num_gpus=8, node_group_affinity=False)
    assert {str(item[0]) for item in prepared} == {"rollout", "teacher"}
    assert all(item[4] is teacher_pg for item in prepared)
    assert all("teacher" in item[5] for item in prepared)

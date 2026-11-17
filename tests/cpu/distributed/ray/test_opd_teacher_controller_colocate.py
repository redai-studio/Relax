# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


def test_managed_teacher_colocate_uses_full_shared_pg(monkeypatch):
    full_pg = ("pg", list(range(8)), list(range(8)))

    core_service = ModuleType("relax.core.service")
    core_service.create_placement_group = lambda *args, **kwargs: full_pg
    monkeypatch.setitem(sys.modules, "relax.core.service", core_service)

    from relax.utils.opd import opd_utils

    create_teacher = Mock(return_value=("teacher-manager-handle", ["http://teacher/generate"]))
    monkeypatch.setattr(opd_utils, "create_managed_opd_teacher_manager", create_teacher)

    config = Namespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        teacher_hf_checkpoint="/teacher",
    )

    runtime_env = {"env_vars": {"A": "B"}}
    shared_pg, teacher_manager = opd_utils.maybe_start_managed_opd_teacher(config, runtime_env=runtime_env)

    assert shared_pg == full_pg
    assert teacher_manager == "teacher-manager-handle"
    create_teacher.assert_called_once_with(
        config, num_replicas=1, gpus_per_replica=4, pg=full_pg, shared_pg=True, runtime_env=runtime_env
    )
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
    monkeypatch.setattr(controller, "maybe_start_managed_opd_teacher", Mock(return_value=(teacher_pg, [])))
    registry = SimpleNamespace(resolve_algorithm=lambda _: controller.ALGOS["grpo"].copy())
    monkeypatch.setitem(sys.modules, "relax.core.registry", registry)
    monkeypatch.setattr(controller, "create_data_source_actor", Mock())
    service = Mock()
    monkeypatch.setattr(controller, "Service", service)

    instance.register_all_serve()

    service.assert_called_once()
    assert service.call_args.kwargs["role"] == "rollout"
    assert service.call_args.kwargs["num_gpus"] == 4
    assert service.call_args.kwargs["actor_rollout_pgs"] is teacher_pg

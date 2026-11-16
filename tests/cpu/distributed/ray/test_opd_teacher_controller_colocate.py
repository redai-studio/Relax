# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType
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

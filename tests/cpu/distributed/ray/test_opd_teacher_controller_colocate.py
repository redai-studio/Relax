# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType


def test_managed_teacher_colocate_uses_full_shared_pg(monkeypatch):
    full_pg = ("pg", list(range(8)), list(range(8)))

    core_service = ModuleType("relax.core.service")
    core_service.create_placement_group = lambda *args, **kwargs: full_pg
    monkeypatch.setitem(sys.modules, "relax.core.service", core_service)

    from relax.utils.opd import opd_utils

    captured = {}

    def fake_create_teacher_manager(
        args,
        *,
        num_replicas,
        gpus_per_replica,
        pg=None,
        shared_pg=False,
        runtime_env=None,
    ):
        captured["num_replicas"] = num_replicas
        captured["gpus_per_replica"] = gpus_per_replica
        captured["pg"] = pg
        captured["shared_pg"] = shared_pg
        captured["runtime_env"] = runtime_env
        return "teacher-manager-handle", ["http://teacher/generate"]

    monkeypatch.setattr(opd_utils, "create_managed_opd_teacher_manager", fake_create_teacher_manager)

    config = Namespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        teacher_hf_checkpoint="/teacher",
    )

    shared_pg, teacher_manager = opd_utils.maybe_start_managed_opd_teacher(
        config,
        runtime_env={"env_vars": {"A": "B"}},
    )

    assert shared_pg == full_pg
    assert teacher_manager == "teacher-manager-handle"
    assert captured == {
        "num_replicas": 1,
        "gpus_per_replica": 4,
        "pg": full_pg,
        "shared_pg": True,
        "runtime_env": {"env_vars": {"A": "B"}},
    }
    assert config.opd_teacher_url == "http://teacher/generate"

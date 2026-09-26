# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType


def test_managed_teacher_colocate_uses_full_shared_pg(monkeypatch):
    full_pg = ("pg", list(range(8)), list(range(8)))

    core_service = ModuleType("relax.core.service")
    core_service.create_placement_group = lambda *args, **kwargs: full_pg
    monkeypatch.setitem(sys.modules, "relax.core.service", core_service)

    from relax.components import inference_gateway
    from relax.utils.opd import opd_utils

    captured = {}

    def fake_create_teacher(
        args,
        *,
        num_replicas,
        gpus_per_replica,
        inference_manager_handle,
        pg=None,
        shared_pg=False,
    ):
        captured["num_replicas"] = num_replicas
        captured["gpus_per_replica"] = gpus_per_replica
        captured["pg"] = pg
        captured["shared_pg"] = shared_pg
        captured["owner"] = inference_manager_handle
        return ("teacher-model",), ["http://teacher/generate"]

    monkeypatch.setattr(opd_utils, "create_managed_opd_teacher", fake_create_teacher)
    monkeypatch.setattr(inference_gateway, "deploy_gateway", lambda role, manager_handle: "http://gateway/teacher")

    config = Namespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        teacher_hf_checkpoint="/teacher",
    )

    shared_pg, teacher_models = opd_utils.maybe_start_managed_opd_teacher(config, inference_manager_handle="owner")

    assert shared_pg == full_pg
    assert teacher_models == ("teacher-model",)
    assert captured == {
        "num_replicas": 1,
        "gpus_per_replica": 4,
        "pg": full_pg,
        "shared_pg": True,
        "owner": "owner",
    }
    assert config.opd_teacher_url == "http://gateway/teacher/generate"


def test_teacher_colocate_layout_accepts_split_and_shared_only():
    import pytest

    from relax.utils.opd.opd_utils import check_teacher_colocate_layout

    check_teacher_colocate_layout(4, 4, 8)  # split: teachers after rollout
    check_teacher_colocate_layout(8, 8, 8)  # shared: teachers reuse the rollout bundles
    check_teacher_colocate_layout(8, 4, 8)
    for rollout, teacher in ((4, 2), (6, 4), (8, 9)):
        with pytest.raises(ValueError, match="split bundles"):
            check_teacher_colocate_layout(rollout, teacher, 8)


def test_shared_teacher_layout_starts_at_bundle_zero_and_defers():
    from relax.engine.inference.phase_plans import PHASE_TEACHER, deferred_opd_enabled, placement_phase
    from relax.utils.opd.opd_utils import teacher_region_offset

    base = dict(use_opd=True, opd_type="sglang", colocate=True, hybrid=False, opd_teacher_routes=None)
    shared = Namespace(
        **base,
        teacher_hf_checkpoint="/ckpt",
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "teacher": [1, 8]},
    )
    split = Namespace(
        **base,
        teacher_hf_checkpoint="/ckpt",
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
    )
    assert teacher_region_offset(shared) == 0 and deferred_opd_enabled(shared)
    assert placement_phase(shared, PHASE_TEACHER) == PHASE_TEACHER
    assert teacher_region_offset(split) == 4 and not deferred_opd_enabled(split)
    assert placement_phase(split, PHASE_TEACHER) != PHASE_TEACHER


def test_opd_utils_shared_teacher_layout_accepts_agentic_rollout():
    import pytest

    from relax.utils.opd.opd_utils import validate_managed_opd_teacher_colocate_args

    def _args(rollout_gpus, teacher_gpus):
        return Namespace(
            use_opd=True,
            opd_type="sglang",
            colocate=True,
            hybrid=False,
            opd_teacher_routes=None,
            teacher_hf_checkpoint="/ckpt",
            use_agentic_rollout=True,
            use_critic=False,
            actor_num_gpus_per_node=8,
            actor_num_nodes=1,
            offload_train=None,
            offload_rollout=None,
            rollout_num_gpus=rollout_gpus,
            resource={"actor": [1, 8], "rollout": [1, rollout_gpus], "teacher": [1, teacher_gpus]},
        )

    # The resident Agentic pipeline scores a shared teacher in its deferred close stage.
    validate_managed_opd_teacher_colocate_args(_args(8, 8))
    validate_managed_opd_teacher_colocate_args(_args(4, 4))
    with pytest.raises(ValueError, match="split bundles"):
        validate_managed_opd_teacher_colocate_args(_args(8, 9))


def test_opd_utils_teacher_layout_counts_only_actor_pg_bundles_with_critic():
    """Critic GPUs are not in the actor placement group the teachers share, so
    validation and placement must agree on the actor bundle count."""
    import pytest

    from relax.utils.opd.opd_utils import teacher_shares_rollout_bundles, validate_managed_opd_teacher_colocate_args

    def _args(rollout_gpus, teacher_gpus):
        return Namespace(
            use_opd=True,
            opd_type="sglang",
            colocate=True,
            hybrid=False,
            opd_teacher_routes=None,
            teacher_hf_checkpoint="/ckpt",
            use_agentic_rollout=False,
            use_critic=True,
            critic_num_gpus_per_node=8,
            critic_num_nodes=1,
            actor_num_gpus_per_node=8,
            actor_num_nodes=1,
            offload_train=None,
            offload_rollout=None,
            rollout_num_gpus=rollout_gpus,
            resource={"actor": [1, 8], "rollout": [1, rollout_gpus], "teacher": [1, teacher_gpus]},
        )

    # Shared layout: rollout covers the 8 actor bundles and the teacher reuses 4 of them.
    shared = _args(8, 4)
    validate_managed_opd_teacher_colocate_args(shared)
    assert teacher_shares_rollout_bundles(shared)
    # 8 + 8 only fits a 16-bundle PG; the actor PG has 8, so it is the shared layout.
    validate_managed_opd_teacher_colocate_args(_args(8, 8))
    assert teacher_shares_rollout_bundles(_args(8, 8))
    with pytest.raises(ValueError):
        validate_managed_opd_teacher_colocate_args(_args(8, 9))

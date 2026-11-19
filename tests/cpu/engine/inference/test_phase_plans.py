# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.inference.config_adapters import static_role_spec
from relax.engine.inference.phase_plans import deferred_genrm_enabled, placement_phase
from relax.engine.inference.types import DeploymentMode, Role
from relax.utils.opd.opd_utils import teacher_shares_rollout_bundles, validate_managed_opd_teacher_colocate_args


@pytest.mark.parametrize(
    "rollout,phase,mode", [(4, "inference", DeploymentMode.SPLIT), (8, "teacher", DeploymentMode.DEFER)]
)
def test_teacher_layout_adapts_existing_flags(rollout, phase, mode):
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        colocate=True,
        hybrid=False,
        resource={"actor": [1, 8], "rollout": [1, rollout], "teacher": [1, 4]},
        rollout_num_gpus=rollout,
        use_critic=False,
        num_gpus_per_node=8,
    )
    assert teacher_shares_rollout_bundles(args) is (rollout == 8)
    spec = static_role_spec(Role.TEACHER, args, {"num_replicas": 1, "gpus_per_replica": 4})
    assert spec.deployment.phase == phase
    assert spec.deployment.mode is mode
    if rollout == 8:
        args.offload_train = args.offload_rollout = None
        args.actor_num_gpus_per_node = 8
        args.actor_num_nodes = 1
        validate_managed_opd_teacher_colocate_args(args)
        assert args.offload_train and args.offload_rollout


@pytest.mark.parametrize(
    "shared,deferred,phase", [(True, True, "genrm"), (False, True, "inference"), (True, False, "genrm")]
)
def test_genrm_phase_is_independent_of_automatic_scoring(shared, deferred, phase):
    args = SimpleNamespace(
        _genrm_instances_resolved=[{"name": "judge"}],
        _genrm_colocate_with_rollout=shared,
        defer_reward_to_post_process=deferred,
        genrm_model_path="judge",
        genrm_num_gpus=4,
        genrm_num_gpus_per_engine=4,
        genrm_engine_config={},
        genrm_sampling_config={},
        num_gpus_per_node=4,
        colocate=True,
        hybrid=False,
    )
    assert placement_phase(args, "genrm") == phase
    assert deferred_genrm_enabled(args) is (shared and deferred)
    spec = static_role_spec(Role.GENRM, args, {})
    assert spec.deployment.phase == phase
    assert spec.deployment.mode is (DeploymentMode.DEFER if shared else DeploymentMode.SPLIT)

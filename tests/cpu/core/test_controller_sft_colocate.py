# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Controller's SFT colocate decision without starting Ray
services."""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from relax.core import controller as module
from relax.core import registry
from relax.distributed.ray import inference_handle


@pytest.mark.parametrize("with_rollout", [True, False])
def test_register_all_serve_sft_colocate_uses_optional_roles(monkeypatch, with_rollout):
    algo = {role: object() for role in [*registry.ROLES_SFT_ONLY, registry.ROLES.rollout]}
    resolve_algorithm = Mock(return_value=algo)
    monkeypatch.setattr(registry, "resolve_algorithm", resolve_algorithm)
    monkeypatch.setattr(inference_handle, "shared_task_enabled", lambda config: False)
    collaborators = {
        "validate_ppo_config": Mock(),
        "resolve_sft_algo_key": Mock(return_value="sft"),
        "ALGOS": {"sft": algo},
        "process_role": Mock(return_value=registry.ROLES_SFT_ONLY),
        "validate_sft_resource": Mock(),
        "register_extra_roles": Mock(return_value=[registry.ROLES.rollout] if with_rollout else []),
        "load_function": Mock(return_value=object),
        "create_data_source_actor": Mock(return_value=object()),
        "_build_train_start_config": Mock(return_value={}),
        "_actor_rollout_pg_roles": Mock(return_value=["actor", "rollout"]),
    }
    for name, value in collaborators.items():
        monkeypatch.setattr(module, name, value)

    class ReachedResourceValidation(Exception):
        pass

    controller = module.Controller.__new__(module.Controller)
    controller.config = Namespace(
        colocate=True,
        resource={role: (1, 1) for role in algo},
        data_source_path="test.data_source",
    )
    controller.runtime_env = None
    controller._health_manager = Mock()
    controller._maybe_resolve_num_rollout = Mock()
    controller._validate_gpu_resources = Mock(side_effect=ReachedResourceValidation)
    with pytest.raises(ReachedResourceValidation):
        controller.register_all_serve()

    roles, colocate, pg_roles = controller._validate_gpu_resources.call_args.args
    resolve_algorithm.assert_called_once_with(controller.config)
    collaborators["_actor_rollout_pg_roles"].assert_called_once_with(controller.config)
    assert pg_roles == ["actor", "rollout"]
    assert colocate is with_rollout
    assert {str(role) for role, *_ in roles} == ({"sft", "actor", "rollout"} if with_rollout else {"sft", "actor"})

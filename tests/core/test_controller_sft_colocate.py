# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Controller's SFT colocate decision without starting Ray
services."""

import ast
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("with_rollout, expected_colocate", [(True, True), (False, False)])
def test_register_all_serve_sft_colocate_uses_optional_roles(with_rollout, expected_colocate):
    # Execute the actual method; only its external collaborators are replaced.
    # Stop at resource validation, before placement groups/services are created.
    path = Path(__file__).resolve().parents[2] / "relax/core/controller.py"
    tree = ast.parse(path.read_text())
    controller_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Controller")
    method = next(
        n for n in controller_class.body if isinstance(n, ast.FunctionDef) and n.name == "register_all_serve"
    )
    algo = {role: object() for role in ("actor", "sft", "rollout")}
    namespace = {
        "validate_ppo_config": Mock(),
        "maybe_start_managed_opd_teacher": Mock(return_value=(None, None)),
        "resolve_sft_algo_key": Mock(return_value="sft"),
        "ALGOS": {"sft": algo},
        "process_role": Mock(return_value=("sft", "actor")),
        "validate_sft_resource": Mock(),
        "register_extra_roles": Mock(return_value=["rollout"] if with_rollout else []),
        "load_function": Mock(return_value=object),
        "create_data_source_actor": Mock(return_value=object()),
        "logger": Mock(),
        "relax_utils": Mock(),
        "_build_train_start_config": Mock(return_value={}),
        "_actor_rollout_pg_roles": Mock(return_value=["actor", "rollout"]),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class ReachedResourceValidation(Exception):
        pass

    controller = Namespace(
        config=Namespace(
            colocate=True,
            resource={role: (1, 1) for role in algo},
            data_source_path="test.data_source",
        ),
        runtime_env=None,
        _health_manager=Mock(),
        _maybe_resolve_num_rollout=Mock(),
        _validate_gpu_resources=Mock(side_effect=ReachedResourceValidation),
    )
    with pytest.raises(ReachedResourceValidation):
        namespace["register_all_serve"](controller)

    roles, colocate, _ = controller._validate_gpu_resources.call_args.args
    assert colocate is expected_colocate
    assert {str(role) for role, *_ in roles} == ({"sft", "actor", "rollout"} if with_rollout else {"sft", "actor"})

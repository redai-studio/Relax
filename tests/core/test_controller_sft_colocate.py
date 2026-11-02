# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Controller's SFT colocate decision without starting Ray
services."""

import ast
import builtins
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(params=[False, True], ids=["stdlib-strenum", "fallback-strenum"])
def registry_roles(request):
    # Execute the production enum definitions without importing backend components.
    path = Path(__file__).resolve().parents[2] / "relax/core/registry.py"
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        or (isinstance(node, ast.ImportFrom) and node.module == "enum")
        or (isinstance(node, ast.ClassDef) and node.name in {"ROLES", "ROLES_SFT_ONLY"})
    ]

    def import_enum(name, globals=None, locals=None, fromlist=(), level=0):
        if request.param and name == "enum" and "StrEnum" in fromlist:
            raise ImportError("Exercise the pre-Python-3.11 fallback")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {"__name__": "_test_registry_roles", "__builtins__": dict(vars(builtins), __import__=import_enum)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ROLES"], namespace["ROLES_SFT_ONLY"]


@pytest.mark.parametrize("with_rollout, expected_colocate", [(True, True), (False, False)])
def test_register_all_serve_sft_colocate_uses_optional_roles(
    monkeypatch, registry_roles, with_rollout, expected_colocate
):
    # Execute the actual method; only its external collaborators are replaced.
    # Stop at resource validation, before placement groups/services are created.
    path = Path(__file__).resolve().parents[2] / "relax/core/controller.py"
    tree = ast.parse(path.read_text())
    controller_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Controller")
    method = next(
        n for n in controller_class.body if isinstance(n, ast.FunctionDef) and n.name == "register_all_serve"
    )
    ROLES, ROLES_SFT_ONLY = registry_roles
    algo = {role: object() for role in [*ROLES_SFT_ONLY, ROLES.rollout]}
    registry = ModuleType("relax.core.registry")
    registry.resolve_algorithm = Mock(return_value=algo)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    namespace = {
        "validate_ppo_config": Mock(),
        "maybe_start_managed_opd_teacher": Mock(return_value=(None, None)),
        "resolve_sft_algo_key": Mock(return_value="sft"),
        "ALGOS": {"sft": algo},
        "process_role": Mock(return_value=ROLES_SFT_ONLY),
        "validate_sft_resource": Mock(),
        "register_extra_roles": Mock(return_value=[ROLES.rollout] if with_rollout else []),
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

    roles, colocate, pg_roles = controller._validate_gpu_resources.call_args.args
    registry.resolve_algorithm.assert_called_once_with(controller.config)
    namespace["_actor_rollout_pg_roles"].assert_called_once_with(controller.config)
    assert pg_roles == ["actor", "rollout"]
    assert colocate is expected_colocate
    assert {str(role) for role, *_ in roles} == ({"sft", "actor", "rollout"} if with_rollout else {"sft", "actor"})

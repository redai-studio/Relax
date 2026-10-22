# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the actual Bridge override loop without Megatron/CUDA imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("force_sparse", [False, True])
def test_bridge_forwards_qsa_force_sparse(force_sparse):
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model_provider.py"
    tree = ast.parse(source.read_text())
    statements = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"bridge_keys", "args_dict"}:
                statements[node.targets[0].id] = node
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "attr":
            if ast.unparse(node.iter) == "vars(provider)":
                statements["override_loop"] = node
    namespace = {
        "args": SimpleNamespace(qsa_force_sparse=force_sparse),
        "provider": SimpleNamespace(qsa_force_sparse=not force_sparse),
        "logger": Mock(),
        "is_kimi_k3": False,
    }
    body = [statements[key] for key in ("bridge_keys", "args_dict", "override_loop")]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["provider"].qsa_force_sparse is force_sparse

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep model operators separate from runtime mutation of third-party
methods."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "forward",
    "backward",
    "fuser_forward",
    "fuser_backward",
    "_make_fused_ops",
    "_checkpointed_forward",
    "_forward_mlp",
    "_flex_forward",
    "_id_sparse_forward",
    "_pool_keys",
    "_select_blocks",
    "get_query_key_value_tensors",
    "pre_gated_delta_rule",
    "gated_delta_rule",
    "process_mtp_loss",
    "_allreduce_non_tensor_model_parallel_grads",
    "_allreduce_layernorm_grads",
    "topk_routing_with_score_function",
}


def test_model_operators_do_not_replace_external_methods():
    violations = []
    for package in ("common", "qwen3_5", "qwen3_8"):
        for path in (ROOT / "relax/models" / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if "MethodType" in names:
                        violations.append((str(path.relative_to(ROOT)), node.lineno, "MethodType"))
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and target.attr in METHODS:
                            violations.append((str(path.relative_to(ROOT)), node.lineno, target.attr))
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in METHODS
                ):
                    violations.append((str(path.relative_to(ROOT)), node.lineno, node.args[1].value))
    assert not violations, violations


def test_backend_has_no_import_time_patch_installer():
    tree = ast.parse((ROOT / "relax/backends/megatron/__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "megatron_patch" for alias in node.names)
    assert not (ROOT / "relax/backends/megatron/megatron_patch").exists()
    assert not (ROOT / "relax/backends/megatron/kernels/qwen3_5").exists()

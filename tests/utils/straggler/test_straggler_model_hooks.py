# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static guards on the Megatron timer hook points.

There are exactly three places in ``relax/backends/megatron/model.py`` where
``config.timers`` is decided:

* ``setup_model_and_optimizer`` — the optimizer reads its own timers, so the
  profiler's object must be assigned here as well;
* ``train`` — the training ``TransformerConfig``;
* ``forward_only`` (the evaluation / log-prob path) — upstream deliberately
  keeps ``None`` there, and Task 11 does not change that.

The guards below exist because two of them are easy to break silently: passing
``timers=`` into the config constructor instead of assigning afterwards would
hand the profiler's CUDA events and threads to Megatron's attention/MoE modules
via ``copy.deepcopy(self.config)``, and instrumenting the evaluation path would
mix evaluation stages into the training windows the detector compares.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MODEL_PATH = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "model.py"

GATED_CALL = "get_straggler_timers()"
EXPECTED_SITES = {
    "setup_model_and_optimizer": GATED_CALL,
    "train": GATED_CALL,
    "forward_only": "None",
}


def parse_model() -> ast.Module:
    """Parse the Megatron backend module without importing it."""
    return ast.parse(MODEL_PATH.read_text(encoding="utf-8"))


def enclosing_function(tree: ast.Module, lineno: int) -> Optional[str]:
    """Return the innermost function name containing ``lineno``."""
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.lineno == lineno:
            current: Optional[ast.AST] = node
            while current is not None and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current = parents.get(current)
            return current.name if current is not None else None
    return None


def timer_assignment_sites(tree: ast.Module) -> List[Tuple[str, int, str]]:
    """Return ``(function, lineno, value_source)`` for every ``config.timers = ...``."""
    sites: List[Tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Attribute) and target.attr == "timers" for target in node.targets):
            continue
        function = enclosing_function(tree, node.lineno) or "<module>"
        sites.append((function, node.lineno, ast.unparse(node.value)))
    return sites


def test_model_decides_timers_in_exactly_three_places() -> None:
    tree = parse_model()

    sites = timer_assignment_sites(tree)

    assert {function for function, _, _ in sites} == set(EXPECTED_SITES)
    for function, _, value in sites:
        assert value == EXPECTED_SITES[function], f"{function} sets config.timers to {value}"


def test_profiler_timers_are_assigned_after_config_construction() -> None:
    """Megatron deepcopies the config while building attention/MoE modules."""
    tree = parse_model()
    sites = {function: lineno for function, lineno, value in timer_assignment_sites(tree) if value == GATED_CALL}

    assert sites, "expected the profiler's timers to be gated in"
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in sites:
            continue
        construction = [
            statement.lineno
            for statement in ast.walk(node)
            if isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "config" for target in statement.targets)
            and statement.lineno < sites[node.name]
        ]
        assert construction, f"{node.name} assigns config.timers before building the config"
        assert max(construction) < sites[node.name]


def test_timers_are_never_passed_into_a_config_constructor() -> None:
    """A constructor argument would reach the deepcopied attention/MoE
    configs."""
    tree = parse_model()

    offenders = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and any(keyword.arg == "timers" for keyword in node.keywords)
    ]

    assert offenders == []


def test_model_imports_the_profiler_factory() -> None:
    tree = parse_model()

    imports = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "relax.utils.straggler"
    ]

    assert imports == ["from relax.utils.straggler import get_straggler_timers"]


def test_evaluation_path_keeps_timers_disabled() -> None:
    """Evaluation stages must not leak into the training windows we compare."""
    tree = parse_model()
    sites = {function: value for function, _, value in timer_assignment_sites(tree)}

    assert sites["forward_only"] == "None"

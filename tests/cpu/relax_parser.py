# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build CPU fixtures from selected Relax CLI declarations."""

import argparse
import ast
import json
import math
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

from relax.algorithms.spec import list_algorithm_names
from relax.utils.tq.config import DEFAULT_TQ_RDMA_MODE


@cache
def parser_for_flags(
    *flags: str,
    source_file: str = "relax/utils/arguments.py",
    constants: tuple[str, ...] = (),
    converters: tuple[str, ...] = (),
) -> argparse.ArgumentParser:
    source = Path(__file__).resolve().parents[2] / source_file
    tree = ast.parse(source.read_text(encoding="utf-8"))
    bindings = {
        statement.targets[0].id: ast.literal_eval(statement.value)
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id in constants
    }
    assert len(bindings) == len(constants)
    declarations = {}
    preparser_declarations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"parser", "group", "temp_parser"}
        ):
            options = node.args
            target = preparser_declarations if node.func.value.id == "temp_parser" else declarations
        elif isinstance(node.func, ast.Name) and node.func.id == "reset_arg":
            options = node.args[1:]
            target = declarations
        else:
            continue
        for argument in options:
            value = argument.value if isinstance(argument, ast.Constant) else None
            if isinstance(argument, ast.Name):
                value = bindings.get(argument.id)
            if isinstance(value, str) and value in flags:
                assert value not in target, value
                target[value] = node
    declarations = {**preparser_declarations, **declarations}
    assert set(declarations) == set(flags), set(flags) - declarations.keys()
    names = set(converters)
    if any(isinstance(call.func, ast.Name) for call in declarations.values()):
        names.add("reset_arg")
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    parser = argparse.ArgumentParser(add_help=False)
    calls = [ast.Expr(node) for node in dict.fromkeys(declarations.values())]
    module = ast.fix_missing_locations(ast.Module(body=[*definitions, *calls], type_ignores=[]))
    namespace = {
        "parser": parser,
        "group": parser,
        "temp_parser": parser,
        "argparse": argparse,
        "json": json,
        "math": math,
        "list_algorithm_names": list_algorithm_names,
        "DEFAULT_TQ_RDMA_MODE": DEFAULT_TQ_RDMA_MODE,
    }
    exec(compile(module, str(source), "exec"), {**namespace, **bindings})
    return parser


def args_for_flags(*flags: str, **overrides: Any) -> argparse.Namespace:
    """Build a configuration from declarations and test overrides."""
    args = deepcopy(parser_for_flags(*flags).parse_args([]))
    vars(args).update(overrides)
    return args


MULTIMODAL_FLAGS = (
    "--image-max-token-num",
    "--image-min-token-num",
    "--image-resize-scale-factor",
    "--video-min-token-num",
    "--video-max-token-num",
    "--video-fps",
    "--video-fps-min-frames",
    "--video-fps-max-frames",
    "--frame-factor",
    "--audio-sample-rate",
)

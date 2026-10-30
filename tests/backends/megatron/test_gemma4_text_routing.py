# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise checkpoint routing without constructing a Ray/Megatron actor."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("config_name", ["processor_config.json", "preprocessor_config.json"])
@pytest.mark.parametrize(
    "model_type,mode,processor,expected",
    [
        ("gemma4", "text", True, False),
        ("gemma4", "vl", True, True),
        ("gemma4", None, True, True),
        ("qwen3_vl", "text", True, True),
        ("gemma4", "text", False, False),
        ("qwen3", None, False, False),
    ],
)
def test_checkpoint_routing_respects_explicit_gemma4_text_conversion(
    tmp_path, monkeypatch, config_name, model_type, mode, processor, expected
):
    if processor:
        (tmp_path / config_name).write_text("{}")
    if mode is None:
        monkeypatch.delenv("GEMMA4_CONVERSION_MODE", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", mode)
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    # Run the production routing assignments with only their lightweight inputs.
    # Importing the complete actor would require unrelated Ray/Megatron services.
    assignments = []
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"has_processor_config", "gemma4_text_conversion"}:
            assignments.append(node)
        elif isinstance(target, ast.Attribute) and target.attr == "is_vl_model":
            assignments.append(node)
    assert len(assignments) == 3
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))
    namespace = {"os": os, "args": args, "self": SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))}
    module = ast.Module(body=sorted(assignments, key=lambda node: node.lineno), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    assert args.is_vl_model is expected

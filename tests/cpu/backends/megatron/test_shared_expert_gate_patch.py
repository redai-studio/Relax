# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execute the image-built shared-gate publication method with CPU stubs."""

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("gate", [False, True])
def test_gate_publication_precedes_stream_wait(monkeypatch, gate, enabled):
    monkeypatch.setenv("RELAX_SFT_RESIDENT_PARAM_OVERLAP", "1" if enabled else "0")
    root = os.environ.get("COMMON_MCORE_SOURCE_DIR")
    if not root:
        pytest.skip("Set COMMON_MCORE_SOURCE_DIR to the fully patched MCore source")
    source = Path(root) / "megatron/core/transformer/moe/shared_experts.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SharedExpertMLP"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "wait_current_stream")
    readiness = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "shared_gate_param_readiness_version" for t in n.targets)
    )
    assert ast.literal_eval(readiness.value) == 1
    events = []
    utils = ModuleType("megatron.core.utils")
    utils.ensure_params_ready = lambda params: events.append(("publish", params))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    scope = {"torch": SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: "current"))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    instance = SimpleNamespace(
        use_shared_expert_gate=gate,
        gate_weight=object(),
        stream=SimpleNamespace(wait_stream=lambda stream: events.append(("wait", stream))),
    )
    scope["wait_current_stream"](instance)
    assert events == ([("publish", [instance.gate_weight])] if gate and enabled else []) + [("wait", "current")]

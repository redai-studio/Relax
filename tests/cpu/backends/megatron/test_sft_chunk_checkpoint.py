# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpointed output-head state restoration without a Ray runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.utils.env import Envs


@pytest.mark.parametrize("mode", ["sft", "cpt"])
@pytest.mark.parametrize("fail", [False, True])
def test_checkpointed_head_disables_fusion_and_restores_state(mode, fail):
    from contextlib import contextmanager

    tree = ast.parse(Path("relax/backends/megatron/model.py").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_bypass_output_layer")
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_bypass_output_layer"
    )
    guard = next(k.value for k in call.keywords if k.arg == "unfused_head_wgrad")
    enabled = eval(
        compile(ast.Expression(guard), "guard", "eval"),
        {
            "args": SimpleNamespace(sft_training_mode=mode, recompute_loss_function=True),
            "mtp_enabled": False,
        },
    )
    assert enabled

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 2))
            self.sequence_parallel = False
            self.gradient_accumulation_fusion = True

        def forward(self, value, **kwargs):
            assert not self.gradient_accumulation_fusion
            if fail:
                raise ValueError("head failure")
            return value @ self.weight.T, None

    head = Head()
    ns = {
        "contextmanager": contextmanager,
        "torch": torch,
        "Envs": Envs,
        "_find_lm_output_layer": lambda model: model,
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "head", "exec"), ns)
    value = torch.randn(2, 2, requires_grad=True)
    with ns["_bypass_output_layer"](head, unfused_head_wgrad=enabled) as forward:
        head(value)
        if fail:
            with pytest.raises(ValueError, match="head failure"):
                forward(value)
        else:
            result, _ = forward(value)
            result.sum().backward()
            assert head.weight.grad is not None
        assert head.gradient_accumulation_fusion
    assert "forward" not in head.__dict__

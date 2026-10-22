# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the patched GDN method without importing CUDA-dependent MCore."""

import ast
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[3]
_TARGET = "megatron/core/ssm/gated_delta_net/gdn.py"
_SOURCE = """class GatedDeltaNet:
    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        original_shape = x.shape
        y = self.out_norm(x.reshape(-1, x.shape[-1])).reshape(original_shape)

        # Output gate. In the fused pre-GDR path, gate is a strided view of
        # qkvzba's Z channels. Keep it 4-D so this pointwise operation reads Z
        # through its strides instead of reshape() materializing a copy.
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

"""


@pytest.mark.parametrize("activation", [torch.sigmoid, F.silu])
def test_qwen38_gdn_output_gate_matches_configured_activation(tmp_path: Path, activation) -> None:
    patch = (_ROOT / "docker/patch/megatron/20261016-qwen38-models.patch").read_text()
    start = patch.index(f"diff --git a/{_TARGET} b/{_TARGET}\n")
    section = patch[start:].split("\ndiff --git ", 1)[0]
    # This fixture covers the output gate only; other GDN methods have separate tests.
    header = section.split("@@", 1)[0]
    hunks = re.findall(r"@@[^\n]*\n(?:(?!@@)[\s\S])*", section)
    section = header + next(hunk for hunk in hunks if "self.gate_act_fn" in hunk)
    target = tmp_path / _TARGET
    target.parent.mkdir(parents=True)
    target.write_text(_SOURCE)
    subprocess.run(["git", "apply", "-"], input=section, text=True, cwd=tmp_path, check=True)
    method = ast.parse(target.read_text()).body[0].body[0]
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(target), "exec"), namespace)
    model = SimpleNamespace(out_norm=torch.nn.Identity(), act_fn=F.silu, gate_act_fn=activation)
    x = torch.tensor([[[[2.0, -3.0, 4.0]]]], requires_grad=True)
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]]]], requires_grad=True)
    output = namespace["_apply_gated_norm"](model, x, gate)
    expected = x * activation(gate)
    torch.testing.assert_close(output, expected)
    actual_grad = torch.autograd.grad(output.sum(), (x, gate), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (x, gate))
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the shipped GDN gate hunk with autograd on CPU."""

import ast
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
TARGET = "megatron/core/ssm/gated_delta_net/gdn.py"
SOURCE = """class GatedDeltaNet:
    def _compute_gates(self, A_log_local_cp, dt_bias_local_cp, batch, seq_len, *gate_feats):
        # ``gate_feats`` arrives in ``in_proj_split_names`` order: beta, then alpha.
        beta, alpha = gate_feats
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.float().sigmoid()
        return g, {"beta": beta.contiguous()}

    @jit_fuser
    def other(self):
        pass
"""


@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gdn_beta_preserves_configured_precision_and_gradient(tmp_path, fp32, dtype):
    patch = (ROOT / "docker/patch/megatron/20261016-qwen38-models.patch").read_text()
    section = patch.split(f"diff --git a/{TARGET} b/{TARGET}\n")[1].split("\ndiff --git ")[0]
    hunks = re.split(r"(?=^@@ )", section, flags=re.MULTILINE)
    hunk = next(h for h in hunks[1:] if "beta.float()" in h)
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE)
    subprocess.run(["git", "apply", "-"], input=hunks[0] + hunk, text=True, cwd=tmp_path, check=True)
    ns = {"torch": torch, "F": F, "jit_fuser": lambda fn: fn}
    exec(compile(ast.parse(target.read_text()), str(target), "exec"), ns)
    model = ns["GatedDeltaNet"]()
    model.config = SimpleNamespace(linear_attention_beta_in_fp32=fp32)
    beta = torch.tensor([-8.0, 0.0, 8.0], dtype=dtype, requires_grad=True)
    reference = beta.detach().clone().requires_grad_()
    _, gates = model._compute_gates(torch.zeros(3), torch.zeros(3), 1, 3, beta, torch.zeros(3))
    expected = (reference.float() if fp32 else reference).sigmoid()
    assert gates["beta"].dtype == expected.dtype
    torch.testing.assert_close(gates["beta"], expected, atol=0, rtol=0)
    gates["beta"].sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(beta.grad, reference.grad, atol=0, rtol=0)
    if fp32:
        assert beta.grad[-1] > 0


def test_projection_dtype_beta_is_opted_in_only_by_qwen38_providers():
    patch = (ROOT / "docker/patch/megatron/20261016-qwen38-models.patch").read_text()
    assert "+    linear_attention_beta_in_fp32: bool = True" in patch
    sections = [s for s in patch.split("diff --git ") if "provider.linear_attention_beta_in_fp32 = False" in s]
    assert len(sections) == 2
    for section in sections:
        assert "qwen4" in section.splitlines()[0].lower()

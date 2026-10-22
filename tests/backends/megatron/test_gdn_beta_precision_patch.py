# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Apply the image patch and verify both configured GDN beta precision
modes."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


_SOURCE = """class GatedDeltaNet:
    def _compute_gates(self, A_log_local_cp, dt_bias_local_cp, batch, seq_len, *gate_feats):
        '''Compute the per-head log-decay g and the write strength beta.'''
        # ``gate_feats`` arrives in ``in_proj_split_names`` order: beta, then alpha.
        beta, alpha = gate_feats
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.float().sigmoid()
        return g, {"beta": beta.contiguous()}

    @jit_fuser
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

    def _run_core(self, qkvzba):
        if True:
            if self.config.gdn_conv_pad_alignment is not None:
                raise ValueError(
                    "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                    "chunk-local causal-conv inputs can change later chunk numerics."
                )
            with torch._dynamo.config.patch(disable=True):
                query, key, value, gate, beta, g = self.pre_gated_delta_rule(
                    qkvzba,
                    batch,
                )

"""


@pytest.mark.parametrize("beta_in_fp32", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gdn_image_patch_preserves_configured_beta_values_and_gradients(
    tmp_path: Path, dtype: torch.dtype, beta_in_fp32: bool
) -> None:
    root = Path(__file__).resolve().parents[3]
    relative = "megatron/core/ssm/gated_delta_net/gdn.py"
    patch = (root / "docker/patch/megatron/20261016-qwen38-models.patch").read_text()
    start = patch.index(f"diff --git a/{relative} b/{relative}\n")
    section = patch[start:].split("\ndiff --git ", 1)[0] + "\n"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(_SOURCE)
    subprocess.run(["git", "apply", "-"], input=section, text=True, cwd=tmp_path, check=True)
    cls = ast.parse(target.read_text()).body[0]
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_gates")
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(target), "exec"), scope)
    beta = torch.linspace(-7, 7, 4112, dtype=dtype).reshape(257, 16).requires_grad_()
    alpha = torch.linspace(-2, 2, 4112, dtype=dtype).reshape(257, 16).requires_grad_()
    a_log = torch.linspace(-1, 1, 16, requires_grad=True)
    dt_bias = torch.linspace(-0.5, 0.5, 16, requires_grad=True)
    model = SimpleNamespace(config=SimpleNamespace(linear_attention_beta_in_fp32=beta_in_fp32))
    decay, features = scope["_compute_gates"](model, a_log, dt_bias, 1, 257, beta, alpha)
    expected_decay = -a_log.exp() * F.softplus(alpha.float() + dt_bias)
    expected_beta = (beta.float() if beta_in_fp32 else beta).sigmoid()
    assert features["beta"].dtype == expected_beta.dtype
    torch.testing.assert_close(features["beta"], expected_beta, rtol=0, atol=0)
    upstream = torch.linspace(-3, 2, beta.numel()).reshape_as(beta)
    actual_grads = torch.autograd.grad(
        (decay, features["beta"]), (beta, alpha, a_log, dt_bias), (upstream, upstream), retain_graph=True
    )
    expected_grads = torch.autograd.grad(
        (expected_decay, expected_beta),
        (beta, alpha, a_log, dt_bias),
        (upstream, upstream.to(expected_beta.dtype)),
        retain_graph=True,
    )
    torch.testing.assert_close(decay, expected_decay, rtol=0, atol=0)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if dtype == torch.bfloat16 and not beta_in_fp32:
        promoted = beta.float().sigmoid()
        promoted_grad = torch.autograd.grad(promoted, beta, upstream)[0]
        assert not torch.equal(promoted, features["beta"])
        assert not torch.equal(promoted_grad, actual_grads[0])

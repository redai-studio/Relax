# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verify both configured beta precision modes on the image's GDN
implementation."""

from inspect import unwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("beta_in_fp32", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gdn_preserves_configured_beta_values_and_gradients(dtype: torch.dtype, beta_in_fp32: bool) -> None:
    gdn = pytest.importorskip("megatron.core.ssm.gated_delta_net")
    beta = torch.linspace(-8, 8, 4112, dtype=dtype).reshape(257, 16).requires_grad_()
    alpha = torch.linspace(-2, 2, 4112, dtype=dtype).reshape(257, 16).requires_grad_()
    a_log = torch.linspace(-1, 1, 16, requires_grad=True)
    dt_bias = torch.linspace(-0.5, 0.5, 16, requires_grad=True)
    model = SimpleNamespace(config=SimpleNamespace(linear_attention_beta_in_fp32=beta_in_fp32))
    compute_gates = unwrap(gdn.GatedDeltaNet._compute_gates)
    decay, features = compute_gates(model, a_log, dt_bias, 1, 257, beta, alpha)
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
    if beta_in_fp32:
        assert actual_grads[0][-1, -1] > 0
    if dtype == torch.bfloat16 and not beta_in_fp32:
        promoted = beta.float().sigmoid()
        promoted_grad = torch.autograd.grad(promoted, beta, upstream)[0]
        assert not torch.equal(promoted, features["beta"])
        assert not torch.equal(promoted_grad, actual_grads[0])

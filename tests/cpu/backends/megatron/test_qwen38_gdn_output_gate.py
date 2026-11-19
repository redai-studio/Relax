# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify the installed GDN output gate and its gradients on CPU."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("activation", [torch.sigmoid, F.silu])
def test_qwen38_gdn_output_gate_matches_configured_activation(activation) -> None:
    from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet

    model = SimpleNamespace(out_norm=torch.nn.Identity(), act_fn=F.silu, gate_act_fn=activation)
    x = torch.tensor([[[[2.0, -3.0, 4.0]]]], requires_grad=True)
    gate = torch.tensor([[[[-2.0, 0.0, 2.0]]]], requires_grad=True)
    output = inspect.unwrap(GatedDeltaNet._apply_gated_norm)(model, x, gate)
    expected = x * activation(gate)
    torch.testing.assert_close(output, expected)
    actual_grad = torch.autograd.grad(output.sum(), (x, gate), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (x, gate))
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)

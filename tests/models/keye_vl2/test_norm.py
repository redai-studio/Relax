# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye RMSNorm requires CUDA")
@pytest.mark.parametrize("width", [128, 2048])
def test_rmsnorm_is_exact_across_token_counts_and_grad_modes(width):
    from relax.models.keye_vl2.norm import RMSNorm

    torch.manual_seed(2113)
    norm = RMSNorm(width, eps=1e-6, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.uniform_(0.1, 3.0)
    x = torch.randn(8192, width, device="cuda", dtype=torch.bfloat16)
    whole = norm(x)
    with torch.no_grad():
        for length in (1, 8, 127, 512, 2113, 8192):
            torch.testing.assert_close(norm(x[:length]), whole[:length], atol=0, rtol=0)
        torch.testing.assert_close(norm(x), whole, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye RMSNorm backward requires CUDA")
@pytest.mark.parametrize("width", [128, 2048])
def test_rmsnorm_gradients_match_torch_formula(width):
    from relax.models.keye_vl2.norm import RMSNorm

    torch.manual_seed(73)
    norm = RMSNorm(width, eps=1e-6, device="cuda", dtype=torch.float32)
    x = torch.randn(17, width, device="cuda", requires_grad=True)
    weight = norm.weight.detach().clone().requires_grad_()
    reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    actual = norm(x)
    dy = torch.randn_like(x)
    expected_dx, expected_dw = torch.autograd.grad(reference, (x, weight), dy)
    dx, dw = torch.autograd.grad(actual, (x, norm.weight), dy)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(dx, expected_dx, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(dw, expected_dw, atol=2e-6, rtol=2e-5)

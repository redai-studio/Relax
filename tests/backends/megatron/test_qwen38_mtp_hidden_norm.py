# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare the installed MCore MTP norm with SGLang's draft implementation."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MCore/TE import requires a CUDA runtime")
def test_qwen38_mtp_hidden_norm_matches_sglang_draft(dtype):
    hc = pytest.importorskip("megatron.core.transformer.hyper_connection")
    if not hasattr(hc, "GatedResidualMTPHiddenNorm"):
        pytest.skip("Requires the Qwen3.8 Megatron runtime")
    config = SimpleNamespace(hidden_size=8, num_residual_streams=4, params_dtype=dtype, sequence_parallel=True)
    model = hc.GatedResidualMTPHiddenNorm(config, hidden_size=8, eps=1e-6)
    layernorm = pytest.importorskip("sglang.srt.layers.layernorm")
    reference = layernorm.GemmaRMSNorm(32, eps=1e-6).to(dtype)
    gain = torch.linspace(-0.2, 0.3, 32, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(gain)
        reference.weight.copy_(gain)
    # Unequal stream scales distinguish one n*H variance from n independent H variances.
    base = torch.arange(1, 9, dtype=dtype).reshape(1, 1, 1, 8)
    scales = torch.tensor([0.1, 1.0, 4.0, 10.0], dtype=dtype).reshape(1, 1, 4, 1)
    inputs = (base * scales).expand(3, 2, 4, 8).clone().requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = model(inputs)
    expected = reference.forward_native(reference_inputs.flatten(-2)).reshape_as(inputs)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == inputs.shape and actual.dtype == inputs.dtype
    assert model.weight.sequence_parallel
    probe = torch.linspace(-1, 1, inputs.numel(), dtype=torch.float32).reshape_as(inputs)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6)

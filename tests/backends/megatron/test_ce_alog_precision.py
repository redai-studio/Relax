# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression coverage for the CE backport and FP32 GDN parameter storage."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="The CE backport requires an SM100 GPU and the CuTe runtime",
)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backport_first", [False, True])
def test_ce_dispatch_preserves_entrypoint_identity(enabled, backport_first):
    script = f"""
import importlib
if {backport_first!r}:
    importlib.import_module('relax.backends.megatron.kernels.linear_cross_entropy')
from relax.backends.megatron import fused_linear_ce
original = fused_linear_ce.linear_cross_entropy
from relax.backends.megatron.kernels import linear_cross_entropy as kernel
assert fused_linear_ce.linear_cross_entropy is original
backport_result, fallback_result = object(), object()
kernel.linear_cross_entropy = lambda *args, **kwargs: backport_result
fused_linear_ce._LinearCE.apply = lambda *args: fallback_result
fused_linear_ce._log_enabled = lambda *args: None
import torch
hidden = torch.ones(1)
expected = backport_result if {enabled!r} else fallback_result
assert fused_linear_ce.linear_cross_entropy(hidden, hidden, hidden, None) is expected
assert fused_linear_ce.linear_cross_entropy is original
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RELAX_SFT_FUSED_LINEAR_CE=str(int(enabled)))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def patched_ce(monkeypatch):
    from relax.backends.megatron import fused_linear_ce

    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    return fused_linear_ce.linear_cross_entropy


def _assert_gradient_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    relative_l2 = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-12)
    # The kernel uses BF16 dLogits/dWeight, unlike this independent FP64 oracle.
    assert relative_l2 < 0.008, f"gradient relative L2 error: {relative_l2.item()}"


@sm100
@pytest.mark.parametrize("accumulate", [False, True], ids=["autograd", "checkpoint-main-grad"])
def test_ce_patch_matches_fp64_loss_and_gradients(patched_ce, accumulate):
    generator = torch.Generator(device="cuda").manual_seed(63)
    weight = torch.nn.Parameter(torch.randn(3136, 256, device="cuda", generator=generator).bfloat16() / 16)
    weight.main_grad = torch.full_like(weight, 0.125, dtype=torch.float32)
    initial = weight.main_grad.clone()
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = True
    expected_total = torch.zeros_like(weight, dtype=torch.float64)

    # Cross the default 3072-vocabulary split and reuse compilation for a tail batch.
    for rows in (129, 17):
        hidden = torch.randn(rows, 256, device="cuda", generator=generator).bfloat16().requires_grad_()
        labels = torch.randint(3136, (rows,), device="cuda", generator=generator)
        labels[:5] = torch.tensor([0, 3071, 3072, 3135, -100], device="cuda")
        upstream = torch.linspace(-0.7, 1.3, rows, device="cuda")
        upstream[::7] = 0
        x64 = hidden.detach().double().requires_grad_()
        w64 = weight.detach().double().requires_grad_()
        expected = F.cross_entropy(F.linear(x64, w64), labels, reduction="none")
        dx64, dw64 = torch.autograd.grad(expected, (x64, w64), upstream.double())

        def loss_fn(value):
            return patched_ce(value, weight, labels, None, accumulate=accumulate)[0]

        if accumulate:
            actual = checkpoint(loss_fn, hidden, use_reentrant=True)
            actual.backward(upstream)
            dx = hidden.grad
            expected_total.add_(dw64)
            _assert_gradient_close(weight.main_grad - initial, expected_total)
            assert weight.main_grad.dtype == torch.float32
            assert weight.grad_added_to_main_grad
            assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0
        else:
            actual, metric_logits = patched_ce(hidden, weight, labels, None, return_logits=True)
            dx, dw = torch.autograd.grad(actual, (hidden, weight), upstream)
            _assert_gradient_close(dw, dw64)
            assert not metric_logits.requires_grad
            torch.testing.assert_close(metric_logits, F.linear(hidden, weight), atol=0, rtol=0)
            torch.testing.assert_close(weight.main_grad, initial, atol=0, rtol=0)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=0)
        _assert_gradient_close(dx, dx64)
        assert actual[4] == 0 and torch.count_nonzero(dx[4]) == 0


def test_ce_patch_preserves_nonunit_temperature_fallback(monkeypatch):
    from relax.backends.megatron import fused_linear_ce

    calls = []
    result = (object(), object())

    def original(*args):
        calls.append(args)
        return result

    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    monkeypatch.setattr(fused_linear_ce._LinearCE, "apply", original)
    monkeypatch.setattr(fused_linear_ce, "_log_enabled", lambda *args: None)
    hidden = torch.ones(1, requires_grad=True)
    hidden.main_grad = torch.zeros(1)
    args = (hidden, hidden, object(), object())
    assert fused_linear_ce.linear_cross_entropy(*args, temperature=0.7, accumulate=True, return_logits=True) is result
    assert calls == [(*args, 1.0 / 0.7, True, True)]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_gdn_a_log_preserves_fp32_checkpoint_through_bf16_wrapper(device):
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.module import Float16Module

    from relax.backends.megatron.model_provider import configure_gdn_a_log_fp32

    config = SimpleNamespace(
        params_dtype=torch.bfloat16, fp16=False, bf16=True, virtual_pipeline_model_parallel_size=None
    )
    # Use the installed GDN type without allocating its unrelated projections.
    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.A_log = torch.nn.Parameter(torch.zeros(16, device=device, dtype=torch.bfloat16))
    layer.A_log.tensor_model_parallel = True
    layer.A_log.partition_dim = 0
    layer.dt_bias = torch.nn.Parameter(torch.ones(16, device=device))
    model = torch.nn.Sequential(layer)
    model.config = config
    parameter = layer.A_log
    configure_gdn_a_log_fp32(model)
    configure_gdn_a_log_fp32(model)
    wrapped = Float16Module(config, model)
    assert wrapped.module[0].A_log is parameter
    assert parameter.dtype == torch.float32 and parameter.keep_in_fp32
    assert parameter.tensor_model_parallel and parameter.partition_dim == 0
    assert layer.dt_bias.dtype == torch.bfloat16

    checkpoint_a = torch.linspace(-2.012345, 2.123456, 16, device=device)
    assert not torch.equal(checkpoint_a, checkpoint_a.bfloat16().float())
    wrapped.load_state_dict({"0.A_log": checkpoint_a, "0.dt_bias": torch.ones(16, device=device)})
    torch.testing.assert_close(parameter, checkpoint_a, atol=0, rtol=0)
    parameter.exp().sum().backward()
    assert parameter.grad.dtype == torch.float32
    torch.testing.assert_close(parameter.grad, checkpoint_a.exp(), atol=0, rtol=0)

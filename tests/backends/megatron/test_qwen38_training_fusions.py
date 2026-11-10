# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Numerical contracts for residual writes and packed PLE convolution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "relax/models/qwen3_8/ops"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Fusion parity requires CUDA and Triton")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PATCHES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_combine_patch_bias_fallback_and_idempotence(monkeypatch, device):
    patch = _load("hc_fused_combine")
    residual = torch.ones(3, 1, 20, device=device)
    gates, output = torch.ones(3, 1, 4, device=device), torch.ones(3, 1, 5, device=device)
    bias = torch.ones(5, device=device)
    assert not patch.supported(residual, gates, output, bias, 4)


@pytest.mark.parametrize("mode", ["cpu", "cp", "streamed"])
def test_ple_install_preserves_native_fallback(monkeypatch, mode):
    from megatron.core import parallel_state

    patch = _load("ple_fused_conv")
    monkeypatch.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "1")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1" if mode == "streamed" else "0")
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 2 if mode == "cp" else 1)
    hidden = torch.ones(3, 1, 20) if mode == "cpu" else SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    assert not patch.supported(SimpleNamespace(), hidden)


@gpu
@pytest.mark.parametrize("n,channels,rows,batch", [(1, 257, 17, 2), (4, 257, 23, 2), (4, 2560, 11, 1), (4, 257, 0, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reentrant", [None, True])
def test_combine_forward_three_gradients_and_checkpoint(n, channels, rows, batch, dtype, reentrant):
    ops = _load("hc_combine")
    torch.manual_seed(45)
    # Sliced tensors exercise noncontiguous storage and view backward.
    residual = torch.randn(rows * 2, batch, n * channels, device="cuda", dtype=dtype)[::2].requires_grad_()
    output = torch.randn(rows * 2, batch, channels, device="cuda", dtype=dtype)[::2].requires_grad_()
    gates = torch.randn(rows, batch, n, device="cuda", requires_grad=True)
    upstream = torch.randn_like(residual)
    reference = residual.float().unflatten(-1, (n, channels)) + gates.unsqueeze(-1) * output.float().unsqueeze(-2)
    reference = reference.flatten(-2).to(dtype)
    expected = torch.autograd.grad(reference, (residual, output, gates), upstream)
    fn = lambda r, y, a: ops.combine(r, y, a, n)
    actual = (
        fn(residual, output, gates)
        if reentrant is None
        else checkpoint(fn, residual, output, gates, use_reentrant=reentrant)
    )
    actual_grads = torch.autograd.grad(actual, (residual, output, gates), upstream) if reentrant is None else None
    if reentrant is not None:
        actual.backward(upstream)
        actual_grads = residual.grad, output.grad, gates.grad
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, want in zip(actual_grads, expected, strict=True):
        torch.testing.assert_close(
            got.float(), want.float(), rtol=0.02 if dtype == torch.bfloat16 else 2e-5, atol=2e-5
        )


def _conv_reference(extended, gated, weight, positions, dilation):
    # Explicit document-aware indexing, independent of the fused helper's slicing.
    rows, batch, channels = gated.shape
    kernel = weight.shape[-1]
    halo = (kernel - 1) * dilation
    outputs = []
    for row in range(rows):
        value = torch.zeros(batch, channels, dtype=torch.float32, device=gated.device)
        for tap in range(kernel):
            offset = (kernel - 1 - tap) * dilation
            source = extended[halo + row - offset].float()
            source = torch.where((positions[row] >= offset).unsqueeze(-1), source, 0.0)
            value = value + source * weight[:, 0, tap].to(gated.dtype).float()
        outputs.append(value.to(gated.dtype))
    conv = torch.stack(outputs)
    return gated + F.silu(conv.float()).to(gated.dtype)


@pytest.mark.parametrize("kernel,dilation", [(1, 1), (3, 2), (4, 3)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_ple_conv_packed_boundaries_rounding_and_gradients(kernel, dilation, device):
    patch = _load("ple_fused_conv")
    torch.manual_seed(52)
    rows, batch, channels = 13, 2, 32
    halo = (kernel - 1) * dilation
    normed = torch.randn(rows, batch, channels, device=device, dtype=torch.bfloat16, requires_grad=True)
    gated = torch.randn_like(normed, requires_grad=True)
    weight = torch.randn(channels, 1, kernel, device=device, dtype=torch.float32, requires_grad=True)
    # Different document starts in the two batch elements.
    positions = torch.stack((torch.arange(rows) % 5, torch.arange(rows) % 7), 1).to(device)
    extended = F.pad(normed, (0, 0, 0, 0, halo, 0))
    expected = _conv_reference(extended, gated, weight, positions, dilation)
    fn = patch._compiled_conv() if device == "cuda" else patch.convolution_values
    actual = fn(extended, normed, gated, weight, positions, dilation)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (normed, gated, weight), upstream, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (normed, gated, weight), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got.float(), want.float(), rtol=0.02, atol=0.04)


@gpu
@pytest.mark.parametrize("frozen", [False, True])
def test_ple_real_forward_install_and_parameter_gradients(monkeypatch, frozen):
    from megatron.core import parallel_state
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding

    torch.manual_seed(58)
    monkeypatch.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "1")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    original = PerLayerEmbedding.forward
    # Use the genuine MCore forward/conv methods without allocating a checkpoint's
    # large pinned embedding table. Only the table lookup is replaced.
    model = PerLayerEmbedding.__new__(PerLayerEmbedding)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.tp_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    model.n, model.hidden_size, model.norm_eps = 4, 64, 1e-6
    model.conv_kernel_size, model.conv_dilation, model.halo = 3, 2, 4
    model.ple_embedding = torch.nn.Embedding(17, 32, device="cuda", dtype=torch.bfloat16)
    model.key_proj = torch.nn.Linear(32, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    model.value_proj = torch.nn.Linear(32, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1)
        setattr(model, name, norm)
    model.conv1d = torch.nn.Module()
    model.conv1d.weight = torch.nn.Parameter(torch.randn(256, 1, 3, device="cuda", dtype=torch.bfloat16) * 0.1)
    model._ngram_ids = torch.randint(17, (13, 2), device="cuda")
    positions = torch.stack((torch.arange(13) % 5, torch.arange(13) % 7), 1).cuda()
    model._position_in_segment = model._position_in_sequence = positions.T.contiguous()
    model._cp_packed_seq_params = None
    if frozen:
        model.requires_grad_(False)
    hidden = torch.randn(26, 2, 256, device="cuda", dtype=torch.bfloat16)[::2].requires_grad_()
    params = (hidden, *(p for p in model.parameters() if p.requires_grad))
    with monkeypatch.context() as context:
        context.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "0")
        expected = original(model, hidden)
    upstream = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, params, upstream)
    patch = _load("ple_fused_conv")
    actual = patch.fused_forward(model, hidden)
    actual_grads = torch.autograd.grad(actual, params, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got.float(), want.float(), rtol=0.02, atol=0.04)

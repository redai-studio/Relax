# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in numerical gates inside the built 280G Blackwell image; no cluster
submission."""

import os

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    os.environ.get("RELAX_RUN_BLACKWELL_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Requires an explicitly selected 280G Blackwell GPU image (RELAX_RUN_BLACKWELL_GPU_TESTS=1)",
)


@pytest.mark.parametrize("lengths", [(37, 80), (2049, 4097)])
def test_fla_packed_convolution_forward_and_backward(lengths):
    from fla.modules.conv import causal_conv1d

    assert torch.cuda.get_device_capability() == (10, 3), "Run this gate on 280G Blackwell"
    torch.manual_seed(42)
    inputs = [
        torch.randn(1, sum(lengths), 64, device="cuda"),
        torch.randn(64, 4, device="cuda"),
        torch.randn(64, device="cuda"),
    ]
    x, weight, bias = [t.clone().requires_grad_() for t in inputs]
    cu_cpu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    output, _ = causal_conv1d(x, weight, bias, activation="silu", cu_seqlens=cu_cpu.cuda(), cu_seqlens_cpu=cu_cpu)
    reference_inputs = [t.clone().requires_grad_() for t in inputs]
    xr, wr, br = reference_inputs
    outputs = []
    for seq in xr.split(lengths, dim=1):
        conv = F.conv1d(seq.transpose(1, 2), wr[:, None, :], br, padding=3, groups=64)
        outputs.append(F.silu(conv[:, :, : seq.size(1)]).transpose(1, 2))
    reference = torch.cat(outputs, dim=1)
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    reference.backward(gradient)
    for actual, expected in zip((x, weight, bias), reference_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-3, atol=5e-3)


def test_te_cross_entropy_forward_and_backward():
    from relax.utils.training.fused_ce import te_cross_entropy

    assert torch.cuda.get_device_capability() == (10, 3), "Run this gate on 280G Blackwell"
    torch.manual_seed(42)
    logits = torch.randn(37, 1, 8192, device="cuda", requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.randint(8192, (1, 64), device="cuda")[:, :37].transpose(0, 1)
    actual = te_cross_entropy(logits, labels)
    expected = F.cross_entropy(reference.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    actual.mean().backward()
    expected.mean().backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-4, atol=1e-6)


def test_qwen38_native_router_checkpoint_matches_sorted_forward_and_gradients(monkeypatch):
    import inspect

    from megatron.core.transformer.moe import moe_utils
    from torch.utils.checkpoint import checkpoint

    native = inspect.unwrap(moe_utils.topk_routing_with_score_function)
    if "qb_histogram" not in inspect.signature(native).parameters:
        pytest.skip("Requires the unified MCore native replay routing API")
    assert torch.cuda.get_device_capability() == (10, 3), "Run this gate on 280G Blackwell"
    assert "use_sorted_topk" in inspect.getsource(native), "Requires rebuilt static model operators"
    monkeypatch.setenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", "1")
    wrapped = native
    torch.manual_seed(42)
    logits = torch.randn(257, 512, device="cuda", requires_grad=True)
    weights = torch.randn_like(logits)
    expected_probs, expected_map = native(logits, 10)
    with torch.no_grad():
        actual_probs, actual_map = wrapped(logits.detach(), 10)
    torch.testing.assert_close(actual_probs, expected_probs.detach(), rtol=0, atol=0)
    torch.testing.assert_close(actual_map, expected_map, rtol=0, atol=0)
    (expected_probs * weights).sum().backward()
    replay_logits = logits.detach().clone().requires_grad_()
    loss = checkpoint(lambda value: (wrapped(value, 10)[0] * weights).sum(), replay_logits, use_reentrant=True)
    loss.backward()
    torch.testing.assert_close(replay_logits.grad, logits.grad, rtol=0, atol=0)

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


@pytest.mark.parametrize("lengths", [(64, 64), (37, 80), (257, 768, 3071)])
def test_fla_packed_gdn_backward_on_misidentified_sm103(monkeypatch, lengths):
    from fla.ops.common import chunk_o
    from fla.ops.common.backends.tilelang import TileLangBackend
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule

    from relax.backends.megatron.compat import patch_fla_gdn_backward_for_sm103

    monkeypatch.setattr(chunk_o, "IS_NVIDIA_HOPPER", chunk_o.IS_NVIDIA_HOPPER)
    monkeypatch.setattr(TileLangBackend, "chunk_bwd_dqkwg_verifier", TileLangBackend.chunk_bwd_dqkwg_verifier)
    if not patch_fla_gdn_backward_for_sm103():
        pytest.skip("Requires the affected FLA 0.5.2 SM103 runtime")

    rng = torch.Generator(device="cuda").manual_seed(1234)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    tokens = boundaries[-1]
    inputs = {
        name: F.normalize(torch.randn(1, tokens, 8, 128, device="cuda", generator=rng), dim=-1)
        .repeat_interleave(2, dim=2)
        .bfloat16()
        .requires_grad_()
        for name in ("q", "k")
    }
    inputs["v"] = torch.randn(1, tokens, 16, 128, device="cuda", dtype=torch.bfloat16, generator=rng).requires_grad_()
    inputs["g"] = (-F.softplus(torch.randn(1, tokens, 16, device="cuda", generator=rng))).requires_grad_()
    inputs["beta"] = torch.randn(1, tokens, 16, device="cuda", generator=rng).sigmoid().requires_grad_()
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    output, _ = chunk_gated_delta_rule(**inputs, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False)
    grad_output = torch.randn(output.shape, device="cuda", dtype=output.dtype, generator=rng) / 256
    gradients = torch.autograd.grad(output, tuple(inputs.values()), grad_output)

    for start, end in zip(boundaries, boundaries[1:]):
        sample = (slice(None), slice(start, end), slice(0, 1))
        reference_inputs = {name: tensor[sample].detach().float().requires_grad_() for name, tensor in inputs.items()}
        reference_output, _ = torch_chunk_gated_delta_rule(**reference_inputs)
        reference_gradients = torch.autograd.grad(
            reference_output, tuple(reference_inputs.values()), grad_output[sample].float()
        )
        for name, actual, expected in zip(
            ("output", *inputs), (output, *gradients), (reference_output, *reference_gradients)
        ):
            relative_error = (actual[sample].detach().float() - expected.detach()).norm() / expected.detach().norm()
            assert relative_error < 0.02, f"{name} relative L2 error for sequence {start}:{end}: {relative_error}"


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

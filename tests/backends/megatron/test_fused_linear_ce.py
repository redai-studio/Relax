# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-row CE gradients and FP32 head accumulation against an independent dense
head."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Fused linear CE requires the Blackwell CuTe runtime",
)


@pytest.mark.parametrize("mode", [None, "disable_grad_reduce", "explicit_expert_comm", "is_expert", "hook", "gather"])
def test_fused_ce_rejects_heads_with_different_communication_or_hooks(monkeypatch, mode):
    from megatron.core.tensor_parallel import layers

    from relax.backends.megatron.fused_linear_ce import supported

    class Column(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = None
            self.gather_output = mode == "gather"
            self.config = SimpleNamespace(defer_embedding_wgrad_compute=False)
            if mode in ("disable_grad_reduce", "explicit_expert_comm", "is_expert"):
                setattr(self, mode, True)
            if mode == "hook":
                self.register_forward_pre_hook(lambda _module, values: values)

    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    monkeypatch.setattr(layers, "ColumnParallelLinear", Column)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    device = torch.device("cuda:0")
    hidden = SimpleNamespace(is_cuda=True, device=device, shape=(8, 1, 64))
    weight = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=2, shape=(512, 64), is_contiguous=lambda: True)
    assert supported(Column(), hidden, weight) is (mode is None)


@gpu
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("frozen", ["none", "hidden", "weight"])
def test_linear_ce_per_row_loss_gradients_and_metrics(temperature, frozen):
    from relax.backends.megatron.fused_linear_ce import linear_cross_entropy

    torch.manual_seed(63)
    hidden = torch.randn(23, 1, 64, device="cuda", dtype=torch.bfloat16).requires_grad_(frozen != "hidden")
    weight = (torch.randn(520, 64, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(frozen != "weight")
    labels = torch.randint(0, weight.shape[0], (23, 1), device="cuda")
    labels[3] = -100
    upstream = torch.rand(23, 1, device="cuda")
    upstream[::3] = 0
    logits = F.linear(hidden, weight)
    expected = F.cross_entropy(logits.flatten(0, 1).float() / temperature, labels.flatten(), reduction="none").view_as(
        labels
    )
    params = tuple(p for p in (hidden, weight) if p.requires_grad)
    reference = torch.autograd.grad((expected * upstream).sum(), params)
    actual, metric_logits = linear_cross_entropy(
        hidden, weight, labels, None, temperature=temperature, return_logits=True
    )
    got = torch.autograd.grad((actual * upstream).sum(), params)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    # MTP uses temperature=1, so its exact BF16 first-index argmax is preserved.
    if temperature == 1:
        torch.testing.assert_close(metric_logits, logits.flatten(0, 1), atol=0, rtol=0)
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a.float(), b.float(), atol=0.004, rtol=0.04)
        assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12) < 0.008


@gpu
def test_linear_ce_fp32_main_grad_accumulates_with_checkpoint():
    from torch.utils.checkpoint import checkpoint

    from relax.backends.megatron.fused_linear_ce import linear_cross_entropy

    torch.manual_seed(71)
    hidden = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(512, 64, device="cuda", dtype=torch.bfloat16) * 0.1)
    weight.main_grad = torch.randn_like(weight, dtype=torch.float32) * 0.01
    initial = weight.main_grad.clone()
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = True
    expected_dw = torch.zeros_like(weight.main_grad)
    for rows in (17, 9):
        labels = torch.randint(0, weight.shape[0], (rows,), device="cuda")
        mask = torch.rand(rows, device="cuda")
        logits = F.linear(hidden[:rows], weight)
        loss = F.cross_entropy(logits.float(), labels, reduction="none")
        dlogits = torch.autograd.grad((loss * mask).sum(), logits)[0]
        expected_dw.add_(torch.mm(dlogits.T, hidden[:rows].detach(), out_dtype=torch.float32))

        def fn(value):
            return linear_cross_entropy(value, weight, labels, None, accumulate=True)[0]

        result = checkpoint(fn, hidden[:rows], use_reentrant=True)
        (result * mask).sum().backward()
    assert weight.grad_added_to_main_grad
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0
    got = weight.main_grad - initial
    torch.testing.assert_close(got, expected_dw, atol=0.004, rtol=0.04)
    assert (got - expected_dw).norm() / expected_dw.norm() < 0.008


@gpu
def test_linear_ce_production_head_dimensions_retain_no_vocab_activations():
    from relax.backends.megatron.fused_linear_ce import linear_cross_entropy

    torch.manual_seed(77)
    rows, hidden_size, local_vocab = 17, 2560, 124160
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(local_vocab, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02)
    labels = torch.randint(local_vocab, (rows,), device="cuda")
    labels[-1] = local_vocab - 1
    upstream = torch.rand(rows, device="cuda")
    upstream[::3] = 0
    expected = F.cross_entropy(F.linear(hidden, weight).float(), labels, reduction="none")
    reference = torch.autograd.grad((expected * upstream).sum(), (hidden, weight))
    saved = []

    def pack(tensor):
        saved.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual, logits = linear_cross_entropy(hidden, weight, labels, None)
    assert logits is None and torch.Size((rows, local_vocab)) not in saved
    gradients = torch.autograd.grad((actual * upstream).sum(), (hidden, weight))
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for got, want in zip(gradients, reference, strict=True):
        assert (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12) < 0.008

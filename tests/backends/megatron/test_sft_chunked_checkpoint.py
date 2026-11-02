# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Chunked SFT must bound saved vocab activations and preserve backward."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import loss as loss_module
except (ImportError, AttributeError) as exc:
    pytest.skip(f"Requires the pinned Megatron runtime: {exc}", allow_module_level=True)


class _MainGradLinear(torch.autograd.Function):
    """Exercise the fused weight-gradient accumulation contract on CPU."""

    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        return F.linear(hidden, weight)

    @staticmethod
    def backward(ctx, grad):
        hidden, weight = ctx.saved_tensors
        weight.main_grad.add_(grad.flatten(0, -2).float().T @ hidden.flatten(0, -2).float())
        return grad @ weight, torch.zeros_like(weight)


def _setup(monkeypatch, labels, chunk_size, temperature):
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module, "validate_opd_topk_gather", lambda *_: None)
    monkeypatch.setattr(loss_module, "get_responses", lambda hidden, **_: iter([(hidden.squeeze(0), labels)]))
    return Namespace(
        sft_logits_chunk_size=chunk_size,
        rollout_temperature=temperature,
        entropy_coef=0,
        allgather_cp=False,
        recompute_loss_function=True,
    )


def _run(args, hidden, weight, labels, *, main_grad=False):
    def head(values):
        output = _MainGradLinear.apply(values, weight) if main_grad else F.linear(values, weight)
        return output, None

    _, result = loss_module.get_log_probs_and_entropy(
        hidden.unsqueeze(0),
        args=args,
        unconcat_tokens=[labels],
        total_lengths=[len(labels)],
        response_lengths=[len(labels)],
        with_entropy=False,
        lm_head_forward=head,
    )
    return result["log_probs"][0]


@pytest.mark.parametrize("frozen_hidden", [False, True])
def test_sft_fused_ce_keeps_response_rows_temperature_without_duplicate_recompute(monkeypatch, frozen_hidden):
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    torch.manual_seed(36)
    hidden = torch.randn(13, 9, requires_grad=not frozen_hidden)
    weight = torch.randn(31, 9, requires_grad=True)
    labels = torch.randint(0, 31, (13,))
    args = _setup(monkeypatch, labels, 5, 0.7)
    calls = []

    def head(_):
        raise AssertionError("The dense head must not run when fused CE is supported")

    def fused(values, targets, temperature):
        calls.append((len(values), temperature))
        return F.cross_entropy(F.linear(values.squeeze(1), weight) / temperature, targets, reduction="none").unsqueeze(
            1
        )

    head._relax_fused_linear_ce = fused
    _, output = loss_module.get_log_probs_and_entropy(
        hidden.unsqueeze(0),
        args=args,
        unconcat_tokens=[labels],
        total_lengths=[13],
        response_lengths=[13],
        with_entropy=False,
        lm_head_forward=head,
    )
    actual = -output["log_probs"][0]
    expected = F.cross_entropy(F.linear(hidden, weight) / 0.7, labels, reduction="none")
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    params = (weight,) if frozen_hidden else (hidden, weight)
    reference = torch.autograd.grad(expected.sum(), params)
    for got, want in zip((p.grad for p in params), reference, strict=True):
        torch.testing.assert_close(got, want)
    assert calls[:3] == [(5, 0.7), (5, 0.7), (3, 0.7)]
    assert len(calls) == 3  # The fused Function handles its own recomputation.


@pytest.mark.parametrize("chunk_size", [1, 7, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("frozen_hidden", [False, True])
def test_sft_chunk_checkpoint_loss_and_gradients(monkeypatch, chunk_size, temperature, frozen_hidden):
    torch.manual_seed(36)
    hidden = torch.randn(23, 9, requires_grad=not frozen_hidden)
    weight = torch.randn(31, 9, requires_grad=True)
    labels = torch.randint(0, 31, (23,))
    mask = (torch.arange(23) % 3 != 0).float()
    args = _setup(monkeypatch, labels, chunk_size, temperature)
    expected = F.cross_entropy(F.linear(hidden, weight) / temperature, labels, reduction="none")
    params = (weight,) if frozen_hidden else (hidden, weight)
    expected_grad = torch.autograd.grad((expected * mask).sum(), params)
    actual = -_run(args, hidden, weight, labels)
    (actual * mask).sum().backward()
    actual_grad = tuple(param.grad for param in params)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, atol=3e-6, rtol=3e-5)


def test_sft_chunk_checkpoint_retains_no_vocab_activations(monkeypatch):
    torch.manual_seed(37)
    hidden = torch.randn(257, 9, requires_grad=True)
    weight = torch.randn(127, 9, requires_grad=True)
    labels = torch.randint(0, 127, (257,))
    args = _setup(monkeypatch, labels, 16, 1.0)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = _run(args, hidden, weight, labels)
    # The old forward-only chunking retains a [chunk, vocab] softmax for
    # every chunk. Checkpoint inputs are only hidden states and token ids.
    assert not any(shape and shape[-1] == 127 for shape in saved_shapes), saved_shapes
    assert sum(torch.tensor(shape).prod().item() for shape in saved_shapes) <= hidden.numel() + labels.numel()
    result.sum().backward()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()


def test_sft_chunk_checkpoint_accumulates_main_grad_once(monkeypatch):
    torch.manual_seed(38)
    hidden = torch.randn(23, 9, requires_grad=True)
    weight = torch.randn(31, 9, requires_grad=True)
    weight.main_grad = torch.zeros_like(weight)
    labels = torch.randint(0, 31, (23,))
    args = _setup(monkeypatch, labels, 7, 1.0)
    expected = F.cross_entropy(F.linear(hidden, weight), labels, reduction="sum")
    grad_hidden, grad_weight = torch.autograd.grad(expected, (hidden, weight))
    actual = -_run(args, hidden, weight, labels, main_grad=True).sum()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, grad_hidden, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(weight.main_grad, grad_weight, atol=3e-6, rtol=3e-5)


def test_sft_chunk_checkpoint_eval_needs_no_grad(monkeypatch):
    torch.manual_seed(39)
    hidden, weight = torch.randn(23, 9), torch.randn(31, 9)
    labels = torch.randint(0, 31, (23,))
    args = _setup(monkeypatch, labels, 7, 1.0)
    with torch.no_grad():
        actual = -_run(args, hidden, weight, labels)
    expected = F.cross_entropy(F.linear(hidden, weight), labels, reduction="none")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("recompute_loss_function_use_reentrant", False),
        ("overlap_grad_reduce", True),
        ("defer_embedding_wgrad_compute", True),
    ],
)
def test_sft_chunk_checkpoint_rejects_unsupported_backward_mode(monkeypatch, flag, value):
    labels = torch.zeros(3, dtype=torch.long)
    args = _setup(monkeypatch, labels, 2, 1.0)
    setattr(args, flag, value)
    with pytest.raises(ValueError, match="Chunked SFT recomputation requires"):
        _run(args, torch.zeros(3, 4, requires_grad=True), torch.ones(7, 4, requires_grad=True), labels)


@pytest.mark.parametrize("mask_kind", ["sparse", "fractional", "all", "zero", "one"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("frozen_hidden", [False, True])
def test_selective_logits_loss_main_grad_and_work(monkeypatch, mask_kind, enabled, frozen_hidden):
    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", str(int(enabled)))
    torch.manual_seed(71)
    hidden = torch.randn(23, 9, requires_grad=not frozen_hidden)
    weight = torch.randn(31, 9, requires_grad=True)
    weight.main_grad = torch.zeros_like(weight)
    labels = torch.randint(0, 31, (23,))
    mask = (torch.arange(23) % 3 != 0).float()
    if mask_kind == "fractional":
        mask *= torch.linspace(0.1, 0.9, 23)
    elif mask_kind == "all":
        mask.fill_(1)
    elif mask_kind == "zero":
        mask.zero_()
    elif mask_kind == "one":
        mask.zero_()[7] = 1
    indices = mask.nonzero(as_tuple=True)[0]
    args = _setup(monkeypatch, labels, 7, 0.7)
    expected = F.cross_entropy(F.linear(hidden, weight) / 0.7, labels, reduction="none")
    params = (weight,) if frozen_hidden else (hidden, weight)
    expected_grads = torch.autograd.grad((expected * mask).sum(), params)
    calls = []

    def head(values):
        calls.append(values.shape[0])
        return _MainGradLinear.apply(values, weight), None

    _, result = loss_module.get_log_probs_and_entropy(
        hidden.unsqueeze(0),
        args=args,
        unconcat_tokens=[labels],
        total_lengths=[23],
        response_lengths=[23],
        with_entropy=False,
        lm_head_forward=head,
        selective_logit_rows=[(23, indices)],
    )
    actual = -result["log_probs"][0]
    torch.testing.assert_close(actual * mask, expected * mask)
    forward_rows = sum(calls)
    wanted_rows = indices.numel() if enabled and indices.numel() else 23
    assert forward_rows == wanted_rows
    (actual * mask).sum().backward()
    assert sum(calls) == 2 * wanted_rows  # checkpoint recomputes exactly the selected head rows
    torch.testing.assert_close(weight.main_grad, expected_grads[-1], atol=3e-6, rtol=3e-5)
    if not frozen_hidden:
        torch.testing.assert_close(hidden.grad, expected_grads[0], atol=3e-6, rtol=3e-5)

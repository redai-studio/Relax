# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical checks for sequence-classification SFT objectives."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import loss as loss_module
except Exception as exc:
    pytest.skip(f"relax.backends.megatron.loss unavailable: {exc}", allow_module_level=True)


def test_single_label_loss_and_gradient_match_torch_reference(monkeypatch):
    logits = torch.tensor([[1.0, -0.5, 0.25], [-1.0, 0.5, 2.0]], requires_grad=True)
    labels = [torch.tensor(0), torch.tensor(2)]
    monkeypatch.setattr(
        loss_module,
        "get_sequence_classification_outputs",
        lambda args, batch, model_logits: (model_logits, [0, 1]),
    )

    loss, metrics = loss_module.sequence_classification_loss_function(
        Namespace(problem_type="single_label_classification", classification_threshold=0.5),
        {"classification_labels": labels},
        logits,
        lambda values: values.sum(),
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_loss = F.cross_entropy(reference_logits, torch.tensor([0, 2]), reduction="sum")
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(loss.detach(), reference_loss.detach())
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert metrics["accuracy"].item() == 2.0


def test_multi_label_loss_and_gradient_match_torch_reference(monkeypatch):
    logits = torch.tensor([[0.2, -0.4, 1.0], [-0.5, 0.75, 0.1]], requires_grad=True)
    labels = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])]
    monkeypatch.setattr(
        loss_module,
        "get_sequence_classification_outputs",
        lambda args, batch, model_logits: (model_logits, [0, 1]),
    )

    loss, metrics = loss_module.sequence_classification_loss_function(
        Namespace(problem_type="multi_label_classification", classification_threshold=0.5),
        {"classification_labels": labels},
        logits,
        lambda values: values.sum(),
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_targets = torch.stack(labels)
    reference_loss = (
        F.binary_cross_entropy_with_logits(
            reference_logits,
            reference_targets,
            reduction="none",
        )
        .mean(dim=-1)
        .sum()
    )
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(loss.detach(), reference_loss.detach())
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert metrics["subset_accuracy"].item() == 1.0


def test_non_owner_rank_returns_zero_connected_loss(monkeypatch):
    logits = torch.randn(1, 2, 3, requires_grad=True)
    monkeypatch.setattr(
        loss_module,
        "get_sequence_classification_outputs",
        lambda args, batch, model_logits: (model_logits.new_empty((0, 3)), []),
    )

    loss, _ = loss_module.sequence_classification_loss_function(
        Namespace(problem_type="single_label_classification", classification_threshold=0.5),
        {"classification_labels": [torch.tensor(1)]},
        logits,
        lambda values: values.sum(),
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0

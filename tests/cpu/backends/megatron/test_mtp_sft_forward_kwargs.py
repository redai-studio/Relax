# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MTP training forwards must keep labels and loss masks aligned."""

import ast
from argparse import Namespace
from pathlib import Path

import pytest
import torch


# This helper only consumes host metadata; execute its actual source without
# importing optional CUDA extensions from the rest of the training module.
_source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/model.py"
_function = next(
    node
    for node in ast.parse(_source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "_attach_mtp_forward_kwargs"
)
_namespace = {"Namespace": Namespace, "torch": torch}
exec(compile(ast.Module(body=[_function], type_ignores=[]), str(_source), "exec"), _namespace)
_attach_mtp_forward_kwargs = _namespace["_attach_mtp_forward_kwargs"]


def _mk_args(enable_mtp_training: bool) -> Namespace:
    return Namespace(enable_mtp_training=enable_mtp_training)


def test_attach_mtp_forward_kwargs_noop_when_disabled():
    forward_kwargs = {"loss_mask": None}
    original = forward_kwargs.copy()

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=False), {}, forward_kwargs)

    assert forward_kwargs == original


def test_attach_mtp_forward_kwargs_preserves_existing_loss_mask():
    tokens = object()
    existing_loss_mask = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = {"loss_mask": existing_loss_mask}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    assert forward_kwargs["loss_mask"] is existing_loss_mask


@pytest.mark.parametrize("unsplit", [False, True])
def test_attach_mtp_forward_kwargs_restores_bridge_unsplit_loss_mask(unsplit):
    tokens = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    if unsplit:
        batch.update(unsplit_mtp_labels=tokens, unsplit_mtp_loss_mask=full_loss_masks)
    forward_kwargs = {"loss_mask": None}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is tokens
    assert forward_kwargs["loss_mask"] is full_loss_masks
    assert forward_kwargs["mtp_kwargs"].get("labels_are_shifted", False) is unsplit


@pytest.mark.parametrize("layout", ["raw", "unsplit_full", "unsplit_legacy"])
@pytest.mark.parametrize("existing_mask", [False, True])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_forward_mask_controls_independent_mtp_gradient(layout, existing_mask, is_dummy, monkeypatch):
    native = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    monkeypatch.setattr(native.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    full_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    local_mask = torch.tensor([[1.0, 0.0]])
    original_mask = local_mask if layout == "unsplit_legacy" else full_mask
    batch = {"tokens": torch.zeros(1, 4, dtype=torch.long), "full_loss_masks": full_mask, "__is_dummy__": is_dummy}
    if layout != "raw":
        batch.update(
            unsplit_mtp_labels=batch["tokens"],
            unsplit_mtp_loss_mask=local_mask,
            unsplit_full_mtp_loss_mask=full_mask,
        )
    forward_kwargs = {"loss_mask": original_mask if existing_mask else None}
    if layout != "raw":
        forward_kwargs["attention_mask"] = torch.ones(1, 4, dtype=torch.bool)
        if layout == "unsplit_full":
            forward_kwargs["position_ids"] = torch.zeros(3, 1, 4, dtype=torch.long)
    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    weight = torch.tensor(2.0, requires_grad=True)
    auxiliary_loss = (weight.square() * forward_kwargs["loss_mask"]).sum()
    output = native.MTPLossAutoScaler.apply(weight.expand(4), auxiliary_loss)
    # Main-loss dummy masking cannot suppress MTPLossAutoScaler's separate backward injection.
    (output.sum() * 0).backward()
    expected_gradient = 0.0 if is_dummy else (4.0 if layout == "unsplit_legacy" else 8.0)
    torch.testing.assert_close(weight.grad, torch.tensor(expected_gradient))
    assert forward_kwargs["loss_mask"].shape == original_mask.shape
    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is batch["tokens"]
    torch.testing.assert_close(full_mask, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    torch.testing.assert_close(local_mask, torch.tensor([[1.0, 0.0]]))

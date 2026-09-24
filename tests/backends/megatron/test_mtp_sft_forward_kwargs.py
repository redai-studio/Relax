# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MTP training forwards must keep labels and loss masks aligned."""

import ast
from argparse import Namespace
from pathlib import Path

import pytest


# This helper only consumes host metadata; execute its actual source without
# importing optional CUDA extensions from the rest of the training module.
_source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
_function = next(
    node
    for node in ast.parse(_source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "_attach_mtp_forward_kwargs"
)
_namespace = {"Namespace": Namespace}
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

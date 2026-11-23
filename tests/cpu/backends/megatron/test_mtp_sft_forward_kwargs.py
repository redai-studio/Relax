# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MTP training forwards must keep labels and loss masks aligned."""

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("embedding_scattered", [False, True])
def test_mtp_input_mask_controls_gradients_after_sequence_scatter(embedding_scattered, monkeypatch):
    native = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    tp_group = SimpleNamespace(size=lambda: 2)

    def scatter(tensor, *, group):
        assert group is tp_group
        return tensor.chunk(2, dim=0)[0]

    monkeypatch.setattr(native, "scatter_to_sequence_parallel_region", scatter)
    layer = native.MultiTokenPredictionLayer.__new__(native.MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.cp_group = None
    layer.tp_group = tp_group
    layer.config = SimpleNamespace(sequence_parallel=True, attention_cp_layout=None, mtp_detach_embedding=False)
    table = torch.nn.Embedding(5, 1)

    def embedding(*, input_ids, position_ids):
        values = table(input_ids).transpose(0, 1)
        return scatter(values, group=tp_group) if embedding_scattered else values

    embedding.scatter_to_sequence_parallel = embedding_scattered
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, False, True]])
    _, _, _, rolled_mask, decoder_input, _ = layer._get_embeddings(
        ids, ids, embedding, torch.zeros(2, 1, 1), mtp_input_mask=mask
    )

    torch.testing.assert_close(rolled_mask, torch.tensor([[True, False, True, False]]))
    torch.testing.assert_close(decoder_input[:, 0], table.weight[[2, 3]])
    decoder_input.sum().backward()
    torch.testing.assert_close(table.weight.grad[:, 0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


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


@pytest.mark.parametrize("labels_are_shifted", [False, True])
def test_hybrid_sft_keeps_mtp_supervision_separate_from_main_labels(monkeypatch, labels_are_shifted):
    from megatron.core.models.hybrid import hybrid_model
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock

    batch = SimpleNamespace(
        tokens=torch.tensor([[1, 2, 3, 4]]),
        shifted_labels=torch.tensor([[8, 7, 6, 0]]),
        mask=torch.tensor([[True, True, False, False]]),
        conditioning=torch.tensor([[True, False, True, True]]),
        hidden=torch.ones(4, 1, 2),
    )
    config = SimpleNamespace(
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        moe_num_hash_layers=0,
        sequence_parallel=False,
        freeze_base_model_for_mtp=False,
        mtp_num_layers=1,
        use_mup=False,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
    )

    class MTPBlock:
        cp_group = SimpleNamespace(size=lambda: 1)
        sequence_parallel = False
        prepare_cp_layout = MultiTokenPredictionBlock.prepare_cp_layout

        def __call__(self, *, input_ids, hidden_states, mtp_input_mask, **kwargs):
            torch.testing.assert_close(input_ids, batch.tokens)
            torch.testing.assert_close(mtp_input_mask, batch.conditioning)
            return hidden_states

    def process_mtp_loss(**kwargs):
        torch.testing.assert_close(kwargs["labels"], batch.shifted_labels)
        torch.testing.assert_close(kwargs["loss_mask"], batch.mask)
        return kwargs["hidden_states"] + 2

    def main_output(**kwargs):
        assert kwargs["labels"] is None
        return kwargs["hidden_states"]

    monkeypatch.setattr(hybrid_model, "process_mtp_loss", process_mtp_loss)
    mtp = MTPBlock()
    mtp.config = config
    model = SimpleNamespace(
        config=config,
        position_embedding_type="none",
        hybrid_layer_config_list=None,
        share_embeddings_and_output_weights=False,
        mtp_process=True,
        mtp=mtp,
        post_process=True,
        pg_collection=SimpleNamespace(cp=None, dp_cp=None),
        tp_group=None,
        decoder=lambda *, hidden_states, **kwargs: hidden_states,
        embedding=None,
        output_layer=None,
        compute_language_model_loss=None,
        _scale_logits=None,
        training=True,
    )
    output = hybrid_model.HybridModel.forward(
        model,
        input_ids=batch.tokens,
        position_ids=None,
        attention_mask=None,
        decoder_input=batch.hidden,
        loss_mask=batch.mask,
        output_processor=main_output,
        mtp_kwargs={
            "mtp_labels": batch.shifted_labels if labels_are_shifted else torch.tensor([[9, 8, 7, 6]]),
            "labels_are_shifted": labels_are_shifted,
            "mtp_input_mask": batch.conditioning,
        },
    )
    torch.testing.assert_close(output, torch.full_like(batch.hidden, 3))

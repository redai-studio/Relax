# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the installed MTP loss chain instead of assuming an obsolete GPT
roll."""

import ast
import inspect
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


try:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import multi_token_prediction as mtp

    from relax.models.common import chunked_mtp_loss as patch
except (ImportError, AssertionError) as exc:
    pytest.skip(f"MCore unavailable: {exc}", allow_module_level=True)

if "tensors" not in inspect.signature(mtp.roll_tensor).parameters:
    pytest.skip("Requires the image's packed list-based MTP roll API", allow_module_level=True)


def _attach():
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/model.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_attach_mtp_forward_kwargs"
    )
    scope = {"Namespace": Namespace, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("chunked", [False, True])
def test_mtp_predicts_two_tokens_ahead_without_crossing_documents(monkeypatch, chunked):
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    batch = {
        "tokens": tokens,
        "full_loss_masks": torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "packed_seq_params": packed,
    }
    kwargs = {"loss_mask": None}
    _attach()(Namespace(enable_mtp_training=True), batch, kwargs)
    # GPT's labels_are_shifted=False contract performs the initial next-token
    # roll before process_mtp_loss advances to the second future token.
    labels = kwargs["mtp_kwargs"]["mtp_labels"]
    if not kwargs["mtp_kwargs"].get("labels_are_shifted", False):
        [labels] = mtp.roll_tensor([labels], cp_group=None, packed_seq_params=packed)
    seen = []
    auxiliary_losses = []

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(16, 2))

        def forward(self, hidden, weight=None, runtime_gather_output=None):
            return hidden @ self.weight.T, None

    def loss(labels, logits):
        seen.append(labels.clone())
        return torch.ones_like(labels, dtype=torch.float)

    monkeypatch.setattr(patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(patch, "_resolve_chunk_size", lambda: 8)

    def capture_auxiliary_loss(hidden, loss):
        auxiliary_losses.append(loss.detach().clone())
        return hidden

    monkeypatch.setattr(mtp.MTPLossAutoScaler, "apply", capture_auxiliary_loss)
    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        mtp_detach_heads=False,
    )
    fn = patch.process_mtp_loss if chunked else mtp.process_mtp_loss
    fn(
        torch.zeros(16, 1, 2),
        labels,
        kwargs["loss_mask"],
        Head(),
        None,
        None,
        False,
        loss,
        config,
        cp_group=None,
        packed_seq_params=packed,
    )
    torch.testing.assert_close(torch.cat(seen, dim=-1), torch.tensor([[3, 4, 0, 0, 7, 8, 0, 0]]))

    assert len(auxiliary_losses) == 1
    torch.testing.assert_close(
        auxiliary_losses[0].ne(0), torch.tensor([[True, True, False, False, True, True, False, False]])
    )

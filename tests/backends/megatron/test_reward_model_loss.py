# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward-model loss, pooling, and metric contracts."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


try:
    from relax.backends.megatron.loss import reward_model_loss_function
    from relax.backends.megatron.model import _restore_micro_batch_output_order
except Exception as exc:
    pytest.skip(f"Megatron reward-model loss unavailable: {exc}", allow_module_level=True)


def test_preference_outputs_restore_original_order_without_dynamic_batch_flag():
    packed_values = [
        "pair-2-chosen",
        "pair-2-rejected",
        "pair-0-chosen",
        "pair-0-rejected",
        "pair-1-chosen",
        "pair-1-rejected",
    ]
    schedule = [[4, 5, 0, 1], [2, 3]]

    assert _restore_micro_batch_output_order(packed_values, schedule) == [
        "pair-0-chosen",
        "pair-0-rejected",
        "pair-1-chosen",
        "pair-1-rejected",
        "pair-2-chosen",
        "pair-2-rejected",
    ]
    assert _restore_micro_batch_output_order(["aggregate-0", "aggregate-1"], schedule) == [
        "aggregate-0",
        "aggregate-1",
    ]


def _batch():
    branches = [
        torch.tensor([10, 11, 12]),
        torch.tensor([20, 21]),
        torch.tensor([30, 31, 32, 33]),
        torch.tensor([40, 41]),
    ]
    lengths = [len(branch) for branch in branches]
    packed = torch.cat(branches)
    return {
        "total_lengths": lengths,
        "response_lengths": lengths,
        "score_positions": [2, 1, 3, 1],
        "raw_loss_masks": [
            torch.tensor([0, 1, 1]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1]),
        ],
        "tokens": packed.unsqueeze(0),
        "unconcat_tokens": branches,
        "packed_seq_params": SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 5, 9, 11])),
        "preference_branch_pair_ids": [7, 8, 7, 8],
        "preference_is_chosen": [False, True, True, False],
    }


def test_reward_model_loss_uses_pair_identity_after_branch_reordering_and_preserves_gradient():
    batch = _batch()
    flat = torch.arange(11, dtype=torch.float32, requires_grad=True)
    loss, metrics = reward_model_loss_function(Namespace(), batch, flat.reshape(1, 11, 1), lambda value: value)

    expected_margins = torch.tensor([8.0 - 2.0, 4.0 - 10.0])
    expected = -torch.nn.functional.logsigmoid(expected_margins)
    assert torch.allclose(loss, expected.sum())
    assert set(metrics) == {
        "rm/loss",
        "rm/score_chosen_mean",
        "rm/score_rejected_mean",
        "rm/score_margin_mean",
        "rm/accuracy",
        "rm/_score_chosen_second_moment",
        "rm/_score_rejected_second_moment",
    }
    loss.backward()
    assert flat.grad is not None
    assert torch.count_nonzero(flat.grad).item() == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda batch: batch["raw_loss_masks"][0].zero_(), "raw_loss_mask"),
        (lambda batch: batch["tokens"][0].__setitem__(2, 999), "terminal token"),
        (
            lambda batch: setattr(batch["packed_seq_params"], "cu_seqlens_q", torch.tensor([0, 2, 5, 9, 11])),
            "cu_seqlens",
        ),
    ],
)
def test_reward_model_pooling_rejects_mask_token_and_segment_misalignment(mutation, match):
    batch = _batch()
    mutation(batch)
    with pytest.raises(ValueError, match=match):
        reward_model_loss_function(Namespace(), batch, torch.zeros(1, 11, 1), lambda value: value)


def test_reward_model_pooling_allows_only_one_trailing_padding_segment():
    batch = _batch()
    batch["tokens"] = torch.nn.functional.pad(batch["tokens"], (0, 5))
    batch["packed_seq_params"].cu_seqlens_q = torch.tensor([0, 3, 5, 9, 11, 16])
    logits = torch.zeros(1, 16, 1)
    reward_model_loss_function(Namespace(), batch, logits, lambda value: value)

    batch["packed_seq_params"].cu_seqlens_q = torch.tensor([0, 3, 5, 9, 11, 14, 16])
    with pytest.raises(ValueError, match="at most one trailing padding"):
        reward_model_loss_function(Namespace(), batch, logits, lambda value: value)

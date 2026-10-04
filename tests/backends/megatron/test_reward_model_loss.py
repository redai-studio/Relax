# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward-model loss, pooling, and metric contracts."""

from argparse import Namespace

import pytest
import torch

from relax.engine.sft.eval.preference import compute_reward_model_eval_step


try:
    from relax.backends.megatron import data as data_module
    from relax.backends.megatron.loss import loss_function, reward_model_loss_function
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


def _batch(monkeypatch, pair_ids=(8, 7), pad_multiplier=1, allgather_cp=False):
    rows = {
        "pair_ids": list(pair_ids),
        "chosen_tokens": [[10, 11, 12], [30, 31, 32, 33]],
        "rejected_tokens": [[20, 21], [40, 41]],
        "chosen_loss_masks": [[0, 1, 1], [0, 0, 1, 1]],
        "rejected_loss_masks": [[0, 1], [0, 1]],
        "chosen_total_lengths": [3, 4],
        "rejected_total_lengths": [2, 2],
    }
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace())
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    iterator = data_module.DataIterator(
        data_module.expand_preference_rollout_data(rows), micro_batch_indices=[[2, 3, 0, 1]]
    )
    return data_module.get_batch(
        iterator,
        [
            "tokens",
            "loss_masks",
            "total_lengths",
            "response_lengths",
            "preference_branch_pair_ids",
            "preference_is_chosen",
        ],
        pad_multiplier=pad_multiplier,
        allgather_cp=allgather_cp,
        pack_device=torch.device("cpu"),
    )


@pytest.mark.parametrize("pair_ids", [(8, 7), (7, 7)])
@pytest.mark.parametrize("pad_multiplier", [1, 8])
@pytest.mark.parametrize("allgather_cp", [False, True])
def test_reward_model_packing_preserves_terminal_scores_loss_and_gradients(
    monkeypatch, pair_ids, pad_multiplier, allgather_cp
):
    batch = _batch(monkeypatch, pair_ids, pad_multiplier, allgather_cp)
    padding = [] if pad_multiplier == 1 else [0] * 5
    assert batch["tokens"].tolist() == [[30, 31, 32, 33, 40, 41, 10, 11, 12, 20, 21, *padding]]
    assert batch["total_lengths"] == [4, 2, 3, 2]
    assert batch["packed_seq_params"].cu_seqlens_q.tolist() == [0, 4, 6, 9, 11, *([16] if padding else [])]

    flat = (batch["tokens"][0].float() / 10).requires_grad_()
    logits = flat.reshape(1, -1, 1)
    args = Namespace(
        loss_type="rm",
        qkv_format="thd",
        calculate_per_token_loss=False,
        global_batch_size=2,
        recompute_loss_function=False,
        allgather_cp=allgather_cp,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    loss, _, log = loss_function(args, batch, num_microbatches=1, logits=logits)
    metrics = dict(zip(log["keys"], log["values"][1:], strict=True))
    # Independent per-branch scores from the last tokens of the input rows.
    expected_scores = torch.tensor([3.3, 4.1, 1.2, 2.1], requires_grad=True)
    expected = -torch.nn.functional.logsigmoid(expected_scores[::2] - expected_scores[1::2]).mean()
    torch.testing.assert_close(loss, expected)
    assert set(metrics) == {
        "rm/loss",
        "rm/score_chosen_mean",
        "rm/score_rejected_mean",
        "rm/score_margin_mean",
        "rm/accuracy",
        "rm/_score_chosen_second_moment",
        "rm/_score_rejected_second_moment",
    }
    torch.testing.assert_close(metrics["rm/score_chosen_mean"], expected_scores[::2].sum())
    torch.testing.assert_close(metrics["rm/score_rejected_mean"], expected_scores[1::2].sum())
    loss.backward()
    expected.backward()
    expected_grad = torch.zeros_like(flat)
    expected_grad[[3, 5, 8, 10]] = expected_scores.grad
    torch.testing.assert_close(flat.grad, expected_grad)

    _, outputs = compute_reward_model_eval_step(logits, total_lengths=batch["total_lengths"])
    torch.testing.assert_close(torch.stack(outputs["scores"]), expected_scores.detach())
    restored = _restore_micro_batch_output_order(outputs["scores"], [[2, 3, 0, 1]])
    torch.testing.assert_close(torch.stack(restored), torch.tensor([1.2, 2.1, 3.3, 4.1]))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda batch: batch.update(preference_branch_pair_ids=[7, 8, 7, 8]), "same pair ID"),
        (lambda batch: batch.update(preference_is_chosen=[False, True, True, False]), "ordered chosen/rejected"),
        (
            lambda batch: batch.update(total_lengths=[], preference_branch_pair_ids=[], preference_is_chosen=[]),
            "at least one preference pair",
        ),
    ],
)
def test_reward_model_loss_rejects_mismatched_pairs(monkeypatch, mutation, match):
    batch = _batch(monkeypatch)
    mutation(batch)
    with pytest.raises(ValueError, match=match):
        reward_model_loss_function(Namespace(), batch, torch.zeros(1, 11, 1), lambda value: value)

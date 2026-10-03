# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.engine.sft.eval.preference import (
    compute_reward_model_eval_step,
    finalize_pair_metrics,
    pair_metric_sums,
    preference_eval_chunk_sizes,
    preference_eval_local_batch_sizes,
)


@pytest.mark.parametrize(
    ("pair_count", "global_batch_size", "dp_size", "expected"),
    [(1, 4, 1, [1]), (6, 4, 2, [2, 1]), (10, 4, 2, [2, 2, 1]), (512, 30, 2, [15] * 17 + [1])],
)
def test_preference_eval_chunks_preserve_actual_pair_count(pair_count, global_batch_size, dp_size, expected):
    assert preference_eval_local_batch_sizes(pair_count, global_batch_size, dp_size) == expected
    chunks = preference_eval_chunk_sizes(pair_count, global_batch_size)
    assert chunks == [size * dp_size for size in expected]
    assert sum(chunks) == pair_count


def test_preference_eval_rejects_chunks_that_cannot_be_split_across_dp():
    with pytest.raises(ValueError, match="divisible by data-parallel size"):
        preference_eval_local_batch_sizes(5, 4, dp_size=2)


def test_reward_model_eval_emits_one_score_per_branch_for_order_restoration():
    _, outputs = compute_reward_model_eval_step(
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        total_lengths=[2, 2],
        score_positions=[1, 1],
    )

    assert len(outputs["scores"]) == 2
    assert all(score.ndim == 0 for score in outputs["scores"])
    assert torch.stack(outputs["scores"]).tolist() == [1.0, 3.0]


def test_pair_metric_sums_and_finalize_keep_ties_explicit():
    chosen = torch.tensor([2.0, 1.0, 1.0])
    rejected = torch.tensor([1.0, 2.0, 1.0])
    losses = torch.tensor([0.1, 0.2, 0.3])

    metrics = finalize_pair_metrics(pair_metric_sums(chosen, rejected, losses), prefix="rm")

    assert metrics["eval/rm_loss"] == pytest.approx(0.2)
    assert metrics["eval/rm_strict_accuracy"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_rate"] == pytest.approx(1 / 3)
    assert metrics["eval/rm_tie_aware_accuracy"] == pytest.approx(0.5)
    assert metrics["eval/rm_pairs"] == 3

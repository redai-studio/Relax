# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.engine.sft.runtime import resolve_sft_split_indices, should_skip_sft_weight_management


def test_resolve_sft_split_indices_is_deterministic_and_disjoint():
    train_indices, eval_indices = resolve_sft_split_indices(100, 0.2, seed=42)
    same_train, same_eval = resolve_sft_split_indices(100, 0.2, seed=42)
    other_train, other_eval = resolve_sft_split_indices(100, 0.2, seed=43)

    assert (train_indices, eval_indices) == (same_train, same_eval)
    assert (train_indices, eval_indices) != (other_train, other_eval)
    assert len(train_indices) == 80
    assert len(eval_indices) == 20
    assert set(train_indices).isdisjoint(eval_indices)
    assert set(train_indices) | set(eval_indices) == set(range(100))


def test_resolve_sft_split_indices_clamps_eval_to_leave_one_train_row():
    train_indices, eval_indices = resolve_sft_split_indices(4, 99, seed=7)

    assert len(train_indices) == 1
    assert len(eval_indices) == 3


def test_resolve_sft_split_indices_matches_ms_swift_rng_order():
    train_indices, eval_indices = resolve_sft_split_indices(100, 0.2, seed=42)

    assert train_indices[:8] == (49, 48, 46, 56, 12, 17, 0, 50)
    assert eval_indices[:8] == (91, 52, 40, 13, 32, 26, 65, 1)


@pytest.mark.parametrize(
    ("task_type", "mtp_only_training"),
    [("causal_lm", False), ("seq_cls", False), ("seq_contrast", False), ("causal_lm", True)],
)
def test_pure_sft_does_not_need_snapshots_for_saving_or_local_eval(task_type, mtp_only_training):
    args = Namespace(
        loss_type="sft",
        task_type=task_type,
        mtp_only_training=mtp_only_training,
        save="checkpoints",
        save_interval=10,
        save_hf="export",
        eval_interval=5,
        eval_prompt_data=["validation"],
    )

    assert should_skip_sft_weight_management(args)


@pytest.mark.parametrize(
    ("overrides", "consumers"),
    [
        ({"loss_type": "policy_loss"}, {}),
        ({"loss_type": "dpo"}, {}),
        ({"loss_type": "rm"}, {}),
        ({"sft_predict_interval": 10}, {}),
        ({"offload_train": True}, {}),
        ({"keep_old_actor": True}, {}),
        ({}, {"with_ref": True}),
        ({}, {"with_opd_teacher": True}),
    ],
)
@pytest.mark.parametrize("mtp_only_training", [False, True])
def test_sft_keeps_weight_management_for_consumers_and_other_losses(overrides, consumers, mtp_only_training):
    args = Namespace(**{"loss_type": "sft", "mtp_only_training": mtp_only_training, **overrides})

    assert not should_skip_sft_weight_management(args, **consumers)

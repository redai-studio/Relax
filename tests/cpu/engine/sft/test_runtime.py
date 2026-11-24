# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pickle
from argparse import Namespace

import pytest

from relax.engine.sft.bootstrap import resolve_sft_num_rollout
from relax.engine.sft.runtime import (
    resolve_sft_split_indices,
    sft_partition_id,
    sft_partition_ids,
    sft_partition_prefix,
    should_skip_sft_weight_management,
)


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


def test_image_sft_attempt_is_shared_with_workers_and_changes_on_restore():
    args = Namespace(
        loss_type="sft",
        generation_task="i2i",
        i2i_sft_config={"dataset_size": 19},
        global_batch_size=8,
        num_epoch=2,
        num_rollout=None,
    )
    resolve_sft_num_rollout(args)
    worker_args = pickle.loads(pickle.dumps(args))
    previous = sft_partition_id(args, 10)
    assert previous == sft_partition_id(worker_args, 10)
    assert sft_partition_ids(worker_args, 10) == [previous]
    assert previous.startswith(sft_partition_prefix(args))

    resolve_sft_num_rollout(args)
    assert sft_partition_id(args, 10) != previous
    assert not previous.startswith(sft_partition_prefix(args))
    assert args.num_rollout == 6


def test_image_sft_partition_requires_shared_attempt():
    with pytest.raises(ValueError, match="initialized before starting workers"):
        sft_partition_id(Namespace(loss_type="sft", generation_task="i2i"), 10)


@pytest.mark.parametrize("loss_type,prefix", [("sft", "sft_"), ("dpo", "sft_"), ("rm", "sft_"), ("policy", "train_")])
def test_partition_naming_preserves_non_image_modes(loss_type, prefix):
    assert sft_partition_id(Namespace(loss_type=loss_type), 10) == f"{prefix}10"

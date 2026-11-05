# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pickle
from argparse import ArgumentParser, Namespace
from types import ModuleType, SimpleNamespace

import pytest

from relax.engine.sft.bootstrap import resolve_sft_num_rollout
from relax.engine.sft.runtime import (
    resolve_sft_split_indices,
    sft_partition_id,
    sft_partition_ids,
    sft_partition_prefix,
    should_bypass_main_output_layer,
    should_skip_sft_weight_management,
    should_use_sft_chunked,
)
from tests.cpu.relax_parser import parser_for_flags
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--loss-type", "sft"], False),
        (["--loss-type", "sft", "--sft-chunked-logits"], True),
        (["--loss-type", "sft", "--sft-chunked-logits", "--no-sft-chunked-logits"], False),
        (["--loss-type", "dpo", "--sft-chunked-logits"], False),
    ],
)
def test_sft_chunked_uses_declared_cli_flag(arguments_module: ModuleType, flags: list[str], expected: bool) -> None:
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    assert should_use_sft_chunked(parser.parse_args(flags)) is expected


def test_sft_chunked_short_circuits_non_sft_namespace() -> None:
    assert should_use_sft_chunked(Namespace(loss_type="policy_loss")) is False


def test_sft_chunked_preserves_explicit_null() -> None:
    assert should_use_sft_chunked(Namespace(loss_type="sft", sft_chunked_logits=None)) is None


@pytest.mark.parametrize(
    ("flags", "null_mtp", "expected"),
    [
        ([], False, False),
        (["--mtp-only-training"], False, True),
        (["--sft-chunked-logits"], False, True),
        ([], True, False),
        (["--sft-chunked-logits"], True, True),
    ],
)
def test_bypass_main_output_uses_initialized_training_switches(
    flags: list[str], null_mtp: bool, expected: bool
) -> None:
    parser = parser_for_flags("--mtp-only-training", "--sft-chunked-logits")
    args = parser.parse_args(flags, namespace=Namespace(loss_type="sft"))
    if null_mtp:
        args.mtp_only_training = None

    assert should_bypass_main_output_layer(args) is expected


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
@pytest.mark.parametrize("offload_argv", [[], ["--no-offload-train"]])
def test_pure_sft_does_not_need_snapshots_for_saving_or_local_eval(task_type, mtp_only_training, offload_argv):
    args = Namespace(
        loss_type="sft",
        task_type=task_type,
        mtp_only_training=mtp_only_training,
        save="checkpoints",
        save_interval=10,
        save_hf="export",
        eval_interval=5,
        eval_prompt_data=["validation"],
        **vars(
            parser_for_flags("--sft-predict-interval", "--offload-train", "--keep-old-actor").parse_args(offload_argv)
        ),
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
    args = parser_for_flags("--sft-predict-interval", "--offload-train", "--keep-old-actor").parse_args([])
    vars(args).update({"loss_type": "sft", "mtp_only_training": mtp_only_training, **overrides})

    assert not should_skip_sft_weight_management(args, **consumers)


def test_image_sft_attempt_is_shared_with_workers_and_changes_on_restore():
    args = Namespace(
        **vars(parser_for_flags("--sft-async-prepack").parse_args([])),
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
    args = parser_for_flags("--generation-task", "--sft-async-prepack").parse_args(
        [], namespace=Namespace(loss_type=loss_type)
    )
    assert sft_partition_id(args, 10) == f"{prefix}10"

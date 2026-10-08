# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import (
    _opd_args,
)
from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def test_sft_async_prepack_rejects_single_in_flight_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --sft-max-in-flight-steps >= 2"):
        arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)


def test_sft_async_prepack_rejects_zero_max_staleness_without_alias(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=None, sft_async_prepack=True, max_staleness=0)

    with pytest.raises(ValueError, match="requires --max-staleness >= 1"):
        arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)


def test_sft_async_prepack_maps_two_in_flight_steps_to_one_stale_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=2, sft_async_prepack=True, max_staleness=0)

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)

    assert args.max_staleness == 1


def test_sft_without_async_prepack_allows_one_in_flight_step(arguments_module):
    args = SimpleNamespace(sft_max_in_flight_steps=1, sft_async_prepack=False, max_staleness=0)

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)

    assert args.max_staleness == 0


def test_sft_train_data_prefetch_rejects_async_prepack(arguments_module):
    args = SimpleNamespace(
        sft_train_data_prefetch=True,
        sft_async_prepack=True,
        per_rank_fetch=True,
        max_staleness=1,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        arguments_module._validate_sft_train_data_prefetch(args, is_offline=True)


def test_sft_train_data_prefetch_allows_raw_only_mode(arguments_module):
    args = SimpleNamespace(
        sft_train_data_prefetch=True,
        sft_async_prepack=False,
        per_rank_fetch=True,
        max_staleness=1,
    )

    arguments_module._validate_sft_train_data_prefetch(args, is_offline=True)


@pytest.mark.parametrize("loss_type", ["sft", "sft_loss", "sft-loss", "dpo", "rm"])
def test_offline_loss_selects_dataset_training(arguments_module, loss_type):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    parsed = parser.parse_args(["--loss-type", loss_type])
    assert not hasattr(parsed, "sft_objective")

    # Reuse the RL configuration to verify that offline selection disables
    # rollout, advantages, and the critic even when an RL estimator is set.
    args = _opd_args()
    args.loss_type = parsed.loss_type
    args.advantage_estimator = "ppo"
    args.prompt_data = ["/train.jsonl"]
    args.eval_interval = 10
    args.eval_size = 0.1
    args.use_dynamic_batch_size = True
    args.max_tokens_per_gpu = 4096
    args.sft_oversize_strategy = "drop"
    args.sft_oversize_custom_function_path = None
    args.n_samples_per_prompt = 1
    args.qkv_format = "thd"
    args.use_gloo_process_groups = True
    args.preference_max_length = parsed.preference_max_length
    args.preference_max_completion_length = parsed.preference_max_completion_length
    args.ref_load = None
    args.dpo_reference_free = loss_type == "dpo"
    args.sft_train_data_prefetch = True
    args.per_rank_fetch = True
    args.sft_max_in_flight_steps = 2

    arguments_module.slime_validate_args(args)

    assert args.loss_type == ("sft" if loss_type in {"sft_loss", "sft-loss"} else loss_type)
    assert not args.compute_advantages_and_returns
    assert not args.use_critic
    assert not args.use_opd
    assert not args.rollout_global_dataset
    assert not hasattr(args, "eval_datasets")
    assert args.balance_data
    assert args.max_staleness == 1
    assert args.sft_tq_timeout_minutes == args.distributed_timeout_minutes


@pytest.mark.parametrize("argv", [["--sft-objective", "dpo"], ["--sft-objective=dpo"]])
def test_removed_sft_objective_is_rejected_before_backend_parsing(arguments_module, monkeypatch, argv):
    monkeypatch.setattr(arguments_module.sys, "argv", ["train", "--loss-type", "sft", *argv])

    with pytest.raises(ValueError, match="--sft-objective has been removed.*--loss-type"):
        arguments_module._parse_args_impl()


@pytest.mark.parametrize("source", ["hf", "megatron"])
@pytest.mark.parametrize("load_mode", ["new", "resume", "finetune"])
def test_dpo_initial_reference_and_resume_keep_separate_checkpoint_roles(
    arguments_module, tmp_path, source, load_mode
):
    reference = tmp_path / "sft"
    reference.mkdir()
    marker = "config.json" if source == "hf" else "latest_checkpointed_iteration.txt"
    (reference / marker).write_text("{}" if source == "hf" else "7")
    resume = tmp_path / "dpo"
    resume.mkdir()
    (resume / "latest_checkpointed_iteration.txt").write_text("100")
    args = _opd_args()
    vars(args).update(
        loss_type="dpo",
        prompt_data=["/train.jsonl"],
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4096,
        n_samples_per_prompt=1,
        qkv_format="thd",
        use_gloo_process_groups=True,
        preference_max_length=1024,
        preference_max_completion_length=512,
        enable_weights_backuper=True,
        sft_oversize_strategy="keep",
        sft_oversize_custom_function_path=None,
        dpo_reference_free=False,
        ref_load=str(reference),
        ref_ckpt_step=7 if source == "megatron" else None,
        ckpt_step=100,
        load=None if load_mode == "new" else str(resume if load_mode == "resume" else reference),
        finetune=load_mode == "finetune",
        no_load_optim=load_mode == "finetune",
        no_load_rng=load_mode == "finetune",
    )

    arguments_module.slime_validate_args(args)

    assert args.ref_load == str(reference)
    assert args.load == str(resume if load_mode == "resume" else reference)
    assert args.finetune is (load_mode != "resume")
    assert args.no_load_optim is (load_mode != "resume")
    assert args.no_load_rng is (load_mode != "resume")
    assert args.dpo_reference_free is False
    assert args.ckpt_step == (7 if source == "megatron" and load_mode == "new" else 100)

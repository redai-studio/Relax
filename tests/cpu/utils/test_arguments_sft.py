# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

from tests.cpu.relax_parser import parser_for_flags
from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    _opd_args,
)
from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def _in_flight_args(*flags: str, **overrides) -> SimpleNamespace:
    parser = parser_for_flags("--sft-max-in-flight-steps", "--sft-async-prepack")
    args = parser.parse_args(flags, namespace=SimpleNamespace(max_staleness=0))
    vars(args).update(overrides)
    return args


def test_sft_legacy_prepack_allows_single_in_flight_step(arguments_module):
    args = _in_flight_args("--sft-max-in-flight-steps", "1", "--sft-async-prepack")

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)
    assert args.max_staleness == 0


def test_sft_legacy_prepack_allows_zero_max_staleness_without_alias(arguments_module):
    args = _in_flight_args("--sft-async-prepack")

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)
    assert args.max_staleness == 0


def test_sft_async_prepack_maps_two_in_flight_steps_to_one_stale_step(arguments_module):
    args = _in_flight_args("--sft-max-in-flight-steps", "2", "--sft-async-prepack")

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)

    assert args.max_staleness == 1


def test_sft_without_async_prepack_allows_one_in_flight_step(arguments_module):
    args = _in_flight_args("--sft-max-in-flight-steps", "1", sft_async_prepack=False)

    arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)

    assert args.max_staleness == 0


def test_sft_rejects_zero_in_flight_steps(arguments_module):
    args = _in_flight_args("--sft-max-in-flight-steps", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        arguments_module._normalize_sft_max_in_flight_steps(args, is_offline=True)


@pytest.mark.parametrize(
    ("loss_type", "timeout_cli", "timeout_yaml", "expected_timeout"),
    [pytest.param(loss, None, None, 47, id=loss) for loss in ("sft", "sft_loss", "sft-loss", "dpo", "rm")]
    + [
        pytest.param("sft", 19, None, 19, id="timeout-cli"),
        pytest.param("sft", 19, "9", 9, id="timeout-yaml-overrides-cli"),
        pytest.param("sft", None, "null", None, id="timeout-yaml-null"),
        pytest.param("sft", None, "false", False, id="timeout-yaml-false"),
        pytest.param("sft", None, "0", 0, id="timeout-yaml-zero"),
        pytest.param("sft", 0, "9", 0, id="timeout-cli-zero-rejected-before-yaml"),
    ],
)
def test_offline_loss_selects_dataset_training(
    arguments_module, tmp_path, loss_type, timeout_cli, timeout_yaml, expected_timeout
):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    argv = ["--loss-type", loss_type]
    if timeout_cli is not None:
        argv.extend(["--sft-tq-timeout-minutes", str(timeout_cli)])
    parsed = parser.parse_args(argv)
    assert not hasattr(parsed, "sft_objective")

    # Reuse the RL configuration to verify that offline selection disables
    # rollout, advantages, and the critic even when an RL estimator is set.
    args = _opd_args()
    args.loss_type = parsed.loss_type
    args.task_type = parsed.task_type
    args.rollout_temperature = parsed.rollout_temperature
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
    args.dpo_beta = parsed.dpo_beta
    args.ref_load = None
    args.dpo_reference_free = loss_type == "dpo"
    args.sft_train_data_prefetch = True
    args.per_rank_fetch = True
    args.sft_max_in_flight_steps = 2
    args.distributed_timeout_minutes = 47
    args.sft_tq_timeout_minutes = parsed.sft_tq_timeout_minutes
    if timeout_yaml is not None:
        custom_config = tmp_path / "config.yaml"
        custom_config.write_text(f"sft_tq_timeout_minutes: {timeout_yaml}\n", encoding="utf-8")
        args.custom_config_path = str(custom_config)

    if timeout_cli == 0:
        with pytest.raises(ValueError, match="--sft-tq-timeout-minutes must be > 0"):
            arguments_module.slime_validate_args(args)
        assert args.sft_tq_timeout_minutes == expected_timeout
        return

    arguments_module.slime_validate_args(args)

    assert args.loss_type == ("sft" if loss_type in {"sft_loss", "sft-loss"} else loss_type)
    assert not args.compute_advantages_and_returns
    assert not args.use_critic
    assert not args.use_opd
    assert not args.rollout_global_dataset
    assert not hasattr(args, "eval_datasets")
    assert args.balance_data
    assert args.max_staleness == 1
    assert type(args.sft_tq_timeout_minutes) is type(expected_timeout)
    assert args.sft_tq_timeout_minutes == expected_timeout


@pytest.mark.parametrize("argv", [["--sft-objective", "dpo"], ["--sft-objective=dpo"]])
def test_removed_sft_objective_is_rejected_before_backend_parsing(arguments_module, monkeypatch, argv):
    monkeypatch.setattr(arguments_module.sys, "argv", ["train", "--loss-type", "sft", *argv])

    with pytest.raises(ValueError, match="--sft-objective has been removed.*--loss-type"):
        arguments_module._parse_args_impl()


@pytest.mark.parametrize(
    ("cli", "targets", "checks_tq"),
    [
        ([], ["linear_qkv", "linear_proj"], False),
        (["--lora-target-modules", "q_proj", "--fully-async"], ["q_proj"], True),
    ],
    ids=["declared-defaults", "explicit-lora-and-async"],
)
def test_main_parser_reads_initialized_options(
    arguments_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    cli: list[str],
    targets: list[str],
    checks_tq: bool,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--debug-train-only", *cli])
    router = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    monkeypatch.setattr(arguments_module, "RouterArgs", router)
    backend = ModuleType("relax.backends.megatron.arguments")

    def parse_backend(
        extra_args_provider: Callable[[argparse.ArgumentParser], argparse.ArgumentParser], **_kwargs: object
    ) -> argparse.Namespace:
        return extra_args_provider(argparse.ArgumentParser()).parse_known_args()[0]

    backend.megatron_parse_args = parse_backend
    backend.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(arguments_module, "slime_validate_args", lambda args, **_kwargs: None)
    monkeypatch.setattr(arguments_module, "configure_sft_data_pipeline", lambda args: None)
    version_checks: list[bool] = []
    monkeypatch.setattr(arguments_module, "check_transfer_queue_version", lambda: version_checks.append(True))

    parsed = arguments_module._parse_args_impl()

    assert parsed.lora_target_modules == targets
    assert parsed.fully_async is checks_tq
    assert version_checks == ([True] if checks_tq else [])


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
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = arguments_module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = _opd_args()
    vars(args).update(
        loss_type="dpo",
        task_type=parser.get_default("task_type"),
        rollout_temperature=parser.get_default("rollout_temperature"),
        prompt_data=["/train.jsonl"],
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=4096,
        n_samples_per_prompt=1,
        qkv_format="thd",
        use_gloo_process_groups=True,
        preference_max_length=parser.get_default("preference_max_length"),
        preference_max_completion_length=parser.get_default("preference_max_completion_length"),
        dpo_beta=parser.get_default("dpo_beta"),
        enable_weights_backuper=parser.get_default("enable_weights_backuper"),
        ref_update_interval=parser.get_default("ref_update_interval"),
        sft_oversize_strategy="keep",
        sft_oversize_custom_function_path=None,
        dpo_reference_free=parser.get_default("dpo_reference_free"),
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


@pytest.mark.parametrize("mode", ["bridge", "raw"])
def test_i2i_sft_preserves_native_checkpoint_and_image_batching(arguments_module, tmp_path, mode):
    args = _opd_args()
    vars(args).update(
        loss_type="sft",
        generation_task="i2i",
        train_backend="fsdp",
        prompt_data=["/train.jsonl"],
        use_dynamic_batch_size=False,
        balance_data=False,
        n_samples_per_prompt=1,
        sft_oversize_strategy="keep",
        sft_oversize_custom_function_path=None,
        load=str(tmp_path / "i2i_sft"),
        start_rollout_id=7,
        megatron_to_hf_mode=mode,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
    )
    arguments_module.slime_validate_args(args)
    assert args.load == str(tmp_path / "i2i_sft")
    assert args.start_rollout_id == 7
    assert not args.balance_data
    assert not args.use_dynamic_batch_size
    assert not args.no_load_optim
    assert not args.no_load_rng
    assert not args.finetune


@pytest.mark.parametrize("cli,expected", [([], "balanced"), (["--sft-packing-strategy", "binpack"], "binpack")])
def test_sft_packing_preserves_default_and_archived_recipe(arguments_module, cli, expected):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    assert parser.parse_args(cli).sft_packing_strategy == expected

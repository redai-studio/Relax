# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.cpu.relax_parser import parser_for_flags


@pytest.fixture()
def arguments_module(monkeypatch):
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], True),
        (["--recompute-loss-function-use-reentrant"], True),
        (["--no-recompute-loss-function-use-reentrant"], False),
    ],
)
def test_recompute_loss_function_use_reentrant_option(arguments_module, argv, expected):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(argv)

    assert args.recompute_loss_function_use_reentrant is expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "error"),
        (["--sft-invalid-multimodal-strategy", "skip"], "skip"),
    ],
)
def test_sft_invalid_multimodal_strategy_option(arguments_module, argv, expected):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(argv)

    assert args.sft_invalid_multimodal_strategy == expected


def _opd_args() -> SimpleNamespace:
    from relax.utils.opd.opd_utils import add_opd_arguments

    parser = parser_for_flags(
        "--task-type",
        "--generation-task",
        "--calculate-per-token-loss",
        "--sft-async-prepack",
        "--mtp-only-training",
        "--mtp-detach-paths",
        "--save-hf",
        "--save-hf-dtype",
        "--save-hf-fp8-quant-mode",
        "--save-hf-fp8-block-size",
        "--save-hf-post-hook-path",
        "--save-lora-only",
        "--sft-training-mode",
        "--sft-cpt-template",
        "--sft-chunked-logits",
        "--lora-rank",
        "--lora-adapter-mode",
        "--lora-merge-mode",
        "--lora-target-modules",
        "--lora-scope",
        "--genrm-instances",
        "--genrm-model-path",
        "--sft-max-in-flight-steps",
        "--sft-tq-timeout-minutes",
        "--use-rollout-indexer-replay",
        "--ref-ckpt-step",
        "--custom-dataset-class-path",
        "--sft-predict-interval",
        "--data-pad-size-multiplier",
        "--rollout-max-context-len",
        "--rm-type-fallback",
        "--num-labels",
        "--selective-offload",
        "--train-env-vars",
        "--sft-train-data-prefetch",
        "--rollout-result-dir",
        constants=("_MTP_DETACH_PATHS",),
    )
    opd_args = add_opd_arguments(argparse.ArgumentParser()).parse_args(
        ["--use-opd", "--opd-type", "sglang", "--opd-teacher-timeout-s", "600", "--teacher-hf-checkpoint", "/teacher"]
    )
    return SimpleNamespace(
        **vars(parser.parse_args([])),
        **vars(opd_args),
        loss_type="grpo",
        eval_config=None,
        eval_prompt_data=None,
        disable_thinking=False,
        max_staleness=0,
        use_agentic_rollout=False,
        partial_rollout=False,
        use_rollout_routing_replay=False,
        distributed_timeout_minutes=30,
        kl_coef=0.0,
        kl_loss_coef=0.0,
        use_kl_loss=False,
        ref_load="/student",
        multimodal_keys=None,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        rm_url="http://teacher/generate",
        megatron_to_hf_mode="bridge",
        load=None,
        hf_checkpoint="/student",
        eval_interval=None,
        eval_size=None,
        save_interval=None,
        save=None,
        advantage_estimator="grpo",
        normalize_advantages=False,
        rollout_batch_size=4,
        n_samples_per_prompt=8,
        global_batch_size=32,
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        load_debug_rollout_data=None,
        critic_num_gpus_per_node=None,
        critic_num_nodes=None,
        critic_load=None,
        critic_lr=None,
        lr=1e-6,
        offload=False,
        debug_rollout_only=False,
        debug_train_only=False,
        balance_data=False,
        rollout_num_gpus=4,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload_train=None,
        offload_rollout=None,
        rollout_function_path="relax.engine.rollout.generate_rollout",
        eval_function_path=None,
        num_steps_per_rollout=None,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=True,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_prompt_len=None,
        qkv_format="sbhd",
        train_backend="megatron",
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        rotate_ckpt=False,
        async_save=False,
        genrm_engine_config=None,
        genrm_sampling_config=None,
        colocate=False,
        hybrid=False,
        fully_async=False,
    )


def test_opd_sampled_token_loss_is_accepted(arguments_module):
    args = _opd_args()
    args.opd_kl_coef = 0.0
    args.opd_loss_coef = 1.0
    args.opd_token_selection = "student_sampled"

    # student_sampled + loss mode is supported via the 1D reverse-KL path
    # (see compute_policy_opd_loss); validation must not reject it.
    arguments_module.slime_validate_args(args)


@pytest.mark.parametrize("is_offline", [False, True], ids=["disabled", "offline"])
def test_opd_validator_does_not_require_fields_on_inactive_paths(is_offline: bool) -> None:
    from relax.utils.opd.opd_utils import validate_opd_args

    args = parser_for_flags("--use-opd", source_file="relax/utils/opd/opd_utils.py").parse_args(
        ["--use-opd"] if is_offline else []
    )
    before = vars(args).copy()

    validate_opd_args(args, is_offline=is_offline)

    assert vars(args) == before


@pytest.mark.parametrize(("kl_coef", "loss_coef"), [(None, 1.0), (1.0, None)], ids=["loss", "advantage"])
def test_opd_validator_preserves_null_coefficients_and_zero_eos(
    kl_coef: float | None, loss_coef: float | None
) -> None:
    from relax.utils.opd.opd_utils import validate_opd_args

    args = _opd_args()
    args.opd_kl_coef = kl_coef
    args.opd_loss_coef = loss_coef
    args.opd_student_eos_token_id = 0
    args.opd_teacher_eos_token_id = 0
    args.opd_teacher_response_prefix_token_ids = []

    validate_opd_args(args, is_offline=False)

    assert (args.opd_kl_coef, args.opd_loss_coef) == (kl_coef, loss_coef)
    assert (args.opd_student_eos_token_id, args.opd_teacher_eos_token_id) == (0, 0)
    assert args.opd_teacher_response_prefix_token_ids == []
    assert args.opd_teacher_key == "data_source"


@pytest.mark.parametrize(
    ("field", "flag"), [("opd_per_token_clip", "--opd-per-token-clip"), ("opd_is_clip", "--opd-is-clip")]
)
def test_opd_validator_rejects_zero_clip_as_an_explicit_value(field: str, flag: str) -> None:
    from relax.utils.opd.opd_utils import validate_opd_args

    args = _opd_args()
    setattr(args, field, 0.0)

    with pytest.raises(ValueError, match=f"{flag} must be > 0"):
        validate_opd_args(args, is_offline=False)


def test_opd_routes_keep_empty_key_and_report_the_effective_default() -> None:
    from relax.utils.opd.opd_utils import validate_opd_args

    args = _opd_args()
    args.opd_teacher_key = ""
    args.opd_teacher_routes = '{"source":"/teacher"}'
    args.teacher_hf_checkpoint = None
    log = Mock()

    validate_opd_args(args, is_offline=False, log=log)

    assert args.opd_teacher_key == ""
    assert args.opd_teacher_routes_map is None
    log.info.assert_called_once_with(
        "MOPD managed multi-teacher enabled: %d teachers, key='%s', sources=%s", 1, "data_source", ["source"]
    )


def test_managed_opd_teacher_colocate_preserves_rollout_resource_split(arguments_module):
    args = _opd_args()
    args.colocate = True

    arguments_module.slime_validate_args(args)

    assert args.rollout_num_gpus == 4


def test_sequence_classification_accepts_multimodal_keys(arguments_module):
    args = _opd_args()
    args.loss_type = "sft"
    args.task_type = "seq_cls"
    args.label_key = "label"
    args.prompt_data = ["/train.jsonl"]
    args.num_labels = 3
    args.classification_threshold = 0.5
    args.multimodal_keys = {"image": "images"}
    args.use_dynamic_batch_size = True
    args.max_tokens_per_gpu = 4096
    args.sft_chunked_logits = False
    args.allgather_cp = False
    args.sft_oversize_strategy = "keep"
    args.custom_dataset_class_path = None
    args.save_hf = None

    arguments_module.slime_validate_args(args)

    assert args.multimodal_keys == {"image": "images"}


def test_arguments_dynamic_context_parallel_allows_sft_eval(arguments_module):
    args = _opd_args()
    args.loss_type = "sft"
    args.dynamic_context_parallel = True
    args.eval_interval = 10
    args.eval_size = 0.1
    args.prompt_data = ["/train.jsonl"]
    args.sft_oversize_strategy = "drop"
    args.sft_oversize_custom_function_path = None
    args.use_dynamic_batch_size = True
    args.max_tokens_per_gpu = 4096
    args.rollout_max_context_len = 4096

    arguments_module.slime_validate_args(args)

    assert args.dynamic_context_parallel is True
    assert args.eval_interval == 10
    assert args.eval_size == 0.1

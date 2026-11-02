# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import importlib
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from tests.support.args import build_relax_args


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


def _opd_args() -> Namespace:
    """Parameter object for the OPD (on-policy distillation) scenario.

    Only the 18 fields that differ from the declared defaults are written; the
    other 87 fields come from the parameter declarations (see
    tests/support/args.py).
    """
    return build_relax_args(
        loss_type="grpo",
        ref_load="/student",
        opd_teacher_timeout_s=600,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        rm_url="http://teacher/generate",
        megatron_to_hf_mode="bridge",
        hf_checkpoint="/student",
        rollout_batch_size=4,
        n_samples_per_prompt=8,
        global_batch_size=32,
        reward_key="reward",
        rollout_num_gpus=4,
        rollout_function_path="relax.engine.rollout.generate_rollout",
        num_rollout=1,
        qkv_format="sbhd",
    )


def test_opd_sampled_token_loss_is_accepted(arguments_module):
    args = _opd_args()
    args.opd_kl_coef = 0.0
    args.opd_loss_coef = 1.0
    args.opd_token_selection = "student_sampled"

    # student_sampled + loss mode is supported via the 1D reverse-KL path
    # (see compute_policy_opd_loss); validation must not reject it.
    arguments_module.slime_validate_args(args)


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

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from tests.cpu.relax_parser import args_for_flags, parser_for_flags
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


def _cpt_args(*flags: str) -> SimpleNamespace:
    parser = parser_for_flags(
        "--sft-training-mode",
        "--sft-cpt-template",
        "--task-type",
        "--label-key",
        "--eval-label-key",
        "--multimodal-keys",
        "--conversation-key-map",
        "--system-prompt",
        "--tool-key",
        "--eval-tool-key",
        "--apply-chat-template-kwargs",
        "--sft-loss-last-turn-only",
        "--sft-ignore-empty-think",
        "--sft-predict-interval",
        "--custom-dataset-class",
        "--sft-oversize-strategy",
    )
    return parser.parse_args(
        ["--sft-training-mode", "cpt", "--tool-key", "", *flags], namespace=SimpleNamespace(loss_type="sft")
    )


def test_cpt_accepts_the_sft_causal_lm_backend(arguments_module):
    arguments_module._validate_cpt_args(_cpt_args())


def test_multimodal_cpt_accepts_supported_dataset_configuration(arguments_module):
    args = _cpt_args(
        "--sft-cpt-template",
        "multimodal",
        "--multimodal-keys",
        '{"image":"images"}',
        "--conversation-key-map",
        '{"from":"role"}',
        "--apply-chat-template-kwargs",
        '{"enable_thinking":false}',
    )
    arguments_module._validate_cpt_args(args)


def test_cpt_treats_explicit_null_zero_and_empty_values_as_disabled(arguments_module):
    args = _cpt_args()
    args.multimodal_keys = None
    args.apply_chat_template_kwargs = {}
    args.sft_predict_interval = 0
    args.sft_ignore_empty_think = False
    arguments_module._validate_cpt_args(args)


def test_default_sft_and_rl_are_unchanged(arguments_module):
    for loss_type in ("sft", "policy_loss"):
        arguments_module._validate_cpt_args(
            args_for_flags("--sft-training-mode", "--sft-cpt-template", loss_type=loss_type)
        )


def test_qwen_template_requires_cpt_mode(arguments_module):
    args = args_for_flags("--sft-training-mode", "--sft-cpt-template", loss_type="sft", sft_cpt_template="qwen3_5")
    with pytest.raises(ValueError, match="requires --sft-training-mode cpt"):
        arguments_module._validate_cpt_args(args)


@pytest.mark.parametrize(
    "option, value",
    [
        ("loss_type", "policy_loss"),
        ("task_type", "seq_cls"),
        ("label_key", "answer"),
        ("eval_label_key", "answer"),
        ("multimodal_keys", {"image": "image_path"}),
        ("system_prompt", "system"),
        ("tool_key", "tools"),
        ("eval_tool_key", "tools"),
        ("conversation_key_map", {"from": "role"}),
        ("apply_chat_template_kwargs", {"enable_thinking": True}),
        ("sft_loss_last_turn_only", True),
        ("sft_ignore_empty_think", True),
        ("sft_predict_interval", 10),
        ("custom_dataset_class_path", "custom.Dataset"),
        ("sft_oversize_strategy", "custom"),
    ],
)
def test_cpt_rejects_unsupported_options(arguments_module, option, value):
    args = _cpt_args()
    setattr(args, option, value)
    with pytest.raises(ValueError):
        arguments_module._validate_cpt_args(args)

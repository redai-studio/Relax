# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


def test_cpt_accepts_the_sft_causal_lm_backend(arguments_module):
    arguments_module._validate_cpt_args(
        SimpleNamespace(sft_training_mode="cpt", loss_type="sft", task_type="causal_lm")
    )


def test_default_sft_and_rl_are_unchanged(arguments_module):
    arguments_module._validate_cpt_args(SimpleNamespace(loss_type="sft"))
    arguments_module._validate_cpt_args(SimpleNamespace(loss_type="policy_loss"))


def test_qwen_template_requires_cpt_mode(arguments_module):
    with pytest.raises(ValueError, match="requires --sft-training-mode cpt"):
        arguments_module._validate_cpt_args(SimpleNamespace(loss_type="sft", sft_cpt_template="qwen3_5"))


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
    args = SimpleNamespace(sft_training_mode="cpt", loss_type="sft", task_type="causal_lm")
    setattr(args, option, value)
    with pytest.raises(ValueError):
        arguments_module._validate_cpt_args(args)

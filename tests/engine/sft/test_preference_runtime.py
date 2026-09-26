# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-fast validation for offline preference objectives."""

from argparse import Namespace

import pytest

from relax.engine.sft.runtime import is_preference_mode, validate_preference_args


def _args(**overrides) -> Namespace:
    values = {
        "loss_type": "sft",
        "sft_objective": "dpo",
        "custom_dataset_class_path": None,
        "multimodal_keys": None,
        "n_samples_per_prompt": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "dynamic_context_parallel": False,
        "qkv_format": "thd",
        "fully_async": False,
        "hybrid": False,
        "use_gloo_process_groups": True,
        "sft_chunked_logits": False,
        "enable_mtp_training": False,
        "calculate_per_token_loss": False,
        "lora_rank": 0,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "sft_predict_interval": None,
        "eval_interval": None,
        "eval_prompt_data": None,
        "eval_size": None,
        "dpo_beta": 0.1,
        "rollout_temperature": 1.0,
        "dpo_reference_free": False,
        "dpo_reference_repository": "Qwen/Qwen3-0.6B",
        "dpo_reference_revision": "fixed-revision",
        "ref_load": None,
        "ref_update_interval": None,
        "enable_weights_backuper": True,
        "preference_max_length": 1024,
        "preference_max_completion_length": 512,
        "seq_length": 2048,
    }
    values.update(overrides)
    return Namespace(**values)


def test_preference_mode_is_nested_under_sft():
    assert is_preference_mode(_args())
    assert not is_preference_mode(_args(loss_type="policy_loss"))
    assert not is_preference_mode(_args(sft_objective="causal_lm"))
    assert not is_preference_mode(_args(sft_objective="reward_model"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sft_async_prepack": True}, "sft-async-prepack"),
        ({"task_type": "seq_cls"}, "task-type causal_lm"),
        ({"mtp_only_training": True}, "MTP-only"),
        ({"n_samples_per_prompt": 2}, "n-samples-per-prompt"),
        ({"tensor_model_parallel_size": 2}, "TP=CP=PP=1"),
        ({"context_parallel_size": 2}, "TP=CP=PP=1"),
        ({"dynamic_context_parallel": True}, "dynamic context"),
        ({"qkv_format": "bshd"}, "qkv-format thd"),
        ({"use_gloo_process_groups": False}, "use-gloo-process-groups"),
        ({"lora_rank": 8}, "LoRA"),
        ({"hidden_dropout": 0.1}, "dropout"),
        ({"ref_update_interval": 10}, "frozen reference"),
        ({"ref_load": "/tmp/ref"}, "do not use --ref-load"),
        ({"dpo_reference_free": True, "ref_load": "/tmp/ref"}, "do not use --ref-load"),
        ({"dpo_beta": float("nan")}, "finite and positive"),
        ({"rollout_temperature": 0.8}, "rollout-temperature 1.0"),
        ({"rollout_temperature": float("nan")}, "rollout-temperature 1.0"),
        ({"rollout_temperature": float("inf")}, "rollout-temperature 1.0"),
        ({"preference_max_completion_length": 2048}, "must not exceed"),
        ({"eval_prompt_data": ["heldout", "eval.jsonl"]}, "follow-up reward-modeling PR"),
    ],
)
def test_preference_validation_rejects_unsupported_configs(overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        validate_preference_args(_args(**overrides))


def test_reference_free_dpo_does_not_require_ref_update_constraint():
    validate_preference_args(_args(dpo_reference_free=True, ref_update_interval=10))


def test_standard_dpo_requires_explicit_reference_repository_and_revision():
    with pytest.raises(ValueError, match="dpo-reference-repository"):
        validate_preference_args(_args(dpo_reference_repository=None))

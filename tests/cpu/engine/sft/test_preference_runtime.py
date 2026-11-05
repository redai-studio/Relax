# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-fast validation for offline preference objectives."""

import json
from argparse import Namespace

import pytest

from relax.engine.sft.bootstrap import resolve_sft_algo_key, resolve_sft_num_rollout, validate_sft_resource
from relax.engine.sft.runtime import is_preference_mode, validate_preference_args
from tests.cpu.relax_parser import parser_for_flags


def _args(**overrides) -> Namespace:
    values = {
        "loss_type": "dpo",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "dynamic_context_parallel": False,
        "use_gloo_process_groups": True,
        "calculate_per_token_loss": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "eval_interval": None,
        "ref_load": None,
        "seq_length": 2048,
    }
    args = parser_for_flags(
        "--eval-size",
        "--sft-predict-interval",
        "--task-type",
        "--rollout-temperature",
        "--dpo-beta",
        "--dpo-reference-free",
        "--preference-max-length",
        "--preference-max-completion-length",
        "--ref-update-interval",
        "--disable-weights-backuper",
        "--save-hf",
        "--sft-async-prepack",
        "--mtp-only-training",
        "--custom-dataset-class",
        "--multimodal-keys",
        "--n-samples-per-prompt",
        "--qkv-format",
        "--fully-async",
        "--hybrid",
        "--sft-chunked-logits",
        "--enable-mtp-training",
        "--lora-rank",
    ).parse_args([])
    vars(args).update(values, **overrides)
    return args


def test_preference_mode_uses_loss_type():
    assert is_preference_mode(_args())
    assert is_preference_mode(_args(loss_type="rm"))
    assert not is_preference_mode(_args(loss_type="policy_loss"))
    assert not is_preference_mode(_args(loss_type="sft"))


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
        ({"ref_update_interval": 0}, "frozen reference"),
        ({"enable_weights_backuper": False}, "requires --enable-weights-backuper"),
        ({"dpo_beta": float("nan")}, "finite and positive"),
        ({"dpo_beta": 0.0}, "finite and positive"),
        ({"preference_max_length": None}, "length limits must be positive"),
        ({"preference_max_completion_length": 0}, "length limits must be positive"),
        ({"rollout_temperature": 0.8}, "rollout-temperature 1.0"),
        ({"rollout_temperature": 0.0}, "rollout-temperature 1.0"),
        ({"rollout_temperature": float("nan")}, "rollout-temperature 1.0"),
        ({"rollout_temperature": float("inf")}, "rollout-temperature 1.0"),
        ({"preference_max_completion_length": 2048}, "must not exceed"),
        ({"multimodal_keys": {}}, "pure text only"),
    ],
)
def test_preference_validation_rejects_unsupported_configs(overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        validate_preference_args(_args(**overrides))


@pytest.mark.parametrize("loss_type", ["dpo", "rm"])
def test_preference_bootstrap_resolves_pair_steps_and_roles(tmp_path, loss_type):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": "Question"}],
            "chosen": {"role": "assistant", "content": "Chosen"},
            "rejected": {"role": "assistant", "content": "Rejected"},
        }
        for _ in range(4)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = _args(
        loss_type=loss_type,
        prompt_data=str(path),
        resource={"sft": [1, 0], "actor": [1, 1]},
        rollout_batch_size=2,
        num_epoch=3,
        num_rollout=None,
    )

    assert resolve_sft_algo_key(config) == "sft"
    validate_sft_resource(config)
    resolve_sft_num_rollout(config)
    assert config.num_rollout_per_epoch == 2
    assert config.num_rollout == 6

    config.resource = {"actor": [1, 1]}
    with pytest.raises(ValueError, match="missing required role 'sft'"):
        validate_sft_resource(config)


def test_reference_free_dpo_does_not_require_ref_update_constraint():
    validate_preference_args(_args(dpo_reference_free=True, ref_update_interval=10))


@pytest.mark.parametrize("ref_load", [None, "/models/local-sft"])
@pytest.mark.parametrize("reference_free", [False, True])
def test_dpo_accepts_local_reference_without_repository_metadata(ref_load, reference_free):
    args = _args(ref_load=ref_load, dpo_reference_free=reference_free)
    validate_preference_args(args)
    assert args.ref_load == ref_load
    assert args.dpo_reference_free is reference_free


def test_reward_model_accepts_hf_load_and_held_out_evaluation():
    validate_preference_args(
        _args(
            loss_type="rm",
            ref_load="/models/Qwen3-0.6B",
            eval_prompt_data=["task31", "heldout.parquet"],
            rollout_temperature=0.8,
            enable_weights_backuper=False,
            save_hf=None,
        )
    )


def test_reward_model_rejects_hf_export():
    with pytest.raises(ValueError, match="RM training does not support --save-hf"):
        validate_preference_args(_args(loss_type="rm", save_hf="/models/rm-{rollout_id}"))


def test_dpo_and_causal_sft_keep_hf_export_support():
    validate_preference_args(_args(save_hf="/models/dpo-{rollout_id}"))
    validate_preference_args(_args(loss_type="sft", save_hf="/models/sft-{rollout_id}"))


@pytest.mark.parametrize(
    "update", [{"sft_async_prepack": True}, {"task_type": "seq_cls"}, {"mtp_only_training": True}]
)
def test_reward_model_rejects_conflicting_sft_paths(update):
    with pytest.raises(ValueError):
        validate_preference_args(_args(loss_type="rm", **update))

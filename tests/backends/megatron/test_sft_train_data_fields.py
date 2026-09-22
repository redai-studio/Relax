# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT-mode train() must request a smaller data_fields list and use sft_*
partition naming."""

from argparse import Namespace


def _mk_actor_args(loss_type: str):
    return Namespace(
        loss_type=loss_type,
        compute_advantages_and_returns=(loss_type != "sft"),
        debug_train_only=False,
        offload_train=False,
        offload_rollout=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        global_batch_size=2,
        use_rollout_routing_replay=False,
        multimodal_keys=None,
        use_opd=False,
        opd_type=None,
        opd_log_prob_top_k=0,
    )


def test_sft_data_fields_excludes_rl_only_keys():
    """In SFT mode, data_fields must NOT include rollout_log_probs / rewards /
    raw_reward."""
    from relax.utils.training.data_fields import build_data_fields

    args = _mk_actor_args(loss_type="sft")
    fields = build_data_fields(args)

    assert "tokens" in fields
    assert "loss_masks" in fields
    assert "total_lengths" in fields
    assert "response_lengths" in fields

    for forbidden in ("rollout_log_probs", "rewards", "raw_reward", "teacher_log_probs"):
        assert forbidden not in fields, f"SFT data_fields leaked RL key: {forbidden}"


def test_sequence_classification_sft_requests_classification_labels():
    from relax.utils.training.data_fields import build_data_fields

    args = _mk_actor_args(loss_type="sft")
    args.task_type = "seq_cls"

    fields = build_data_fields(args)

    assert "classification_labels" in fields
    assert "sample_weights" not in fields


def test_rl_data_fields_include_replay_metadata():
    """RL training exposes sample identity and reward metadata for replay."""
    from relax.utils.training.data_fields import build_data_fields

    args = _mk_actor_args(loss_type="policy_loss")
    fields = build_data_fields(args)

    for required in (
        "tokens",
        "loss_masks",
        "rollout_log_probs",
        "rewards",
        "sample_indices",
        "sample_index_mask_sums",
        "raw_reward",
        "group_index",
    ):
        assert required in fields


def test_sft_partition_naming_uses_sft_prefix():
    """SFT mode → 'sft_{step}' / 'sft_train'; RL mode → 'train_{step}' /
    'train'."""
    from relax.engine.sft.runtime import sft_partition_id, sft_task_name

    args = _mk_actor_args(loss_type="sft")
    assert sft_partition_id(args, 7) == "sft_7"
    assert sft_task_name(args, component="backend") == "sft_train"

    rl_args = _mk_actor_args(loss_type="policy_loss")
    assert sft_partition_id(rl_args, 7) == "train_7"
    assert sft_task_name(rl_args, component="backend") == "train"

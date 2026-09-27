# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace


def _base_rollout_fields(args: Namespace) -> list[str]:
    fields = [
        "tokens",
        "total_lengths",
        "response_lengths",
        "loss_masks",
        "rollout_log_probs",
        "rewards",
        "sample_indices",
        "sample_index_mask_sums",
        "raw_reward",
        "group_index",
    ]
    if getattr(args, "use_rollout_routing_replay", False):
        fields.append("rollout_routed_experts")
    if getattr(args, "multimodal_keys", None) is not None:
        fields.append("multimodal_train_inputs")
    return fields


def build_data_fields(args: Namespace, *, consumer: str = "actor") -> list[str]:
    """Decide which fields to pull from TransferQueue for training.

    ``consumer`` is only meaningful for PPO (``critic``, ``advantages``,
    ``actor``); other algorithms ignore it and receive the base rollout fields.
    """
    if getattr(args, "loss_type", None) == "sft":
        fields = ["tokens", "total_lengths", "response_lengths", "loss_masks"]
        if getattr(args, "task_type", "causal_lm") == "seq_cls":
            fields.append("classification_labels")
        if args.multimodal_keys is not None:
            fields.append("multimodal_train_inputs")
        return fields

    from relax.algorithms import algorithm_needs_critic

    has_critic = algorithm_needs_critic(args)

    if has_critic and consumer == "critic":
        return _base_rollout_fields(args)

    if has_critic and consumer == "advantages":
        # PPO colocate never runs actor_fwd, so ref/log_probs are only
        # requested when kl_coef != 0 (i.e. an actor_fwd role is present).
        fields = _base_rollout_fields(args)
        fields.append("values")
        if getattr(args, "kl_coef", 0.0) != 0:
            if not getattr(args, "use_rollout_logprobs", False) and not getattr(args, "true_on_policy_mode", False):
                fields.append("log_probs")
            fields.append("ref_log_probs")
        if getattr(args, "use_opd", False):
            from relax.utils.opd.opd_utils import consume_opd_train_data

            consume_opd_train_data(fields, args)
        return fields

    fields = _base_rollout_fields(args)
    if has_critic:
        # PPO colocate: actor consumes critic's ``values`` and computes GAE
        # inline. Fully_async: standalone Advantages service produces
        # ``advantages``/``returns``, actor just pulls the finished tensors.
        if getattr(args, "fully_async", False) and not getattr(args, "hybrid", False):
            fields.extend(["advantages", "returns"])
        else:
            fields.append("values")
            if getattr(args, "kl_coef", 0.0) != 0:
                fields.append("ref_log_probs")
        # log_probs is produced inline by actor's forward pass in train_actor
        # (see relax/backends/megatron/actor.py). PPO does not deploy an
        # actor_fwd role in any mode, so nothing writes log_probs to the
        # TransferQueue; requesting it here would hang async_get_meta.
        if getattr(args, "kl_coef", 0.0) != 0 or getattr(args, "use_kl_loss", False):
            if "ref_log_probs" not in fields:
                fields.append("ref_log_probs")
    if getattr(args, "use_opd", False):
        from relax.utils.opd.opd_utils import consume_opd_train_data

        consume_opd_train_data(fields, args)
    return fields

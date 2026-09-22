# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mode predicates and naming helpers shared across the SFT path.

These are the bits previously duplicated as ``_sft_*`` private functions in
``backends/megatron/actor.py`` and ``components/actor.py``. Centralising them
here keeps the dispatchers in those files to one-line calls.
"""

from argparse import Namespace

import numpy as np

from relax.utils.env import Envs


def resolve_sft_eval_split(total_size: int, eval_size: float | int | None) -> tuple[int, int]:
    """Return ``(train_size, eval_size)`` for an SFT split."""
    if eval_size is None:
        return total_size, 0
    if eval_size < 1:
        n_eval = max(1, int(total_size * eval_size))
    else:
        n_eval = int(eval_size)
    n_eval = min(n_eval, max(total_size - 1, 0))
    return total_size - n_eval, n_eval


def resolve_sft_split_indices(
    total_size: int, eval_size: float | int | None, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return deterministic, disjoint train/eval row IDs in ms-swift order.

    ms-swift draws a seed from ``RandomState(data_seed)`` for the Hugging Face
    ``train_test_split`` call. The test rows are the leading part of that
    permutation and the train rows are the remainder. It then draws separate
    seeds to shuffle the train and eval datasets. Training applies its shuffle
    later in :class:`IndexManager`; eval has no sampler, so its dataset shuffle
    is applied here.
    """
    train_size, n_eval = resolve_sft_eval_split(total_size, eval_size)
    random_state = np.random.RandomState(seed)
    seed_max = np.iinfo(np.int32).max
    split_seed = int(random_state.randint(0, seed_max))
    split_indices = np.random.default_rng(split_seed).permutation(total_size)
    eval_indices_array = split_indices[:n_eval]
    train_indices_array = split_indices[n_eval : n_eval + train_size]

    # Consume the train dataset-shuffle seed here to preserve ms-swift's RNG
    # sequence. SFTStreamingDataset applies this draw via
    # ``dataset_seed_offset=1`` before the Megatron sampler permutation.
    random_state.randint(0, seed_max)
    eval_shuffle_seed = int(random_state.randint(0, seed_max))
    eval_order = np.random.default_rng(eval_shuffle_seed).permutation(n_eval)

    train_indices = tuple(int(index) for index in train_indices_array)
    eval_indices = tuple(int(index) for index in eval_indices_array[eval_order])
    return train_indices, eval_indices


def is_sft_mode(args: Namespace) -> bool:
    """Single source of truth for the "are we training SFT?" check.

    ``args.loss_type == "sft"`` is the canonical signal across argparse,
    controller wiring, components, and the Megatron backend.
    """
    return getattr(args, "loss_type", None) == "sft"


def should_skip_mtp_only_weight_management(
    args: Namespace,
    *,
    with_ref: bool = False,
    with_opd_teacher: bool = False,
) -> bool:
    """Return whether a pure MTP-only SFT actor needs no weight snapshots or
    rollout sync."""
    return bool(
        getattr(args, "mtp_only_training", False)
        and is_sft_mode(args)
        and getattr(args, "sft_predict_interval", None) is None
        and not getattr(args, "offload_train", False)
        and not with_ref
        and not with_opd_teacher
        and not getattr(args, "keep_old_actor", False)
    )


def should_bypass_main_output_layer(args: Namespace) -> bool:
    """Return whether training needs hidden states instead of main language-
    model logits."""
    return getattr(args, "mtp_only_training", False) or should_use_sft_chunked(args)


def should_use_sft_chunked(args: Namespace) -> bool:
    """Return whether regular SFT explicitly enabled chunked language-model
    logits."""
    return is_sft_mode(args) and getattr(args, "sft_chunked_logits", False)


def sft_partition_id(args: Namespace, step: int) -> str:
    return f"sft_{step}" if is_sft_mode(args) else f"train_{step}"


def sft_tq_num_shards(args: Namespace) -> int:
    """Number of TQ shard partitions per SFT step.

    Kept as an env knob while this path is experimental so launch scripts can
    do A/B tests without adding a public CLI surface.
    """
    if not is_sft_mode(args) or not getattr(args, "sft_async_prepack", False):
        return 1
    return max(1, Envs.RELAX_SFT_TQ_SHARDS)


def sft_partition_ids(args: Namespace, step: int) -> list[str]:
    base_partition_id = sft_partition_id(args, step)
    num_shards = sft_tq_num_shards(args)
    if num_shards == 1:
        return [base_partition_id]
    return [f"{base_partition_id}_shard_{shard_id}_of_{num_shards}" for shard_id in range(num_shards)]


def sft_logical_partition_id(partition_id: str) -> str:
    marker = "_shard_"
    if partition_id.startswith("sft_") and marker in partition_id:
        return partition_id.split(marker, 1)[0]
    return partition_id


def sft_task_name(args: Namespace, *, component: str = "actor") -> str:
    """Return the TransferQueue task name.

    ``component`` distinguishes ``components/actor.py`` (uses ``train_actor``
    for RL reset/clear) from ``backends/megatron/actor.py`` (uses ``train``
    when consuming). Both collapse to ``sft_train`` under SFT.
    """
    if is_sft_mode(args):
        return "sft_train"
    if component == "actor":
        return "train_actor"
    return "train"


def should_run_sft_eval(args: Namespace, rollout_id: int) -> bool:
    """SFT PPL eval triggers every ``--eval-interval`` steps under SFT mode
    when an eval source is configured (either ``--eval-prompt-data`` or
    ``--eval-size``, mutually exclusive — see ``utils/arguments.py``).

    Pure Megatron path; no Rollout/SGLang involvement.
    """
    if not is_sft_mode(args):
        return False
    has_eval_source = bool(getattr(args, "eval_prompt_data", None)) or (getattr(args, "eval_size", None) is not None)
    if not has_eval_source:
        return False
    interval = getattr(args, "eval_interval", None)
    if interval is None or interval <= 0:
        return False
    return (rollout_id + 1) % interval == 0


def should_run_sft_predict(args: Namespace, rollout_id: int) -> bool:
    """SFT periodic predict triggers every ``--sft-predict-interval`` steps.

    Argparse already validated ``--loss-type sft``, ``--save``, and the eval
    data source, so we only need the interval check here.
    """
    interval = getattr(args, "sft_predict_interval", None)
    if interval is None or interval <= 0:
        return False
    return (rollout_id + 1) % interval == 0

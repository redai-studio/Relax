# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller-side SFT wiring helpers.

Extracted from ``relax/core/controller.py`` so the controller stays a thin
orchestrator. Three responsibilities:

* ``resolve_sft_num_rollout(config)`` — fill in ``config.num_rollout`` /
  ``config.num_rollout_per_epoch`` for the SFT path before any actor is
  launched. RL goes through ``placement_group.py`` after RolloutManager
  init, so it doesn't go through here.
* ``resolve_sft_algo_key(config)`` — single source of truth for the
  ``ALGOS`` lookup key, parallel to ``relax.core.registry.process_role``.
* ``validate_sft_resource(config)`` — fail-fast on a missing ``sft``
  entry in ``--resource``.
"""

from argparse import Namespace

from relax.engine.sft.runtime import resolve_sft_eval_split
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _is_sft(config: Namespace) -> bool:
    return getattr(config, "loss_type", None) == "sft"


def resolve_sft_algo_key(config: Namespace) -> str:
    """Pick the ``ALGOS`` lookup key. SFT > advantage_estimator (RL).

    Same priority order as ``process_role`` in ``relax.core.registry`` so the
    ROLES set and the ALGOS dict stay consistent.
    """
    if _is_sft(config):
        return "sft"
    return config.advantage_estimator


def resolve_sft_num_rollout(config: Namespace) -> None:
    """Fill in ``config.num_rollout`` (and ``num_rollout_per_epoch``) for SFT.

    Must run before any actor is launched. No-op outside SFT.
    """
    if not _is_sft(config):
        return

    custom_dataset_class_path = getattr(config, "custom_dataset_class_path", None)
    if custom_dataset_class_path:
        if config.num_rollout is None:
            raise ValueError("--loss-type sft with --custom-dataset-class requires --num-rollout.")
        if config.num_epoch is not None:
            logger.warning("--num-epoch is ignored with --custom-dataset-class; use --num-rollout instead.")
        config.num_rollout_per_epoch = None
        assert config.num_rollout > 0
        return

    # Lazy import: pulling streaming dataset at module load would drag heavy
    # multimodal deps into every controller import.
    from relax.engine.sft.dataset.streaming import SFTStreamingDataset

    # Sized-only construction: no tokenizer/processor needed because we never
    # call get_batch — we just need len() to derive num_rollout.
    sizing_dataset = SFTStreamingDataset(path=config.prompt_data, prefetch_max_cached=0)
    total_size = len(sizing_dataset)
    dataset_size, eval_size = resolve_sft_eval_split(total_size, getattr(config, "eval_size", None))
    assert dataset_size >= config.rollout_batch_size, (
        f"SFT dataset size {dataset_size} < rollout_batch_size {config.rollout_batch_size}"
    )
    num_per_epoch, remainder = divmod(dataset_size, config.rollout_batch_size)
    config.num_rollout_per_epoch = num_per_epoch if remainder == 0 else None
    if config.num_epoch is not None:
        epoch_rollout = dataset_size * config.num_epoch // config.rollout_batch_size
        config.num_rollout = (
            min(config.num_rollout, epoch_rollout) if config.num_rollout is not None else epoch_rollout
        )
    assert config.num_rollout is not None and config.num_rollout > 0
    logger.info(
        f"SFT num_rollout resolved: {config.num_rollout} "
        f"(num_rollout_per_epoch={config.num_rollout_per_epoch}, train_size={dataset_size}, "
        f"eval_size={eval_size}, total_size={total_size})"
    )


def validate_sft_resource(config: Namespace) -> None:
    """Fail fast when ``--resource`` is missing the SFT producer role.

    Without this, the train workers would silently block on TransferQueue
    forever (the missing role's data never lands).
    """
    if not _is_sft(config):
        return
    resource = config.resource or {}
    if "sft" not in resource:
        raise ValueError(
            f"--resource is missing required role 'sft' for loss_type=sft "
            f'(SFT producer is CPU-only, e.g. "sft": [1, 0]). '
            f"Got roles: {sorted(resource.keys())}"
        )

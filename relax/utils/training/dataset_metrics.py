# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Additive per-source SFT statistics, normalized only after step reduction."""

from collections.abc import Sequence
from urllib.parse import quote

import torch


DATASET_METRIC_PREFIX = "_sft_dataset/"


@torch.no_grad()
def dataset_loss_sums(
    names: Sequence[str],
    dataset_ids: Sequence[int],
    token_losses: torch.Tensor,
    local_masks: Sequence[torch.Tensor],
    *,
    count_samples: bool,
) -> dict[str, torch.Tensor]:
    """Collect detached local sums with identical keys on every DP/CP rank.

    Loss masks and losses must already use matching CP-local coordinates. Only
    one rank of each actual CP subgroup counts samples. Empty sources retain
    zero entries so the existing step all-reduce has a fixed shape.
    """
    if len(dataset_ids) != len(local_masks):
        raise ValueError("dataset ids and loss masks must have the same sample count")
    if any(not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(names) for idx in dataset_ids):
        raise ValueError("dataset ids must be integers in the configured source vocabulary")
    lengths = [mask.numel() for mask in local_masks]
    losses = token_losses.detach().reshape(-1).float()
    if sum(lengths) != losses.numel():
        raise ValueError("per-token losses and CP-local loss masks must have the same length")
    masks = [mask.to(device=losses.device, dtype=torch.float32) for mask in local_masks]
    # Exclude masked NaN/Inf before multiplication: diagnostics must not abort
    # an otherwise recoverable nonfinite-gradient skip. Keep supervised NaN/Inf.
    row_nll = (
        torch.stack(
            [
                (torch.where(mask != 0, row, 0.0) * mask).sum()
                for row, mask in zip(losses.split(lengths), masks, strict=True)
            ]
        )
        if masks
        else losses.new_empty(0)
    )
    row_tokens = torch.stack([mask.sum() for mask in masks]) if masks else losses.new_empty(0)
    ids = torch.tensor(dataset_ids, dtype=torch.long, device=losses.device)
    sums = losses.new_zeros((3, len(names)))
    sums[0].scatter_add_(0, ids, row_nll)
    sums[1].scatter_add_(0, ids, row_tokens)
    if count_samples:
        sums[2].scatter_add_(0, ids, torch.ones_like(row_tokens))
    return {
        f"{DATASET_METRIC_PREFIX}{quote(name, safe='')}/{stat}": sums[row, idx]
        for idx, name in enumerate(names)
        for row, stat in enumerate(("nll_sum", "tokens", "samples"))
    }


def finalize_dataset_loss_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Remove raw source sums and return token-weighted losses and counts.

    Called on host scalars after microbatch accumulation and DP+CP reduction. A
    source without supervised tokens has counts but no defined loss.
    """
    sources: dict[str, dict[str, float]] = {}
    for key in list(metrics):
        if key.startswith(DATASET_METRIC_PREFIX):
            source, stat = key[len(DATASET_METRIC_PREFIX) :].rsplit("/", 1)
            sources.setdefault(source, {})[stat] = metrics.pop(key)
    result: dict[str, float] = {}
    total_tokens = sum(stats["tokens"] for stats in sources.values())
    total_samples = sum(stats["samples"] for stats in sources.values())
    for source, stats in sources.items():
        tokens = stats["tokens"]
        result[f"dataset/{source}/tokens"] = tokens
        result[f"dataset/{source}/samples"] = stats["samples"]
        result[f"dataset/{source}/token_fraction"] = tokens / total_tokens if total_tokens else 0.0
        result[f"dataset/{source}/sample_fraction"] = stats["samples"] / total_samples if total_samples else 0.0
        if tokens > 0:
            result[f"dataset/{source}/loss"] = stats["nll_sum"] / tokens
        elif stats["nll_sum"] != 0:
            raise RuntimeError(f"Dataset {source!r} has nonzero NLL with no supervised tokens")
    return result

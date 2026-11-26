# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from relax.utils.misc import load_function
from relax.utils.types import Sample


class DatasetStateMismatchError(ValueError):
    """A saved cursor does not belong to the current data sequence."""


class EvalJudgeError(RuntimeError):
    """An evaluation failed during scoring or reporting."""


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None
    # True after the producer's final sample write to this partition is acknowledged.
    partition_complete: bool | None = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    # Apply --rollout-sample-filter-path (train only). The filter sets
    # sample.remove_sample=True in-place; downstream (relax/utils/utils.py:126)
    # zeros loss_mask for those samples so they don't contribute gradient, while
    # keeping reward in GRPO group-normalization. Reloaded every call
    # (ReloadScope.IMMEDIATE per reload_utils.py:180) to support hot-swap.
    if not evaluation and isinstance(output, RolloutFnTrainOutput):
        train_args = args[0] if args else kwargs.get("args")
        filter_path = getattr(train_args, "rollout_sample_filter_path", None)
        if filter_path:
            load_function(filter_path)(train_args, output.samples)

    return output


def request_slots(group: list[Sample]) -> Iterator[int]:
    """Map physical rows back to their original sampling slots, including after
    a partial rollout."""
    slots = {}
    for row in group:
        # Unindexed inputs are independent requests.
        identity = row.index if row.index is not None else object()
        yield slots.setdefault(identity, len(slots))

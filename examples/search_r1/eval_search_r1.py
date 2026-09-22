# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search-R1 evaluation metric grouping."""

from __future__ import annotations

from typing import Any


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    del rollout_id, args, extra_metrics
    for dataset_data in tuple(data.values()):
        grouped: dict[str, dict[str, list[Any]]] = {}
        for sample, reward, truncated in zip(
            dataset_data["samples"],
            dataset_data["rewards"],
            dataset_data["truncated"],
            strict=True,
        ):
            data_source = sample.metadata["data_source"]
            source_data = grouped.setdefault(data_source, {"samples": [], "rewards": [], "truncated": []})
            source_data["samples"].append(sample)
            source_data["rewards"].append(reward)
            source_data["truncated"].append(truncated)
        for data_source, source_data in grouped.items():
            data[data_source] = source_data
    return False

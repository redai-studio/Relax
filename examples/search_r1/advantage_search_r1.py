# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Role-aware group advantage for Search-R1."""

from __future__ import annotations

import statistics
from typing import Any


_SEARCHER_ADVANTAGE_TOTAL_SCALE = 0.5


def _z_scores(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    return [(value - mean) / standard_deviation for value in values]


def advantage_func(metadata_by_slot: list[dict[str, dict[str, Any]]]) -> list[dict[str, float]] | None:
    main_rows = [
        next((name, metadata) for name, metadata in slot.items() if metadata["role"] == "main")
        for slot in metadata_by_slot
    ]
    main_scores = [float(metadata["reward/shaped"]) for _, metadata in main_rows]
    if max(main_scores) == min(main_scores):
        return None

    normalized = _z_scores(main_scores)
    result: list[dict[str, float]] = []
    for slot, (main_name, _), main_advantage in zip(metadata_by_slot, main_rows, normalized):
        slot_result = {main_name: main_advantage}
        searcher_names = [name for name, metadata in slot.items() if metadata["role"] == "searcher"]
        if searcher_names:
            per_searcher = _SEARCHER_ADVANTAGE_TOTAL_SCALE * main_advantage / len(searcher_names)
            slot_result.update({name: per_searcher for name in searcher_names})
        result.append(slot_result)
    return result

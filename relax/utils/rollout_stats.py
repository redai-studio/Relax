# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import Counter
from math import isfinite
from numbers import Integral, Real
from typing import Any, Protocol, Sequence

import numpy as np


class _SampleWithMetadata(Protocol):
    @property
    def metadata(self) -> dict[str, Any] | None: ...


_PERCENTILES = (50, 90, 95, 99)


def _normalize_stop_reason(metadata: dict[str, Any]) -> str:
    for key in ("rollout_stop_reason", "stop_reason"):
        reason = metadata.get(key)
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return "unknown"


def _normalize_turn_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        return 1
    if isinstance(value, Integral) or float(value).is_integer():
        return int(value)
    return 1


def compute_rollout_stop_and_turn_metrics(samples: Sequence[_SampleWithMetadata]) -> dict[str, float | int]:
    """Aggregate one complete rollout batch without mutating its samples.

    Stop reasons prefer ``rollout_stop_reason`` and fall back to
    ``stop_reason``; missing or invalid values are reported as ``unknown``.
    Percentiles use NumPy's default linear interpolation over normalized turn
    counts.
    """

    if not samples:
        return {}

    stop_reasons: Counter[str] = Counter()
    turn_counts: list[int] = []
    for sample in samples:
        metadata = sample.metadata or {}
        stop_reasons[_normalize_stop_reason(metadata)] += 1
        turn_counts.append(_normalize_turn_count(metadata.get("rollout_turns", 1)))

    sample_count = len(samples)
    metrics: dict[str, float | int] = {}
    for reason, count in sorted(stop_reasons.items()):
        metrics[f"stop_reason/{reason}/count"] = float(count)
        metrics[f"stop_reason/{reason}/ratio"] = count / sample_count

    percentile_values = np.percentile(turn_counts, _PERCENTILES).tolist()
    metrics.update(
        {
            "num_turn/mean": np.mean(turn_counts).item(),
            "num_turn/max": np.max(turn_counts).item(),
            "num_turn/min": np.min(turn_counts).item(),
            **{
                f"num_turn/p{percentile}": float(value)
                for percentile, value in zip(_PERCENTILES, percentile_values, strict=True)
            },
        }
    )
    return metrics

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration
from relax.utils.types import Sample


try:
    from relax.distributed.ray.rollout import _compute_spec_metrics
except ImportError:
    pytestmark = pytest.mark.skip(reason="Missing rollout dependencies")


def _record(generation_id: str, counts: SpeculativeCounts) -> dict:
    return SpeculativeGeneration("session", generation_id, f"state-{generation_id}", counts).to_dict()


def _sample(*records: dict) -> Sample:
    return Sample(spec_generations=list(records), session_id="session")


def test_compute_spec_metrics_emits_weighted_metrics() -> None:
    args = SimpleNamespace(sglang_speculative_algorithm="ngram")
    samples = [
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2)), _record("b", SpeculativeCounts(9, 10, 2, 11))),
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2))),
    ]

    metrics = _compute_spec_metrics(args, samples)

    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == 10 / 12


def test_compute_spec_metrics_keeps_disabled_non_speculative_batches_empty() -> None:
    args = SimpleNamespace(sglang_speculative_algorithm=None)
    samples = [
        Sample(
            spec_info=Sample.SpecInfo(
                counts=SpeculativeCounts(None, None, None, 5),
                legacy_counts=False,
                completion_token_num=5,
            ),
        )
    ]

    assert _compute_spec_metrics(args, samples) == {}

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples
from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration
from relax.utils.types import Sample


def _args(algorithm: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        sglang_speculative_algorithm=algorithm,
        advantage_estimator="ppo",
        log_reward_category=None,
        reward_key=None,
        log_passrate=False,
    )


def _record(generation_id: str, counts: SpeculativeCounts) -> dict:
    return SpeculativeGeneration("session", generation_id, f"state-{generation_id}", counts).to_dict()


def _sample(*records: dict) -> Sample:
    return Sample(spec_generations=list(records), session_id="session")


def test_rollout_metrics_emits_weighted_speculative_metrics() -> None:
    args = _args("ngram")
    samples = [
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2)), _record("b", SpeculativeCounts(9, 10, 2, 11))),
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2))),
    ]

    metrics = compute_metrics_from_samples(args, samples)

    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == 10 / 12
    assert metrics["spec/tokens_per_verify"] == 13 / 3


def test_rollout_metrics_omits_disabled_non_speculative_metrics() -> None:
    args = _args(None)
    samples = [
        Sample(
            spec_info=Sample.SpecInfo(
                counts=SpeculativeCounts(None, None, None, 5),
                legacy_counts=False,
                completion_token_num=5,
            ),
        )
    ]

    metrics = compute_metrics_from_samples(args, samples)
    assert not any(key.startswith(("spec/", "spec_")) for key in metrics)
    assert compute_metrics_from_samples(args, []) == {}

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight-version audit: generation spans must stay attributable to a policy
version, and unknown versions must never be silently coerced."""

from types import SimpleNamespace

from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples
from relax.utils.types import GenerationSegment, Sample
from relax.utils.weight_version_audit import parse_weight_version, sample_version_report


def _args(**overrides):
    values = dict(
        advantage_estimator="grpo",
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        n_samples_per_prompt=2,
        partial_rollout=False,
        fully_async=False,
        rollout_num_gpus=2,
        sglang_speculative_algorithm=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(response_length: int, segments: list[GenerationSegment] | None = None) -> Sample:
    return Sample(response_length=response_length, generation_segments=segments or [])


def _meta(version: str | None) -> dict:
    meta_info = {"finish_reason": {"type": "stop"}}
    if version is not None:
        meta_info["weight_version"] = version
    return meta_info


def test_update_from_meta_info_records_the_committed_span():
    sample = Sample()
    sample.response_length = 4

    sample.update_from_meta_info(_args(), _meta("7"), response_delta=4)

    assert sample.weight_versions == ["7"]
    assert len(sample.generation_segments) == 1
    segment = sample.generation_segments[0]
    assert (segment.token_start, segment.token_end) == (0, 4)
    assert segment.weight_versions == ["7"]


def test_partial_rollout_appends_adjacent_segments():
    sample = Sample()
    sample.response_length = 4
    sample.update_from_meta_info(_args(), _meta("7"), response_delta=4)

    sample.response_length = 9
    sample.update_from_meta_info(_args(), _meta("8"), response_delta=5)

    assert [(s.token_start, s.token_end) for s in sample.generation_segments] == [(0, 4), (4, 9)]
    assert [s.weight_versions for s in sample.generation_segments] == [["7"], ["8"]]


def test_metrics_separate_known_from_missing_tokens():
    samples = [
        _sample(10, [GenerationSegment(0, 0, 10, ["7"])]),
        _sample(10, [GenerationSegment(0, 0, 10, ["not-a-version"])]),
        _sample(5),
    ]

    metrics = compute_metrics_from_samples(_args(), samples, consumption_weight_version="10")

    assert metrics["weight_version/coverage"] == 0.4
    assert metrics["weight_version/missing_ratio"] == 0.6
    # 10 known tokens are the only ones that carry a lag.
    assert metrics["weight_version/lag_token_weighted_avg"] == 3.0


def test_metrics_report_generation_span_within_a_segment():
    samples = [_sample(4, [GenerationSegment(0, 0, 4, ["5", "9"])])]

    metrics = compute_metrics_from_samples(_args(), samples, consumption_weight_version="10")

    assert metrics["weight_version/span_max"] == 4.0
    assert metrics["weight_version/lag_avg"] == 3.0


def test_metrics_without_consumption_version_report_batch_age():
    samples = [_sample(10, [GenerationSegment(0, 0, 10, ["7"])])]

    metrics = compute_metrics_from_samples(_args(), samples)

    assert metrics["weight_version/batch_age_max"] == 0.0
    assert not any(key.startswith("weight_version/lag") for key in metrics)


def test_metrics_stay_empty_without_any_audited_segment():
    metrics = compute_metrics_from_samples(_args(), [_sample(10)])

    assert not any(key.startswith("weight_version/") for key in metrics)


def test_parse_weight_version_rejects_unknown_values():
    assert parse_weight_version("12") == 12
    assert parse_weight_version(None) is None
    assert parse_weight_version("v12") is None


def test_sample_dict_roundtrip_keeps_segments():
    sample = _sample(6, [GenerationSegment(1, 0, 6, ["9"], state_hash="abc")])

    restored = Sample.from_dict(sample.to_dict())

    assert restored.generation_segments == sample.generation_segments


def test_sample_report_keeps_turn_and_token_range():
    sample = _sample(6, [GenerationSegment(2, 0, 6, ["9"], state_hash="abc")])

    report = sample_version_report(sample)

    assert report["known_tokens"] == 6
    assert report["unknown_tokens"] == 0
    assert report["segments"][0]["turn_index"] == 2
    assert report["segments"][0]["token_end"] == 6
    assert report["segments"][0]["state_hash"] == "abc"

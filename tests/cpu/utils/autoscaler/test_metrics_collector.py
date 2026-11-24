# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for autoscaler metrics collection."""

import asyncio

from relax.utils.autoscaler.config import AutoscalerConfig
from relax.utils.autoscaler.metrics_collector import (
    AggregatedMetrics,
    EngineMetrics,
    HistogramQuantile,
    MetricsCollector,
    extract_histogram_quantile,
    parse_prometheus_metrics,
)


def _collector() -> MetricsCollector:
    return MetricsCollector(AutoscalerConfig())


def _engine_metric(engine_id: str, token_usage: float = 0.0) -> EngineMetrics:
    return EngineMetrics(
        engine_url=f"http://{engine_id}",
        engine_id=engine_id,
        timestamp=0.0,
        token_usage=token_usage,
    )


def test_get_aggregated_metrics_no_history_is_empty():
    collector = _collector()
    agg = collector.get_aggregated_metrics()
    assert agg.is_empty is True
    assert agg.coverage == 0.0


def test_collect_all_full_failure_yields_is_empty():
    """core: engines exist but every /metrics fetch fails -> empty snapshot.

    Must be flagged is_empty=True (frozen), not a zero-load AggregatedMetrics.
    """
    collector = _collector()
    # Session is never started -> collect_from_engine returns None for each,
    # emulating a total collection failure while engines are still active.
    engines = [{"id": "e1", "url": "http://e1"}, {"id": "e2", "url": "http://e2"}]
    snapshot = asyncio.run(collector.collect_all(engines))
    assert snapshot == {}

    collector.add_snapshot(snapshot)  # uses candidate count recorded by collect_all
    agg = collector.get_aggregated_metrics()
    assert agg.is_empty is True
    assert agg.coverage == 0.0
    # Sanity: this is the fail-safe distinction -- values are 0 but is_empty flags it.
    assert agg.avg_token_usage == 0.0


def test_coverage_full_when_all_report():
    collector = _collector()
    snapshot = {
        "e1": _engine_metric("e1", token_usage=0.5),
        "e2": _engine_metric("e2", token_usage=0.5),
    }
    collector.add_snapshot(snapshot, num_candidates=2)
    agg = collector.get_aggregated_metrics()
    assert agg.is_empty is False
    assert agg.coverage == 1.0
    assert agg.num_engines == 2


def test_coverage_partial_when_some_fail():
    """Only 1 of 2 active engines reported -> coverage 0.5 (denominator is the
    active-candidate count, not the reporting count)."""
    collector = _collector()
    snapshot = {"e1": _engine_metric("e1", token_usage=0.5)}
    collector.add_snapshot(snapshot, num_candidates=2)
    agg = collector.get_aggregated_metrics()
    assert agg.is_empty is False
    assert agg.coverage == 0.5
    assert agg.num_engines == 1


def test_coverage_denominator_recorded_by_collect_all():
    """add_snapshot without explicit num_candidates falls back to the count
    from the most recent collect_all (the active engines we tried to reach)."""
    collector = _collector()
    engines = [{"id": "e1", "url": "http://e1"}, {"id": "e2", "url": "http://e2"}]
    # Session not started -> both fail, but collect_all records 2 candidates.
    asyncio.run(collector.collect_all(engines))
    # Simulate only e1 reporting on the next stored snapshot.
    collector.add_snapshot({"e1": _engine_metric("e1")})
    agg = collector.get_aggregated_metrics()
    assert agg.coverage == 0.5


def test_aggregated_metrics_to_dict_exposes_c1_signals():
    agg = AggregatedMetrics(num_engines=1, is_empty=False, coverage=0.5)
    d = agg.to_dict()
    assert d["is_empty"] is False
    assert d["coverage"] == 0.5


def test_parse_bucket_different_le_not_summed():
    """Histogram *_bucket series with different le boundaries must NOT be
    summed together; each le is kept as a distinct key."""
    text = "\n".join(
        [
            'sglang:queue_time_seconds_bucket{le="0.5"} 3',
            'sglang:queue_time_seconds_bucket{le="1.0"} 7',
            'sglang:queue_time_seconds_bucket{le="5.0"} 10',
        ]
    )
    metrics = parse_prometheus_metrics(text)
    assert metrics['sglang:queue_time_seconds_bucket{le="0.5"}'] == 3.0
    assert metrics['sglang:queue_time_seconds_bucket{le="1.0"}'] == 7.0
    assert metrics['sglang:queue_time_seconds_bucket{le="5.0"}'] == 10.0
    # The bare (le-less) key must not exist -- otherwise buckets got merged.
    assert "sglang:queue_time_seconds_bucket" not in metrics


def test_parse_gauge_same_name_still_summed():
    """Non-bucket gauges with the same name across ranks are still aggregated
    by summing (existing behavior preserved)."""
    text = "\n".join(
        [
            'sglang:num_running_reqs{tp_rank="0"} 4',
            'sglang:num_running_reqs{tp_rank="1"} 6',
        ]
    )
    metrics = parse_prometheus_metrics(text)
    assert metrics["sglang:num_running_reqs"] == 10.0


def test_parse_bucket_inf_label_preserved():
    """The le=\"+Inf\" bucket must be retained (the [^\"]+ pattern matches.

    +Inf).
    """
    text = "\n".join(
        [
            'sglang:queue_time_seconds_bucket{le="1.0"} 5',
            'sglang:queue_time_seconds_bucket{le="+Inf"} 10',
        ]
    )
    metrics = parse_prometheus_metrics(text)
    assert metrics['sglang:queue_time_seconds_bucket{le="+Inf"}'] == 10.0


def test_parse_bucket_same_le_across_ranks_summed():
    """Same le across different ranks (tp_rank/pp_rank) still aggregates:

    cumulative counts add up so the total matches _count.
    """
    text = "\n".join(
        [
            'sglang:queue_time_seconds_bucket{le="1.0",tp_rank="0"} 3',
            'sglang:queue_time_seconds_bucket{le="1.0",tp_rank="1"} 4',
            'sglang:queue_time_seconds_bucket{le="+Inf",tp_rank="0"} 5',
            'sglang:queue_time_seconds_bucket{le="+Inf",tp_rank="1"} 6',
            'sglang:queue_time_seconds_count{tp_rank="0"} 5',
            'sglang:queue_time_seconds_count{tp_rank="1"} 6',
        ]
    )
    metrics = parse_prometheus_metrics(text)
    assert metrics['sglang:queue_time_seconds_bucket{le="1.0"}'] == 7.0
    # +Inf bucket total equals total _count (5+6=11), consistency check.
    assert metrics['sglang:queue_time_seconds_bucket{le="+Inf"}'] == 11.0
    assert metrics["sglang:queue_time_seconds_count"] == 11.0


_METRIC = "sglang:queue_time_seconds"
_COUNT = "sglang:queue_time_seconds_count"


def _hist(bucket_pairs, count):
    """Build a parsed-metrics dict from (le, cumulative_count) pairs + total
    count."""
    metrics = {_COUNT: float(count)}
    for le, c in bucket_pairs:
        metrics[f'{_METRIC}_bucket{{le="{le}"}}'] = float(c)
    return metrics


def test_extract_quantile_real_sglang_fragment_nonzero_p95():
    """Regression for a realistic sglang histogram fragment must yield a non-
    zero p95 (before the le label was dropped and this was always 0)."""
    text = "\n".join(
        [
            'sglang:queue_time_seconds_bucket{le="0.01"} 50',
            'sglang:queue_time_seconds_bucket{le="0.05"} 80',
            'sglang:queue_time_seconds_bucket{le="0.1"} 90',
            'sglang:queue_time_seconds_bucket{le="0.5"} 97',
            'sglang:queue_time_seconds_bucket{le="1.0"} 99',
            'sglang:queue_time_seconds_bucket{le="5.0"} 100',
            'sglang:queue_time_seconds_bucket{le="+Inf"} 100',
            "sglang:queue_time_seconds_count 100",
            "sglang:queue_time_seconds_sum 12.5",
        ]
    )
    metrics = parse_prometheus_metrics(text)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert isinstance(result, HistogramQuantile)
    assert result.overflow is False
    # target_count = 95 -> falls in (0.1, 0.5] bucket -> interpolated within it.
    assert 0.1 < result.value <= 0.5


def test_extract_quantile_overflow_when_target_in_inf_bucket():
    """falsification: max finite bucket 5s, but 95th percentile
    falls into +Inf -> overflow=True.

    A max_finite+EPS return (~5s) would NOT trip a 10s threshold, so the
    explicit flag is mandatory.
    """
    # Only 90/100 observations are <= 5s; the top 10 are unbounded (+Inf).
    metrics = _hist([("1.0", 50), ("5.0", 90), ("+Inf", 100)], count=100)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.overflow is True
    # value is a finite reference (max finite le), never inf (JSON-safe).
    assert result.value == 5.0


def test_extract_quantile_no_overflow_when_finite_covers_target():
    """When the target quantile is covered by finite buckets,
    overflow=False."""
    metrics = _hist([("1.0", 50), ("5.0", 99), ("+Inf", 100)], count=100)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.overflow is False
    assert result.value <= 5.0


def test_extract_quantile_only_inf_bucket_degrades_safely():
    """Only a +Inf bucket, no finite buckets -> safe degradation (value 0.0, no
    spurious overflow that would cause a永久 scale-out)."""
    metrics = _hist([("+Inf", 100)], count=100)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.value == 0.0
    assert result.overflow is False


def test_extract_quantile_zero_count_returns_zero():
    metrics = _hist([("1.0", 0), ("+Inf", 0)], count=0)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.value == 0.0
    assert result.overflow is False


def test_extract_quantile_cross_label_aggregation_matches_count():
    """Same le across ranks aggregates and stays consistent with the summed
    _count so the quantile is computed against the correct total."""
    text = "\n".join(
        [
            'sglang:queue_time_seconds_bucket{le="1.0",tp_rank="0"} 40',
            'sglang:queue_time_seconds_bucket{le="1.0",tp_rank="1"} 40',
            'sglang:queue_time_seconds_bucket{le="5.0",tp_rank="0"} 50',
            'sglang:queue_time_seconds_bucket{le="5.0",tp_rank="1"} 50',
            'sglang:queue_time_seconds_bucket{le="+Inf",tp_rank="0"} 50',
            'sglang:queue_time_seconds_bucket{le="+Inf",tp_rank="1"} 50',
            'sglang:queue_time_seconds_count{tp_rank="0"} 50',
            'sglang:queue_time_seconds_count{tp_rank="1"} 50',
        ]
    )
    metrics = parse_prometheus_metrics(text)
    # total count = 100, cumulative at le=1.0 is 80, at le=5.0 is 100.
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.overflow is False
    # target 95 within (1.0, 5.0] bucket.
    assert 1.0 < result.value <= 5.0


def test_extract_quantile_non_monotonic_buckets_do_not_crash():
    """Illegal / non-monotonic cumulative counts must be handled defensively
    (no crash, sane finite result)."""
    metrics = _hist([("1.0", 90), ("5.0", 10), ("+Inf", 100)], count=100)
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert isinstance(result, HistogramQuantile)
    assert result.value >= 0.0


def test_extract_quantile_no_buckets_returns_zero():
    """No histogram buckets present at all -> degrade to 0.0, no overflow."""
    metrics = {_COUNT: 100.0}
    result = extract_histogram_quantile(metrics, _METRIC, 0.95, _COUNT)
    assert result.value == 0.0
    assert result.overflow is False


def test_aggregated_overflow_or_reduced_across_engines():
    """Engine-level overflow flags must OR-reduce into AggregatedMetrics so the
    decision engine can treat latency as high."""
    collector = _collector()
    m1 = EngineMetrics(engine_url="http://e1", engine_id="e1", timestamp=0.0, queue_time_p95=1.0)
    m2 = EngineMetrics(
        engine_url="http://e2",
        engine_id="e2",
        timestamp=0.0,
        queue_time_p95=5.0,
        queue_time_p95_overflow=True,
    )
    collector.add_snapshot({"e1": m1, "e2": m2}, num_candidates=2)
    agg = collector.get_aggregated_metrics()
    assert agg.max_queue_time_p95_overflow is True
    assert agg.max_ttft_p95_overflow is False

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for per-field metrics validity (Task 4 change 3).

A scrape that returns HTTP 200 but omits individual Prometheus series used to
zero-fill the missing fields, making "TTFT not exposed" indistinguishable
from "TTFT is 0". EngineMetrics now records per-field
``present/observed_at/sample_count`` and exposes ``is_field_valid``; numeric
aggregation is unchanged (legacy consumers keep their behavior) while
``AggregatedMetrics.field_coverage`` annotates which zeros are real.

Run: python -m unittest tests.utils.autoscaler.test_metrics_validity -v
"""

import asyncio
import unittest
from typing import Any

from tests.utils._dep_stubs import install_web_framework_stubs


install_web_framework_stubs()  # relax.utils.autoscaler's package __init__ imports fastapi

from relax.utils.autoscaler.config import AutoscalerConfig  # noqa: E402
from relax.utils.autoscaler.metrics_collector import (  # noqa: E402
    EngineMetrics,
    MetricFieldValidity,
    MetricsCollector,
)


_FULL_SCRAPE = "\n".join(
    [
        "sglang:token_usage 0.42",
        "sglang:num_queue_reqs 3",
        "sglang:num_running_reqs 7",
        "sglang:gen_throughput 120.5",
        "sglang:max_total_num_tokens 1000000",
        "sglang:num_used_tokens 420000",
        "sglang:num_prefill_prealloc_queue_reqs 0",
        "sglang:num_prefill_inflight_queue_reqs 1",
        "sglang:num_decode_prealloc_queue_reqs 0",
        "sglang:num_decode_transfer_queue_reqs 2",
        'sglang:queue_time_seconds_bucket{le="0.5"} 3',
        'sglang:queue_time_seconds_bucket{le="1.0"} 7',
        'sglang:queue_time_seconds_bucket{le="+Inf"} 10',
        "sglang:queue_time_seconds_count 10",
        'sglang:time_to_first_token_seconds_bucket{le="0.5"} 5',
        'sglang:time_to_first_token_seconds_bucket{le="+Inf"} 5',
        "sglang:time_to_first_token_seconds_count 5",
        'sglang:inter_token_latency_seconds_bucket{le="0.1"} 8',
        'sglang:inter_token_latency_seconds_bucket{le="+Inf"} 8',
        "sglang:inter_token_latency_seconds_count 8",
        'sglang:e2e_request_latency_seconds_bucket{le="2.0"} 6',
        'sglang:e2e_request_latency_seconds_bucket{le="+Inf"} 6',
        "sglang:e2e_request_latency_seconds_count 6",
    ]
)

_TTFT_MISSING_SCRAPE = "\n".join(
    [
        "sglang:token_usage 0.42",
        "sglang:num_queue_reqs 3",
        "sglang:num_running_reqs 7",
        "sglang:queue_time_seconds_count 0",
        # no time_to_first_token, inter_token, e2e series at all
    ]
)

# Histograms exposed but carrying zero samples (idle engine).
_ZERO_SAMPLE_SCRAPE = "\n".join(
    [
        "sglang:token_usage 0.0",
        "sglang:num_queue_reqs 0",
        "sglang:num_running_reqs 0",
        'sglang:inter_token_latency_seconds_bucket{le="+Inf"} 0',
        "sglang:inter_token_latency_seconds_count 0",
        "sglang:e2e_request_latency_seconds_count 0",
    ]
)


class _FakeResponse:
    def __init__(self, status: int, text: str):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, text: str, status: int = 200):
        self._text = text
        self._status = status

    def get(self, url: str):
        return _FakeResponse(self._status, self._text)


def _collect(text: str) -> Any:
    collector = MetricsCollector(AutoscalerConfig())
    collector._session = _FakeSession(text)
    return asyncio.run(collector.collect_from_engine("http://engine", "engine_0"))


class TestFieldValidityOnCollection(unittest.TestCase):
    def test_present_fields_are_valid(self):
        metrics = _collect(_FULL_SCRAPE)
        self.assertTrue(metrics.is_field_valid("token_usage"))
        self.assertTrue(metrics.is_field_valid("num_queue_reqs"))
        self.assertTrue(metrics.is_field_valid("queue_time_p95"))
        self.assertTrue(metrics.is_field_valid("ttft_p95"))
        validity = metrics.field_validity["ttft_p95"]
        self.assertTrue(validity.present)
        self.assertEqual(validity.sample_count, 5)
        self.assertIsNotNone(validity.observed_at)

    def test_zero_sample_histogram_is_invalid(self):
        """An exposed-but-sampleless histogram (count=0) must not count as a
        valid latency observation."""
        metrics = _collect(_ZERO_SAMPLE_SCRAPE)
        self.assertFalse(metrics.is_field_valid("itl_p95"))
        self.assertFalse(metrics.is_field_valid("e2e_latency_p95"))
        self.assertEqual(metrics.field_validity["itl_p95"].sample_count, 0)
        self.assertTrue(metrics.is_field_valid("token_usage"))  # gauges unaffected

    def test_missing_series_is_invalid_not_zero_filled(self):
        """TTFT entirely absent from an HTTP-200 scrape: the value stays 0.0
        (legacy numeric behavior) but the field is flagged invalid, so it can
        no longer pollute latency conditions as a fake low value."""
        metrics = _collect(_TTFT_MISSING_SCRAPE)
        self.assertEqual(metrics.ttft_p95, 0.0)
        self.assertFalse(metrics.is_field_valid("ttft_p95"))
        self.assertFalse(metrics.field_validity["ttft_p95"].present)
        self.assertTrue(metrics.is_field_valid("token_usage"))  # present fields unaffected

    def test_missing_token_usage_no_longer_fails_collection(self):
        """Regression: the debug log used to format a missing token_usage with
        :.3f (TypeError), failing the whole collection; missing series must
        still yield a snapshot with validity annotations."""
        metrics = _collect("sglang:num_running_reqs 1\n")
        self.assertIsNotNone(metrics)
        self.assertFalse(metrics.is_field_valid("token_usage"))
        self.assertEqual(metrics.num_running_reqs, 1)

    def test_validity_serializes_in_to_dict(self):
        as_dict = _collect(_FULL_SCRAPE).to_dict()
        self.assertIn("field_validity", as_dict)
        self.assertEqual(as_dict["field_validity"]["ttft_p95"]["sample_count"], 5)
        self.assertTrue(as_dict["field_validity"]["ttft_p95"]["present"])


class TestLegacyCompatibility(unittest.TestCase):
    def test_legacy_construction_without_validity_stays_valid(self):
        """EngineMetrics built the old way (no field_validity) keeps every
        field valid so existing consumers see unchanged behavior."""
        metrics = EngineMetrics(engine_url="http://e", engine_id="e", timestamp=0.0, token_usage=0.5)
        self.assertTrue(metrics.is_field_valid("token_usage"))
        self.assertTrue(metrics.is_field_valid("ttft_p95"))

    def test_aggregated_numeric_behavior_unchanged(self):
        """Numeric aggregation (values, is_empty, coverage) is unchanged; only
        the field_coverage annotation is new."""
        collector = MetricsCollector(AutoscalerConfig())
        m1 = _collect(_FULL_SCRAPE)
        m2 = _collect(_TTFT_MISSING_SCRAPE)
        collector.add_snapshot({"e1": m1, "e2": m2}, num_candidates=2)
        agg = collector.get_aggregated_metrics()
        self.assertEqual(agg.num_engines, 2)
        self.assertAlmostEqual(agg.avg_token_usage, 0.42)  # both engines report 0.42
        self.assertAlmostEqual(agg.max_ttft_p95, 0.475)  # e1's interpolated p95; e2's zero-fill filtered (>0 rule)
        self.assertFalse(agg.max_ttft_p95_overflow)
        self.assertFalse(agg.is_empty)
        self.assertEqual(agg.coverage, 1.0)


class TestFieldCoverageAggregation(unittest.TestCase):
    def test_partial_field_coverage_distinguishes_missing_from_zero(self):
        """One of two engines missing TTFT: field_coverage is 0.5, so a
        consumer can refuse to act on ttft while numeric max stays legacy."""
        collector = MetricsCollector(AutoscalerConfig())
        m1 = _collect(_FULL_SCRAPE)
        m2 = _collect(_TTFT_MISSING_SCRAPE)
        collector.add_snapshot({"e1": m1, "e2": m2}, num_candidates=2)
        agg = collector.get_aggregated_metrics()
        self.assertAlmostEqual(agg.field_coverage["ttft_p95"], 0.5)
        self.assertAlmostEqual(agg.field_coverage["token_usage"], 1.0)
        self.assertAlmostEqual(agg.field_coverage["itl_p95"], 0.5)  # e1 sampled, e2 missing
        self.assertEqual(agg.to_dict()["field_coverage"]["ttft_p95"], 0.5)

    def test_all_fields_valid_when_scrape_complete(self):
        collector = MetricsCollector(AutoscalerConfig())
        collector.add_snapshot({"e1": _collect(_FULL_SCRAPE)}, num_candidates=1)
        agg = collector.get_aggregated_metrics()
        for field_name, coverage in agg.field_coverage.items():
            self.assertEqual(coverage, 1.0, f"field {field_name} should be fully covered")


class TestStaleness(unittest.TestCase):
    def test_max_age_rejects_stale_observations(self):
        metrics = EngineMetrics(
            engine_url="http://e",
            engine_id="e",
            timestamp=1000.0,
            token_usage=0.5,
            field_validity={"token_usage": MetricFieldValidity(present=True, observed_at=900.0, sample_count=1)},
        )
        self.assertTrue(metrics.is_field_valid("token_usage", max_age_secs=100.0))  # age 100s, bound inclusive
        self.assertFalse(metrics.is_field_valid("token_usage", max_age_secs=50.0))  # 100s > 50s bound
        self.assertTrue(metrics.is_field_valid("token_usage"))  # no bound -> no staleness check


class TestMetricFieldValidityDefaults(unittest.TestCase):
    def test_defaults_are_invalid(self):
        validity = MetricFieldValidity()
        self.assertFalse(validity.present)
        self.assertIsNone(validity.observed_at)
        self.assertEqual(validity.sample_count, 0)


if __name__ == "__main__":
    unittest.main()

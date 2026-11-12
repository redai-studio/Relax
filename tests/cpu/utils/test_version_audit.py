# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the weight version audit offline tool."""

from __future__ import annotations

import pytest

from relax.utils.agentic_obs.version_audit import (
    compute_run_summary,
    compute_sample_stats,
)


def _span(version, start, end, attempt=0):
    return {"version": version, "token_start": start, "token_end": end, "attempt_seq": attempt}


class TestSampleStats:
    def test_known_versions_full_metrics(self):
        # 两轮生成：v1 产 2 token，v3 产 2 token；消费时 current=7。
        spans = [_span("1", 0, 2), _span("3", 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="7")
        assert stats["oldest_version"] == "1"
        assert stats["newest_version"] == "3"
        assert stats["generation_span"] == 2  # 3 - 1
        assert stats["oldest_lag"] == 6  # 7 - 1
        assert stats["newest_lag"] == 4  # 7 - 3
        assert stats["token_weighted_lag"] == pytest.approx(5.0)  # (6*2+4*2)/4
        assert stats["post_generation_lag"] == 4  # == newest_lag
        assert stats["known_token_count"] == 4
        assert stats["unknown_token_count"] == 0
        assert stats["span_missing_rate"] == 0.0
        assert stats["token_missing_rate"] == 0.0

    def test_unknown_versions_excluded_from_lag_means(self):
        # 一半 token 未知：lag 只按已知 token 计，缺失率单列。
        spans = [_span("1", 0, 2), _span(None, 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="3")
        assert stats["oldest_version"] == "1"
        assert stats["token_weighted_lag"] == 2.0  # (2*2)/2
        assert stats["known_token_count"] == 2
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 0.5
        assert stats["token_missing_rate"] == pytest.approx(0.5)

    def test_all_unknown_reports_null_not_zero(self):
        stats = compute_sample_stats([_span(None, 0, 3)], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["newest_version"] is None
        assert stats["generation_span"] is None
        assert stats["oldest_lag"] is None
        assert stats["newest_lag"] is None
        assert stats["token_weighted_lag"] is None
        assert stats["post_generation_lag"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 3
        assert stats["span_missing_rate"] == 1.0

    def test_non_numeric_version_treated_as_unknown(self):
        # 非数字版本字符串按未知处理（进缺失率），不猜测。
        spans = [_span("v1", 0, 2)]
        stats = compute_sample_stats(spans, consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 1.0

    def test_missing_consumed_version_lags_null(self):
        # 消费点版本缺失：跨度可算，lag 全 None。
        stats = compute_sample_stats([_span("1", 0, 2), _span("3", 2, 4)], consumed_weight_version=None)
        assert stats["generation_span"] == 2
        assert stats["oldest_lag"] is None
        assert stats["token_weighted_lag"] is None

    def test_empty_spans(self):
        stats = compute_sample_stats([], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["span_missing_rate"] is None  # 无 span 样本不计缺失率


class TestRunSummary:
    def test_run_level_aggregates_exclude_unknown_from_means(self):
        rows = [
            {  # 全已知
                "oldest_lag": 6,
                "newest_lag": 4,
                "token_weighted_lag": 5.0,
                "generation_span": 2,
                "known_token_count": 4,
                "unknown_token_count": 0,
                "spans": 2,
                "unknown_spans": 0,
            },
            {  # 消费点缺失 → lag None，不计入均值
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "generation_span": 2,
                "known_token_count": 4,
                "unknown_token_count": 0,
                "spans": 2,
                "unknown_spans": 0,
            },
            {  # 全未知 → lag None
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "generation_span": None,
                "known_token_count": 0,
                "unknown_token_count": 3,
                "spans": 1,
                "unknown_spans": 1,
            },
        ]
        summary = compute_run_summary(rows)
        assert summary["sample_count"] == 3
        assert summary["lag_sample_count"] == 1  # 只有第一个进均值
        assert summary["oldest_lag_mean"] == 6.0
        assert summary["newest_lag_mean"] == 4.0
        assert summary["token_weighted_lag_mean"] == 5.0
        assert summary["generation_span_mean"] == 2.0  # 前两个有已知版本
        assert summary["span_missing_rate"] == pytest.approx(1 / 5)
        assert summary["token_missing_rate"] == pytest.approx(3 / 11)
        assert summary["sample_missing_rate"] == pytest.approx(1 / 3)

    def test_median_and_max(self):
        rows = [
            {
                "oldest_lag": 1,
                "newest_lag": 1,
                "token_weighted_lag": 1.0,
                "generation_span": 0,
                "known_token_count": 1,
                "unknown_token_count": 0,
                "spans": 1,
                "unknown_spans": 0,
            },
            {
                "oldest_lag": 3,
                "newest_lag": 2,
                "token_weighted_lag": 2.5,
                "generation_span": 1,
                "known_token_count": 2,
                "unknown_token_count": 0,
                "spans": 2,
                "unknown_spans": 0,
            },
            {
                "oldest_lag": 10,
                "newest_lag": 5,
                "token_weighted_lag": 8.0,
                "generation_span": 4,
                "known_token_count": 3,
                "unknown_token_count": 0,
                "spans": 3,
                "unknown_spans": 0,
            },
        ]
        summary = compute_run_summary(rows)
        assert summary["oldest_lag_median"] == 3
        assert summary["oldest_lag_max"] == 10
        assert summary["newest_lag_median"] == 2
        assert summary["newest_lag_max"] == 5

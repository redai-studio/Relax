# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metrics collection module for autoscaler.

This module provides functionality to collect, parse, and aggregate metrics
from SGLang inference engines. It supports Prometheus-format metrics and
computes aggregated statistics for scaling decisions.
"""

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.utils.autoscaler.config import AutoscalerConfig

logger = get_logger(__name__)


class HistogramQuantile(NamedTuple):
    """Histogram quantile value and whether it exceeds all finite buckets."""

    value: float
    overflow: bool


@dataclass
class MetricFieldValidity:
    """Per-field observation metadata for one engine metrics snapshot.

    A field is *valid* for condition evaluation when it was present in the
    scraped series and carries at least one sample; gauges report
    ``sample_count=1`` when present, histogram-derived fields report the
    histogram's ``_count`` (so an exposed-but-sampleless histogram is invalid,
    matching "histogram 无样本不参与条件判断"). ``observed_at`` enables
    staleness checks via ``EngineMetrics.is_field_valid(max_age_secs=...)``.
    """

    present: bool = False
    observed_at: Optional[float] = None
    sample_count: int = 0


# EngineMetrics field name -> Prometheus gauge series it is read from.
_GAUGE_FIELD_SOURCES: Dict[str, str] = {
    "token_usage": "sglang:token_usage",
    "num_queue_reqs": "sglang:num_queue_reqs",
    "num_running_reqs": "sglang:num_running_reqs",
    "gen_throughput": "sglang:gen_throughput",
    "max_total_num_tokens": "sglang:max_total_num_tokens",
    "num_used_tokens": "sglang:num_used_tokens",
    "num_prefill_prealloc_queue_reqs": "sglang:num_prefill_prealloc_queue_reqs",
    "num_prefill_inflight_queue_reqs": "sglang:num_prefill_inflight_queue_reqs",
    "num_decode_prealloc_queue_reqs": "sglang:num_decode_prealloc_queue_reqs",
    "num_decode_transfer_queue_reqs": "sglang:num_decode_transfer_queue_reqs",
}

# EngineMetrics field name -> histogram _count series backing its quantile.
_HISTOGRAM_FIELD_SOURCES: Dict[str, str] = {
    "queue_time_p95": "sglang:queue_time_seconds_count",
    "ttft_p95": "sglang:time_to_first_token_seconds_count",
    "itl_p95": "sglang:inter_token_latency_seconds_count",
    "e2e_latency_p95": "sglang:e2e_request_latency_seconds_count",
}

_ALL_VALIDITY_FIELDS = tuple(_GAUGE_FIELD_SOURCES) + tuple(_HISTOGRAM_FIELD_SOURCES)

# Fields whose series are REQUIRED before an engine may be judged idle/low-load
# by the scale-in decision path: they back avg_token_usage / total_queue_reqs /
# throughput_variance, the three scale-in conditions. A scrape that answers
# HTTP 200 but omits any of these series leaves the engine "unknown" -- its
# zero-filled values are placeholders, never evidence of idleness.
CRITICAL_METRICS_FIELDS: Tuple[str, ...] = ("token_usage", "num_queue_reqs", "gen_throughput")


@dataclass
class EngineMetrics:
    """Single engine metrics snapshot.

    This dataclass holds a snapshot of metrics from a single SGLang engine
    at a point in time. It includes both primary scaling metrics and
    auxiliary metrics for observability.

    Attributes:
        engine_url: URL of the engine (e.g., "http://localhost:30000").
        engine_id: Unique identifier for the engine.
        timestamp: Unix timestamp when metrics were collected.
        token_usage: KV cache token usage ratio (0.0 - 1.0).
        num_queue_reqs: Number of requests waiting in queue.
        num_running_reqs: Number of requests currently being processed.
        gen_throughput: Generation throughput in tokens/second.
        max_total_num_tokens: Maximum total tokens in KV cache pool.
        num_used_tokens: Currently used tokens in KV cache.
        queue_time_p95: P95 queue waiting time in seconds.
        ttft_p95: P95 time-to-first-token in seconds.
        itl_p95: P95 inter-token latency in seconds.
        e2e_latency_p95: P95 end-to-end request latency in seconds.
        num_prefill_prealloc_queue_reqs: Requests in prefill prealloc queue.
        num_prefill_inflight_queue_reqs: Requests in prefill inflight queue.
        num_decode_prealloc_queue_reqs: Requests in decode prealloc queue.
        num_decode_transfer_queue_reqs: Requests in decode transfer queue.
    """

    engine_url: str
    engine_id: str
    timestamp: float

    # Primary scaling metrics
    token_usage: float = 0.0
    num_queue_reqs: int = 0
    num_running_reqs: int = 0
    gen_throughput: float = 0.0

    # Resource metrics
    max_total_num_tokens: int = 0
    num_used_tokens: int = 0

    # Latency metrics (from histogram quantiles)
    queue_time_p95: float = 0.0
    ttft_p95: float = 0.0
    itl_p95: float = 0.0
    e2e_latency_p95: float = 0.0

    queue_time_p95_overflow: bool = False
    ttft_p95_overflow: bool = False
    itl_p95_overflow: bool = False
    e2e_latency_p95_overflow: bool = False

    # Detailed queue metrics
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_inflight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0

    # Per-field observation metadata (Task 4): which fields were actually
    # present in the scraped series. Empty for legacy constructions, in which
    # case ``is_field_valid`` conservatively returns True so existing
    # consumers see unchanged behavior.
    field_validity: Dict[str, MetricFieldValidity] = field(default_factory=dict)

    def is_field_valid(self, field: str, max_age_secs: Optional[float] = None) -> bool:
        """Whether ``field`` was observed and may drive condition evaluation.

        A field is invalid when it was missing from the scrape or its
        histogram carried no samples; with ``max_age_secs`` it is also invalid
        when observed too long before this snapshot's ``timestamp``. Fields
        without a validity record (legacy construction) stay valid.

        Args:
            field: EngineMetrics field name (e.g. "ttft_p95", "token_usage").
            max_age_secs: Optional staleness bound relative to ``self.timestamp``.

        Returns:
            True if the field is valid for evaluation.
        """
        validity = self.field_validity.get(field)
        if validity is None:
            return True
        if not validity.present or validity.sample_count <= 0:
            return False
        if max_age_secs is not None and validity.observed_at is not None:
            if self.timestamp - validity.observed_at > max_age_secs:
                return False
        return True

    def has_critical_metrics(self) -> bool:
        """Whether every critical field was validly observed for this engine.

        Critical fields (``CRITICAL_METRICS_FIELDS``) are the series the
        scale-in (idle/low-load) decision path consumes. An engine answering
        HTTP 200 but missing one of them is "unknown": its zero-filled values
        must not be read as real idleness. Legacy constructions without
        ``field_validity`` count as complete so existing consumers keep their
        behavior.

        Returns:
            True if all critical fields are valid for condition evaluation.
        """
        return all(self.is_field_valid(name) for name in CRITICAL_METRICS_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "engine_url": self.engine_url,
            "engine_id": self.engine_id,
            "timestamp": self.timestamp,
            "token_usage": self.token_usage,
            "num_queue_reqs": self.num_queue_reqs,
            "num_running_reqs": self.num_running_reqs,
            "gen_throughput": self.gen_throughput,
            "max_total_num_tokens": self.max_total_num_tokens,
            "num_used_tokens": self.num_used_tokens,
            "queue_time_p95": self.queue_time_p95,
            "ttft_p95": self.ttft_p95,
            "itl_p95": self.itl_p95,
            "e2e_latency_p95": self.e2e_latency_p95,
            "queue_time_p95_overflow": self.queue_time_p95_overflow,
            "ttft_p95_overflow": self.ttft_p95_overflow,
            "itl_p95_overflow": self.itl_p95_overflow,
            "e2e_latency_p95_overflow": self.e2e_latency_p95_overflow,
            "num_prefill_prealloc_queue_reqs": self.num_prefill_prealloc_queue_reqs,
            "num_prefill_inflight_queue_reqs": self.num_prefill_inflight_queue_reqs,
            "num_decode_prealloc_queue_reqs": self.num_decode_prealloc_queue_reqs,
            "num_decode_transfer_queue_reqs": self.num_decode_transfer_queue_reqs,
            "field_validity": {
                name: {"present": v.present, "observed_at": v.observed_at, "sample_count": v.sample_count}
                for name, v in self.field_validity.items()
            },
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all engines.

    This dataclass holds aggregated statistics computed from individual
    engine metrics, suitable for scaling decisions.

    Attributes:
        num_engines: Total number of engines reporting metrics.
        total_queue_reqs: Sum of queue requests across all engines.
        total_running_reqs: Sum of running requests across all engines.
        avg_token_usage: Average token usage ratio across engines.
        total_throughput: Total generation throughput in tokens/second.
        max_queue_time_p95: Maximum P95 queue time across engines.
        max_ttft_p95: Maximum P95 TTFT across engines.
        max_itl_p95: Maximum P95 ITL across engines.
        throughput_variance: Relative variance in throughput over time window.
        is_empty: True when no engine reported metrics this cycle (empty snapshot).
            This distinguishes "no data" (collection fully failed -> freeze scaling)
            from "data says load is 0" (engines idle -> scale-in is legitimate).
        coverage: Fraction of active engines that successfully reported metrics
            (num_reporting / num_active_candidates), in [0.0, 1.0]. Used by the
            decision engine to gate scaling when metrics are only partially available.
        unknown_engines: Engines that answered HTTP 200 but were missing at
            least one critical series (token usage / queue depth / throughput).
            Their zero-filled values are excluded from the load aggregates and
            the decision engine must treat them as "unknown" -- never as idle --
            when judging scale-in.
        timestamp: Unix timestamp of aggregation.
    """

    num_engines: int = 0
    total_queue_reqs: int = 0
    total_running_reqs: int = 0
    avg_token_usage: float = 0.0
    total_throughput: float = 0.0
    max_queue_time_p95: float = 0.0
    max_ttft_p95: float = 0.0
    max_itl_p95: float = 0.0
    throughput_variance: float = 0.0
    # Overflow flags OR-reduced across engines: True when any engine's p95 for
    # that latency fell into the ``le="+Inf"`` bucket. Consumed by the decision
    # engine's latency scale-out conditions (value>threshold OR overflow).
    max_queue_time_p95_overflow: bool = False
    max_ttft_p95_overflow: bool = False
    max_itl_p95_overflow: bool = False
    is_empty: bool = False
    coverage: float = 1.0
    timestamp: float = field(default_factory=time.time)
    # Fraction of reporting engines for which each field was validly observed
    # (present + sampled). Lets consumers distinguish "field says 0" from
    # "field was never exposed" without changing the numeric aggregation.
    field_coverage: Dict[str, float] = field(default_factory=dict)
    # Engines whose scrape answered HTTP 200 but missed at least one critical
    # series. Excluded from the load aggregates above; the decision engine
    # freezes scale-in while this list is non-empty (missing != idle).
    unknown_engines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_engines": self.num_engines,
            "total_queue_reqs": self.total_queue_reqs,
            "total_running_reqs": self.total_running_reqs,
            "avg_token_usage": self.avg_token_usage,
            "total_throughput": self.total_throughput,
            "max_queue_time_p95": self.max_queue_time_p95,
            "max_ttft_p95": self.max_ttft_p95,
            "max_itl_p95": self.max_itl_p95,
            "throughput_variance": self.throughput_variance,
            "max_queue_time_p95_overflow": self.max_queue_time_p95_overflow,
            "max_ttft_p95_overflow": self.max_ttft_p95_overflow,
            "max_itl_p95_overflow": self.max_itl_p95_overflow,
            "is_empty": self.is_empty,
            "coverage": self.coverage,
            "field_coverage": dict(self.field_coverage),
            "unknown_engines": list(self.unknown_engines),
            "timestamp": self.timestamp,
        }


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text format into a metric dictionary.

    This function parses the Prometheus exposition format and extracts
    metric values into a flat dictionary. For most metrics with the same name
    but different labels (e.g., from different tp_rank/pp_rank), values are
    aggregated by summing (appropriate for gauges like num_running_reqs).

    Histogram bucket boundaries are preserved while equal boundaries across
    ranks are summed.

    Args:
        text: Raw Prometheus metrics text.

    Returns:
        Dictionary mapping metric keys to aggregated values. Bucket keys carry
        the ``le`` label (``name{le="..."}``); all other keys are the bare name.
    """
    metrics: Dict[str, float] = {}
    # Track metrics that appear multiple times (different labels)
    metric_counts: Dict[str, int] = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse: metric_name{labels} value or metric_name value
        # Match the metric name and optional labels
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)((?:\{[^}]*\})?)?\s+(.+)$", line)
        if not match:
            continue

        name = match.group(1)
        labels = match.group(2) or ""
        value_str = match.group(3)

        try:
            value = float(value_str)
        except ValueError:
            continue

        if name.endswith("_bucket"):
            le_match = re.search(r'le="([^"]+)"', labels)
            key = f'{name}{{le="{le_match.group(1)}"}}' if le_match else name
        else:
            key = name

        if key in metrics:
            metrics[key] += value
            metric_counts[key] += 1
        else:
            metrics[key] = value
            metric_counts[key] = 1

    # Log aggregation info for debugging
    aggregated_metrics = {k: v for k, v in metric_counts.items() if v > 1}
    if aggregated_metrics:
        logger.debug(f"Aggregated metrics with multiple label combinations: {aggregated_metrics}")

    return metrics


def extract_histogram_quantile(
    metrics: Dict[str, float],
    metric_name: str,
    quantile: float,
    count_key: str,
) -> HistogramQuantile:
    """Extract approximate quantile from Prometheus histogram buckets.

    This function computes an approximate quantile value from histogram
    bucket cumulative counts. It uses linear interpolation within buckets.

    If the target falls into ``le="+Inf"``, returns the largest finite boundary
    with ``overflow=True``.

    Args:
        metrics: Dictionary of metric key to value (bucket keys carry the le
            label, ``name{le="..."}``, as produced by parse_prometheus_metrics).
        metric_name: Base name of the histogram metric (e.g., "sglang:queue_time_seconds").
        quantile: Target quantile (e.g., 0.95 for P95).
        count_key: Key for the total count metric.

    Returns:
        HistogramQuantile(value, overflow). Value is always finite.
    """
    total_count = metrics.get(count_key, 0)
    if total_count == 0:
        return HistogramQuantile(0.0, False)

    finite_buckets: List[tuple] = []
    bucket_prefix = f"{metric_name}_bucket"
    le_pattern = re.compile(r'le="([^"]+)"')

    for key, value in metrics.items():
        if not key.startswith(bucket_prefix):
            continue
        match = le_pattern.search(key)
        if not match:
            continue
        le_str = match.group(1)
        # +Inf bucket is not a finite boundary; its presence just means overflow
        # is possible (detected below when the loop exhausts finite buckets).
        if le_str.lstrip("+").lower() in ("inf", "infinity"):
            continue
        try:
            le = float(le_str)
        except ValueError:
            continue
        finite_buckets.append((le, float(value)))

    if not finite_buckets:
        # No finite buckets (e.g. only +Inf present). Cannot estimate a quantile;
        # degrade safely to 0.0 rather than emit a spurious overflow that would
        # pin scale-out on forever.
        return HistogramQuantile(0.0, False)

    # Sort by bucket boundary
    finite_buckets.sort(key=lambda x: x[0])
    max_finite_le = finite_buckets[-1][0]

    # Find the bucket containing the target quantile
    target_count = total_count * quantile

    prev_count = 0.0
    prev_le = 0.0

    for le, cumulative_count in finite_buckets:
        # Defend against non-monotonic / illegal cumulative counts: clamp so the
        # sequence stays non-decreasing (a bad bucket contributes no new mass).
        if cumulative_count < prev_count:
            cumulative_count = prev_count

        if cumulative_count >= target_count:
            # Linear interpolation within bucket
            if cumulative_count == prev_count:
                return HistogramQuantile(le, False)  # Empty bucket, use boundary

            bucket_range = le - prev_le
            count_in_bucket = cumulative_count - prev_count
            count_needed = target_count - prev_count

            if count_in_bucket > 0:
                interpolation = count_needed / count_in_bucket
                return HistogramQuantile(prev_le + bucket_range * interpolation, False)
            return HistogramQuantile(le, False)

        prev_count = cumulative_count
        prev_le = le

    # Target quantile exceeds every finite bucket -> it lies in the +Inf bucket.
    # Overflow: real quantile is >= max_finite_le, possibly far larger. Return an
    # explicit flag with a finite reference value (JSON-safe, never inf).
    return HistogramQuantile(max_finite_le, True)


class MetricsCollector:
    """Collects metrics from SGLang engines periodically.

    This class manages the collection of metrics from multiple engines,
    maintains a history of metrics snapshots, and provides aggregated
    statistics for scaling decisions.

    Attributes:
        config: AutoscalerConfig instance.
        _history: Deque of historical metrics snapshots.
        _session: aiohttp ClientSession for HTTP requests.
    """

    def __init__(self, config: "AutoscalerConfig"):
        """Initialize the metrics collector.

        Args:
            config: Autoscaler configuration.
        """
        self.config = config
        history_size = max(
            1,
            int(config.condition_window_secs / config.metrics_interval_secs),
        )
        self._history: deque = deque(maxlen=history_size)
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        # Number of active engines used as the coverage denominator.
        self._last_num_candidates: int = 0

    async def start(self) -> None:
        """Initialize async resources (HTTP session)."""
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0),
            )
            logger.info("MetricsCollector started")
        except ImportError:
            logger.error("aiohttp not installed, metrics collection will fail")
            raise

    async def stop(self) -> None:
        """Clean up async resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("MetricsCollector stopped")

    async def collect_from_engine(
        self,
        engine_url: str,
        engine_id: str,
    ) -> Optional[EngineMetrics]:
        """Fetch and parse metrics from a single engine.

        Args:
            engine_url: Base URL of the engine.
            engine_id: Unique identifier for the engine.

        Returns:
            EngineMetrics snapshot, or None if collection failed.
        """
        if self._session is None:
            logger.warning("HTTP session not initialized")
            return None

        metrics_url = f"{engine_url.rstrip('/')}/metrics"

        try:
            async with self._session.get(metrics_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch metrics from {engine_url}: HTTP {response.status}")
                    return None

                text = await response.text()
                raw_metrics = parse_prometheus_metrics(text)

                # Debug log key metrics for troubleshooting
                token_usage_raw = raw_metrics.get("sglang:token_usage")
                token_usage_repr = f"{token_usage_raw:.3f}" if token_usage_raw is not None else "N/A"
                logger.debug(
                    f"Collected metrics from {engine_id}: "
                    f"num_running_reqs={raw_metrics.get('sglang:num_running_reqs', 'N/A')}, "
                    f"num_queue_reqs={raw_metrics.get('sglang:num_queue_reqs', 'N/A')}, "
                    f"token_usage={token_usage_repr}"
                )

                # Extract histogram quantiles (structured: value + overflow flag)
                queue_time_q = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:queue_time_seconds",
                    0.95,
                    "sglang:queue_time_seconds_count",
                )
                ttft_q = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:time_to_first_token_seconds",
                    0.95,
                    "sglang:time_to_first_token_seconds_count",
                )
                itl_q = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:inter_token_latency_seconds",
                    0.95,
                    "sglang:inter_token_latency_seconds_count",
                )
                e2e_q = extract_histogram_quantile(
                    raw_metrics,
                    "sglang:e2e_request_latency_seconds",
                    0.95,
                    "sglang:e2e_request_latency_seconds_count",
                )

                # Per-field validity: gauges are valid when their series was
                # present; histogram quantiles additionally require samples
                # (count > 0). Values above keep their legacy zero-fill for
                # backward compatibility; validity tells consumers which
                # zeros are real.
                timestamp = time.time()
                field_validity: Dict[str, MetricFieldValidity] = {
                    name: MetricFieldValidity(
                        present=series in raw_metrics,
                        observed_at=timestamp,
                        sample_count=1 if series in raw_metrics else 0,
                    )
                    for name, series in _GAUGE_FIELD_SOURCES.items()
                }
                for name, count_series in _HISTOGRAM_FIELD_SOURCES.items():
                    count = int(raw_metrics.get(count_series, 0))
                    field_validity[name] = MetricFieldValidity(
                        present=count_series in raw_metrics,
                        observed_at=timestamp,
                        sample_count=count,
                    )

                return EngineMetrics(
                    engine_url=engine_url,
                    engine_id=engine_id,
                    timestamp=timestamp,
                    token_usage=raw_metrics.get("sglang:token_usage", 0.0),
                    num_queue_reqs=int(raw_metrics.get("sglang:num_queue_reqs", 0)),
                    num_running_reqs=int(raw_metrics.get("sglang:num_running_reqs", 0)),
                    gen_throughput=raw_metrics.get("sglang:gen_throughput", 0.0),
                    max_total_num_tokens=int(raw_metrics.get("sglang:max_total_num_tokens", 0)),
                    num_used_tokens=int(raw_metrics.get("sglang:num_used_tokens", 0)),
                    queue_time_p95=queue_time_q.value,
                    ttft_p95=ttft_q.value,
                    itl_p95=itl_q.value,
                    e2e_latency_p95=e2e_q.value,
                    queue_time_p95_overflow=queue_time_q.overflow,
                    ttft_p95_overflow=ttft_q.overflow,
                    itl_p95_overflow=itl_q.overflow,
                    e2e_latency_p95_overflow=e2e_q.overflow,
                    num_prefill_prealloc_queue_reqs=int(raw_metrics.get("sglang:num_prefill_prealloc_queue_reqs", 0)),
                    num_prefill_inflight_queue_reqs=int(raw_metrics.get("sglang:num_prefill_inflight_queue_reqs", 0)),
                    num_decode_prealloc_queue_reqs=int(raw_metrics.get("sglang:num_decode_prealloc_queue_reqs", 0)),
                    num_decode_transfer_queue_reqs=int(raw_metrics.get("sglang:num_decode_transfer_queue_reqs", 0)),
                    field_validity=field_validity,
                )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching metrics from {engine_url}")
            return None
        except Exception as e:
            logger.warning(f"Error collecting metrics from {engine_url}: {e}")
            return None

    async def collect_all(
        self,
        engines: List[Dict[str, str]],
    ) -> Dict[str, EngineMetrics]:
        """Collect metrics from all engines concurrently.

        Args:
            engines: List of engine info dicts with 'id' and 'url' keys.

        Returns:
            Dictionary mapping engine_id to EngineMetrics.
        """
        if not engines:
            return {}

        # Remember how many active engines we are trying to collect from so the
        # coverage denominator reflects active candidates (not reporting count).
        self._last_num_candidates = len(engines)

        tasks = [self.collect_from_engine(engine["url"], engine["id"]) for engine in engines]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics_dict: Dict[str, EngineMetrics] = {}
        for engine, result in zip(engines, results):
            if isinstance(result, BaseException):
                logger.warning(f"Exception collecting from {engine['id']}: {result}")
            elif result is not None:
                metrics_dict[result.engine_id] = result

        return metrics_dict

    def add_snapshot(self, snapshot: Dict[str, EngineMetrics], num_candidates: Optional[int] = None) -> None:
        """Add a metrics snapshot to history.

        Args:
            snapshot: Dictionary mapping engine_id to EngineMetrics.
            num_candidates: Number of active engines we attempted to collect from
                this cycle (coverage denominator). Defaults to the count recorded
                by the most recent ``collect_all`` call.
        """
        if num_candidates is None:
            num_candidates = self._last_num_candidates
        self._history.append(
            {
                "timestamp": time.time(),
                "metrics": snapshot,
                "num_candidates": num_candidates,
            }
        )

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Compute aggregated metrics from recent history.

        Returns:
            AggregatedMetrics instance with computed statistics.
        """
        if not self._history:
            return AggregatedMetrics(is_empty=True, coverage=0.0)

        # Get latest snapshot
        latest_entry = self._history[-1]
        latest = latest_entry["metrics"]
        # Coverage denominator: number of active engines we tried to collect from.
        num_candidates = latest_entry.get("num_candidates", len(latest))

        if not latest:
            return AggregatedMetrics(is_empty=True, coverage=0.0)

        # Aggregate across engines.
        num_engines = len(latest)

        # Engines whose scrape was missing a critical series are "unknown":
        # their zero-filled token_usage / queue / throughput values are missing
        # data, not real idleness, so they are excluded from every load
        # aggregate the scale-in (idle/low-load) decision path consumes.
        unknown_engines = [engine_id for engine_id, m in latest.items() if not m.has_critical_metrics()]
        known_metrics = [m for m in latest.values() if m.has_critical_metrics()]

        # Aggregation denominators (capacity-ratio semantics):
        # - avg_token_usage divides by the number of KNOWN engines only, so a
        #   missing (zero-filled) series cannot drag the mean down and fake
        #   "low load". When every engine is known this equals the legacy
        #   num_engines denominator, so the healthy path is unchanged.
        # - Load totals (queue/running/throughput) sum KNOWN engines only: an
        #   unknown engine contributes no evidence in either direction.
        # - num_engines and coverage keep their legacy meaning (reporting
        #   engines over active candidates), so scale-out gates such as
        #   queue_backlog's per-engine denominator are unchanged.
        total_queue = sum(m.num_queue_reqs for m in known_metrics)
        total_running = sum(m.num_running_reqs for m in known_metrics)
        total_throughput = sum(m.gen_throughput for m in known_metrics)
        avg_token_usage = sum(m.token_usage for m in known_metrics) / len(known_metrics) if known_metrics else 0.0

        # Compute max P95 latencies
        all_queue_times = [m.queue_time_p95 for m in latest.values() if m.queue_time_p95 > 0]
        all_ttft = [m.ttft_p95 for m in latest.values() if m.ttft_p95 > 0]
        all_itl = [m.itl_p95 for m in latest.values() if m.itl_p95 > 0]

        max_queue_time_p95 = max(all_queue_times) if all_queue_times else 0.0
        max_ttft_p95 = max(all_ttft) if all_ttft else 0.0
        max_itl_p95 = max(all_itl) if all_itl else 0.0

        # Any engine overflow makes the aggregate latency overflow.
        max_queue_time_p95_overflow = any(m.queue_time_p95_overflow for m in latest.values())
        max_ttft_p95_overflow = any(m.ttft_p95_overflow for m in latest.values())
        max_itl_p95_overflow = any(m.itl_p95_overflow for m in latest.values())

        # Compute throughput variance over time window
        throughput_variance = 0.0
        if len(self._history) >= 3:
            # Same known-engine scope as total_throughput: a frame where an
            # engine's critical series went missing must not inject a fake
            # 0 tok/s sample into the variance window.
            throughputs = [
                sum(m.gen_throughput for m in h["metrics"].values() if m.has_critical_metrics())
                for h in list(self._history)[-10:]
            ]
            if throughputs:
                mean_t = sum(throughputs) / len(throughputs)
                if mean_t > 0:
                    max_deviation = max(abs(t - mean_t) for t in throughputs)
                    throughput_variance = max_deviation / mean_t

        # Coverage: fraction of active candidates that reported metrics.
        if num_candidates > 0:
            coverage = min(1.0, num_engines / num_candidates)
        else:
            coverage = 1.0

        # Per-field coverage: fraction of reporting engines whose field was
        # validly observed (present + sampled). Numeric aggregation above is
        # intentionally unchanged; this only annotates which zeros are real.
        field_coverage: Dict[str, float] = {}
        if num_engines > 0:
            field_coverage = {
                field_name: sum(1 for m in latest.values() if m.is_field_valid(field_name)) / num_engines
                for field_name in _ALL_VALIDITY_FIELDS
            }

        return AggregatedMetrics(
            num_engines=num_engines,
            total_queue_reqs=total_queue,
            total_running_reqs=total_running,
            avg_token_usage=avg_token_usage,
            total_throughput=total_throughput,
            max_queue_time_p95=max_queue_time_p95,
            max_ttft_p95=max_ttft_p95,
            max_itl_p95=max_itl_p95,
            throughput_variance=throughput_variance,
            max_queue_time_p95_overflow=max_queue_time_p95_overflow,
            max_ttft_p95_overflow=max_ttft_p95_overflow,
            max_itl_p95_overflow=max_itl_p95_overflow,
            is_empty=False,
            coverage=coverage,
            field_coverage=field_coverage,
            unknown_engines=unknown_engines,
            timestamp=time.time(),
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the metrics history for debugging/observability.

        Returns:
            List of historical snapshots.
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self._history.clear()

    def get_condition_duration(
        self,
        condition_name: str,
        check_fn: Callable[[Dict[str, EngineMetrics]], bool],
    ) -> float:
        """Check how long a condition has been continuously true.

        Args:
            condition_name: Name of the condition (for logging).
            check_fn: Function that takes a metrics snapshot and returns True
                if the condition is met.

        Returns:
            Duration in seconds that the condition has been continuously true.
        """
        duration = 0.0
        interval = self.config.metrics_interval_secs

        for snapshot in reversed(list(self._history)):
            if check_fn(snapshot["metrics"]):
                duration += interval
            else:
                break

        return duration

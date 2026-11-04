# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from relax.utils.metrics.sglang_exporter import SGLangMetricsExporter


METRICS = """# HELP sglang:num_running_reqs Running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="a\\\"b",tp_rank="0"} 3
# HELP sglang:requests_total Requests
# TYPE sglang:requests_total counter
sglang:requests_total{model_name="a\\\"b"} 12
# HELP sglang:latency_seconds Latency
# TYPE sglang:latency_seconds histogram
sglang:latency_seconds_bucket{le="0.5"} 2
sglang:latency_seconds_bucket{le="+Inf"} 4
sglang:latency_seconds_sum 2.5
sglang:latency_seconds_count 4
# HELP process_cpu_seconds_total CPU
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 99
"""


def _pool(*urls: str, complete: bool = True) -> dict:
    return {
        "models": {
            "default": {
                "engine_groups": [
                    {
                        "worker_type": "regular",
                        "engines": [{"rank": i, "status": "active", "url": url} for i, url in enumerate(urls)],
                    }
                ]
            }
        },
        "observation": {
            "complete": complete,
            "issues": [] if complete else [{"scope": "default", "reason": "pending"}],
        },
    }


def _samples(body: bytes, name: str) -> list:
    return [
        sample
        for family in text_string_to_metric_families(body.decode())
        for sample in family.samples
        if sample.name == name
    ]


def _value(body: bytes, name: str) -> float:
    return _samples(body, name)[0].value


@pytest.fixture
def upstream(monkeypatch):
    client = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            "relax.utils.metrics.sglang_exporter.httpx.AsyncClient",
            lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
        )

    return install


async def test_sglang_metrics_preserve_families_labels_and_buckets(upstream):
    upstream(lambda request: httpx.Response(200, text=METRICS))
    discovery = _pool("http://engine-a", "http://engine-b")
    discovery["models"]["reward"] = _pool("http://engine-c")["models"]["default"]
    body = await SGLangMetricsExporter(AsyncMock(return_value=discovery)).scrape()

    assert _value(body, "relax_sglang_scrape_success") == 1
    assert _value(body, "relax_sglang_engines_discovered") == 3
    running = _samples(body, "sglang:num_running_reqs")
    assert [sample.value for sample in running] == [3, 3, 3]
    assert {sample.labels["relax_engine"] for sample in running} == {
        "http://engine-a",
        "http://engine-b",
        "http://engine-c",
    }
    assert {sample.labels["relax_model"] for sample in running} == {"default", "reward"}
    assert all(sample.labels["model_name"] == 'a"b' and sample.labels["tp_rank"] == "0" for sample in running)
    assert len(_samples(body, "sglang:requests_total")) == 3
    assert len(_samples(body, "sglang:latency_seconds_bucket")) == 6
    assert {sample.labels["le"] for sample in _samples(body, "sglang:latency_seconds_bucket")} == {"0.5", "+Inf"}
    assert _value(body, "sglang:latency_seconds_sum") == 2.5
    assert _value(body, "sglang:latency_seconds_count") == 4
    assert body.count(b"# HELP sglang:latency_seconds ") == 1
    assert body.count(b"# TYPE sglang:latency_seconds ") == 1
    assert b"process_cpu_seconds" not in body


async def test_sglang_metrics_deduplicate_urls_and_skip_non_http_ranks(upstream):
    urls = []

    def respond(request):
        urls.append(str(request.url))
        return httpx.Response(200, text=METRICS)

    upstream(respond)
    data = _pool("http://engine-a", "http://engine-a/")
    data["models"]["default"]["engine_groups"][0]["engines"] += [
        {"rank": 2, "status": "active"},
        {"rank": 3, "status": "dead", "url": "http://dead"},
    ]
    body = await SGLangMetricsExporter(AsyncMock(return_value=data)).scrape()
    assert urls == ["http://engine-a/metrics"]
    assert _value(body, "relax_sglang_engines_discovered") == 1


@pytest.mark.parametrize("failure", ["http", "timeout", "invalid", "empty", "duplicate", "labels", "schema", "type"])
async def test_sglang_metrics_partial_failure_preserves_successful_engines(upstream, failure):
    def respond(request):
        if request.url.host == "good":
            return httpx.Response(200, text=METRICS)
        if failure == "http":
            return httpx.Response(503)
        if failure == "timeout":
            raise httpx.ReadTimeout("upstream timed out")
        text = {
            "invalid": 'sglang:broken{model="oops} 1',
            "empty": "process_cpu_seconds_total 1\n",
            "duplicate": "sglang:duplicate 1\nsglang:duplicate 2\n",
            "labels": 'sglang:reserved{relax_engine="fake"} 1\n',
            "schema": METRICS.replace("Running requests", "Conflicting documentation"),
            "type": METRICS.replace("sglang:requests_total counter", "sglang:requests_total gauge"),
        }[failure]
        return httpx.Response(200, text=text)

    upstream(respond)
    body = await SGLangMetricsExporter(AsyncMock(return_value=_pool("http://good", "http://bad"))).scrape()
    assert _value(body, "relax_sglang_discovery_success") == 1
    assert _value(body, "relax_sglang_scrape_success") == 0
    assert {s.labels["relax_engine"]: s.value for s in _samples(body, "relax_sglang_engine_up")} == {
        "http://good": 1,
        "http://bad": 0,
    }
    for family in text_string_to_metric_families(body.decode()):
        if family.name.startswith("sglang:"):
            assert all(s.labels["relax_engine"] == "http://good" for s in family.samples)


@pytest.mark.parametrize("data", [None, {}, {"models": {}}, _pool("file:///tmp/metrics"), _pool("http://user@host")])
async def test_sglang_metrics_invalid_discovery_is_not_a_healthy_empty_pool(data):
    body = await SGLangMetricsExporter(AsyncMock(return_value=data)).scrape()
    assert _value(body, "relax_sglang_discovery_success") == 0
    assert _value(body, "relax_sglang_scrape_success") == 0


async def test_sglang_metrics_empty_pool_and_incomplete_discovery(upstream):
    upstream(lambda request: httpx.Response(200, text=METRICS))
    discover = AsyncMock(side_effect=[_pool(), _pool("http://good", complete=False)])
    exporter = SGLangMetricsExporter(discover)
    empty = await exporter.scrape()
    assert _value(empty, "relax_sglang_discovery_success") == 1
    assert _value(empty, "relax_sglang_engines_discovered") == 0
    assert not _samples(empty, "relax_sglang_engine_up")
    partial = await exporter.scrape()
    assert _value(partial, "relax_sglang_discovery_success") == 0
    assert _value(partial, "relax_sglang_scrape_success") == 0
    assert _value(partial, "relax_sglang_engine_up") == 1
    assert _samples(partial, "sglang:num_running_reqs")


async def test_sglang_metrics_recover_after_startup_and_remove_stale_engines(upstream):
    upstream(lambda request: httpx.Response(200, text=METRICS))
    discover = AsyncMock(side_effect=[RuntimeError("not deployed"), _pool("http://a", "http://b"), _pool("http://b")])
    exporter = SGLangMetricsExporter(discover)
    assert _value(await exporter.scrape(), "relax_sglang_scrape_success") == 0
    assert _value(await exporter.scrape(), "relax_sglang_engines_discovered") == 2
    body = await exporter.scrape()
    assert _value(body, "relax_sglang_engines_discovered") == 1
    assert {s.labels["relax_engine"] for s in _samples(body, "sglang:num_running_reqs")} == {"http://b"}


async def test_sglang_metrics_discovery_has_a_deadline():
    cancelled = asyncio.Event()

    async def discover():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    body = await asyncio.wait_for(SGLangMetricsExporter(discover, discovery_timeout=0.02).scrape(), 1)
    assert cancelled.is_set()
    assert _value(body, "relax_sglang_discovery_success") == 0


async def test_sglang_metrics_bound_concurrency_and_cancel_over_budget_scrapes(upstream):
    running = peak = cancelled = 0

    async def respond(request):
        nonlocal running, peak, cancelled
        running += 1
        peak = max(peak, running)
        try:
            if request.url.host == "fast":
                return httpx.Response(200, text=METRICS)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            running -= 1

    upstream(respond)
    exporter = SGLangMetricsExporter(
        AsyncMock(return_value=_pool("http://fast", "http://slow1", "http://slow2", "http://queued")),
        concurrency=2,
        scrape_timeout=0.1,
    )
    body = await asyncio.wait_for(exporter.scrape(), 1)
    assert peak == 2 and running == 0 and cancelled == 2
    assert [s.value for s in _samples(body, "relax_sglang_engine_up")] == [1, 0, 0, 0]
    assert _value(body, "relax_sglang_scrape_success") == 0


async def test_sglang_metrics_share_inflight_scrapes_and_survive_caller_cancellation(upstream):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, text=METRICS)

    upstream(respond)
    discover = AsyncMock(return_value=_pool("http://engine"))
    exporter = SGLangMetricsExporter(discover)
    first = asyncio.create_task(exporter.scrape())
    await entered.wait()
    second = asyncio.create_task(exporter.scrape())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert _value(await second, "relax_sglang_scrape_success") == 1
    assert discover.await_count == calls == 1

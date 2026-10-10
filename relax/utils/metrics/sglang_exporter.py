# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Expose SGLang metrics without flattening labels or histogram buckets."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from prometheus_client import generate_latest
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.parser import text_string_to_metric_families

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_SOURCE_LABELS = ("relax_engine", "relax_model", "relax_worker_type")


@dataclass(frozen=True)
class _Engine:
    url: str
    model: str
    worker_type: str

    @property
    def labels(self) -> dict[str, str]:
        return dict(zip(_SOURCE_LABELS, (self.url, self.model, self.worker_type)))


@dataclass
class _Snapshot:
    families: list[Metric]

    def collect(self) -> Iterable[Metric]:
        return iter(self.families)


def _discover_engines(data: dict[str, Any]) -> tuple[list[_Engine], bool]:
    """Validate discovery; an incomplete observation is never a healthy empty
    pool."""
    models = data["models"]
    observation = data["observation"]
    if not isinstance(models, dict) or not isinstance(observation, dict):
        raise ValueError("Invalid engine discovery response")
    complete, issues = observation["complete"], observation["issues"]
    if type(complete) is not bool or not isinstance(issues, list) or complete != (not issues):
        raise ValueError("Invalid engine observation")

    engines: dict[str, _Engine] = {}
    for model, info in models.items():
        if not isinstance(info["engine_groups"], list):
            raise ValueError("Invalid engine groups")
        for group in info["engine_groups"]:
            if not isinstance(group["engines"], list):
                raise ValueError("Invalid engines")
            for engine in group["engines"]:
                # Multi-node engines expose HTTP only on node_rank 0.
                if engine.get("status") != "active" or not engine.get("url"):
                    continue
                url = engine["url"].rstrip("/")
                address = urlsplit(url)
                if address.scheme not in ("http", "https") or not address.hostname or address.username:
                    raise ValueError("Invalid engine URL")
                worker_type = group.get("worker_type", "regular")
                if not isinstance(worker_type, str):
                    raise ValueError("Invalid engine worker type")
                engines.setdefault(url, _Engine(url, model, worker_type))
    return list(engines.values()), complete


def _parse_metrics(text: str, engine: _Engine) -> list[Metric]:
    families = []
    seen = set()
    for family in text_string_to_metric_families(text):
        if not family.name.startswith("sglang:"):
            continue
        for i, sample in enumerate(family.samples):
            if any(label in sample.labels for label in _SOURCE_LABELS):
                raise ValueError("Upstream metric uses a reserved source label")
            labels = {**sample.labels, **engine.labels}
            key = (sample.name, tuple(sorted(labels.items())))
            if key in seen:
                raise ValueError("Duplicate upstream metric sample")
            seen.add(key)
            family.samples[i] = sample._replace(labels=labels)
        families.append(family)
    if not seen:
        raise ValueError("No SGLang metric samples in upstream response")
    return families


def _exposed_name(family: Metric) -> str:
    # The parser strips _total from counter family names; exposition restores it.
    return family.name + "_total" if family.type == "counter" else family.name


def _render_metrics(engines: list[_Engine], results: list[list[Metric] | None], discovery_ok: bool) -> bytes:
    families: dict[str, Metric] = {}
    engine_up = GaugeMetricFamily(
        "relax_sglang_engine_up", "Whether SGLang metrics were fetched and merged successfully.", labels=_SOURCE_LABELS
    )
    scrape_ok = discovery_ok
    for engine, result in zip(engines, results):
        if result is not None:
            # Validate all families before merging any samples from this engine.
            schemas = {name: (metric.type, metric.documentation) for name, metric in families.items()}
            for family in result:
                schema = (family.type, family.documentation)
                if schemas.setdefault(_exposed_name(family), schema) != schema:
                    logger.warning("Conflicting SGLang metric schema from %s: %s", engine.url, family.name)
                    result = None
                    break
        success = result is not None
        engine_up.add_metric(list(engine.labels.values()), int(success))
        scrape_ok = scrape_ok and success
        if result is not None:
            for family in result:
                name = _exposed_name(family)
                if name in families:
                    families[name].samples.extend(family.samples)
                else:
                    families[name] = family

    status = [
        GaugeMetricFamily(
            "relax_sglang_discovery_success", "Whether engine discovery was complete.", value=discovery_ok
        ),
        GaugeMetricFamily(
            "relax_sglang_scrape_success", "Whether discovery and all engine scrapes succeeded.", value=scrape_ok
        ),
        GaugeMetricFamily(
            "relax_sglang_engines_discovered",
            "Number of unique active engine HTTP endpoints discovered.",
            value=len(engines),
        ),
        engine_up,
    ]
    return generate_latest(_Snapshot([*status, *families.values()]))


class SGLangMetricsExporter:
    """Bounded asynchronous scrapes, sharing in-flight work across callers.

    Each scrape discovers the current pool and uses a fresh snapshot; failed
    targets never retain old values. HTTP 200 carries explicit discovery and
    per-engine success gauges even when no upstream metrics are available.
    """

    def __init__(
        self,
        discover: Callable[[], Awaitable[dict[str, Any]]],
        *,
        discovery_timeout: float = 2.0,
        engine_timeout: float = 2.0,
        scrape_timeout: float = 5.0,
        concurrency: int = 16,
    ) -> None:
        self._discover = discover
        self._discovery_timeout = discovery_timeout
        self._engine_timeout = engine_timeout
        self._scrape_timeout = scrape_timeout
        self._concurrency = concurrency
        self._task: asyncio.Task[bytes] | None = None

    async def scrape(self) -> bytes:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._scrape())
        return await asyncio.shield(self._task)

    async def _fetch(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, engine: _Engine
    ) -> list[Metric] | None:
        try:
            async with semaphore:
                response = await client.get(f"{engine.url}/metrics", headers={"Accept": "text/plain; version=0.0.4"})
                response.raise_for_status()
                return await asyncio.to_thread(_parse_metrics, response.text, engine)
        except Exception as error:
            logger.warning("SGLang metrics scrape failed for %s: %s", engine.url, type(error).__name__)
            return None

    async def _scrape(self) -> bytes:
        try:
            data = await asyncio.wait_for(self._discover(), timeout=self._discovery_timeout)
            engines, discovery_ok = _discover_engines(data)
        except Exception as error:
            logger.warning("SGLang engine discovery failed: %s", type(error).__name__)
            return _render_metrics([], [], False)

        results: list[list[Metric] | None] = []
        if engines:
            async with httpx.AsyncClient(
                timeout=self._engine_timeout, trust_env=False, follow_redirects=True
            ) as client:
                semaphore = asyncio.Semaphore(self._concurrency)
                tasks = [asyncio.create_task(self._fetch(client, semaphore, engine)) for engine in engines]
                try:
                    done, _ = await asyncio.wait(tasks, timeout=self._scrape_timeout)
                    results = [task.result() if task in done else None for task in tasks]
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        return await asyncio.to_thread(_render_metrics, engines, results, discovery_ok)

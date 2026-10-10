# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pluggable web-search backends for the DeepEyesV2 ``search`` tool.

Backend selection and tuning are environment-driven (``SearchConfig.from_env``)
so machine-local endpoints and API keys stay inside the gitignored ``env.sh``
and never reach committed code:

* ``mock`` (default) — deterministic offline snippets, no network, no randomness.
* ``retriever`` — Search-R1 compatible retrieval service
  (``POST {url}`` with ``{"queries": [...], "topk": k, "return_scores": false}``,
  mirroring ``examples/search_r1/retrieval_server.py``).
* ``external`` — generic JSON web-search API. Defaults match the Serper.dev
  ``POST /search`` preset; endpoint, auth header, request fields and response
  field mapping are all configurable.

Every backend returns the uniform shape consumed by ``env_deepeyes_v2``::

    {"elapsed_time": float, "data": [{"title": str, "link": str, "snippet": str, "date": str | None}, ...]}

or the literal ``"Error"`` string on failure — the env's existing convention
(``_dispatch_search`` turns it into ``{"status": "error", ...}`` and the agent
process keeps running). Failures are logged, never silently swallowed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import requests


logger = logging.getLogger(__name__)

ERROR = "Error"

DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 5.0

RETRYABLE_STATUS_CODES = (500, 502, 503, 504)

RESULT_FIELDS = ("title", "link", "snippet", "date")


@dataclass(frozen=True)
class SearchConfig:
    """Resolved search configuration; built from environment variables."""

    backend: str = "mock"
    top_k: int = DEFAULT_TOP_K
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    retriever_url: str = ""
    external_endpoint: str = ""
    external_api_key: str = ""
    external_auth_header: str = "X-API-KEY"
    external_query_field: str = "q"
    external_topk_field: str = "num"
    external_results_field: str = "organic"
    external_field_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SearchConfig":
        """Read config from ``environ`` (default ``os.environ``).

        Raises ``ValueError`` on malformed numeric or JSON values so that
        configuration mistakes surface as the env's ``"Error"`` tool result
        (with a logged reason) instead of a silently wrong request.
        """
        env = os.environ if environ is None else environ
        return cls(
            backend=env.get("DEEPEYES_V2_SEARCH_BACKEND", "mock").strip() or "mock",
            top_k=_env_int(env, "DEEPEYES_V2_SEARCH_TOP_K", DEFAULT_TOP_K),
            timeout_seconds=_env_float(env, "DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_retries=_env_int(env, "DEEPEYES_V2_SEARCH_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            retry_delay_seconds=_env_float(env, "DEEPEYES_V2_SEARCH_RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY_SECONDS),
            retriever_url=env.get("DEEPEYES_V2_RETRIEVER_URL", "").strip(),
            external_endpoint=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT", "").strip(),
            external_api_key=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_API_KEY", ""),
            external_auth_header=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_AUTH_HEADER", "").strip() or "X-API-KEY",
            external_query_field=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_QUERY_FIELD", "").strip() or "q",
            external_topk_field=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_TOPK_FIELD", "").strip() or "num",
            external_results_field=env.get("DEEPEYES_V2_EXTERNAL_SEARCH_RESULTS_FIELD", "").strip() or "organic",
            external_field_map=_env_field_map(env),
        )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_field_map(env: Mapping[str, str]) -> dict[str, str]:
    """Parse ``DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP`` (JSON object).

    Keys must be the unified schema fields (``title``/``link``/``snippet``/
    ``date``); values are the corresponding keys in the external API's result
    rows. Unmapped fields fall back to the unified field name itself.
    """
    raw = env.get("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP must be a JSON object")
    field_map: dict[str, str] = {}
    for key, value in parsed.items():
        if key not in RESULT_FIELDS:
            raise ValueError(f"DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP has unsupported key {key!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP[{key!r}] must be a non-empty string")
        field_map[key] = value
    return field_map


def normalize_results(
    rows: list[Any],
    field_map: Mapping[str, str] | None = None,
    snippet_fallback_field: str | None = None,
) -> list[dict[str, Any]]:
    """Map raw result rows onto the unified schema.

    ``title``/``link``/``snippet`` are always strings (missing or non-string
    values become ``""``); ``date`` stays ``None`` when absent — the env
    renders both cases. With ``snippet_fallback_field`` set, a row without a
    snippet falls back to that field (Search-R1 corpus documents carry their
    text under ``contents``). Rows that are not dicts are dropped with a
    warning; structural failures higher up (non-list result containers) are
    handled by the callers as ``"Error"``.
    """
    mapping = field_map or {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            logger.warning(f"[search_backends] dropping non-dict result row: {type(row).__name__}")
            continue

        def pick(unified_field: str) -> str | None:
            value = row.get(mapping.get(unified_field, unified_field))
            return value if isinstance(value, str) else None

        snippet = pick("snippet")
        if not snippet and snippet_fallback_field:
            fallback = row.get(snippet_fallback_field)
            if isinstance(fallback, str):
                snippet = fallback
        normalized.append(
            {
                "title": pick("title") or "",
                "link": pick("link") or "",
                "snippet": snippet or "",
                "date": pick("date"),
            }
        )
    return normalized


def _post_json_with_retries(
    *,
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: float,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    """POST JSON and return the parsed JSON object, retrying transient
    failures.

    Mirrors the semantics of the existing retrieval client
    (``examples/on_policy_distillation/.../retrieval_client.py``): connection
    errors, timeouts, 5xx statuses and unparsable JSON bodies are retried; any
    other non-2xx status fails immediately. Raises ``RuntimeError`` once the
    attempts are exhausted — callers translate that into the ``"Error"``
    convention.
    """
    attempts = max(1, max_retries)
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, headers=dict(headers), json=payload, timeout=timeout_seconds)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"server error {resp.status_code}"
            else:
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as exc:
                    last_error = f"invalid JSON response status={resp.status_code} body={resp.text[:200]!r}: {exc}"
                else:
                    if isinstance(data, dict):
                        return data
                    last_error = (
                        f"invalid JSON response type={type(data).__name__} status={resp.status_code} "
                        f"bytes={len(resp.content)}"
                    )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
        except requests.exceptions.RequestException as exc:
            # Non-retryable request failure (e.g. 4xx via raise_for_status).
            last_error = str(exc)
            break
        if attempt < attempts:
            delay = min(retry_delay_seconds * attempt, MAX_RETRY_DELAY_SECONDS)
            logger.warning(
                f"[search_backends] attempt {attempt}/{attempts} failed ({last_error}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"search request failed after {attempts} attempts: {last_error}")


class MockSearchBackend:
    """Deterministic offline backend (default).

    Same query and top_k always produce the same results; no network and no
    randomness, so recipes run without any retrieval service or keys. The
    constructor accepts the shared ``SearchConfig`` for factory uniformity
    (only ``top_k`` is relevant, resolved by the caller).
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        started = time.monotonic()
        data = [
            {
                "title": f"Mock result {i + 1} for: {query}",
                "link": f"https://mock.local/search/{i + 1}",
                "snippet": f"Deterministic mock snippet {i + 1} for query: {query}",
                "date": None,
            }
            for i in range(max(0, top_k))
        ]
        return {"elapsed_time": time.monotonic() - started, "data": data}


class RetrieverSearchBackend:
    """Single-query client for a Search-R1 compatible retrieval service.

    Wire protocol (vendored ``examples/search_r1/retrieval_server.py``):
    ``POST {url}`` with ``{"queries": [<query>], "topk": <k>,
    "return_scores": false}`` → ``{"result": [[doc, ...], ...]}``; we send one
    query per call and read ``result[0]``. Corpus documents expose their text
    under ``contents``; any ``title``/``link``/``date`` keys a corpus carries
    are propagated, missing ones degrade to the unified schema defaults.

    Like the historical placeholder, ``search`` returns ``"Error"`` (after
    retries) instead of raising, so callers keep running.
    """

    def __init__(self, config: SearchConfig) -> None:
        if not config.retriever_url:
            raise ValueError("DEEPEYES_V2_RETRIEVER_URL is required when DEEPEYES_V2_SEARCH_BACKEND=retriever")
        self.config = config

    def search(self, query: str, top_k: int) -> dict[str, Any] | str:
        try:
            return self._search(query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[search_backends] retriever query={query!r} failed: {exc}")
            return ERROR

    def _search(self, query: str, top_k: int) -> dict[str, Any]:
        started = time.monotonic()
        payload: dict[str, Any] = {"queries": [query], "topk": top_k, "return_scores": False}
        response = _post_json_with_retries(
            url=self.config.retriever_url,
            payload=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            retry_delay_seconds=self.config.retry_delay_seconds,
        )
        rows = response.get("result")
        if not isinstance(rows, list):
            raise RuntimeError(f"retriever response field 'result' is {type(rows).__name__}, expected list")
        documents = rows[0] if rows else []
        if not isinstance(documents, list):
            raise RuntimeError(f"retriever response 'result[0]' is {type(documents).__name__}, expected list")
        data = normalize_results(documents, snippet_fallback_field="contents")
        return {"elapsed_time": time.monotonic() - started, "data": data}


class ExternalSearchBackend:
    """Client for a generic JSON web-search API (Serper.dev preset by default).

    Request: ``POST {endpoint}`` with ``{<query_field>: query,
    <topk_field>: topk}`` and the API key in ``<auth_header>``. Response: the
    top-level ``<results_field>`` list is mapped onto the unified schema via
    ``external_field_map``. All of these knobs come from environment variables;
    the API key is only ever read from the environment, never hardcoded.

    Like the retriever backend, ``search`` returns ``"Error"`` (after retries)
    instead of raising, so callers keep running.
    """

    def __init__(self, config: SearchConfig) -> None:
        if not config.external_endpoint:
            raise ValueError(
                "DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT is required when DEEPEYES_V2_SEARCH_BACKEND=external"
            )
        self.config = config

    def search(self, query: str, top_k: int) -> dict[str, Any] | str:
        try:
            return self._search(query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[search_backends] external query={query!r} failed: {exc}")
            return ERROR

    def _search(self, query: str, top_k: int) -> dict[str, Any]:
        started = time.monotonic()
        cfg = self.config
        payload: dict[str, Any] = {cfg.external_query_field: query, cfg.external_topk_field: top_k}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if cfg.external_api_key:
            headers[cfg.external_auth_header] = cfg.external_api_key
        response = _post_json_with_retries(
            url=cfg.external_endpoint,
            payload=payload,
            headers=headers,
            timeout_seconds=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
            retry_delay_seconds=cfg.retry_delay_seconds,
        )
        rows = response.get(cfg.external_results_field)
        if not isinstance(rows, list):
            raise RuntimeError(
                f"external search response field {cfg.external_results_field!r} is "
                f"{type(rows).__name__}, expected list"
            )
        return {"elapsed_time": time.monotonic() - started, "data": normalize_results(rows, cfg.external_field_map)}


_BACKENDS: dict[str, type] = {
    "mock": MockSearchBackend,
    "retriever": RetrieverSearchBackend,
    "external": ExternalSearchBackend,
}


def get_search_backend(config: SearchConfig | None = None) -> Any:
    """Instantiate the configured backend; raises ``ValueError`` on unknown
    backend names or missing mandatory endpoint configuration."""
    cfg = config if config is not None else SearchConfig.from_env()
    backend_cls = _BACKENDS.get(cfg.backend)
    if backend_cls is None:
        raise ValueError(f"Unknown DEEPEYES_V2_SEARCH_BACKEND {cfg.backend!r}; expected one of {sorted(_BACKENDS)}")
    return backend_cls(cfg)

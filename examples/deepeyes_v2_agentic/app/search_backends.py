# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pluggable web-search backends for the DeepEyesV2 ``search`` tool.

The backend is selected with ``DEEPEYES_V2_SEARCH_BACKEND``:

* ``mock`` (default) — deterministic canned snippets; fully offline, so the
  recipe runs end-to-end with no retrieval service, network, or API keys.
* ``retriever`` — a Search-R1 compatible retrieval service speaking
  ``POST {url} {"queries": [q], "topk": n}`` → ``{"result": [[doc, ...]]}``
  (same wire protocol as ``examples/search_r1/retrieval_server.py``).
* ``external`` — a configurable external search API. Endpoint, auth header and
  request/response field mapping come from a JSON config file (built-in
  defaults target the Serper Google Search API); the API key is read from an
  environment variable and never hardcoded, written to the config or logged.

Every backend returns the unified result shape::

    {"elapsed_time": float, "data": [{"title": str, "link": str, "snippet": str, "date": str | None}, ...]}

Backends raise on failure; the retry loop and the ``"Error"`` sentinel live in
:mod:`app.search_utils` so the env-side handling stays unchanged.

Environment variables (empty or unset values fall back to the defaults):

* ``DEEPEYES_V2_SEARCH_BACKEND`` — backend selection (default ``mock``).
* ``DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS`` — HTTP timeout for real backends (default ``10``).
* ``DEEPEYES_V2_SEARCH_MAX_RETRIES`` — attempts made by ``app.search_utils.search`` (default ``3``).
* ``DEEPEYES_V2_SEARCH_RETRIEVER_URL`` — full ``/retrieve`` endpoint
  (default ``http://127.0.0.1:17389/retrieve``).
* ``DEEPEYES_V2_SEARCH_RETRIEVER_TOPK`` — top-k override (default: the ``size`` argument).
* ``DEEPEYES_V2_SEARCH_EXTERNAL_CONFIG`` — JSON config file for ``external`` (default: built-in Serper mapping).
* ``DEEPEYES_V2_SEARCH_EXTERNAL_API_KEY`` — API key injected into the auth header (no default).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


logger = logging.getLogger(__name__)

BACKEND_ENV = "DEEPEYES_V2_SEARCH_BACKEND"
TIMEOUT_ENV = "DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS"
MAX_RETRIES_ENV = "DEEPEYES_V2_SEARCH_MAX_RETRIES"
RETRIEVER_URL_ENV = "DEEPEYES_V2_SEARCH_RETRIEVER_URL"
RETRIEVER_TOPK_ENV = "DEEPEYES_V2_SEARCH_RETRIEVER_TOPK"
EXTERNAL_CONFIG_ENV = "DEEPEYES_V2_SEARCH_EXTERNAL_CONFIG"
EXTERNAL_API_KEY_ENV = "DEEPEYES_V2_SEARCH_EXTERNAL_API_KEY"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:17389/retrieve"

_JSON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[search_backends] ignoring invalid {name}={raw!r}, using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[search_backends] ignoring invalid {name}={raw!r}, using {default}")
        return default


def get_max_retries() -> int:
    """Attempts used by :func:`app.search_utils.search` before giving up."""
    return max(1, _env_int(MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES))


def _new_http_session() -> Any:
    """Build the pooled ``requests.Session`` shared by the HTTP backends.

    Mirrors the OPD retrieval client: proxy-bypassing (``trust_env=False``),
    connection pooling, retries handled by the caller. ``requests`` is imported
    lazily so the default offline ``mock`` path never needs it.
    """
    import requests

    session = requests.Session()
    session.trust_env = False
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=512,
        pool_maxsize=512,
        max_retries=0,  # retries are handled explicitly by app.search_utils
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class SearchBackend:
    """Base class for text web-search backends.

    Implementations return the unified result dict and raise on any failure;
    retries and the ``"Error"`` sentinel convention are handled by the caller.
    """

    def search(self, query: str, size: int = 5) -> dict:
        raise NotImplementedError


class MockSearchBackend(SearchBackend):
    """Deterministic offline backend — canned snippets, no network, no keys."""

    def search(self, query: str, size: int = 5) -> dict:
        return {
            "elapsed_time": 0.0,
            "data": [
                {
                    "title": f"Placeholder Title {i}",
                    "link": f"http://example.com/{i}",
                    "snippet": f"This is a placeholder snippet for query: {query}",
                    "date": None,
                }
                for i in range(size)
            ],
        }


class RetrieverSearchBackend(SearchBackend):
    """Search-R1 compatible retriever client (one query per call).

    Wire protocol: ``POST {url} {"queries": [q], "topk": n}`` →
    ``{"result": [[<corpus row> | {"document": {...}, "score": ...}, ...]]}``.
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_RETRIEVER_URL,
        topk: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Any = None,
    ) -> None:
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.session = session if session is not None else _new_http_session()

    def search(self, query: str, size: int = 5) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("empty query")

        start = time.perf_counter()
        resp = self.session.post(
            self.url,
            headers=_JSON_HEADERS,
            json={"queries": [query], "topk": self.topk or size},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        elapsed = time.perf_counter() - start

        rows = body.get("result") if isinstance(body, dict) else None
        if not isinstance(rows, list) or (rows and not isinstance(rows[0], list)):
            raise ValueError(f"unexpected retriever response shape: {str(body)[:200]}")

        data: list[dict] = []
        for idx, row in enumerate(rows[0] if rows else []):
            if not isinstance(row, dict):
                continue
            doc = row["document"] if isinstance(row.get("document"), dict) else row
            data.append(
                {
                    "title": _as_text(doc.get("title")) or f"Document {idx + 1}",
                    "link": _as_text(doc.get("url")) or _as_text(doc.get("link")) or "",
                    "snippet": _as_text(doc.get("contents")) or _as_text(doc.get("text")) or "",
                    "date": _as_text(doc.get("date")),
                }
            )
        return {"elapsed_time": elapsed, "data": data}


@dataclass(frozen=True)
class ExternalSearchConfig:
    """Endpoint / auth / field-mapping description for the external backend.

    ``request_map`` maps the logical request fields (``query``, ``size``) onto
    the API's parameter names — an empty target name omits the field.
    ``response_map["results"]`` is a dot path to the result list inside the
    response JSON; the remaining entries map each result item onto the unified
    page fields.
    """

    endpoint: str = "https://google.serper.dev/search"
    method: str = "POST"
    auth_header: str = "X-API-KEY"
    auth_scheme: str = ""  # e.g. "Bearer " for Authorization-style headers
    request_map: Mapping[str, str] = field(default_factory=lambda: {"query": "q", "size": "num"})
    response_map: Mapping[str, str] = field(
        default_factory=lambda: {
            "results": "organic",
            "title": "title",
            "link": "link",
            "snippet": "snippet",
            "date": "date",
        }
    )


def _dig(payload: Any, path: str) -> Any:
    node = payload
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _load_external_config() -> ExternalSearchConfig:
    """Merge the JSON config file (if any) over the Serper-flavoured
    defaults."""
    path = _env_str(EXTERNAL_CONFIG_ENV)
    if not path:
        return ExternalSearchConfig()

    try:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {EXTERNAL_CONFIG_ENV}={path!r}: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"{EXTERNAL_CONFIG_ENV}={path!r} must contain a JSON object")

    known: dict[str, Any] = {}
    for name in ("endpoint", "method", "auth_header", "auth_scheme"):
        if name in overrides:
            known[name] = overrides.pop(name)
    unknown = set(overrides) - {"request_map", "response_map"}
    if unknown:
        logger.warning(f"[search_backends] ignoring unknown keys in external config: {sorted(unknown)}")
    for name in ("request_map", "response_map"):
        value = overrides.get(name)
        if value is None:
            continue
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise ValueError(f"external config field {name!r} must be a JSON object mapping to strings")
        known[name] = value

    config = replace(ExternalSearchConfig(), **known)
    if not config.endpoint.strip():
        raise ValueError("external config is missing 'endpoint'")
    if config.method.upper() not in ("GET", "POST"):
        raise ValueError(f"external config 'method' must be GET or POST, got {config.method!r}")
    return config


class ExternalSearchBackend(SearchBackend):
    """Configurable external search API client.

    The JSON config file (``DEEPEYES_V2_SEARCH_EXTERNAL_CONFIG``) overrides any
    subset of :class:`ExternalSearchConfig`; the API key arrives via
    ``DEEPEYES_V2_SEARCH_EXTERNAL_API_KEY`` and is only placed in the auth
    header — never logged or persisted.
    """

    def __init__(
        self,
        *,
        config: ExternalSearchConfig | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Any = None,
    ) -> None:
        self.config = config if config is not None else ExternalSearchConfig()
        self.api_key = api_key
        self.timeout = timeout
        self.session = session if session is not None else _new_http_session()

    def search(self, query: str, size: int = 5) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("empty query")

        fields: dict[str, Any] = {}
        query_key = self.config.request_map.get("query", "q")
        if query_key:
            fields[query_key] = query
        size_key = self.config.request_map.get("size", "num")
        if size_key:
            fields[size_key] = size

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers[self.config.auth_header] = f"{self.config.auth_scheme}{self.api_key}"

        start = time.perf_counter()
        if self.config.method.upper() == "GET":
            resp = self.session.get(self.config.endpoint, params=fields, headers=headers, timeout=self.timeout)
        else:
            resp = self.session.post(
                self.config.endpoint,
                headers={**headers, "Content-Type": "application/json"},
                json=fields,
                timeout=self.timeout,
            )
        resp.raise_for_status()
        body = resp.json()
        elapsed = time.perf_counter() - start

        results = _dig(body, self.config.response_map.get("results", ""))
        if not isinstance(results, list):
            raise ValueError(f"unexpected external response shape: {str(body)[:200]}")

        mapping = self.config.response_map
        data = [
            {
                "title": _as_text(item.get(mapping.get("title", "title"))) or "",
                "link": _as_text(item.get(mapping.get("link", "link"))) or "",
                "snippet": _as_text(item.get(mapping.get("snippet", "snippet"))) or "",
                "date": _as_text(item.get(mapping.get("date", "date"))),
            }
            for item in results[:size]
            if isinstance(item, dict)
        ]
        return {"elapsed_time": elapsed, "data": data}


def _build_backend_from_env() -> SearchBackend:
    name = _env_str(BACKEND_ENV, "mock").lower()
    timeout = _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
    if name == "retriever":
        return RetrieverSearchBackend(
            url=_env_str(RETRIEVER_URL_ENV, DEFAULT_RETRIEVER_URL),
            topk=_env_int(RETRIEVER_TOPK_ENV, 0) or None,
            timeout=timeout,
        )
    if name == "external":
        return ExternalSearchBackend(
            config=_load_external_config(),
            api_key=_env_str(EXTERNAL_API_KEY_ENV) or None,
            timeout=timeout,
        )
    if name != "mock":
        logger.warning(f"[search_backends] unknown {BACKEND_ENV}={name!r}, falling back to mock")
    return MockSearchBackend()


_BACKEND: SearchBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_search_backend() -> SearchBackend:
    """Return the process-wide backend selected by
    ``DEEPEYES_V2_SEARCH_BACKEND``.

    The instance is cached (reset by assigning ``None`` to the module-global
    ``_BACKEND``, which tests rely on). Raises when a real backend's
    configuration is invalid — the caller turns that into the ``"Error"``
    sentinel without retrying.
    """
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = _build_backend_from_env()
    return _BACKEND

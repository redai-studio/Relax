# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search tool helpers for the DeepEyesV2 env.

* :func:`search` is a **pluggable** web-search entry point. It dispatches to one
  of three backends selected by ``DEEPEYES_V2_SEARCH_BACKEND``:

  - ``mock`` (default) — deterministic, offline canned results so the recipe
    runs end-to-end without any retrieval service, network, or API key.
  - ``retriever`` — a Search-R1 compatible ``POST /retrieve`` HTTP service
    (see ``examples/search_r1/retrieval_server.py``).
  - ``brave`` — Brave Search API (https://api.search.brave.com). Endpoint and
    subscription token come from env; the API key is never hardcoded.

  Regardless of backend, :func:`search` returns the uniform shape::

      {"elapsed_time": float, "data": [{"title", "link", "snippet", "date"}, ...]}

  and degrades to the string ``"Error"`` on any timeout / exception / invalid
  response, matching the env's existing failure convention so the agent process
  never crashes.

* :func:`image_search` serves cached results keyed by ``data_idx`` from JSON
  files listed in ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` (colon/comma-separated).
  Missing / unparsable caches degrade to returning ``"Error"`` so the env
  surfaces a clean failure instead of crashing at import.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol
from urllib.parse import quote_plus

import requests


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Environment-variable names (see env.sh.example for docs)
# --------------------------------------------------------------------------- #
ENV_BACKEND = "DEEPEYES_V2_SEARCH_BACKEND"

ENV_TIMEOUT = "DEEPEYES_V2_SEARCH_TIMEOUT"
ENV_MAX_RETRIES = "DEEPEYES_V2_SEARCH_MAX_RETRIES"
ENV_RETRY_BUDGET = "DEEPEYES_V2_SEARCH_RETRY_BUDGET"
ENV_TRUST_ENV = "DEEPEYES_V2_SEARCH_TRUST_ENV"

# retriever backend
ENV_RETRIEVER_URL = "DEEPEYES_V2_SEARCH_RETRIEVER_URL"
ENV_TOPK = "DEEPEYES_V2_SEARCH_TOPK"

# brave backend (https://brave.com/search/api/)
ENV_BRAVE_API_KEY = "DEEPEYES_V2_SEARCH_BRAVE_API_KEY"
ENV_BRAVE_ENDPOINT = "DEEPEYES_V2_SEARCH_BRAVE_ENDPOINT"
BRAVE_DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

DEFAULT_SIZE = 5
_RETRYABLE_STATUS = {500, 502, 503, 504}
_INITIAL_RETRY_DELAY = 1.0


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def _env(name: str, default: str | None = None) -> str | None:
    """Return env var ``name`` treating empty/whitespace as unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Shared HTTP client (sync; runs inside the env's asyncio.to_thread)
# --------------------------------------------------------------------------- #
def _build_session(trust_env: bool) -> requests.Session:
    """Pooled session; retries are handled explicitly in
    :func:`_request_json`."""
    session = requests.Session()
    session.trust_env = trust_env
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=64,
        pool_maxsize=64,
        max_retries=0,
        pool_block=False,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _request_json(
    session: requests.Session,
    *,
    method: str,
    url: str,
    timeout: float,
    max_retries: int,
    retry_budget: float,
    **kwargs: Any,
) -> Any:
    """POST/GET with explicit retry on transient failures.

    Retries connection errors, timeouts and 5xx within ``retry_budget``
    seconds. Non-retryable failures (bad <500 status, invalid JSON) raise
    immediately. Always raises ``RuntimeError`` on exhaustion; the caller turns
    it into ``"Error"``.
    """
    last_error: str | None = None
    deadline = time.monotonic() + retry_budget
    attempts = 0
    while attempts < max(1, max_retries):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempts += 1
        try:
            resp = session.request(method, url=url, timeout=min(timeout, remaining), **kwargs)
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"server error {resp.status_code}"
            else:
                resp.raise_for_status()
                return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
        except (requests.exceptions.RequestException, ValueError) as exc:
            # bad status (<500), invalid JSON, etc. — not worth retrying.
            raise RuntimeError(f"request failed: {exc}") from exc

        remaining = deadline - time.monotonic()
        if attempts >= max(1, max_retries) or remaining <= 0:
            break
        time.sleep(min(_INITIAL_RETRY_DELAY * attempts, 5.0, remaining))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class SearchBackend(Protocol):
    """Produce raw search result rows; normalization + timing live in
    :func:`search`."""

    def search(self, query: str, size: int) -> list[dict]: ...


class MockBackend:
    """Deterministic, offline canned results (default backend)."""

    def search(self, query: str, size: int) -> list[dict]:
        items: list[dict] = []
        for i in range(max(0, size)):
            items.append(
                {
                    "title": f"Mock result {i + 1} for {query!r}",
                    "link": f"https://example.com/search?q={quote_plus(query)}&r={i + 1}",
                    "snippet": f"Deterministic offline snippet {i + 1} for query {query!r}.",
                    "date": None,
                }
            )
        return items


class RetrieverBackend:
    """Client for a Search-R1 compatible ``POST /retrieve`` endpoint.

    POST {url}  {"queries": [q], "topk": k, "return_scores": true}
    -> {"result": [[{"document": {"contents": str, ...}, "score": float}, ...]]}
    """

    def __init__(self) -> None:
        self.url = _env(ENV_RETRIEVER_URL)
        if not self.url:
            raise ValueError(f"{ENV_RETRIEVER_URL} is required for the retriever backend")
        self.topk = int(_env(ENV_TOPK, "5"))
        self.timeout = float(_env(ENV_TIMEOUT, "30"))
        self.max_retries = int(_env(ENV_MAX_RETRIES, "3"))
        self.retry_budget = float(_env(ENV_RETRY_BUDGET, "60"))
        self.session = _build_session(_env_bool(ENV_TRUST_ENV, False))

    def search(self, query: str, size: int) -> list[dict]:
        # Prefer the caller's ``size``; fall back to env topk when size is unset/0.
        topk = size if size and size > 0 else self.topk
        payload = {"queries": [query], "topk": topk, "return_scores": True}
        data = _request_json(
            self.session,
            method="POST",
            url=self.url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_budget=self.retry_budget,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        rows = data.get("result", []) if isinstance(data, dict) else []
        if not rows:
            return []
        docs_for_query = rows[0] if isinstance(rows[0], list) else rows
        items: list[dict] = []
        for entry in docs_for_query:
            document = entry.get("document", entry) if isinstance(entry, dict) else {}
            if not isinstance(document, dict):
                continue
            contents = str(document.get("contents", "")).strip()

            lines = contents.split("\n") if contents else []
            title = lines[0].strip() if lines and lines[0].strip() else str(document.get("title") or "(untitled)")
            snippet = contents or None
            link = document.get("id") or document.get("url") or ""
            items.append({"title": title, "link": str(link), "snippet": snippet, "date": document.get("date")})
        return items


class BraveBackend:
    """Brave Search API client (the ``brave`` backend).

    Wire protocol (Brave Web Search)::

        GET {endpoint}?q=<query>&count=<size>
        Header: X-Subscription-Token: <api_key>
        -> {"web": {"results": [{"title", "url", "description", "age"}, ...]}}

    Configurable via env (key is never hardcoded)::

        DEEPEYES_V2_SEARCH_BRAVE_API_KEY  # required
        DEEPEYES_V2_SEARCH_BRAVE_ENDPOINT  # optional override
        DEEPEYES_V2_SEARCH_TIMEOUT / _MAX_RETRIES / _RETRY_BUDGET / _TRUST_ENV
    """

    def __init__(self) -> None:
        self.api_key = _env(ENV_BRAVE_API_KEY)
        if not self.api_key:
            raise ValueError(f"{ENV_BRAVE_API_KEY} is required for the brave backend")
        self.endpoint = _env(ENV_BRAVE_ENDPOINT, BRAVE_DEFAULT_ENDPOINT)
        self.timeout = float(_env(ENV_TIMEOUT, "30"))
        self.max_retries = int(_env(ENV_MAX_RETRIES, "3"))
        self.retry_budget = float(_env(ENV_RETRY_BUDGET, "60"))
        self.session = _build_session(_env_bool(ENV_TRUST_ENV, False))

    def search(self, query: str, size: int) -> list[dict]:
        count = size if size and size > 0 else DEFAULT_SIZE
        data = _request_json(
            self.session,
            method="GET",
            url=self.endpoint,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_budget=self.retry_budget,
            params={"q": query, "count": count},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        web = data.get("web") if isinstance(data, dict) else None
        rows = web.get("results") if isinstance(web, dict) else None
        if not isinstance(rows, list):
            return []
        items: list[dict] = []
        for row in rows[:count]:
            if not isinstance(row, dict):
                continue
            # Brave fields -> uniform {title, link, snippet, date}.
            items.append(
                {
                    "title": row.get("title"),
                    "link": row.get("url"),
                    "snippet": row.get("description"),
                    "date": row.get("age"),
                }
            )
        return items


_BACKENDS: dict[str, type] = {
    "mock": MockBackend,
    "retriever": RetrieverBackend,
    "brave": BraveBackend,
}
_BACKEND_CACHE: dict[str, SearchBackend] = {}


def _get_backend() -> SearchBackend:
    name = (_env(ENV_BACKEND, "mock") or "mock").lower()
    if name not in _BACKENDS:
        logger.warning(f"[search] unknown backend {name!r}; falling back to mock")
        name = "mock"
    if name not in _BACKEND_CACHE:
        _BACKEND_CACHE[name] = _BACKENDS[name]()
    return _BACKEND_CACHE[name]


def _reset_backend_cache() -> None:
    """Drop cached backend singletons (config is read at construction time)."""
    _BACKEND_CACHE.clear()


def _normalize(items: list[dict], size: int) -> list[dict]:
    """Coerce raw rows into the uniform ``{title, link, snippet, date}`` shape.

    ``title`` / ``link`` are always non-null strings (the env indexes them
    directly); ``snippet`` / ``date`` may be ``None``.
    """
    out: list[dict] = []
    limit = size if size and size > 0 else len(items)
    for item in items[:limit]:
        title = item.get("title")
        link = item.get("link")
        snippet = item.get("snippet")
        date = item.get("date")
        out.append(
            {
                "title": "" if title is None else str(title),
                "link": "" if link is None else str(link),
                "snippet": None if snippet is None else str(snippet),
                "date": None if date is None else str(date),
            }
        )
    return out


def search(query: str, size: int = DEFAULT_SIZE):
    """Pluggable web-search. Returns::

        {"elapsed_time": float, "data": [{"title", "link", "snippet", "date"}, ...]}

    or the string ``"Error"`` on any failure (timeout / exception / invalid
    response), so the env surfaces a clean ``search_failed`` and the agent keeps
    running. Backend is chosen via ``DEEPEYES_V2_SEARCH_BACKEND`` (default mock).
    """
    start = time.monotonic()
    try:
        backend = _get_backend()
        data = _normalize(backend.search(query, size), size)
    except Exception as exc:
        logger.warning(f"[search] backend failed for query={query!r}: {exc}")
        return "Error"
    return {"elapsed_time": time.monotonic() - start, "data": data}


def _load_image_search_cache() -> dict:
    """Load image-search caches from JSON files, controlled by env var.

    Set ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` to a colon- or comma-separated list
    of JSON files. Missing or invalid files are skipped with a warning.
    """
    raw = os.environ.get("DEEPEYES_V2_SEARCH_CACHE_PATHS", "")
    if not raw.strip():
        return {}

    paths: list[str] = []
    for chunk in raw.replace(",", ":").split(":"):
        chunk = chunk.strip()
        if chunk:
            paths.append(chunk)

    merged: dict = {}
    for p in paths:
        if not os.path.isfile(p):
            logger.warning(f"[search_utils] image-search cache not found: {p} (skipping)")
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
        except Exception as exc:
            logger.warning(f"[search_utils] failed to load cache {p}: {exc}")
    return merged


# Lazily-initialised global so the module remains importable even when no cache
# is configured.
_IMAGE_SEARCH_CACHE: dict | None = None


def _get_image_search_cache() -> dict:
    global _IMAGE_SEARCH_CACHE
    if _IMAGE_SEARCH_CACHE is None:
        _IMAGE_SEARCH_CACHE = _load_image_search_cache()
    return _IMAGE_SEARCH_CACHE


def image_search(_query, data_idx: str | None = None):
    """Image-search via cached results, keyed by ``data_idx``.

    Only ``fvqa`` indexed entries are served. If the cache is empty (no
    ``DEEPEYES_V2_SEARCH_CACHE_PATHS``), returns ``"Error"`` so the env
    propagates a clean failure.
    """
    if data_idx is None or "fvqa" not in str(data_idx):
        logger.warning("image_search failed, no fvqa found in data index")
        return "Error"

    cache = _get_image_search_cache()
    cached_data = cache.get(data_idx, {})
    if not cached_data:
        logger.warning(f"image_search: data_idx={data_idx} not in cache (cache size={len(cache)})")
        return "Error"

    tool_returned_web_title = cached_data.get("tool_returned_web_title", [])
    cached_images_path = cached_data.get("cached_images_path", [])

    return_cached_images_path: list[str] = []
    return_tool_returned_web_title: list[str] = []
    for title, path in zip(tool_returned_web_title, cached_images_path):
        if path is not None and os.path.exists(path):
            return_cached_images_path.append(path)
            return_tool_returned_web_title.append(title)

    return {
        "tool_returned_web_title": return_tool_returned_web_title,
        "cached_images_path": return_cached_images_path,
    }

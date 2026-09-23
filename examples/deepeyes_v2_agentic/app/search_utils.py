# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search tool helpers for the DeepEyesV2 env.

* :func:`search` dispatches to a pluggable backend (see
  ``app.search_backends``): ``mock`` (default, deterministic and offline),
  ``retriever`` (Search-R1 compatible HTTP service) or ``external``
  (configurable web-search API). It returns the uniform
  ``{"elapsed_time", "data"}`` shape on success or ``"Error"`` on any failure —
  the env's existing convention — so backend failures surface as a clean
  in-trajectory tool error instead of crashing the process.
* :func:`image_search` serves cached results keyed by ``data_idx`` from JSON
  files listed in ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` (colon/comma-separated).
  Missing / unparsable caches degrade to returning ``"Error"`` so the env
  surfaces a clean failure instead of crashing at import.
"""

from __future__ import annotations

import json
import logging
import os

from app.search_backends import ERROR, SearchConfig, get_search_backend


logger = logging.getLogger(__name__)


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


def search(query: str, size: int | None = None):
    """Web search via the configured backend, returning::

        {"elapsed_time": float, "data": [{"title", "link", "snippet", "date"?}, ...]}

    Backends (``DEEPEYES_V2_SEARCH_BACKEND``): ``mock`` (default —
    deterministic offline snippets), ``retriever`` (Search-R1 compatible HTTP
    service) and ``external`` (configurable web-search API); see
    ``app.search_backends`` for the configuration surface. Explicit ``size``
    wins over the configured ``DEEPEYES_V2_SEARCH_TOP_K`` (default 5, matching
    the historical signature). Any configuration or backend failure is logged
    and mapped to ``"Error"``; callers keep running.
    """
    try:
        config = SearchConfig.from_env()
        top_k = size if size is not None else config.top_k
        return get_search_backend(config).search(query, top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[search] query={query!r} failed: {exc}")
        return ERROR


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

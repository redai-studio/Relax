# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search tool helpers for the DeepEyesV2 env.

* :func:`search` dispatches to the backend selected by
  ``DEEPEYES_V2_SEARCH_BACKEND``. The default ``mock`` backend is fully
  offline and deterministic.
* :func:`image_search` serves cached results keyed by ``data_idx`` from JSON
  files listed in ``DEEPEYES_V2_SEARCH_CACHE_PATHS`` (colon/comma-separated).
  Missing / unparsable caches degrade to returning ``"Error"`` so the env
  surfaces a clean failure instead of crashing at import.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote


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


def _mock_search(query: str, size: int) -> dict[str, object]:
    """Return deterministic, fully offline results for development and tests."""
    encoded_query = quote(query, safe="")
    data = [
        {
            "title": f'Mock result {index + 1} for "{query}"',
            "link": f"https://example.com/deepeyes-v2/mock/{encoded_query}/{index + 1}",
            "snippet": f'Offline mock search result {index + 1} for query "{query}".',
            "date": None,
        }
        for index in range(size)
    ]
    return {"elapsed_time": 0.0, "data": data}


def _retriever_search(_query: str, _size: int) -> str:
    """Reserved entry point for the retriever backend."""
    return "Error"


def _external_search(_query: str, _size: int) -> str:
    """Reserved entry point for the external backend."""
    return "Error"


_SEARCH_BACKENDS = {
    "mock": _mock_search,
    "retriever": _retriever_search,
    "external": _external_search,
}


def search(query: str, size: int = 5) -> dict[str, object] | str:
    """Search with the configured backend, defaulting to the offline mock."""
    backend_name = os.environ.get("DEEPEYES_V2_SEARCH_BACKEND", "mock").strip().lower()
    backend = _SEARCH_BACKENDS.get(backend_name)
    if backend is None:
        logger.warning(f"[search] unknown backend: {backend_name}")
        return "Error"

    try:
        return backend(query, size)
    except Exception as exc:
        logger.warning(f"[search] backend {backend_name} failed: {exc}")
        return "Error"


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

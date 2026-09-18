# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search tool helpers for the DeepEyesV2 env.

* :func:`search` dispatches to the pluggable backend selected by
  ``DEEPEYES_V2_SEARCH_BACKEND`` (mock / retriever / external — see
  :mod:`app.search_backends`), retrying transient failures and keeping the
  ``"Error"`` sentinel contract the env relies on.
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

from app.search_backends import get_max_retries, get_search_backend


logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SECONDS = 1.0


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


def search(query: str, size: int = 5):
    """Text web-search via the configured backend (see
    ``app.search_backends``).

    Returns the unified shape::

        {"elapsed_time": float, "data": [{"title", "link", "snippet", "date"}, ...]}

    Backend configuration errors fail fast; transient backend failures are
    retried ``DEEPEYES_V2_SEARCH_MAX_RETRIES`` times with a fixed backoff.
    Every failure path returns the ``"Error"`` sentinel so the env surfaces a
    clean ``search_failed`` observation instead of crashing the agent process.
    """
    try:
        backend = get_search_backend()
    except Exception as e:
        logger.warning(f"[search] failed to initialise search backend: {e}")
        return "Error"

    max_try = get_max_retries()
    result = "Error"
    for try_idx in range(max_try):
        try:
            result = backend.search(query, size)
            break
        except Exception as e:
            logger.warning(f"[search] attempt {try_idx + 1}/{max_try} failed: {e}")
            result = "Error"
            if try_idx < max_try - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (try_idx + 1))
    return result


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

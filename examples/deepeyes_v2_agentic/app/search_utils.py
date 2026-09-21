# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlencode

from pydantic import ValidationError

from relax.utils.logging_utils import get_logger

from .search_config import (
    SEARCH_RESPONSE_ADAPTER,
    ExternalSearchConfig,
    SearchConfig,
    SearchError,
    SearchRequest,
    SearchResponse,
    load_search_config,
)
from .search_external import search_external
from .search_retriever import search_retriever


logger = get_logger(__name__)


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


def _search_mock(query: str, size: int, _config: SearchConfig) -> SearchResponse:
    encoded_query = urlencode({"q": query})
    return {
        "elapsed_time": 0.0,
        "data": [
            {
                "snippet": f"[mock] 离线测试资料，查询：{query}",
                "title": f"[mock] 离线搜索结果 {rank}",
                "link": f"https://example.com/mock/search?{encoded_query}&rank={rank}",
                "date": None,
            }
            for rank in range(1, size + 1)
        ],
    }


SearchBackend = Callable[[str, int, SearchConfig], SearchResponse | Literal["Error"]]
_SEARCH_BACKENDS: dict[str, SearchBackend] = {
    "mock": _search_mock,
    "retriever": search_retriever,
    "external": search_external,
}


def search(query: str, size: int | None = None) -> SearchResponse | Literal["Error"]:
    stage = "config"
    try:
        config = load_search_config()
        stage = "request"
        request = SearchRequest(query=query, size=config.topk if size is None else size)
        if isinstance(config, ExternalSearchConfig):
            maximum = config.request.max_size
            if maximum is not None and request.size > maximum:
                raise SearchError("size_exceeds_maximum")
        stage = "backend"
        backend = _SEARCH_BACKENDS.get(config.backend)
        if backend is None:
            raise SearchError("backend_unavailable")
        result = backend(request.query, request.size, config)
        if result == "Error":
            return "Error"
        stage = "response"
        response = SEARCH_RESPONSE_ADAPTER.validate_python(result, strict=True)
        response["data"] = response["data"][: request.size]
        return response
    except (SearchError, ValidationError):
        logger.warning("[search] failed during %s", stage)
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

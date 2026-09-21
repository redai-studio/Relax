# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import NoReturn

from relax.utils.logging_utils import get_logger

from .search_config import RetrieverSearchConfig, SearchConfig, SearchError, SearchResponse, SearchResult
from .search_http import request_search


logger = get_logger(__name__)


def _invalid_response(field: str) -> NoReturn:
    logger.warning("[search_retriever] invalid response field=%s", field)
    raise SearchError("invalid_retriever_response")


def parse_retriever_results(payload: object) -> list[SearchResult]:
    if not isinstance(payload, dict):
        _invalid_response("response")
    batches = payload.get("result")
    if not isinstance(batches, list) or len(batches) != 1:
        _invalid_response("result")
    if not isinstance(batches[0], list):
        _invalid_response("result[0]")

    results: list[SearchResult] = []
    for index, item in enumerate(batches[0]):
        location = f"result[0][{index}]"
        if not isinstance(item, dict):
            _invalid_response(location)
        if "document" in item:
            document = item["document"]
            location += ".document"
        else:
            document = item
        if not isinstance(document, dict):
            _invalid_response(location)
        contents = document.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            _invalid_response(f"{location}.contents")
        for field in ("title", "link", "url"):
            if field in document and not isinstance(document[field], str):
                _invalid_response(f"{location}.{field}")
        date = document.get("date")
        if date is not None and not isinstance(date, str):
            _invalid_response(f"{location}.date")

        title, separator, snippet = contents.partition("\n")
        results.append(
            {
                "title": document.get("title", title),
                "link": document.get("link", document.get("url", "")),
                "snippet": snippet if separator else contents,
                "date": date,
            }
        )
    return results


def search_retriever(query: str, size: int, config: SearchConfig) -> SearchResponse:
    if not isinstance(config, RetrieverSearchConfig):
        raise SearchError("invalid_retriever_config")
    return request_search(
        config,
        method="POST",
        json_body={"queries": [query], "topk": size, "return_scores": True},
        headers={"Accept": "application/json"},
        parse_results=parse_retriever_results,
    )

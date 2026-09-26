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
    """转换单条查询的 Search-R1 result 列表，兼容 document 包装形式.

    contents 首行为默认标题，其余内容为 snippet；单行内容全部作为 snippet. link 依次读取 link、url
    或空字符串，date 缺失时使用 None；非法结构或字段类型抛出 SearchError.
    """

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
    """以 POST 提交单条 Search-R1 查询，使用 size 作为 topk 并请求分数包装.

    返回统一搜索结果；配置类型、HTTP 请求或响应错误抛出 SearchError.
    """

    if not isinstance(config, RetrieverSearchConfig):
        raise SearchError("invalid_retriever_config")
    return request_search(
        config,
        method="POST",
        json_body={"queries": [query], "topk": size, "return_scores": True},
        headers={"Accept": "application/json"},
        parse_results=parse_retriever_results,
    )

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
from functools import partial
from typing import NoReturn

import httpx
from pydantic import JsonValue

from relax.utils.logging_utils import get_logger

from .search_config import (
    ExternalSearchConfig,
    ResponseMapping,
    SearchConfig,
    SearchError,
    SearchResponse,
    SearchResult,
)
from .search_http import request_search


logger = get_logger(__name__)


def _invalid_response(field: str, index: int | None = None, step: int | None = None) -> NoReturn:
    logger.warning("[search_external] invalid response field=%s index=%s step=%s", field, index, step)
    raise SearchError("invalid_external_response")


def _read_path(
    value: object,
    path: list[str],
    *,
    field: str,
    index: int | None = None,
    optional: bool = False,
) -> object:
    for step, key in enumerate(path):
        if not isinstance(value, dict):
            _invalid_response(field, index, step)
        if key not in value:
            if optional:
                return None
            _invalid_response(field, index, step)
        value = value[key]
    return value


def parse_external_results(payload: object, *, mapping: ResponseMapping) -> list[SearchResult]:
    """按 mapping 的对象键路径转换外部结果，必需字段均应为字符串.

    date 未配置、键缺失或值为 None 时返回 None；路径中间结构及字段类型错误抛出 SearchError.
    """

    items = _read_path(payload, mapping.items_path, field="items")
    if not isinstance(items, list):
        _invalid_response("items")
    results: list[SearchResult] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _invalid_response("item", index)
        values: dict[str, str] = {}
        for field in ("title", "link", "snippet"):
            value = _read_path(item, getattr(mapping.fields, field), field=field, index=index)
            if not isinstance(value, str):
                _invalid_response(field, index)
            values[field] = value
        date = None
        if mapping.fields.date is not None:
            date = _read_path(item, mapping.fields.date, field="date", index=index, optional=True)
            if date is not None and not isinstance(date, str):
                _invalid_response("date", index)
        results.append({"title": values["title"], "link": values["link"], "snippet": values["snippet"], "date": date})
    return results


def _query_params(config: ExternalSearchConfig, fields: dict[str, JsonValue]) -> httpx.QueryParams:
    for value in fields.values():
        values = value if isinstance(value, list) else [value]
        if any(isinstance(item, (dict, list)) for item in values):
            raise SearchError("invalid_query_parameter")
    try:
        return httpx.URL(str(config.endpoint)).params.merge(fields)
    except (httpx.InvalidURL, UnicodeError):
        raise SearchError("invalid_external_url") from None


def search_external(query: str, size: int, config: SearchConfig) -> SearchResponse:
    """按配置构建外部搜索请求，从环境变量读取认证值并返回统一结果.

    固定字段与 query、size 一起写入 query 参数或 JSON 请求体；认证 prefix 直接拼接凭据.
    配置类型、认证、请求或响应错误抛出 SearchError，HTTP 超时和重试由共用请求函数处理.
    """

    if not isinstance(config, ExternalSearchConfig):
        raise SearchError("invalid_external_config")
    headers = dict(config.headers)
    if config.auth is not None:
        token = os.environ.get(config.auth.env)
        if token is None or not token.strip():
            raise SearchError("missing_external_auth")
        headers[config.auth.header] = config.auth.prefix + token
    fields: dict[str, JsonValue] = {
        **config.request.static_fields,
        config.request.query_field: query,
        config.request.size_field: size,
    }
    params = _query_params(config, fields) if config.request.location == "query" else None
    return request_search(
        config,
        method=config.method,
        params=params,
        json_body=fields if config.request.location == "json" else None,
        headers=headers,
        parse_results=partial(parse_external_results, mapping=config.response),
    )

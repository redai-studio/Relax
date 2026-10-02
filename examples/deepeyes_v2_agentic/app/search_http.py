# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Literal, NoReturn

import httpx
from pydantic import JsonValue, ValidationError

from relax.utils.logging_utils import get_logger

from .search_config import (
    SEARCH_RESPONSE_ADAPTER,
    ExternalSearchConfig,
    RetrieverSearchConfig,
    SearchError,
    SearchResponse,
    SearchResult,
)


logger = get_logger(__name__)
HttpSearchConfig = RetrieverSearchConfig | ExternalSearchConfig
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def _failure(reason: str, attempt: int, total: int) -> SearchError:
    logger.warning("[search_http] failed reason=%s attempt=%d/%d", reason, attempt, total)
    return SearchError(reason)


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("invalid_json_constant")


def _create_client(config: HttpSearchConfig) -> httpx.Client:
    try:
        return httpx.Client(
            timeout=httpx.Timeout(config.timeout_s),
            trust_env=config.trust_env,
            follow_redirects=False,
        )
    except (OSError, ValueError, httpx.InvalidURL):
        raise _failure("invalid_client_config", 0, config.max_retries + 1) from None


def _request_json(
    client: httpx.Client,
    config: HttpSearchConfig,
    *,
    method: Literal["GET", "POST"],
    params: Mapping[str, JsonValue] | None,
    json_body: Mapping[str, JsonValue] | None,
    headers: Mapping[str, str] | None,
) -> tuple[object, int]:
    total = config.max_retries + 1
    delay = min(config.retry_delay_s, config.retry_max_delay_s)
    for attempt in range(1, total + 1):
        try:
            with client.stream(
                method, str(config.endpoint), params=params, json=json_body, headers=headers
            ) as response:
                if response.is_success:
                    response.read()
                    try:
                        return response.json(parse_constant=_reject_json_constant), attempt
                    except ValueError:
                        raise _failure("invalid_json", attempt, total) from None
                reason = f"http_{response.status_code}"
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise _failure(reason, attempt, total)
        except RETRYABLE_EXCEPTIONS as exc:
            reason = type(exc).__name__
        except (httpx.RequestError, httpx.InvalidURL, UnicodeError) as exc:
            raise _failure(type(exc).__name__, attempt, total) from None
        if attempt == total:
            raise _failure(reason, attempt, total)
        logger.warning("[search_http] retry reason=%s attempt=%d/%d", reason, attempt, total)
        if delay > 0:
            time.sleep(delay)
        delay = min(delay * 2, config.retry_max_delay_s)
    raise AssertionError("unreachable_retry_state")


def request_search(
    config: HttpSearchConfig,
    *,
    method: Literal["GET", "POST"],
    parse_results: Callable[[object], list[SearchResult]],
    params: Mapping[str, JsonValue] | None = None,
    json_body: Mapping[str, JsonValue] | None = None,
    headers: Mapping[str, str] | None = None,
) -> SearchResponse:
    """执行 HTTP 搜索，返回统一结果及包含重试等待和客户端关闭的总耗时.

    每次调用创建并关闭客户端，HTTP 各阶段使用 timeout_s，最多请求 max_retries + 1 次.
    408、429、500、502、503、504 及超时、指定连接、读写、协议错误触发重试. parse_results
    转换服务响应；预期请求或结果验证错误抛出 SearchError.
    """

    started = time.monotonic()
    attempt = 0
    total = config.max_retries + 1
    try:
        with _create_client(config) as client:
            payload, attempt = _request_json(
                client, config, method=method, params=params, json_body=json_body, headers=headers
            )
            try:
                response = SEARCH_RESPONSE_ADAPTER.validate_python(
                    {"elapsed_time": 0.0, "data": parse_results(payload)}, strict=True
                )
            except (SearchError, ValidationError):
                raise _failure("invalid_response", attempt, total) from None
    except httpx.RequestError as exc:
        raise _failure(type(exc).__name__, attempt, total) from None
    response["elapsed_time"] = time.monotonic() - started
    return response

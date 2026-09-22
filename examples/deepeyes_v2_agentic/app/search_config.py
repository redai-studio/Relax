# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, StringConstraints, TypeAdapter, model_validator
from typing_extensions import Self, TypedDict


SEARCH_CONFIG_ENV = "DEEPEYES_V2_SEARCH_CONFIG_PATH"
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
NonEmptyString = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
FieldPath = Annotated[list[NonEmptyString], Field(min_length=1)]


class SearchError(ValueError):
    """表示搜索配置、请求或服务响应中的预期错误."""

    pass


class SearchResult(TypedDict):
    """保存 title、link、snippet 字符串及可为 None 的 date 字符串."""

    title: str
    link: str
    snippet: str
    date: str | None


class SearchResponse(TypedDict):
    """保存结果列表及以秒计量的非负 elapsed_time，mock 耗时固定为 0.0."""

    elapsed_time: NonNegativeFloat
    data: list[SearchResult]


SEARCH_RESPONSE_ADAPTER = TypeAdapter(SearchResponse)


class StrictConfig(BaseModel):
    """严格验证字段类型、默认值和有限数值，并拒绝未知配置字段."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True, allow_inf_nan=False)


class SearchRequest(StrictConfig):
    """验证非空查询与正整数结果数量，并清理查询首尾空白."""

    query: NonEmptyString
    size: PositiveInt


class CommonSearchConfig(StrictConfig):
    """默认请求 5 条结果，HTTP 各阶段超时为 10 秒，失败后最多重试 2 次.

    重试等待从 0.5 秒开始倍增，上限为 2 秒；默认禁用 HTTPX 环境配置.
    """

    topk: PositiveInt = 5
    timeout_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10.0
    max_retries: NonNegativeInt = 2
    retry_delay_s: NonNegativeFloat = 0.5
    retry_max_delay_s: NonNegativeFloat = 2.0
    trust_env: bool = False


class MockSearchConfig(CommonSearchConfig):
    """配置无需网络或认证信息的确定性离线搜索后端."""

    backend: Literal["mock"] = "mock"


class RetrieverSearchConfig(CommonSearchConfig):
    """配置 Search-R1 兼容 HTTP 服务的 endpoint 与请求选项."""

    backend: Literal["retriever"]
    endpoint: HttpUrl


class AuthConfig(StrictConfig):
    """指定认证 header、读取凭据的环境变量及可选认证前缀."""

    header: NonEmptyString
    env: NonEmptyString
    prefix: str = ""


class RequestMapping(StrictConfig):
    """将查询和数量映射到 query 或 JSON 字段，并配置固定字段及可选数量上限."""

    location: Literal["query", "json"]
    query_field: NonEmptyString
    size_field: NonEmptyString
    max_size: PositiveInt | None = None
    static_fields: dict[NonEmptyString, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_field_names(self) -> Self:
        """拒绝重复请求字段、固定字段冲突及非有限 JSON 数值."""

        if self.query_field == self.size_field:
            raise ValueError("duplicate_request_fields")
        if {self.query_field, self.size_field} & self.static_fields.keys():
            raise ValueError("conflicting_static_fields")
        json.dumps(self.static_fields, allow_nan=False)
        return self


class ResponseFields(StrictConfig):
    """指定统一结果字段在服务条目中的逐级对象键路径."""

    title: FieldPath
    link: FieldPath
    snippet: FieldPath
    date: FieldPath | None = None


class ResponseMapping(StrictConfig):
    """指定结果列表的对象键路径及条目映射；空 items_path 表示响应本身为列表."""

    items_path: list[NonEmptyString]
    fields: ResponseFields
    optional_items_paths: list[FieldPath] = Field(default_factory=list)
    snippet_optional: bool = False

    @model_validator(mode="after")
    def validate_optional_paths(self) -> Self:
        """仅允许将结果路径的非空前缀声明为可选节点."""

        if any(path != self.items_path[: len(path)] for path in self.optional_items_paths):
            raise ValueError("invalid_optional_items_path")
        return self


class ExternalSearchConfig(CommonSearchConfig):
    """配置外部搜索 API 的 endpoint、认证、HTTP 方法及请求响应映射."""

    backend: Literal["external"]
    endpoint: HttpUrl
    method: Literal["GET", "POST"]
    headers: dict[NonEmptyString, str] = Field(default_factory=dict)
    auth: AuthConfig | None = None
    request: RequestMapping
    response: ResponseMapping

    @model_validator(mode="after")
    def validate_request_options(self) -> Self:
        """验证 GET 参数位置，并按大小写不敏感规则检查 header 冲突."""

        if self.method == "GET" and self.request.location != "query":
            raise ValueError("get_requires_query_parameters")
        header_names = [name.lower() for name in self.headers]
        if len(header_names) != len(set(header_names)):
            raise ValueError("duplicate_headers")
        if self.auth is not None and self.auth.header.lower() in header_names:
            raise ValueError("conflicting_auth_header")
        return self


SearchConfig = Annotated[
    MockSearchConfig | RetrieverSearchConfig | ExternalSearchConfig,
    Field(discriminator="backend"),
]
SEARCH_CONFIG_ADAPTER = TypeAdapter(SearchConfig)


def load_search_config() -> SearchConfig:
    """读取 DEEPEYES_V2_SEARCH_CONFIG_PATH 指定的 YAML，每次调用重新验证内容.

    未设置环境变量时使用默认 mock；文件省略 backend 时同样选择 mock. 文件读取、YAML 格式或顶层结构错误抛出
    SearchError，字段验证错误抛出 Pydantic ValidationError.
    """

    config_path = os.environ.get(SEARCH_CONFIG_ENV)
    if config_path is None:
        return MockSearchConfig()
    try:
        with Path(config_path).open(encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise SearchError("invalid_config_file") from exc
    if not isinstance(values, dict):
        raise SearchError("config_requires_object")
    return SEARCH_CONFIG_ADAPTER.validate_python({"backend": "mock", **values})

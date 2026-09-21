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
    pass


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    date: str | None


class SearchResponse(TypedDict):
    elapsed_time: NonNegativeFloat
    data: list[SearchResult]


SEARCH_RESPONSE_ADAPTER = TypeAdapter(SearchResponse)


class StrictConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True, allow_inf_nan=False)


class SearchRequest(StrictConfig):
    query: NonEmptyString
    size: PositiveInt


class CommonSearchConfig(StrictConfig):
    topk: PositiveInt = 5
    timeout_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10.0
    max_retries: NonNegativeInt = 2
    retry_delay_s: NonNegativeFloat = 0.5
    retry_max_delay_s: NonNegativeFloat = 2.0
    trust_env: bool = False


class MockSearchConfig(CommonSearchConfig):
    backend: Literal["mock"] = "mock"


class RetrieverSearchConfig(CommonSearchConfig):
    backend: Literal["retriever"]
    endpoint: HttpUrl


class AuthConfig(StrictConfig):
    header: NonEmptyString
    env: NonEmptyString
    prefix: str = ""


class RequestMapping(StrictConfig):
    location: Literal["query", "json"]
    query_field: NonEmptyString
    size_field: NonEmptyString
    max_size: PositiveInt | None = None
    static_fields: dict[NonEmptyString, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_field_names(self) -> Self:
        if self.query_field == self.size_field:
            raise ValueError("duplicate_request_fields")
        if {self.query_field, self.size_field} & self.static_fields.keys():
            raise ValueError("conflicting_static_fields")
        json.dumps(self.static_fields, allow_nan=False)
        return self


class ResponseFields(StrictConfig):
    title: FieldPath
    link: FieldPath
    snippet: FieldPath
    date: FieldPath | None = None


class ResponseMapping(StrictConfig):
    items_path: list[NonEmptyString]
    fields: ResponseFields


class ExternalSearchConfig(CommonSearchConfig):
    backend: Literal["external"]
    endpoint: HttpUrl
    method: Literal["GET", "POST"]
    headers: dict[NonEmptyString, str] = Field(default_factory=dict)
    auth: AuthConfig | None = None
    request: RequestMapping
    response: ResponseMapping

    @model_validator(mode="after")
    def validate_request_options(self) -> Self:
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

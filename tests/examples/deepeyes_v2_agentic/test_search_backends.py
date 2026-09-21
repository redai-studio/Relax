# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_http, search_utils  # noqa: E402
from app.search_config import SEARCH_CONFIG_ENV  # noqa: E402


@pytest.fixture
def http_handler(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    handler = Mock()
    clients = []

    def create_client(config: search_http.HttpSearchConfig) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handler), timeout=config.timeout_s)
        clients.append(client)
        return client

    monkeypatch.setattr(search_http, "_create_client", create_client)
    yield handler
    assert all(client.is_closed for client in clients)


@pytest.fixture
def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def write(backend: str, **overrides: Any) -> None:
        template = "brave" if backend == "external" else backend
        values = yaml.safe_load((EXAMPLE_DIR / f"search_config.{template}.yaml").read_text(encoding="utf-8"))
        values.update(overrides)
        path = tmp_path / "search.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        monkeypatch.setenv(SEARCH_CONFIG_ENV, str(path))

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-token")
    return write


@pytest.mark.parametrize("wrapped", [False, True])
def test_retriever_protocol_and_normalization(
    configure: Callable[..., None], http_handler: Mock, wrapped: bool
) -> None:
    configure("retriever", endpoint="https://retriever.example.test/custom", topk=7, timeout_s=0.25)
    documents = [
        {"contents": "Title\nFirst service text\nMore text"},
        {"contents": "Single line", "title": "Explicit", "url": "https://source.test", "date": "2026-09-20"},
        {"contents": "Third\nExtra result"},
    ]
    items = [{"document": item, "score": index} for index, item in enumerate(documents)] if wrapped else documents
    http_handler.return_value = httpx.Response(200, json={"result": [items]})
    response = search_utils.search("  中文 & query\n", size=2)
    assert isinstance(response, dict)
    assert response["elapsed_time"] >= 0
    assert response["data"] == [
        {"title": "Title", "link": "", "snippet": "First service text\nMore text", "date": None},
        {"title": "Explicit", "link": "https://source.test", "snippet": "Single line", "date": "2026-09-20"},
    ]
    request = http_handler.call_args.args[0]
    assert request.method == "POST" and str(request.url) == "https://retriever.example.test/custom"
    assert json.loads(request.content) == {"queries": ["中文 & query"], "topk": 2, "return_scores": True}
    assert request.headers["Accept"] == "application/json"
    assert request.extensions["timeout"] == {name: 0.25 for name in ("connect", "read", "write", "pool")}


@pytest.mark.parametrize(("method", "location"), [("GET", "query"), ("POST", "json")])
def test_external_request_and_response_mapping(
    configure: Callable[..., None], http_handler: Mock, method: str, location: str
) -> None:
    configure(
        "external",
        endpoint="https://external.example.test/search?source=web",
        method=method,
        auth={"header": "Authorization", "env": "BRAVE_SEARCH_API_KEY", "prefix": "Bearer "},
        request={"location": location, "query_field": "term", "size_field": "limit", "static_fields": {"lang": "zh"}},
        response={
            "items_path": ["payload", "hits"],
            "fields": {"title": ["meta", "heading"], "link": ["url"], "snippet": ["text"], "date": ["published"]},
        },
    )
    rows = [{"meta": {"heading": "Title"}, "url": "", "text": "Service text", "published": "2026-09-20"}]
    rows.append({"meta": {"heading": "Second"}, "url": "", "text": "Other text"})
    http_handler.return_value = httpx.Response(200, json={"payload": {"hits": rows}})
    response = search_utils.search(" query & text ", size=2)
    assert isinstance(response, dict)
    assert response["data"] == [
        {"title": "Title", "link": "", "snippet": "Service text", "date": "2026-09-20"},
        {"title": "Second", "link": "", "snippet": "Other text", "date": None},
    ]
    request = http_handler.call_args.args[0]
    assert request.method == method
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.url.params["source"] == "web"
    fields = {"term": "query & text", "limit": 2, "lang": "zh"}
    if location == "json":
        assert json.loads(request.content) == fields
    else:
        assert dict(request.url.params) == {"source": "web", **{key: str(value) for key, value in fields.items()}}
        assert request.content == b""


def test_brave_template_and_auth_refresh(
    configure: Callable[..., None], http_handler: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure("external")
    row = {"title": "Brave title", "url": "https://source.test", "description": "Brave service text"}
    http_handler.side_effect = lambda request: httpx.Response(200, json={"web": {"results": [row]}})
    for token in ("first-token", "second-token"):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", token)
        response = search_utils.search("query", size=20)
        assert isinstance(response, dict)
        assert response["data"] == [
            {"title": row["title"], "link": row["url"], "snippet": row["description"], "date": None}
        ]
        request = http_handler.call_args.args[0]
        assert request.url.params["q"] == "query" and request.url.params["count"] == "20"
        assert request.headers["X-Subscription-Token"] == token


def test_external_root_results_without_auth(configure: Callable[..., None], http_handler: Mock) -> None:
    configure(
        "external",
        auth=None,
        response={"items_path": [], "fields": {"title": ["title"], "link": ["url"], "snippet": ["text"]}},
    )
    http_handler.return_value = httpx.Response(200, json=[{"title": "", "url": "", "text": "Root service text"}])
    response = search_utils.search("query")
    assert isinstance(response, dict)
    assert response["data"] == [{"title": "", "link": "", "snippet": "Root service text", "date": None}]
    assert "X-Subscription-Token" not in http_handler.call_args.args[0].headers


@pytest.mark.parametrize("failure", ["missing_auth", "size_limit", "nested_query"])
def test_external_rejects_invalid_request_before_http(
    configure: Callable[..., None], http_handler: Mock, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    overrides = {}
    if failure == "missing_auth":
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY")
    elif failure == "nested_query":
        overrides["request"] = {
            "location": "query",
            "query_field": "q",
            "size_field": "count",
            "static_fields": {"options": {"lang": "zh"}},
        }
    configure("external", **overrides)
    assert search_utils.search("query", size=21 if failure == "size_limit" else 1) == "Error"
    http_handler.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "payload"), [("retriever", {"result": [[]]}), ("external", {"web": {"results": []}})]
)
def test_remote_empty_response_is_success(
    configure: Callable[..., None], http_handler: Mock, backend: str, payload: Any
) -> None:
    configure(backend)
    http_handler.return_value = httpx.Response(200, json=payload)
    response = search_utils.search("query")
    assert isinstance(response, dict) and response["data"] == []
    http_handler.assert_called_once()


@pytest.mark.parametrize(
    ("backend", "payload"),
    [
        ("retriever", {"result": [[], []]}),
        ("retriever", {"result": [[{"contents": ""}]]}),
        ("retriever", {"result": [[{"document": {"contents": "Text", "date": 7}}]]}),
        ("external", {"web": {"results": None}}),
        ("external", {"web": {"results": [{"title": "Title", "url": ""}]}}),
        ("external", {"web": {"results": [{"title": "Title", "url": "", "description": {}}]}}),
    ],
)
def test_remote_invalid_response_returns_error_without_retry(
    configure: Callable[..., None], http_handler: Mock, backend: str, payload: Any
) -> None:
    configure(backend)
    http_handler.return_value = httpx.Response(200, json=payload)
    assert search_utils.search("query") == "Error"
    http_handler.assert_called_once()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_config, search_utils  # noqa: E402


@pytest.fixture(autouse=True)
def clear_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(search_config.SEARCH_CONFIG_ENV, raising=False)


def use_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> Path:
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    monkeypatch.setenv(search_config.SEARCH_CONFIG_ENV, str(path))
    return path


def test_default_mock_is_deterministic_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("默认搜索访问了 HTTP 客户端或图片缓存")

    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(search_utils, "_get_image_search_cache", forbidden)
    query = ' 中文 "quotes" & + 🙂 '
    response = search_utils.search(query)
    assert isinstance(response, dict)
    assert response == search_utils.search(query)
    assert response["elapsed_time"] == 0.0
    assert len(response["data"]) == 5
    for index, item in enumerate(response["data"], 1):
        assert set(item) == {"title", "link", "snippet", "date"}
        assert item["title"].startswith("[mock]") and query.strip() in item["snippet"]
        assert item["date"] is None
        assert parse_qs(urlsplit(item["link"]).query) == {"q": [query.strip()], "rank": [str(index)]}
    assert search_utils.search("another query") != response


def test_mock_config_size_priority_and_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default = search_config.load_search_config()
    monkeypatch.setenv(search_config.SEARCH_CONFIG_ENV, str(EXAMPLE_DIR / "search_config.mock.yaml"))
    assert search_config.load_search_config() == default
    path = use_config(tmp_path, monkeypatch, {"topk": 3})
    assert len(search_utils.search("query")["data"]) == 3
    assert len(search_utils.search("query", 2)["data"]) == 2
    path.write_text("topk: 1\n", encoding="utf-8")
    assert len(search_utils.search("query")["data"]) == 1


@pytest.mark.parametrize(("query", "size"), [(" ", 1), (None, 1), ("query", 0), ("query", True)])
def test_invalid_request_stops_before_backend(monkeypatch: pytest.MonkeyPatch, query: Any, size: Any) -> None:
    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "mock", lambda *args: pytest.fail("非法请求进入后端"))
    assert search_utils.search(query, size) == "Error"


@pytest.mark.parametrize(
    "values",
    [
        {"backend": "unknown"},
        {"topk": "5"},
        {"timeout_s": 0},
        {"max_retries": -1},
        {"retry_delay_s": float("nan")},
        {"backend": "retriever", "endpoint": "ftp://example.test/index"},
        {"unexpected": True},
    ],
)
def test_invalid_config_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    use_config(tmp_path, monkeypatch, values)
    for backend in ("mock", "retriever", "external"):
        monkeypatch.setitem(search_utils._SEARCH_BACKENDS, backend, lambda *args: pytest.fail("非法配置进入后端"))
    assert search_utils.search("query") == "Error"


@pytest.mark.parametrize("content", [None, "backend: [\n", "[]\n"])
def test_config_file_failures_return_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    path = tmp_path / "search.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv(search_config.SEARCH_CONFIG_ENV, str(path))
    assert search_utils.search("query") == "Error"


@pytest.mark.parametrize("conflict", ["method", "fields", "static", "headers", "auth"])
def test_external_config_rejects_conflicting_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: str
) -> None:
    values = yaml.safe_load((EXAMPLE_DIR / "search_config.brave.yaml").read_text(encoding="utf-8"))
    if conflict == "method":
        values["request"]["location"] = "json"
    elif conflict == "fields":
        values["request"]["size_field"] = values["request"]["query_field"]
    elif conflict == "static":
        values["request"]["static_fields"] = {"q": "fixed"}
    elif conflict == "headers":
        values["headers"] = {"Accept": "application/json", "accept": "text/plain"}
    else:
        values["headers"] = {"x-subscription-token": "configured-value"}
    use_config(tmp_path, monkeypatch, values)
    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "external", lambda *args: pytest.fail("非法配置进入后端"))
    assert search_utils.search("query") == "Error"


@pytest.mark.parametrize(
    "options",
    [
        {"optional_items_paths": [[]]},
        {"optional_items_paths": [["results"]]},
        {"optional_items_paths": [["web", "results", "extra"]]},
        {"optional_items_paths": "web"},
        {"optional_items_paths": [[1]]},
        {"items_path": [], "optional_items_paths": [["web"]]},
        {"snippet_optional": "true"},
    ],
)
def test_external_config_rejects_invalid_response_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    values = yaml.safe_load((EXAMPLE_DIR / "search_config.brave.yaml").read_text(encoding="utf-8"))
    values["response"].update(options)
    use_config(tmp_path, monkeypatch, values)
    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "external", lambda *args: pytest.fail("非法配置进入后端"))
    assert search_utils.search("query") == "Error"


@pytest.mark.parametrize("response", ["Error", {"elapsed_time": 0.0, "data": []}])
def test_search_preserves_backend_error_and_empty_success(monkeypatch: pytest.MonkeyPatch, response: Any) -> None:
    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "mock", lambda *args: response)
    assert search_utils.search("query") == response


@pytest.mark.parametrize("exception", [search_config.SearchError, RuntimeError, KeyboardInterrupt])
def test_search_handles_only_expected_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, exception: type[BaseException]
) -> None:
    def backend(*args: Any) -> None:
        raise exception("private-test-value")

    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "mock", backend)
    monkeypatch.setattr(search_utils.logger, "handlers", [*search_utils.logger.handlers, caplog.handler])
    if exception is search_config.SearchError:
        assert search_utils.search("private query") == "Error"
        assert "failed during backend" in caplog.text
        assert "private" not in caplog.text
    else:
        with pytest.raises(exception):
            search_utils.search("query")


def test_search_validates_all_results_before_limiting_count(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {"title": "Title", "link": "", "snippet": "Body", "date": None}
    result = {"elapsed_time": 0.0, "data": [item, {**item, "date": 2}]}
    monkeypatch.setitem(search_utils._SEARCH_BACKENDS, "mock", lambda *args: result)
    assert search_utils.search("query", 1) == "Error"

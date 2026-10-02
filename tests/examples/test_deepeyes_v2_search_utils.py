# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from examples.deepeyes_v2_agentic.app.search_utils import search


def test_search_defaults_to_mock_when_backend_is_not_configured(monkeypatch):
    monkeypatch.delenv("DEEPEYES_V2_SEARCH_BACKEND", raising=False)

    result = search("default backend", size=2)

    assert result != "Error"
    assert len(result["data"]) == 2
    assert all(item["title"].startswith("Mock result") for item in result["data"])


def test_mock_search_returns_uniform_schema(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "mock")

    result = search("schema", size=2)

    assert set(result) == {"elapsed_time", "data"}
    assert isinstance(result["elapsed_time"], float)
    assert isinstance(result["data"], list)
    assert len(result["data"]) == 2
    for item in result["data"]:
        assert set(item) == {"title", "link", "snippet", "date"}
        assert isinstance(item["title"], str)
        assert isinstance(item["link"], str)
        assert isinstance(item["snippet"], str)
        assert item["date"] is None


def test_mock_search_data_is_deterministic(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "mock")

    first = search("repeatable query", size=3)
    second = search("repeatable query", size=3)

    assert first["data"] == second["data"]


def test_unavailable_backends_return_error(monkeypatch):
    for backend in ("retriever", "external"):
        monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", backend)
        assert search("not implemented") == "Error"


def test_unknown_backend_returns_error(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "unknown")

    assert search("query") == "Error"

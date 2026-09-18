# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the pluggable DeepEyesV2 web-search backends.

Covers mock / retriever / external adaptation, env-driven selection and the
failure paths (timeout, retries, malformed responses, invalid config). HTTP is
faked with a stub session so everything runs offline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import requests as requests_lib


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_backends as sb  # noqa: E402
from app.search_utils import search  # noqa: E402


_SEARCH_ENV_VARS = (
    sb.BACKEND_ENV,
    sb.TIMEOUT_ENV,
    sb.MAX_RETRIES_ENV,
    sb.RETRIEVER_URL_ENV,
    sb.RETRIEVER_TOPK_ENV,
    sb.EXTERNAL_CONFIG_ENV,
    sb.EXTERNAL_API_KEY_ENV,
)


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests_lib.exceptions.HTTPError(f"{self.status_code} error")


class _FakeSession:
    """Stub standing in for ``requests.Session``; records every call."""

    def __init__(self, outcomes) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple] = []

    def _handle(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def post(self, url, **kwargs):
        return self._handle("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._handle("GET", url, **kwargs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _SEARCH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    sb._BACKEND = None
    yield
    sb._BACKEND = None


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)


# ---- mock backend --------------------------------------------------------------


def test_default_backend_is_offline_mock():
    result = search("what is RL")
    assert isinstance(result, dict)
    assert set(result) == {"elapsed_time", "data"}
    assert result["elapsed_time"] == 0.0
    assert len(result["data"]) == 5
    for page in result["data"]:
        assert set(page) == {"title", "link", "snippet", "date"}
        assert page["date"] is None
        assert "what is RL" in page["snippet"]


def test_mock_is_deterministic_and_honors_size():
    assert search("q") == search("q")
    assert len(search("q", size=2)["data"]) == 2


# ---- retriever backend ---------------------------------------------------------


def _retriever_response(rows):
    return _FakeResponse(payload={"result": [rows]})


def test_retriever_normalizes_raw_corpus_rows():
    session = _FakeSession(
        [
            _retriever_response(
                [
                    {"title": "Alpha", "url": "http://a", "contents": "alpha text", "date": "2024-01-01"},
                    {"contents": "beta text"},
                ]
            )
        ]
    )
    backend = sb.RetrieverSearchBackend(url="http://retriever/retrieve", session=session)
    result = backend.search("query")
    assert [p["title"] for p in result["data"]] == ["Alpha", "Document 2"]
    assert result["data"][0]["link"] == "http://a"
    assert result["data"][0]["snippet"] == "alpha text"
    assert result["data"][0]["date"] == "2024-01-01"
    assert result["data"][1]["link"] == ""
    assert result["data"][1]["date"] is None
    assert result["elapsed_time"] >= 0.0
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://retriever/retrieve")
    assert kwargs["json"] == {"queries": ["query"], "topk": 5}


def test_retriever_accepts_document_score_rows():
    session = _FakeSession([_retriever_response([{"document": {"contents": "doc text"}, "score": 0.9}])])
    result = sb.RetrieverSearchBackend(session=session).search("q")
    assert result["data"][0]["snippet"] == "doc text"


def test_retriever_empty_result_is_valid():
    session = _FakeSession([_retriever_response([])])
    result = sb.RetrieverSearchBackend(session=session).search("q")
    assert result == {"elapsed_time": pytest.approx(0.0, abs=10.0), "data": []}


def test_retriever_empty_query_raises():
    backend = sb.RetrieverSearchBackend(session=_FakeSession([]))
    with pytest.raises(ValueError):
        backend.search("   ")


def test_retriever_topk_env_overrides_size(monkeypatch):
    monkeypatch.setenv(sb.BACKEND_ENV, "retriever")
    monkeypatch.setenv(sb.RETRIEVER_TOPK_ENV, "3")
    session = _FakeSession([_retriever_response([])])
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)
    search("q")
    assert session.calls[0][2]["json"] == {"queries": ["q"], "topk": 3}


@pytest.mark.parametrize(
    "outcome",
    [
        _FakeResponse(status_code=500),
        _FakeResponse(status_code=503),
        requests_lib.exceptions.Timeout("timed out"),
        requests_lib.exceptions.ConnectionError("refused"),
        _FakeResponse(payload=None, text="not json"),
        _FakeResponse(payload={"nope": 1}),
        _FakeResponse(payload={"result": "not-a-list"}),
    ],
)
def test_retriever_failures_return_error_after_retries(monkeypatch, outcome):
    monkeypatch.setenv(sb.BACKEND_ENV, "retriever")
    session = _FakeSession([outcome] * 3)
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)
    assert search("q") == "Error"
    assert len(session.calls) == 3


# ---- external backend ----------------------------------------------------------


def test_external_default_mapping_targets_serper():
    session = _FakeSession(
        [
            _FakeResponse(
                payload={"organic": [{"title": "T", "link": "http://t", "snippet": "S", "date": "2025-05-01"}]}
            )
        ]
    )
    backend = sb.ExternalSearchBackend(api_key="secret-key", session=session)
    result = backend.search("query", size=3)
    assert result["data"] == [{"title": "T", "link": "http://t", "snippet": "S", "date": "2025-05-01"}]
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://google.serper.dev/search")
    assert kwargs["json"] == {"q": "query", "num": 3}
    assert kwargs["headers"]["X-API-KEY"] == "secret-key"


def test_external_missing_date_becomes_none():
    session = _FakeSession([_FakeResponse(payload={"organic": [{"title": "T", "link": "L", "snippet": "S"}]})])
    result = sb.ExternalSearchBackend(session=session).search("q")
    assert result["data"][0]["date"] is None


def test_external_custom_config_via_env(monkeypatch, tmp_path):
    config_path = tmp_path / "search_external.json"
    config_path.write_text(
        json.dumps(
            {
                "endpoint": "https://search.example.com/v1",
                "method": "GET",
                "auth_header": "Authorization",
                "auth_scheme": "Bearer ",
                "request_map": {"query": "keyword", "size": ""},
                "response_map": {
                    "results": "data.items",
                    "title": "name",
                    "link": "url",
                    "snippet": "excerpt",
                    "date": "published",
                },
            }
        )
    )
    monkeypatch.setenv(sb.BACKEND_ENV, "external")
    monkeypatch.setenv(sb.EXTERNAL_CONFIG_ENV, str(config_path))
    monkeypatch.setenv(sb.EXTERNAL_API_KEY_ENV, "tok-123")
    session = _FakeSession(
        [
            _FakeResponse(
                payload={
                    "data": {"items": [{"name": "N", "url": "http://n", "excerpt": "E", "published": "2026-01-01"}]}
                }
            )
        ]
    )
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)

    result = search("hello", size=4)
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", "https://search.example.com/v1")
    assert kwargs["params"] == {"keyword": "hello"}
    assert kwargs["headers"]["Authorization"] == "Bearer tok-123"
    assert result["data"][0] == {"title": "N", "link": "http://n", "snippet": "E", "date": "2026-01-01"}


def test_external_invalid_config_fails_fast_without_http(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv(sb.BACKEND_ENV, "external")
    monkeypatch.setenv(sb.EXTERNAL_CONFIG_ENV, str(bad))
    session = _FakeSession([])
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)

    assert search("q") == "Error"
    assert session.calls == []


def test_external_unauthorized_returns_error(monkeypatch):
    monkeypatch.setenv(sb.BACKEND_ENV, "external")
    monkeypatch.setenv(sb.EXTERNAL_API_KEY_ENV, "bad-key")
    session = _FakeSession([_FakeResponse(status_code=401)] * 3)
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)

    assert search("q") == "Error"
    assert len(session.calls) == 3


# ---- dispatch ------------------------------------------------------------------


def test_unknown_backend_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv(sb.BACKEND_ENV, "bogus")
    result = search("q")
    assert isinstance(result, dict)
    assert result["data"][0]["title"] == "Placeholder Title 0"


def test_max_retries_env_override(monkeypatch):
    monkeypatch.setenv(sb.BACKEND_ENV, "retriever")
    monkeypatch.setenv(sb.MAX_RETRIES_ENV, "2")
    session = _FakeSession([_FakeResponse(status_code=500)] * 2)
    monkeypatch.setattr(sb, "_new_http_session", lambda: session)

    assert search("q") == "Error"
    assert len(session.calls) == 2

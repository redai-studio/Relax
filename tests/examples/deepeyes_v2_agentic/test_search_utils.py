# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for the pluggable DeepEyesV2 search backends.

All HTTP is stubbed; no real retrieval service or network is contacted. Covers
the three backends (mock / retriever / brave), their adaptation to the uniform
``{title, link, snippet, date}`` structure, and the timeout / invalid-response
/ server-error paths that must degrade to ``"Error"`` without crashing the
agent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import requests


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_utils  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes + fixtures
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, json_exc=None):
        self.status_code = status_code
        self._json_data = json_data
        self._json_exc = json_exc

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data


class FakeSession:
    def __init__(self, handler):
        self._handler = handler
        self.trust_env = False
        self.calls: list[tuple] = []

    def mount(self, *args, **kwargs):  # pragma: no cover - noop
        pass

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._handler(method, url, kwargs)


def _install_session(monkeypatch, handler) -> FakeSession:
    session = FakeSession(handler)
    monkeypatch.setattr(search_utils, "_build_session", lambda trust_env: session)
    return session


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DEEPEYES_V2_SEARCH"):
            monkeypatch.delenv(key, raising=False)
    search_utils._reset_backend_cache()
    yield
    search_utils._reset_backend_cache()


def _assert_uniform(result):
    assert isinstance(result, dict)
    assert set(result) == {"elapsed_time", "data"}
    assert isinstance(result["elapsed_time"], float)
    for item in result["data"]:
        assert set(item) == {"title", "link", "snippet", "date"}
        assert isinstance(item["title"], str)
        assert isinstance(item["link"], str)
        assert item["snippet"] is None or isinstance(item["snippet"], str)
        assert item["date"] is None or isinstance(item["date"], str)


# --------------------------------------------------------------------------- #
# mock backend
# --------------------------------------------------------------------------- #
def test_search_mock_offline_deterministic():
    result_a = search_utils.search("dogs", size=3)
    search_utils._reset_backend_cache()
    result_b = search_utils.search("dogs", size=3)

    _assert_uniform(result_a)
    assert len(result_a["data"]) == 3
    for item in result_a["data"]:
        assert item["title"]
        assert item["link"]
    assert result_a["data"] == result_b["data"]


def test_search_unknown_backend_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "does-not-exist")
    result = search_utils.search("x", size=2)
    _assert_uniform(result)
    assert len(result["data"]) == 2


# --------------------------------------------------------------------------- #
# retriever backend (Search-R1 protocol)
# --------------------------------------------------------------------------- #
def test_search_retriever_maps_search_r1_response(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_RETRIEVER_URL", "http://retriever/retrieve")

    def handler(method, url, kwargs):
        assert method == "POST"
        assert kwargs["json"]["queries"] == ["cats"]
        assert kwargs["json"]["return_scores"] is True
        return FakeResponse(
            json_data={
                "result": [
                    [
                        {"document": {"id": "doc-1", "contents": "Cats\nCats are small felines."}, "score": 0.9},
                        {"document": {"id": "doc-2", "contents": "Kittens\nBaby cats are kittens."}, "score": 0.8},
                    ]
                ]
            }
        )

    _install_session(monkeypatch, handler)
    result = search_utils.search("cats", size=2)

    _assert_uniform(result)
    assert result["data"][0]["title"] == "Cats"
    assert result["data"][0]["snippet"] == "Cats\nCats are small felines."
    assert result["data"][0]["link"] == "doc-1"
    assert "Placeholder" not in json.dumps(result["data"])


def test_search_retriever_missing_url_returns_error(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")
    assert search_utils.search("cats") == "Error"


# --------------------------------------------------------------------------- #
# brave backend
# --------------------------------------------------------------------------- #
def test_search_brave_maps_response(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "brave")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BRAVE_API_KEY", "secret-token")

    captured: dict = {}

    def handler(method, url, kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["params"] = kwargs.get("params", {})
        return FakeResponse(
            json_data={
                "web": {
                    "results": [
                        {
                            "title": "T1",
                            "url": "https://a",
                            "description": "S1",
                            "age": "2024-01-01",
                        },
                        {
                            "title": "T2",
                            "url": "https://b",
                            "description": "S2",
                        },
                    ]
                }
            }
        )

    _install_session(monkeypatch, handler)
    result = search_utils.search("weather", size=5)

    _assert_uniform(result)
    assert captured["method"] == "GET"
    assert captured["url"] == search_utils.BRAVE_DEFAULT_ENDPOINT
    assert captured["headers"]["X-Subscription-Token"] == "secret-token"
    assert captured["params"] == {"q": "weather", "count": 5}
    assert result["data"][0] == {"title": "T1", "link": "https://a", "snippet": "S1", "date": "2024-01-01"}
    assert result["data"][1]["date"] is None
    assert "Placeholder" not in json.dumps(result["data"])


def test_search_brave_missing_key_returns_error(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "brave")
    assert search_utils.search("q") == "Error"


def test_search_brave_custom_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "brave")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BRAVE_API_KEY", "k")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BRAVE_ENDPOINT", "https://proxy.example/brave")

    captured: dict = {}

    def handler(method, url, kwargs):
        captured["url"] = url
        return FakeResponse(json_data={"web": {"results": []}})

    _install_session(monkeypatch, handler)
    search_utils.search("q", size=1)
    assert captured["url"] == "https://proxy.example/brave"


# --------------------------------------------------------------------------- #
# error handling — must degrade to "Error", never raise
# --------------------------------------------------------------------------- #
def _retriever_env(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_RETRIEVER_URL", "http://retriever/retrieve")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_MAX_RETRIES", "2")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_RETRY_BUDGET", "0.05")


def test_search_timeout_returns_error(monkeypatch):
    _retriever_env(monkeypatch)

    def handler(method, url, kwargs):
        raise requests.exceptions.Timeout("timed out")

    _install_session(monkeypatch, handler)
    assert search_utils.search("q") == "Error"


def test_search_retry_budget_prevents_extra_request(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_RETRIEVER_URL", "http://retriever/retrieve")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_TIMEOUT", "30")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_MAX_RETRIES", "2")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_RETRY_BUDGET", "1")

    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(search_utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(search_utils.time, "sleep", clock.sleep)

    def handler(method, url, kwargs):
        clock.now = 0.95
        raise requests.exceptions.Timeout("timed out")

    session = _install_session(monkeypatch, handler)
    assert search_utils.search("q") == "Error"
    assert len(session.calls) == 1
    assert session.calls[0][2]["timeout"] == pytest.approx(1.0)
    assert clock.sleeps == [pytest.approx(0.05)]


def test_search_bad_json_returns_error(monkeypatch):
    _retriever_env(monkeypatch)

    def handler(method, url, kwargs):
        return FakeResponse(json_exc=ValueError("no json here"))

    _install_session(monkeypatch, handler)
    assert search_utils.search("q") == "Error"


def test_search_server_error_returns_error(monkeypatch):
    _retriever_env(monkeypatch)

    def handler(method, url, kwargs):
        return FakeResponse(status_code=503)

    _install_session(monkeypatch, handler)
    assert search_utils.search("q") == "Error"


if __name__ == "__main__":
    import pprint

    # Preserve the Brave key across the wipe below.
    #   export DEEPEYES_V2_SEARCH_BRAVE_API_KEY=your-brave-token
    #   PYTHONPATH=examples/deepeyes_v2_agentic \
    #     python tests/examples/deepeyes_v2_agentic/test_search_utils.py
    brave_key = os.environ.get("DEEPEYES_V2_SEARCH_BRAVE_API_KEY", "").strip()
    for _key in list(os.environ):
        if _key.startswith("DEEPEYES_V2_SEARCH"):
            del os.environ[_key]
    search_utils._reset_backend_cache()
    _real_build_session = search_utils._build_session

    print("=== [1/3] mock ===")
    pprint.pp(search_utils.search("Knowledge Graph", size=3))

    print("\n=== [2/3] retriever stub ===")
    os.environ["DEEPEYES_V2_SEARCH_BACKEND"] = "retriever"
    os.environ["DEEPEYES_V2_SEARCH_RETRIEVER_URL"] = "http://127.0.0.1:17389/retrieve"
    search_utils._reset_backend_cache()

    def _retriever_handler(method, url, kwargs):
        return FakeResponse(
            json_data={
                "result": [[{"document": {"id": "doc-1", "contents": "Title line\nBody snippet."}, "score": 0.9}]]
            }
        )

    search_utils._build_session = lambda trust_env: FakeSession(_retriever_handler)  # type: ignore[assignment]
    pprint.pp(search_utils.search("cats", size=2))
    search_utils._build_session = _real_build_session

    print("\n=== [3/3] brave LIVE ===")
    if not brave_key:
        print("SKIP: export DEEPEYES_V2_SEARCH_BRAVE_API_KEY=... then re-run.")
    else:
        for _key in list(os.environ):
            if _key.startswith("DEEPEYES_V2_SEARCH"):
                del os.environ[_key]
        os.environ["DEEPEYES_V2_SEARCH_BACKEND"] = "brave"
        os.environ["DEEPEYES_V2_SEARCH_BRAVE_API_KEY"] = brave_key
        os.environ["DEEPEYES_V2_SEARCH_TRUST_ENV"] = "1"
        search_utils._reset_backend_cache()
        pprint.pp(search_utils.search("who is the president of France", size=3))

    print("\nDone. Breakpoints: MockBackend / RetrieverBackend / BraveBackend / search().")

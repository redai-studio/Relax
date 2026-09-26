# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for the DeepEyesV2 pluggable search backends.

Covers the three backends (mock / retriever / external) against a local stub
HTTP service — no real network, no keys — plus the exception paths (timeout,
connection failure, non-2xx, invalid JSON, missing fields) and the env-level
``"Error"`` convention. Smoke for the rest lives in the launch scripts.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

import app.search_utils as search_utils  # noqa: E402
from app.env_deepeyes_v2 import DeepEyesV2Env  # noqa: E402
from app.search_backends import (  # noqa: E402
    ExternalSearchBackend,
    MockSearchBackend,
    RetrieverSearchBackend,
    SearchConfig,
    get_search_backend,
    normalize_results,
)


# ---------------------------------------------------------------------------
# Local stub HTTP service
# ---------------------------------------------------------------------------
class _StubSearchService:
    """Minimal localhost stub for the retriever/external backends.

    Tests configure ``status`` / ``body`` / ``raw_body`` / ``delay_seconds``
    and inspect ``requests`` (method, path, headers, parsed JSON body).
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.status = 200
        self.body: object = {"result": []}
        self.raw_body: bytes | None = None
        self.delay_seconds = 0.0
        self.fail_first_n = 0  # respond 500 for the first N requests, then normally
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> str:
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                stub.requests.append({"path": self.path, "headers": dict(self.headers.items()), "payload": payload})
                time.sleep(stub.delay_seconds)
                status = 500 if len(stub.requests) <= stub.fail_first_n else stub.status
                body = stub.raw_body if stub.raw_body is not None else json.dumps(stub.body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A002
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}/retrieve"

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _assert_uniform_shape(result: dict) -> None:
    assert isinstance(result["elapsed_time"], float)
    assert result["elapsed_time"] >= 0.0
    assert isinstance(result["data"], list)
    for row in result["data"]:
        assert isinstance(row["title"], str)
        assert isinstance(row["link"], str)
        assert isinstance(row["snippet"], str)
        assert row["date"] is None or isinstance(row["date"], str)


# ---------------------------------------------------------------------------
# mock backend (default)
# ---------------------------------------------------------------------------
def test_mock_backend_is_deterministic_and_uniform():
    first = MockSearchBackend().search("relax rl", 3)
    second = MockSearchBackend().search("relax rl", 3)
    assert first["data"] == second["data"]  # elapsed_time is measured, data is deterministic
    _assert_uniform_shape(first)
    assert len(first["data"]) == 3
    assert first["data"][0]["title"] == "Mock result 1 for: relax rl"
    assert all(row["date"] is None for row in first["data"])


def test_mock_backend_respects_top_k_and_handles_zero():
    assert len(MockSearchBackend().search("q", 5)["data"]) == 5
    assert MockSearchBackend().search("q", 0)["data"] == []


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------
def test_config_from_env_defaults():
    config = SearchConfig.from_env({})
    assert config.backend == "mock"
    assert config.top_k == 5
    assert config.timeout_seconds == 30.0
    assert config.max_retries == 3
    assert config.external_auth_header == "X-API-KEY"
    assert config.external_field_map == {}


def test_config_from_env_parses_overrides():
    config = SearchConfig.from_env(
        {
            "DEEPEYES_V2_SEARCH_BACKEND": "retriever",
            "DEEPEYES_V2_SEARCH_TOP_K": "7",
            "DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS": "1.5",
            "DEEPEYES_V2_SEARCH_MAX_RETRIES": "2",
            "DEEPEYES_V2_RETRIEVER_URL": "http://127.0.0.1:9/retrieve",
            "DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP": '{"date": "published"}',
        }
    )
    assert config.backend == "retriever"
    assert config.top_k == 7
    assert config.timeout_seconds == 1.5
    assert config.max_retries == 2
    assert config.retriever_url == "http://127.0.0.1:9/retrieve"
    assert config.external_field_map == {"date": "published"}


def test_config_from_env_rejects_malformed_values():
    for name, raw in [
        ("DEEPEYES_V2_SEARCH_TOP_K", "five"),
        ("DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS", "soon"),
        ("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP", "not-json"),
        ("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP", '["title"]'),
        ("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP", '{"unknown": "x"}'),
        ("DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP", '{"title": ""}'),
    ]:
        try:
            SearchConfig.from_env({name: raw})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {name}={raw!r}")


def test_get_search_backend_unknown_name_raises():
    try:
        get_search_backend(SearchConfig(backend="nope"))
    except ValueError as exc:
        assert "DEEPEYES_V2_SEARCH_BACKEND" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown backend")


def test_normalize_results_maps_and_defaults():
    rows = [
        {"title": "T", "link": "L", "snippet": "S", "published": "2026-01-01"},
        {"title": 42, "snippet": None},
        "not-a-dict",
    ]
    data = normalize_results(rows, {"date": "published"})
    assert data[0] == {"title": "T", "link": "L", "snippet": "S", "date": "2026-01-01"}
    assert data[1] == {"title": "", "link": "", "snippet": "", "date": None}
    assert len(data) == 2  # non-dict row dropped with a warning


# ---------------------------------------------------------------------------
# retriever backend (Search-R1 protocol) against the local stub
# ---------------------------------------------------------------------------
def _retriever_config(url: str, **overrides) -> SearchConfig:
    values = {
        "backend": "retriever",
        "retriever_url": url,
        "timeout_seconds": 5.0,
        "max_retries": 1,
        "retry_delay_seconds": 0.0,
    }
    values.update(overrides)
    return SearchConfig(**values)


def test_retriever_backend_maps_search_r1_response():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.body = {
            "result": [
                [
                    {"contents": "Doc text one", "title": "Title One", "link": "http://a/1", "date": "2026-01-02"},
                    {"contents": "Doc text two"},
                ]
            ]
        }
        result = RetrieverSearchBackend(_retriever_config(url)).search("who wrote relax", 2)
        _assert_uniform_shape(result)
        assert result["data"][0] == {
            "title": "Title One",
            "link": "http://a/1",
            "snippet": "Doc text one",
            "date": "2026-01-02",
        }
        assert result["data"][1] == {"title": "", "link": "", "snippet": "Doc text two", "date": None}
        request = stub.requests[0]
        assert request["payload"] == {"queries": ["who wrote relax"], "topk": 2, "return_scores": False}
    finally:
        stub.stop()


def test_retriever_backend_empty_result_is_success_with_empty_data():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.body = {"result": [[]]}
        result = RetrieverSearchBackend(_retriever_config(url)).search("q", 3)
        assert result["data"] == []
    finally:
        stub.stop()


def test_retriever_backend_retries_then_succeeds():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.fail_first_n = 2
        stub.body = {"result": [[{"contents": "doc"}]]}
        result = RetrieverSearchBackend(_retriever_config(url, max_retries=3)).search("q", 1)
        assert result["data"][0]["snippet"] == "doc"
        assert len(stub.requests) == 3
    finally:
        stub.stop()


def test_retriever_backend_retries_exhausted_returns_error():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.status = 500
        result = RetrieverSearchBackend(_retriever_config(url, max_retries=3)).search("q", 1)
        assert result == "Error"
        assert len(stub.requests) == 3
    finally:
        stub.stop()


def test_retriever_backend_does_not_retry_client_errors():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.status = 403
        result = RetrieverSearchBackend(_retriever_config(url, max_retries=3)).search("q", 1)
        assert result == "Error"
        assert len(stub.requests) == 1
    finally:
        stub.stop()


def test_retriever_backend_invalid_json_returns_error():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.raw_body = b"<html>not json</html>"
        result = RetrieverSearchBackend(_retriever_config(url)).search("q", 1)
        assert result == "Error"
    finally:
        stub.stop()


def test_retriever_backend_structural_failures_return_error():
    stub = _StubSearchService()
    url = stub.start()
    config = _retriever_config(url)
    try:
        for body in [{"result": "not-a-list"}, {"result": ["not-a-list-of-lists"]}, {"unexpected": {}}]:
            stub.requests.clear()
            stub.body = body
            result = RetrieverSearchBackend(config).search("q", 1)
            assert result == "Error", f"expected Error for body={body!r}"
    finally:
        stub.stop()


def test_retriever_backend_connection_refused_returns_error():
    backend = RetrieverSearchBackend(_retriever_config(f"http://127.0.0.1:{_free_port()}/retrieve"))
    assert backend.search("q", 1) == "Error"


def test_retriever_backend_timeout_returns_error():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.delay_seconds = 1.0
        result = RetrieverSearchBackend(_retriever_config(url, timeout_seconds=0.2)).search("q", 1)
        assert result == "Error"
    finally:
        stub.stop()


def test_retriever_backend_missing_url_config_raises_at_construction():
    try:
        RetrieverSearchBackend(SearchConfig(backend="retriever"))
    except ValueError as exc:
        assert "DEEPEYES_V2_RETRIEVER_URL" in str(exc)
    else:
        raise AssertionError("expected ValueError when retriever URL is missing")


# ---------------------------------------------------------------------------
# external backend (Serper.dev preset + custom mapping) against the stub
# ---------------------------------------------------------------------------
def _external_config(url: str, **overrides) -> SearchConfig:
    values = {
        "backend": "external",
        "external_endpoint": url,
        "timeout_seconds": 5.0,
        "max_retries": 1,
        "retry_delay_seconds": 0.0,
    }
    values.update(overrides)
    return SearchConfig(**values)


_SERPER_ROW = {"title": "Relax", "link": "http://r/1", "snippet": "A Ray RL framework", "date": "2026-09-01"}


def test_external_backend_serper_preset_mapping():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.body = {"organic": [_SERPER_ROW, {"title": "No date row", "link": "http://r/2", "snippet": "s"}]}
        config = _external_config(url, external_api_key="stub-key")
        result = ExternalSearchBackend(config).search("relax", 2)
        _assert_uniform_shape(result)
        assert result["data"][0] == {
            "title": "Relax",
            "link": "http://r/1",
            "snippet": "A Ray RL framework",
            "date": "2026-09-01",
        }
        assert result["data"][1]["date"] is None
        request = stub.requests[0]
        assert request["payload"] == {"q": "relax", "num": 2}
        assert request["headers"].get("X-API-KEY") == "stub-key"
    finally:
        stub.stop()


def test_external_backend_without_api_key_omits_auth_header():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.body = {"organic": []}
        ExternalSearchBackend(_external_config(url)).search("q", 1)
        assert "X-API-KEY" not in stub.requests[0]["headers"]
    finally:
        stub.stop()


def test_external_backend_custom_request_and_response_mapping():
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.body = {"data": [{"name": "N", "url": "U", "description": "D", "published": "2026-05-01"}]}
        config = _external_config(
            url,
            external_auth_header="Authorization",
            external_api_key="Bearer stub-token",
            external_query_field="search_term",
            external_topk_field="limit",
            external_results_field="data",
            external_field_map={"title": "name", "link": "url", "snippet": "description", "date": "published"},
        )
        result = ExternalSearchBackend(config).search("relax", 4)
        assert result["data"][0] == {"title": "N", "link": "U", "snippet": "D", "date": "2026-05-01"}
        request = stub.requests[0]
        assert request["payload"] == {"search_term": "relax", "limit": 4}
        assert request["headers"].get("Authorization") == "Bearer stub-token"
    finally:
        stub.stop()


def test_external_backend_structural_failures_return_error():
    stub = _StubSearchService()
    url = stub.start()
    config = _external_config(url)
    try:
        for body in [{"organic": "not-a-list"}, {"results": []}, [1, 2, 3]]:
            stub.requests.clear()
            stub.body = body
            result = ExternalSearchBackend(config).search("q", 1)
            assert result == "Error", f"expected Error for body={body!r}"
    finally:
        stub.stop()


def test_external_backend_missing_endpoint_config_raises_at_construction():
    try:
        ExternalSearchBackend(SearchConfig(backend="external"))
    except ValueError as exc:
        assert "DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT" in str(exc)
    else:
        raise AssertionError("expected ValueError when external endpoint is missing")


# ---------------------------------------------------------------------------
# search() entry point + env-level "Error" convention
# ---------------------------------------------------------------------------
def test_search_entry_defaults_to_mock(monkeypatch):
    for name in [n for n in os.environ if n.startswith("DEEPEYES_V2_")]:
        monkeypatch.delenv(name)
    result = search_utils.search("offline query")
    _assert_uniform_shape(result)
    assert len(result["data"]) == 5  # historical default size
    assert result["data"][0]["snippet"].startswith("Deterministic mock snippet")


def test_search_entry_explicit_size_wins(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_TOP_K", "2")
    assert len(search_utils.search("q")["data"]) == 2
    assert len(search_utils.search("q", size=4)["data"]) == 4


def test_search_entry_config_failures_map_to_error(monkeypatch):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")  # no URL configured
    assert search_utils.search("q") == "Error"
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "unknown")
    assert search_utils.search("q") == "Error"
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "mock")
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_TOP_K", "not-a-number")
    assert search_utils.search("q") == "Error"


def test_env_exec_tool_renders_mock_results():
    """The mock backend flows through exec_tool into a normal observation."""
    env = DeepEyesV2Env(data_index="idx", sandbox_executor=None, image=None)
    response = '<tool_call>{"name": "search", "arguments": {"query": "relax"}}</tool_call>'
    obs = asyncio.run(env.exec_tool(response))
    assert obs.error is None
    assert "A Google search for 'relax' found 5 results" in obs.body_text
    assert "[Mock result 1 for: relax](https://mock.local/search/1)" in obs.body_text


def test_env_exec_tool_maps_backend_failure_to_error_obs(monkeypatch):
    """A broken backend keeps the agent process alive: the env renders its
    existing "Error: search returned no result" observation."""
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "external")  # endpoint unset
    env = DeepEyesV2Env(data_index="idx", sandbox_executor=None, image=None)
    response = '<tool_call>{"name": "search", "arguments": {"query": "relax"}}</tool_call>'
    obs = asyncio.run(env.exec_tool(response))
    assert obs.error == "search_failed"
    assert obs.body_text.startswith("Error: search returned no result")


def test_env_dispatch_search_error_convention_with_live_failure(monkeypatch):
    """A retriever service returning garbage reaches _dispatch_search as
    "Error" and comes back as {"status": "error"} instead of raising."""
    stub = _StubSearchService()
    url = stub.start()
    try:
        stub.raw_body = b"not json"
        monkeypatch.setenv("DEEPEYES_V2_SEARCH_BACKEND", "retriever")
        monkeypatch.setenv("DEEPEYES_V2_RETRIEVER_URL", url)
        monkeypatch.setenv("DEEPEYES_V2_SEARCH_MAX_RETRIES", "1")
        env = DeepEyesV2Env(data_index="idx", sandbox_executor=None, image=None)
        dispatch_result = env._dispatch_search("search", {"query": "q"})
        assert dispatch_result == {"status": "error", "result": "Error", "images": []}
    finally:
        stub.stop()


def test_launch_scripts_forward_search_env_vars():
    """The run scripts whitelist which env vars reach the Ray workers, so a
    knob missing from that whitelist would silently leave training on the mock
    backend.

    Every documented search knob must be forwarded.
    """
    search_env_vars = [
        "DEEPEYES_V2_SEARCH_BACKEND",
        "DEEPEYES_V2_SEARCH_TOP_K",
        "DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS",
        "DEEPEYES_V2_SEARCH_MAX_RETRIES",
        "DEEPEYES_V2_SEARCH_RETRY_DELAY_SECONDS",
        "DEEPEYES_V2_RETRIEVER_URL",
        "DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT",
        "DEEPEYES_V2_EXTERNAL_SEARCH_API_KEY",
        "DEEPEYES_V2_EXTERNAL_SEARCH_AUTH_HEADER",
        "DEEPEYES_V2_EXTERNAL_SEARCH_QUERY_FIELD",
        "DEEPEYES_V2_EXTERNAL_SEARCH_TOPK_FIELD",
        "DEEPEYES_V2_EXTERNAL_SEARCH_RESULTS_FIELD",
        "DEEPEYES_V2_EXTERNAL_SEARCH_FIELD_MAP",
    ]
    for script_name in ("run_deepeyes_v2_agentic.sh", "run_deepeyes_v2_agentic_klx.sh"):
        body = (EXAMPLE_DIR / script_name).read_text(encoding="utf-8")
        for var in search_env_vars:
            assert var in body, f"{script_name} does not forward {var} into the Ray workers"

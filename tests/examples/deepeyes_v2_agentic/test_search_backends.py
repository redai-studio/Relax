# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Search contract and local HTTP integration tests; no external services."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import agent, search_backends, search_utils  # noqa: E402
from app.env_deepeyes_v2 import DeepEyesV2Env  # noqa: E402


@pytest.fixture(autouse=True)
def clean_search_env(monkeypatch):
    monkeypatch.delenv("DEEPEYES_V2_SEARCH_CONFIG", raising=False)
    monkeypatch.delenv("DEEPEYES_V2_SEARCH_API_KEY", raising=False)


@pytest.fixture
def configure(tmp_path, monkeypatch):
    def write(config):
        path = tmp_path / "search.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv("DEEPEYES_V2_SEARCH_CONFIG", str(path))
        return path

    return write


@pytest.fixture
def server():
    state = SimpleNamespace(requests=[], status=200, body={"result": [[]]}, delay=0.0)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(parse_qs(urlsplit(self.path).query))

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, payload):
            state.requests.append((self.command, payload, dict(self.headers)))
            time.sleep(state.delay)
            status = state.status.pop(0) if isinstance(state.status, list) else state.status
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Location", "/redirect-target")
            self.end_headers()
            body = state.body if isinstance(state.body, bytes) else json.dumps(state.body).encode()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_port}/retrieve"
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


def retriever_config(server, **options):
    return {"backend": "retriever", "retriever": {"url": server.url}, "backoff_s": 0, **options}


def external_config(server, **options):
    return {
        "backend": "external",
        "backoff_s": 0,
        "external": {
            "endpoint": server.url,
            "method": "POST",
            "auth": {"header": "Authorization", "prefix": "Bearer "},
            "request_map": {"query": "query", "size": "max_results"},
            "response_map": {
                "results": "results",
                "title": "title",
                "link": "url",
                "snippet": "content",
                "date": "published_date",
            },
            **options,
        },
    }


def test_mock_is_deterministic_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mock attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    first = search_utils.search("搜索", 2)
    assert first == search_utils.search("搜索", 2)
    assert first["elapsed_time"] == 0.0
    assert len(first["data"]) == 2
    assert all(set(row) == {"title", "link", "snippet", "date"} and row["date"] is None for row in first["data"])
    assert "搜索" in first["data"][0]["snippet"]


def test_config_top_k_and_explicit_size(configure):
    assert len(search_utils.search("q")["data"]) == 5
    configure({"top_k": 2})
    assert len(search_utils.search("q")["data"]) == 2
    assert len(search_utils.search("q", 1)["data"]) == 1
    configure({"top_k": 3})
    assert len(search_utils.search("q")["data"]) == 3


@pytest.mark.parametrize("query", [None, {}, [], "", "   ", 42])
def test_invalid_query_returns_error(query):
    assert search_utils.search(query) == "Error"


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "2"])
def test_invalid_size_returns_error(size):
    assert search_utils.search("q", size) == "Error"


@pytest.mark.parametrize(
    "config",
    [
        [],
        None,
        {"backend": "typo"},
        {"backend": []},
        {"top_k": True},
        {"top_k": 0},
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"timeout_s": float("inf")},
        {"max_retries": -1},
        {"max_retries": False},
        {"backoff_s": -1},
        {"trust_env": "false"},
        {"topk": 1},
        {"backend": "retriever", "retriever": {"url": None}},
        {"backend": "retriever", "retriever": {"url": "file:///tmp/data"}},
        {"backend": "retriever", "retriever": {"url": "http://user:secret@host"}},
    ],
)
def test_bad_config_does_not_request(configure, monkeypatch, config):
    configure(config)
    monkeypatch.setattr(
        requests.Session, "request", lambda *a, **kw: pytest.fail("invalid configuration made a request")
    )
    assert search_utils.search("q") == "Error"


def test_config_read_errors_are_diagnostic_and_redacted(configure, monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_CONFIG", str(tmp_path / "missing.yaml"))
    assert search_utils.search("q") == "Error"
    assert "FileNotFoundError" in caplog.text
    path = configure({})
    path.write_text("external: [sensitive-value", encoding="utf-8")
    assert search_utils.search("q") == "Error"
    assert "DEEPEYES_V2_SEARCH_CONFIG" in caplog.text
    assert "sensitive-value" not in caplog.text


@pytest.mark.parametrize("scored", [False, True])
def test_search_r1_protocol_and_normalization(server, configure, scored):
    docs = [
        {"contents": '"Relax"\nA real passage.', "id": "123"},
        {"title": "Second", "url": "https://example.org", "contents": "More", "date": "2026-09-19"},
    ]
    server.body = {"result": [[{"document": doc, "score": 0.9} for doc in docs] if scored else docs]}
    configure(retriever_config(server, top_k=2))
    result = search_utils.search("a query")
    assert result["data"] == [
        {"title": "Relax", "link": "", "snippet": '"Relax"\nA real passage.', "date": None},
        {"title": "Second", "link": "https://example.org", "snippet": "More", "date": "2026-09-19"},
    ]
    assert result["elapsed_time"] >= 0
    assert server.requests[0][:2] == ("POST", {"queries": ["a query"], "topk": 2, "return_scores": True})
    assert len(search_utils.search("q", 1)["data"]) == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"result": []},
        {"result": [[], []]},
        {"result": [None]},
        {"result": [[None]]},
        {"result": [[{"document": []}]]},
        {"result": [[{"contents": 5}]]},
        {"result": [[{"contents": "text", "date": 12}]]},
        b"not json",
    ],
)
def test_malformed_retriever_response_is_not_empty_success(server, configure, body):
    server.body = body
    configure(retriever_config(server))
    assert search_utils.search("q") == "Error"
    assert len(server.requests) == 1


def test_empty_results_are_successful(server, configure):
    configure(retriever_config(server))
    assert search_utils.search("q")["data"] == []
    assert len(server.requests) == 1


@pytest.mark.parametrize("method", ["POST", "GET"])
def test_external_mapping_and_auth(server, configure, monkeypatch, method):
    configure(external_config(server, method=method))
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_API_KEY", "test-credential")
    server.body = {"results": [{"title": "Page", "url": "https://example.org", "content": "Retrieved text"}]}
    result = search_utils.search("中文 & query", 1)
    assert result["data"] == [
        {"title": "Page", "link": "https://example.org", "snippet": "Retrieved text", "date": None}
    ]
    expected = (
        {"query": "中文 & query", "max_results": 1}
        if method == "POST"
        else {"query": ["中文 & query"], "max_results": ["1"]}
    )
    assert server.requests[0][:2] == (method, expected)
    assert server.requests[0][2]["Authorization"] == "Bearer test-credential"


def test_external_nested_mapping_without_auth(server, configure):
    config = external_config(server)
    del config["external"]["auth"]
    config["external"]["response_map"] = {
        "results": "data.items",
        "title": "page.title",
        "link": "url",
        "snippet": "text",
    }
    configure(config)
    server.body = {
        "data": {"items": [{"page": {"title": "Nested"}, "url": "https://example.org", "text": "Actual text"}]}
    }
    assert search_utils.search("q")["data"][0]["title"] == "Nested"
    assert "Authorization" not in server.requests[0][2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", 5),
        ("endpoint", None),
        ("request_map", {"query": "q", "size": "q"}),
        ("response_map", {"results": "results"}),
        ("auth", {"header": "Authorization", "prefix": "\n"}),
    ],
)
def test_invalid_external_config_is_rejected_before_http(server, configure, monkeypatch, field, value):
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_API_KEY", "test-credential")
    configure(external_config(server, **{field: value}))
    assert search_utils.search("q") == "Error"
    assert not server.requests


def test_missing_key_is_not_sent(server, configure):
    configure(external_config(server))
    assert search_utils.search("q") == "Error"
    assert not server.requests


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"results": {}},
        {"results": [None]},
        {"results": [{"title": "x", "url": "u", "content": []}]},
        {"results": [{"title": "x", "url": "u", "content": "s", "published_date": 42}]},
    ],
)
def test_external_malformed_results(server, configure, monkeypatch, body):
    configure(external_config(server))
    monkeypatch.setenv("DEEPEYES_V2_SEARCH_API_KEY", "test-credential")
    server.body = body
    assert search_utils.search("q") == "Error"
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_status_is_retried(server, configure, status):
    configure(retriever_config(server))
    server.status = [status, 200]
    assert search_utils.search("q")["data"] == []
    assert len(server.requests) == 2


@pytest.mark.parametrize("status,attempts", [(401, 1), (403, 1), (302, 1), (503, 3)])
def test_http_failure_is_bounded_and_redacted(server, configure, caplog, status, attempts):
    configure(retriever_config(server))
    server.status = status
    server.body = {"detail": "sensitive-response"}
    assert search_utils.search("q") == "Error"
    assert len(server.requests) == attempts
    assert str(status) in caplog.text
    assert "sensitive-response" not in caplog.text


def test_read_timeout(server, configure, caplog):
    configure(retriever_config(server, timeout_s=0.05, max_retries=0))
    server.delay = 0.2
    assert search_utils.search("q") == "Error"
    assert "ReadTimeout" in caplog.text


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
def test_transport_retry_and_backoff(configure, monkeypatch, error):
    configure({"backend": "retriever", "retriever": {"url": "http://localhost/retrieve"}})
    calls, sleeps = [], []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error("sensitive-transport-error")

    monkeypatch.setattr(requests.Session, "request", fail)
    monkeypatch.setattr(search_backends.time, "sleep", sleeps.append)
    assert search_utils.search("q") == "Error"
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert all(c["timeout"] == 10 and c["allow_redirects"] is False for c in calls)


@pytest.mark.parametrize("failed", [False, True])
async def test_agent_continues_after_search(server, configure, monkeypatch, failed):
    class APIStatusError(Exception):
        pass

    configure(retriever_config(server, max_retries=0))
    server.status = 503 if failed else 200
    server.body = {"result": [[{"contents": "Evidence from the retriever."}]]}
    monkeypatch.setenv("OPENAI_API_KEY", "local-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://unused.invalid/v1")
    monkeypatch.setattr(agent, "_build_executor", lambda *args: None)
    seen = []

    async def create(**kwargs):
        seen.append(list(kwargs["messages"]))
        text = (
            '<tool_call>{"name":"search","arguments":{"query":"q"}}</tool_call>'
            if len(seen) == 1
            else "<answer>done</answer>"
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])

    # The scripted model does not require the optional OpenAI SDK.
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            APIStatusError=APIStatusError,
            AsyncOpenAI=lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        ),
    )
    result = await agent.run_session([{"role": "user", "content": "Search then answer"}], {})
    assert len(server.requests) == 1
    assert result["metadata"]["stop_reason"] == "env_done"
    assert result["metadata"]["final_answer"] == "done"
    assert result["metadata"]["last_error"] == ("search_failed" if failed else None)
    observation = seen[1][-1]
    assert observation["role"] == "tool"
    assert ("Error" if failed else "Evidence from the retriever.") in observation["content"]


async def test_env_default_offline_and_invalid_args(monkeypatch):
    env = DeepEyesV2Env(data_index="test", sandbox_executor=None, image=None)
    for args in (None, {}, {"query": []}):
        obs = await env.exec_tool("<tool_call>" + json.dumps({"name": "search", "arguments": args}) + "</tool_call>")
        assert obs.error == "search_failed" and not obs.done
    obs = await env.exec_tool('<tool_call>{"name":"search","arguments":{"query":"q"}}</tool_call>')
    assert obs.error is None and not obs.done
    assert "Offline mock" in obs.body_text
    monkeypatch.setattr(search_utils, "_IMAGE_SEARCH_CACHE", {})
    obs = await env.exec_tool('<tool_call>{"name":"image_search"}</tool_call>')
    assert obs.error == "search_failed" and not obs.done

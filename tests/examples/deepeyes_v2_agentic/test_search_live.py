# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.parse import quote

import httpx
import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR / "scripts"))
sys.path.insert(1, str(EXAMPLE_DIR))

import verify_search_live as live  # noqa: E402
from app import search_http, search_utils  # noqa: E402
from app.search_config import SEARCH_CONFIG_ENV  # noqa: E402


@pytest.fixture
def configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    values = {
        "backend": "external",
        "method": "GET",
        "endpoint": "https://private-user:private-password@service.example.test/private/search?token=private-query",
        "max_retries": 1,
        "retry_delay_s": 0,
        "retry_max_delay_s": 0,
        "trust_env": False,
        "headers": {"X-Custom": "private-header", "Accept": "application/json"},
        "auth": {"header": "X-Search-Token", "env": "SEARCH_LIVE_TEST_TOKEN"},
        "request": {"location": "query", "query_field": "q", "size_field": "count"},
        "response": {
            "items_path": ["data", "items"],
            "fields": {"title": ["heading"], "link": ["url"], "snippet": ["text"]},
        },
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    monkeypatch.setenv("SEARCH_LIVE_TEST_TOKEN", "private-auth")
    monkeypatch.setenv(SEARCH_CONFIG_ENV, "previous-config")
    return config


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            query = json.loads(request.content)["queries"][0]
            return httpx.Response(200, json={"result": [[{"document": {"contents": f"Title\n{query}"}}]]})
        assert request.headers["X-Search-Token"] == os.environ["SEARCH_LIVE_TEST_TOKEN"]
        row = {"heading": "Title", "url": "", "text": request.url.params["q"]}
        return httpx.Response(200, json={"data": {"items": [row]}})

    handler = Mock(side_effect=respond)
    clients: list[httpx.Client] = []

    def factory(config: Any) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(search_http, "_create_client", factory)
    yield handler
    assert all(client.is_closed for client in clients)
    assert search_http._create_client is factory
    assert os.environ[SEARCH_CONFIG_ENV] == "previous-config"


def arguments(config: Path, output: Path, queries: tuple[str, ...] = ("first query", "second query")) -> list[str]:
    result = ["--config", str(config), "--service-version", "fixture-version", "--output-dir", str(output)]
    for query in queries:
        result.extend(("--query", query))
    return result


def read_artifact(output: Path, name: str = "summary.json") -> dict[str, Any]:
    return json.loads((output / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend", ["retriever", "external"])
def test_live_verification_records_sources_and_version(
    tmp_path: Path, configuration: Path, service: Mock, backend: str, capsys: pytest.CaptureFixture[str]
) -> None:
    if backend == "retriever":
        configuration.write_text(
            "backend: retriever\nendpoint: https://service.example.test/search\n", encoding="utf-8"
        )
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 0
    assert json.loads(capsys.readouterr().out) == {"passed": True, "query_count": 2}
    summary = read_artifact(output)
    assert summary["passed"] is True
    assert summary["backend"] == backend
    assert summary["service_version"] == "fixture-version"
    assert summary["config_sha256"] == hashlib.sha256(configuration.read_bytes()).hexdigest()
    for name, digest in summary["implementation_sha256"].items():
        assert digest == hashlib.sha256((EXAMPLE_DIR / name).read_bytes()).hexdigest()
    for reference, query in zip(summary["queries"], ("first query", "second query"), strict=True):
        evidence = read_artifact(output, reference["evidence"])
        assert evidence["query"] == query
        assert evidence["passed"] is evidence["field_source_matches"] is True
        assert evidence["normalized"]["data"] == [{"title": "Title", "link": "", "snippet": query, "date": None}]
        assert evidence["elapsed_time"] >= evidence["normalized"]["elapsed_time"] >= 0
        assert evidence["request_count"] == 1
        assert evidence["attempts"][0]["status_code"] == 200
        assert query in json.dumps(evidence["attempts"][0]["raw_response"])
    assert service.call_count == 2
    assert os.environ["SEARCH_LIVE_TEST_TOKEN"] == "private-auth"


@pytest.mark.parametrize("failure", ["http", "invalid_json", "empty"])
def test_live_verification_reports_failed_queries(
    tmp_path: Path, configuration: Path, service: Mock, failure: str
) -> None:
    responses = {
        "http": httpx.Response(401),
        "invalid_json": httpx.Response(200, text="invalid JSON"),
        "empty": httpx.Response(200, json={"data": {"items": []}}),
    }
    service.side_effect = lambda request: responses[failure]
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 1
    assert read_artifact(output)["passed"] is False
    assert all(read_artifact(output, f"query-{index:03d}.json")["passed"] is False for index in (1, 2))


def test_live_verification_observes_retry(tmp_path: Path, configuration: Path, service: Mock) -> None:
    respond = service.side_effect
    service.side_effect = lambda request: httpx.Response(503) if service.call_count == 1 else respond(request)
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 0
    evidence = read_artifact(output, "query-001.json")
    assert evidence["request_count"] == 2
    assert [attempt["status_code"] for attempt in evidence["attempts"]] == [503, 200]


def test_live_verification_rejects_unrelated_normalization(
    tmp_path: Path, configuration: Path, service: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = search_utils.search

    def modified_search(query: str) -> dict[str, Any]:
        result = original(query)
        result["data"][0]["snippet"] = "unrelated content"
        return result

    monkeypatch.setattr(search_utils, "search", modified_search)
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 1
    assert read_artifact(output, "query-001.json")["error"] == "source_mismatch"
    assert read_artifact(output)["passed"] is False


def test_live_verification_redacts_values_and_preserves_json_structure(
    tmp_path: Path,
    configuration: Path,
    service: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SEARCH_LIVE_TEST_TOKEN", "data")
    markers = ["private-user", "private-password", "private-query", "private-header", "/private/search"]
    text = " ".join(markers + [quote(value, safe="") for value in markers] + ["data", "application/json"])
    respond = service.side_effect

    def response(request: httpx.Request) -> httpx.Response:
        payload = respond(request).json()
        payload["nested"] = [{"private-user": [text, True, 3, None]}]
        return httpx.Response(200, json=payload)

    service.side_effect = response
    output = tmp_path / "evidence"
    args = arguments(configuration, output, (f"first {text}", f"second {text}"))
    args.extend(("--sensitive-query-param", "token", "--sensitive-header", "x-custom"))
    assert live.main(args) == 0
    artifacts = "".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    captured = capsys.readouterr()
    for marker in markers:
        assert marker not in artifacts + captured.out + captured.err
        assert quote(marker, safe="") not in artifacts
    evidence = read_artifact(output, "query-001.json")
    assert set(evidence["normalized"]) == {"elapsed_time", "data"}
    assert set(evidence["normalized"]["data"][0]) == {"title", "link", "snippet", "date"}
    assert "data" not in evidence["query"]
    assert "application/json" in evidence["query"]
    assert evidence["attempts"][0]["raw_response"]["nested"][0]["[REDACTED]"][1:] == [True, 3, None]
    summary = read_artifact(output)
    assert summary["config_sha256"] == hashlib.sha256(configuration.read_bytes()).hexdigest()
    assert all((output / item["evidence"]).is_file() for item in summary["queries"])


def test_live_verification_rejects_redacted_key_collision(
    tmp_path: Path, configuration: Path, service: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    respond = service.side_effect

    def response(request: httpx.Request) -> httpx.Response:
        payload = respond(request).json()
        payload["nested"] = {"private-user": "first", "[REDACTED]": "second"}
        return httpx.Response(200, json=payload)

    service.side_effect = response
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 1
    assert "redacted_key_collision" in capsys.readouterr().err
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("filename", ["query-001.json", "summary.json"])
def test_live_verification_rejects_corrupted_evidence(
    tmp_path: Path, configuration: Path, service: Mock, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    original_write = live.write_json

    def write_corrupted(path: Path, value: Any) -> None:
        original_write(path, value)
        if path.name == filename:
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(live, "write_json", write_corrupted)
    output = tmp_path / "evidence"
    assert live.main(arguments(configuration, output)) == 1
    assert service.call_count == 2
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize(
    "failure", ["single_query", "duplicate_query", "empty_query", "mock", "missing_auth", "existing"]
)
def test_live_verification_rejects_invalid_inputs_before_requests(
    tmp_path: Path, configuration: Path, service: Mock, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output = tmp_path / "evidence"
    queries = {"single_query": ("only",), "duplicate_query": ("same", " same "), "empty_query": ("first", " ")}
    if failure == "mock":
        configuration.write_text("backend: mock\n", encoding="utf-8")
    elif failure == "missing_auth":
        monkeypatch.delenv("SEARCH_LIVE_TEST_TOKEN")
    elif failure == "existing":
        output.mkdir()
        (output / "sentinel").write_text("keep", encoding="utf-8")
    assert live.main(arguments(configuration, output, queries.get(failure, ("first", "second")))) == 1
    service.assert_not_called()
    assert not (output / "summary.json").exists()
    if failure == "existing":
        assert (output / "sentinel").read_text(encoding="utf-8") == "keep"
    if failure == "missing_auth":
        assert "SEARCH_LIVE_TEST_TOKEN" not in os.environ

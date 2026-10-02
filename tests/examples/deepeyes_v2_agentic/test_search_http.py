# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_http  # noqa: E402
from app.search_config import RetrieverSearchConfig, SearchError, SearchResult  # noqa: E402


@pytest.fixture
def config() -> RetrieverSearchConfig:
    return RetrieverSearchConfig(
        backend="retriever", endpoint="https://search.example.test/retrieve", max_retries=2, retry_delay_s=0.0
    )


@pytest.fixture
def http_handler(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    handler = Mock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(search_http, "_create_client", lambda config: client)
    yield handler
    assert client.is_closed


def parse_results(payload: object) -> list[SearchResult]:
    if payload != []:
        raise SearchError("invalid-service-response")
    return []


@pytest.mark.parametrize("trust_env", [False, True])
def test_http_client_options(monkeypatch: pytest.MonkeyPatch, config: RetrieverSearchConfig, trust_env: bool) -> None:
    config.trust_env = trust_env
    config.timeout_s = 0.25
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])))
    factory = Mock(return_value=client)
    monkeypatch.setattr(search_http.httpx, "Client", factory)
    assert search_http.request_search(config, method="GET", parse_results=parse_results)["data"] == []
    options = factory.call_args.kwargs
    assert options["timeout"].as_dict() == {name: 0.25 for name in ("connect", "read", "write", "pool")}
    assert options["trust_env"] is trust_env and options["follow_redirects"] is False
    assert client.is_closed


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_http_transient_status_recovers(config: RetrieverSearchConfig, http_handler: Mock, status: int) -> None:
    http_handler.side_effect = [httpx.Response(status), httpx.Response(200, json=[])]
    assert search_http.request_search(config, method="POST", parse_results=parse_results)["data"] == []
    assert http_handler.call_count == 2


@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ReadTimeout])
def test_http_transient_exception_recovers(
    config: RetrieverSearchConfig, http_handler: Mock, exception: type[httpx.RequestError]
) -> None:
    http_handler.side_effect = [exception("private-test-value"), httpx.Response(200, json=[])]
    assert search_http.request_search(config, method="POST", parse_results=parse_results)["data"] == []
    assert http_handler.call_count == 2


@pytest.mark.parametrize("retries", [0, 2])
def test_http_retry_exhaustion_is_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    config: RetrieverSearchConfig,
    http_handler: Mock,
    retries: int,
) -> None:
    config.max_retries = retries
    config.retry_delay_s = 0.5
    waits = Mock()
    monkeypatch.setattr(search_http.time, "sleep", waits)
    monkeypatch.setattr(search_http.logger, "handlers", [*search_http.logger.handlers, caplog.handler])
    http_handler.side_effect = httpx.ReadTimeout("private-test-value")
    with pytest.raises(SearchError, match="^ReadTimeout$"):
        search_http.request_search(config, method="GET", parse_results=parse_results)
    assert http_handler.call_count == retries + 1
    assert [call.args[0] for call in waits.call_args_list] == ([0.5, 1.0] if retries else [])
    assert f"attempt={retries + 1}/{retries + 1}" in caplog.text
    assert "private-test-value" not in caplog.text


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(303, headers={"Location": "/next"}), "http_303"),
        (httpx.Response(401, text="private-test-value"), "http_401"),
        (httpx.Response(400), "http_400"),
        (httpx.LocalProtocolError("private-test-value"), "LocalProtocolError"),
        (httpx.Response(200, text="<html>private-test-value</html>"), "invalid_json"),
        (httpx.Response(200, content=b"[NaN]"), "invalid_json"),
        (httpx.Response(200, json={}), "invalid_response"),
    ],
)
def test_http_terminal_failures_do_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    config: RetrieverSearchConfig,
    http_handler: Mock,
    response: httpx.Response | httpx.RequestError,
    reason: str,
) -> None:
    http_handler.side_effect = [response]
    monkeypatch.setattr(search_http.logger, "handlers", [*search_http.logger.handlers, caplog.handler])
    with pytest.raises(SearchError, match=f"^{reason}$"):
        search_http.request_search(config, method="GET", parse_results=parse_results)
    http_handler.assert_called_once()
    assert "private-test-value" not in caplog.text


class ResponseStream(httpx.SyncByteStream):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        if self.fail:
            raise httpx.ReadError("interrupted-response")
        yield b"[]"

    def close(self) -> None:
        self.closed = True


def test_http_closes_responses_before_backoff_and_measures_total_elapsed(
    monkeypatch: pytest.MonkeyPatch, config: RetrieverSearchConfig, http_handler: Mock
) -> None:
    streams = [ResponseStream(), ResponseStream(fail=True), ResponseStream()]
    responses = [httpx.Response(status, stream=stream) for status, stream in zip([503, 200, 200], streams)]
    clock = [0.0]
    waits = []
    config.retry_delay_s = 0.5
    config.retry_max_delay_s = 0.75

    def handle(request: httpx.Request) -> httpx.Response:
        clock[0] += 0.25
        return responses[http_handler.call_count - 1]

    def sleep(delay: float) -> None:
        assert streams[http_handler.call_count - 1].closed
        waits.append(delay)
        clock[0] += delay

    http_handler.side_effect = handle
    monkeypatch.setattr(search_http.time, "sleep", sleep)
    monkeypatch.setattr(search_http.time, "monotonic", lambda: clock[0])
    response = search_http.request_search(config, method="POST", parse_results=parse_results)
    assert response == {"elapsed_time": 2.0, "data": []}
    assert waits == [0.5, 0.75] and http_handler.call_count == 3
    assert all(stream.closed for stream in streams)
    assert all(response.is_closed for response in responses)

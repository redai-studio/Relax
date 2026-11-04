# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace

import httpx
import pytest

from relax.utils.http_utils import (
    _post,
    is_expected_sglang_499,
    is_router_no_available_workers,
    router_worker_base_url,
    router_worker_base_urls,
)


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _response(status_code: int, *, body: str | dict):
    request = httpx.Request("POST", "http://test/post")
    if isinstance(body, dict):
        return httpx.Response(status_code, request=request, json=body)
    return httpx.Response(status_code, request=request, text=body)


def _status_error(status_code: int, *, body: str | dict) -> httpx.HTTPStatusError:
    response = _response(status_code, body=body)
    return httpx.HTTPStatusError("boom", request=response.request, response=response)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (499, {"error": {"message": "Request abc123 was aborted"}}, True),
        (499, {"error": {"message": "upstream connection closed"}}, False),
        (499, "<html>499 client closed request</html>", False),
        (499, {"error": {"message": "upstream request was aborted while proxying"}}, False),
        (499, {"error": {"message": "the request was aborted"}}, False),
        (499, {"error": {"message": "Request abc was aborted by the user"}}, False),
        (499, {"error": {"message": "Request was aborted"}}, False),
        (499, {"error": "client closed request"}, False),
        (499, {"error": ["x"]}, False),
        (499, {"error": 123}, False),
        (500, {"error": {"message": "Request abc123 was aborted"}}, False),
    ],
)
def test_is_expected_sglang_499(status, body, expected):
    assert is_expected_sglang_499(_status_error(status, body=body)) is expected


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (503, {"error": {"code": "no_available_workers"}}, True),
        (503, {"error": {"code": "other"}}, False),
        (503, {"error": "no_available_workers"}, False),
        (500, {"error": {"code": "no_available_workers"}}, False),
    ],
)
def test_is_router_no_available_workers(status, body, expected):
    assert is_router_no_available_workers(_status_error(status, body=body)) is expected


def test_post_does_not_retry_non_retryable_400():
    client = _StubClient(
        [
            _response(
                400,
                body={"error": {"message": "Requested token count exceeds the model's maximum context length."}},
            )
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert client.calls == 1


def test_post_retries_retryable_503_then_succeeds():
    client = _StubClient(
        [
            _response(503, body={"error": {"message": "No available workers"}}),
            _response(200, body={"ok": True}),
        ]
    )

    result = asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert result == {"ok": True}
    assert client.calls == 2


def test_post_can_fail_fast_on_router_no_workers():
    client = _StubClient([_response(503, body={"error": {"code": "no_available_workers"}})])

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://test/post", {}, max_retries=6, fail_fast_no_workers=True))

    assert client.calls == 1


@pytest.mark.parametrize(
    ("worker_url", "expected"),
    [
        ("http://worker:8000", "http://worker:8000"),
        ("http://worker:8000@0", "http://worker:8000"),
        ("http://[2001:db8::1]:8000@12", "http://[2001:db8::1]:8000"),
        ("http://user:password@worker:8000", "http://user:password@worker:8000"),
        ("http://user:p%40ss@worker:8000@2", "http://user:p%40ss@worker:8000"),
        ("http://user@123", "http://user@123"),
        ("http://worker:8000@rank0", "http://worker:8000@rank0"),
        ("http://worker:8000@-1", "http://worker:8000@-1"),
        ("http://worker:8000@²", "http://worker:8000@²"),
        ("http://worker:8000/path@0", "http://worker:8000/path@0"),
        ("http://worker:8000?rank=@0", "http://worker:8000?rank=@0"),
        ("", ""),
    ],
)
def test_router_worker_base_url(worker_url, expected):
    assert router_worker_base_url(worker_url) == expected


def test_router_worker_base_urls_stably_deduplicates_dp_ranks():
    urls = [
        "http://worker-a:8000@0",
        "http://worker-b:8000@0",
        "http://worker-a:8000@1",
        "http://worker-b:8000@1",
    ]

    assert router_worker_base_urls(urls) == [
        "http://worker-a:8000",
        "http://worker-b:8000",
    ]


# --- distributed POST: a remote HTTP response must not trigger a local re-send --


class _RaisingRef:
    """A Ray-ObjectRef stand-in whose ``await`` re-raises a stored error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __await__(self):
        raise self._error
        yield  # pragma: no cover  # makes this a generator function


class _RemoteActorThatRaises:
    """A Ray-actor stand-in whose ``do_post.remote`` returns a raising ref."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def do_post(self):
        error = self._error
        return SimpleNamespace(remote=lambda *args, **kwargs: _RaisingRef(error))


class _ReturningRef:
    def __init__(self, value) -> None:
        self._value = value

    def __await__(self):
        async def result():
            return pickle.loads(pickle.dumps(self._value))

        return result().__await__()


@pytest.mark.parametrize(
    ("status_code", "body", "expected_abort"),
    [
        (499, {"error": {"message": "Request abc was aborted"}}, True),
        (400, {"error": {"message": "Bad request"}}, False),
    ],
)
def test_distributed_post_http_error_survives_ray_serialization_without_resend(
    monkeypatch, status_code, body, expected_abort
):
    from relax.utils import http_utils

    client = _StubClient([_response(status_code, body=body)])
    envelope = asyncio.run(http_utils._post_with_error_envelope(client, "http://test/generate", {}))
    actor = SimpleNamespace(do_post=SimpleNamespace(remote=lambda *args, **kwargs: _ReturningRef(envelope)))
    local = _StubClient([_response(200, body={"ok": True})])
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [actor])
    monkeypatch.setattr(http_utils, "_http_client", local)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(http_utils.post("http://test/generate", {}))

    assert exc_info.value.response.status_code == status_code
    assert is_expected_sglang_499(exc_info.value) is expected_abort
    assert local.calls == 0


def test_distributed_post_success_envelope_preserves_payload(monkeypatch):
    from relax.utils import http_utils

    client = _StubClient([_response(200, body={"ok": True})])
    envelope = asyncio.run(http_utils._post_with_error_envelope(client, "http://test/generate", {}))
    actor = SimpleNamespace(do_post=SimpleNamespace(remote=lambda *args, **kwargs: _ReturningRef(envelope)))
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [actor])

    assert asyncio.run(http_utils.post("http://test/generate", {})) == {"ok": True}


def test_post_does_fall_back_to_local_on_remote_transport_error(monkeypatch):
    # A non-HTTP transport/actor fault (e.g. Ray actor died) MUST still fall back
    # to the local client — only a real HTTP response skips the fallback.
    from relax.utils import http_utils

    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [_RemoteActorThatRaises(ConnectionError("actor died"))])
    local = _StubClient([_response(200, body={"ok": True})])
    monkeypatch.setattr(http_utils, "_http_client", local)

    result = asyncio.run(http_utils.post("http://test/generate", {}))

    assert result == {"ok": True}
    assert local.calls == 1  # transport fault -> local fallback

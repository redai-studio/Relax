# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from relax.utils import http_utils
from relax.utils.http_utils import _post, router_worker_base_url, router_worker_base_urls


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


@pytest.mark.parametrize("fallback_to_local", [False, True])
def test_distributed_post_lost_reply_respects_fallback_policy(monkeypatch, fallback_to_local):
    remote_client = _StubClient([_response(200, body={"accepted": True})])
    local_client = _StubClient([_response(200, body={"local": True})])
    lost_reply = RuntimeError("Ray reply lost after HTTP request completed")

    async def remote(url, payload, max_retries, headers=None):
        await _post(remote_client, url, payload, max_retries, headers=headers)
        raise lost_reply

    actor = SimpleNamespace(do_post=SimpleNamespace(remote=remote))
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [actor])
    monkeypatch.setattr(http_utils, "_next_actor", lambda: actor)
    monkeypatch.setattr(http_utils, "_http_client", local_client)
    options = {} if fallback_to_local else {"fallback_to_local": False}
    if fallback_to_local:
        assert asyncio.run(http_utils.post("http://test/post", {}, max_retries=1, **options)) == {"local": True}
    else:
        with pytest.raises(RuntimeError) as error:
            asyncio.run(http_utils.post("http://test/post", {}, max_retries=1, **options))
        assert error.value is lost_reply
    assert remote_client.calls == 1
    assert local_client.calls == int(fallback_to_local)


def test_post_without_distributed_actor_still_sends_once_when_fallback_disabled(monkeypatch):
    client = _StubClient([_response(200, body={"ok": True})])
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_http_client", client)
    assert asyncio.run(http_utils.post("http://test/post", {}, max_retries=1, fallback_to_local=False)) == {"ok": True}
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

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import gzip
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["gzip", "server_error", "read_timeout", "actor_error"])
async def test_admitted_distributed_post_never_replays_unknown_outcome(monkeypatch, outcome):
    from relax.utils import http_utils

    calls = []

    def upstream(request):
        calls.append(request)
        if outcome == "read_timeout":
            raise httpx.ReadTimeout("reply lost", request=request)
        if outcome == "gzip":
            return httpx.Response(
                200,
                headers={"content-encoding": "gzip", "content-type": "application/json"},
                content=gzip.compress(b'{"text":"generated"}'),
            )
        return httpx.Response(500, text="unknown outcome")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:

        async def remote(url, body, headers, request_id):
            if outcome == "actor_error":
                raise RuntimeError("poster actor died")
            return await http_utils._request_once(client, url, body, headers)

        monkeypatch.setattr(http_utils, "_post_actors", [SimpleNamespace(do_request=SimpleNamespace(remote=remote))])
        monkeypatch.setattr(http_utils, "_http_client", _StubClient([]))
        request = client.build_request("POST", "http://engine/generate", json={"rid": "one-execution"})
        if outcome == "gzip":
            response = await http_utils.send_distributed_request(request)
            assert response.json() == {"text": "generated"}
            assert response.headers["content-type"] == "application/json"
        elif outcome == "server_error":
            response = await http_utils.send_distributed_request(request)
            assert response.status_code == 500
        else:
            with pytest.raises(httpx.ReadError if outcome == "read_timeout" else RuntimeError):
                await http_utils.send_distributed_request(request)
        assert len(calls) == (0 if outcome == "actor_error" else 1)


@pytest.mark.asyncio
async def test_poster_failure_before_dispatch_is_reported_as_connection_error(monkeypatch):
    import ray

    from relax.utils import http_utils

    async def process_info():
        raise ray.exceptions.ActorDiedError()

    post = Mock()
    stop = Mock()
    poster = SimpleNamespace(do_request=SimpleNamespace(remote=post), stop_requests=SimpleNamespace(remote=stop))
    healthy_send = AsyncMock(return_value=(200, [], b"ok", None))
    healthy_poster = SimpleNamespace(do_request=SimpleNamespace(remote=healthy_send))
    monkeypatch.setattr(http_utils, "_post_actors", [poster, healthy_poster])
    monkeypatch.setattr(http_utils, "_poster_processes", {id(poster): process_info()})
    request = httpx.Request("POST", "http://engine/generate", json={})

    with pytest.raises(httpx.ConnectError, match="before dispatch"):
        await http_utils.send_distributed_request(request)

    post.assert_not_called()
    stop.assert_not_called()
    for _ in range(2):
        response = await http_utils.send_distributed_request(request)
        assert response.content == b"ok"
    assert healthy_send.await_count == 2


@pytest.mark.asyncio
async def test_distributed_post_cancellation_waits_for_poster_completion(monkeypatch):
    from relax.utils import http_utils

    submitted, cancellation_requested, acknowledge = (asyncio.Event() for _ in range(3))
    ids = []
    active = None

    async def remote(url, body, headers, request_id):
        nonlocal active
        ids.append(request_id)
        if len(ids) == 1:
            active = asyncio.current_task()
            submitted.set()
            await asyncio.Future()
        return 200, [], b"ok", None

    async def cancel(request_id):
        assert ids == [request_id]
        cancellation_requested.set()
        await acknowledge.wait()
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)
        return True

    actor = SimpleNamespace(do_request=SimpleNamespace(remote=remote), cancel_request=SimpleNamespace(remote=cancel))
    monkeypatch.setattr(http_utils, "_post_actors", [actor])
    task = asyncio.create_task(http_utils.send_distributed_request(httpx.Request("POST", "http://engine/generate")))
    try:
        await asyncio.wait_for(submitted.wait(), 1)
        task.cancel()
        await asyncio.wait_for(cancellation_requested.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        acknowledge.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    response = await http_utils.send_distributed_request(httpx.Request("POST", "http://engine/generate"))
    assert response.content == b"ok" and len(ids) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("join_before_ack", [True, False])
async def test_distributed_post_shares_poster_stop_receipt(monkeypatch, join_before_ack):
    import ray

    from relax.utils import http_utils

    both_sent, lose_second, stopping, confirmed = (asyncio.Event() for _ in range(4))
    sent = 0
    stop_calls = 0

    async def remote(*args):
        nonlocal sent
        sent += 1
        if sent == 1:
            await both_sent.wait()
            raise ray.exceptions.ActorUnavailableError("connection lost", None)
        both_sent.set()
        await lose_second.wait()
        raise ray.exceptions.TaskCancelledError()

    async def stop():
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls > 1:
            raise ray.exceptions.ActorDiedError()
        stopping.set()
        await confirmed.wait()

    actor = SimpleNamespace(do_request=SimpleNamespace(remote=remote), stop_requests=SimpleNamespace(remote=stop))
    monkeypatch.setattr(http_utils, "_post_actors", [actor])
    request = lambda: httpx.Request("POST", "http://engine/generate")  # noqa: E731
    first, second = [asyncio.create_task(http_utils.send_distributed_request(request())) for _ in range(2)]
    try:
        await asyncio.wait_for(stopping.wait(), 1)
        if join_before_ack:
            lose_second.set()
            # If the second request starts a second stop, it can fail even
            # while the original stop call is waiting for acknowledgement.
            done, _ = await asyncio.wait({second}, timeout=0.05)
            assert not done
        confirmed.set()
        with pytest.raises(ray.exceptions.ActorUnavailableError):
            await asyncio.wait_for(first, 1)
        lose_second.set()
        with pytest.raises(ray.exceptions.TaskCancelledError):
            await asyncio.wait_for(second, 1)
        assert stop_calls == 1
    finally:
        confirmed.set()
        lose_second.set()
        await asyncio.gather(first, second, return_exceptions=True)


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


def test_post_retry_marks_generation_token_count_incomplete(monkeypatch):
    from unittest.mock import AsyncMock

    from relax.utils.http_utils import PostObservation, post_observation

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _StubClient([_response(503, body={}), _response(200, body={"ok": True})])

    async def scenario():
        observation = PostObservation()
        token = post_observation.set(observation)
        try:
            assert await _post(client, "http://test/post", {}) == {"ok": True}
        finally:
            post_observation.reset(token)
        assert not observation.token_count_complete
        assert post_observation.get() is None

    asyncio.run(scenario())


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


@pytest.mark.asyncio
async def test_shutdown_requires_process_exit_after_poster_loses_acknowledgement(monkeypatch):
    import ray

    from relax.utils import http_utils

    async def lost_ack():
        raise ray.exceptions.ActorDiedError()

    actor = SimpleNamespace(stop_requests=SimpleNamespace(remote=lost_ack))
    receipt = http_utils._PosterProcess("poster-node", 123, 456, 789)
    target = http_utils._PosterShutdownTarget(actor, receipt)
    exited = False

    async def check_process(process):
        assert process == receipt
        return exited

    monkeypatch.setattr(http_utils, "_confirm_poster_process_exit", check_process)
    with pytest.raises(http_utils.DistributedPostUnconfirmed) as error:
        await http_utils.stop_distributed_poster(target)
    assert error.value.poster == target
    exited = True
    await http_utils.stop_distributed_poster(target)


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

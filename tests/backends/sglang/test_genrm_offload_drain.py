# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression tests for the GenRM colocate offload drain state machine.

These cover the two failure modes that motivated the drain rewrite:

* Admission race — SGLang's ``release_memory_occupation`` asserts the scheduler
  is idle, so a straggler ``/generate`` admitted between flush and release
  crashes the engine. The drain closes admission via ``/pause_generation``
  (abort mode) before touching the engine, and re-opens it on resume.
* Unbounded hang — every HTTP round-trip and the whole drain must be bounded by
  a wall-clock deadline; a wedged scheduler must surface as ``TimeoutError``
  rather than blocking rank-0 (and every rank at the downstream barrier).

The engine is exercised without a live SGLang server by constructing a bare
instance and faking the module-level ``requests`` / ``time`` so the pure
control flow is observable.
"""

from __future__ import annotations

import pytest


pytest.importorskip("sglang.srt.server_args")

try:
    import relax.backends.sglang.sglang_engine as m
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.sglang.sglang_engine unavailable: {_exc}", allow_module_level=True)


class _FakeResp:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise m.requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return {}


class _FakeRequests:
    """Records calls and returns programmable responses per endpoint.

    ``exceptions`` is aliased to the real ``requests.exceptions`` so
    ``raise_for_status`` / ``_make_request`` error handling stays realistic.
    """

    def __init__(self, flush_statuses=None, post_raises=None):
        import requests as _real_requests

        self.exceptions = _real_requests.exceptions
        self.calls: list[str] = []
        # Queue of status codes to return from successive /flush_cache GETs;
        # the last value is repeated once exhausted.
        self._flush_statuses = list(flush_statuses or [200])
        # Mapping endpoint -> Exception to raise on POST (e.g. flaky abort).
        self._post_raises = dict(post_raises or {})

    @staticmethod
    def _endpoint(url: str) -> str:
        return url.rsplit("/", 1)[-1]

    def get(self, url, timeout=None):
        ep = self._endpoint(url)
        self.calls.append(ep)
        assert timeout is not None, "flush_cache GET must be bounded by a timeout"
        if ep == "flush_cache":
            status = self._flush_statuses[0] if len(self._flush_statuses) == 1 else self._flush_statuses.pop(0)
            return _FakeResp(status)
        return _FakeResp(200)

    def post(self, url, json=None, timeout=None):  # noqa: A002 — mirror requests signature
        ep = self._endpoint(url)
        self.calls.append(ep)
        if ep in self._post_raises:
            raise self._post_raises[ep]
        return _FakeResp(200)


class _FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it, no real
    waiting."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _make_engine():
    engine = m.SGLangEngine.__new__(m.SGLangEngine)
    engine.role = m.Role.GENRM
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.unregister_from_router = lambda **kwargs: True
    engine.register_to_router = lambda: True
    engine._released_memory = set()
    return engine


def _released_engine():
    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES

    engine = _make_engine()
    engine._released_memory = set(GPU_MEMORY_ALL_TYPES)
    engine._generation_paused = True
    return engine


@pytest.fixture
def patched(monkeypatch):
    """Install fakes; return a helper that wires a scenario and returns
    state."""

    def _install(flush_statuses=None, post_raises=None, drain_timeout=5.0):
        fake_requests = _FakeRequests(flush_statuses=flush_statuses, post_raises=post_raises)
        fake_clock = _FakeClock()
        monkeypatch.setattr(m, "requests", fake_requests)
        monkeypatch.setattr(m, "time", fake_clock)
        monkeypatch.setattr(m, "_OFFLOAD_DRAIN_TIMEOUT_S", drain_timeout)
        return fake_requests, fake_clock

    return _install


def test_release_retries_flush_until_ready_then_releases(patched):
    # 400 (pending) twice, then 200 -> drain converges and release runs once.
    fake_requests, _ = patched(flush_statuses=[400, 400, 200])
    _make_engine().release_memory_occupation()

    assert fake_requests.calls.count("flush_cache") == 3
    assert fake_requests.calls.count("release_memory_occupation") == 1


def test_release_closes_admission_before_touching_engine(patched):
    # pause_generation (admission gate) must precede any flush and the release.
    fake_requests, _ = patched(flush_statuses=[200])
    _make_engine().release_memory_occupation()

    assert "pause_generation" in fake_requests.calls
    assert fake_requests.calls.index("pause_generation") < fake_requests.calls.index("flush_cache")
    assert fake_requests.calls.index("pause_generation") < fake_requests.calls.index("release_memory_occupation")


def test_teacher_release_closes_raw_generation_and_full_resume_reopens_it(patched):
    fake_requests, _ = patched(flush_statuses=[200])
    engine = _make_engine()
    engine.role = m.Role.TEACHER
    events = []
    engine.pause_generation = lambda **kwargs: events.append(("pause", kwargs))
    engine.unregister_from_router = lambda **kwargs: events.append(("unregister", kwargs)) or True
    engine.register_to_router = lambda: events.append(("register", {})) or True

    engine.release_memory_occupation()
    assert [name for name, _ in events] == ["pause", "unregister"]
    assert events[1] == ("unregister", {"wait_for_removal": True})
    assert fake_requests.calls.index("flush_cache") < fake_requests.calls.index("release_memory_occupation")

    engine.resume_memory_occupation(tags=["weights"])
    assert "continue_generation" not in fake_requests.calls
    assert all(name != "register" for name, _ in events)
    engine.resume_memory_occupation(tags=None)
    assert fake_requests.calls.index("resume_memory_occupation") < fake_requests.calls.index("continue_generation")
    assert events[-1] == ("register", {})


@pytest.mark.parametrize("role", [m.Role.TEACHER, m.Role.GENRM])
def test_release_refuses_to_offload_while_router_worker_is_registered(patched, role):
    fake_requests, _ = patched()
    engine = _make_engine()
    engine.role = role
    engine.unregister_from_router = lambda **kwargs: False
    engine._router_lists_worker = lambda url: True

    with pytest.raises(RuntimeError, match="still lists it"):
        engine.release_memory_occupation()

    assert "release_memory_occupation" not in fake_requests.calls


@pytest.mark.parametrize("role", [m.Role.ROLLOUT, m.Role.GENRM])
def test_release_proceeds_when_the_router_never_listed_the_engine(patched, role):
    # A policy engine before its first weight sync was never registered.
    fake_requests, _ = patched()
    engine = _make_engine()
    engine.role = role
    engine.unregister_from_router = lambda **kwargs: False
    engine._router_lists_worker = lambda url: False

    engine.release_memory_occupation()
    assert fake_requests.calls[-1] == "release_memory_occupation"


def test_release_refuses_when_router_membership_is_unknown(patched):
    fake_requests, _ = patched()
    engine = _make_engine()
    engine.role = m.Role.ROLLOUT
    engine.unregister_from_router = lambda **kwargs: False

    def unreachable(url):
        raise m.requests.exceptions.ConnectionError("router down")

    engine._router_lists_worker = unreachable
    with pytest.raises(RuntimeError, match=r"Router .* is unreachable.*keeps its GPU memory"):
        engine.release_memory_occupation()
    assert "release_memory_occupation" not in fake_requests.calls


def test_full_resume_does_not_register_if_continue_generation_fails(patched):
    import requests as _real_requests

    fake_requests, _ = patched(post_raises={"continue_generation": _real_requests.exceptions.ConnectionError("down")})
    engine = _make_engine()
    calls = []
    engine.register_to_router = lambda: calls.append("registered") or True

    with pytest.raises(_real_requests.exceptions.ConnectionError, match="down"):
        engine.resume_memory_occupation(tags=None)

    assert calls == []
    assert "continue_generation" in fake_requests.calls


def test_full_resume_fails_if_router_registration_fails(patched):
    fake_requests, _ = patched()
    engine = _make_engine()
    engine.register_to_router = lambda: False

    with pytest.raises(RuntimeError, match="Cannot register resumed genrm"):
        engine.resume_memory_occupation(tags=None)

    assert "continue_generation" in fake_requests.calls
    assert fake_requests.calls.index("continue_generation") < fake_requests.calls.index("pause_generation")


def test_release_raises_timeout_and_skips_release_when_never_idle(patched):
    # flush never returns 200 -> bounded deadline surfaces as TimeoutError and
    # release is NOT attempted (would crash the scheduler assert).
    fake_requests, _ = patched(flush_statuses=[400], drain_timeout=5.0)
    with pytest.raises(TimeoutError):
        _make_engine().release_memory_occupation()

    assert "release_memory_occupation" not in fake_requests.calls


def test_release_tolerates_abort_failure(patched):
    # A flaky /abort_request must not abort the offload; drain still completes.
    import requests as _real_requests

    fake_requests, _ = patched(
        flush_statuses=[200],
        post_raises={"abort_request": _real_requests.exceptions.ConnectionError("boom")},
    )
    _make_engine().release_memory_occupation()

    assert fake_requests.calls.count("release_memory_occupation") == 1


@pytest.mark.parametrize("role", [m.Role.ROLLOUT, m.Role.GENRM, m.Role.TEACHER])
def test_release_fails_before_releasing_when_the_pause_fails(patched, role):
    # Without the pause a request can reach the scheduler between the last
    # flush and the release, so the release does not go ahead.
    import requests as _real_requests

    fake_requests, _ = patched(
        flush_statuses=[200],
        post_raises={"pause_generation": _real_requests.exceptions.ConnectionError("boom")},
    )
    engine = _make_engine()
    engine.role = role
    with pytest.raises(_real_requests.exceptions.ConnectionError, match="boom"):
        engine.release_memory_occupation()

    assert "flush_cache" not in fake_requests.calls
    assert "release_memory_occupation" not in fake_requests.calls
    assert not engine._released_memory


def test_release_is_skipped_once_released_and_resume_once_resumed(patched):
    fake_requests, _ = patched()
    engine = _make_engine()
    engine.role = m.Role.ROLLOUT
    engine.release_memory_occupation()
    engine.release_memory_occupation()
    assert fake_requests.calls.count("release_memory_occupation") == 1
    assert fake_requests.calls.count("pause_generation") == 1

    engine.resume_memory_occupation(tags=["weights"])
    engine.resume_memory_occupation(tags=["weights"])
    engine.resume_memory_occupation(tags=None)
    engine.resume_memory_occupation(tags=None)
    assert fake_requests.calls.count("resume_memory_occupation") == 2
    # Generation reopens once, after the KV cache is back.
    assert fake_requests.calls.count("continue_generation") == 1
    assert fake_requests.calls[-1] == "continue_generation"


def test_resume_of_an_engine_whose_release_failed_only_reopens_it(patched):
    fake_requests, _ = patched(flush_statuses=[400], drain_timeout=5.0)
    engine = _make_engine()
    engine.role = m.Role.ROLLOUT
    with pytest.raises(TimeoutError):
        engine.release_memory_occupation()

    engine.resume_memory_occupation(tags=None)
    assert "resume_memory_occupation" not in fake_requests.calls
    assert fake_requests.calls[-1] == "continue_generation"


@pytest.mark.parametrize("endpoint", ["release_memory_occupation", "resume_memory_occupation"])
def test_unconfirmed_memory_request_refuses_every_later_release_and_resume(patched, endpoint):
    import requests as _real_requests

    fake_requests, _ = patched(post_raises={endpoint: _real_requests.exceptions.ReadTimeout("lost")})
    engine = _make_engine() if endpoint == "release_memory_occupation" else _released_engine()
    engine.role = m.Role.ROLLOUT
    call = getattr(engine, endpoint)
    with pytest.raises(m.EngineMemoryStateUnknown, match="not confirmed"):
        call()

    fake_requests._post_raises.clear()
    for retry in (engine.release_memory_occupation, engine.resume_memory_occupation):
        with pytest.raises(m.EngineMemoryStateUnknown, match="must be restarted"):
            retry()
    assert fake_requests.calls.count(endpoint) == 1


def test_full_resume_reopens_admission(patched):
    fake_requests, _ = patched()
    _released_engine().resume_memory_occupation(tags=None)

    assert "continue_generation" in fake_requests.calls
    assert fake_requests.calls.index("resume_memory_occupation") < fake_requests.calls.index("continue_generation")


def test_partial_resume_does_not_reopen_admission(patched):
    # A weights-only resume must not re-enable generation before KV cache is back.
    fake_requests, _ = patched()
    _released_engine().resume_memory_occupation(tags=["weights"])

    assert "continue_generation" not in fake_requests.calls


def test_rollout_release_fences_aborts_and_waits_until_the_engine_is_idle(patched):
    # A request whose caller is gone keeps the scheduler busy; the release
    # closes the engine, aborts it and waits for /flush_cache to report idle.
    fake_requests, _ = patched(flush_statuses=[400, 400, 200])
    engine = _make_engine()
    engine.role = m.Role.ROLLOUT
    left = []
    engine.unregister_from_router = lambda **kwargs: left.append(kwargs) or True
    engine.release_memory_occupation()

    assert left == [{"wait_for_removal": True}]
    calls = fake_requests.calls
    assert calls[0] == "pause_generation" and calls[-1] == "release_memory_occupation"
    assert calls.count("flush_cache") == 3 and calls.count("abort_request") == 3


def test_rollout_release_raises_and_skips_release_when_never_idle(patched):
    fake_requests, _ = patched(flush_statuses=[400])
    engine = _make_engine()
    engine.role = m.Role.ROLLOUT
    with pytest.raises(TimeoutError, match="draining rollout"):
        engine.release_memory_occupation()

    assert "release_memory_occupation" not in fake_requests.calls

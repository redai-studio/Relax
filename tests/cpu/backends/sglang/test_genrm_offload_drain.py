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
    engine = m.GenRMEngine.__new__(m.GenRMEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
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
        monkeypatch.setattr(m, "_GENRM_OFFLOAD_DRAIN_TIMEOUT_S", drain_timeout)
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


def test_release_tolerates_pause_failure_and_still_drains(patched):
    # If pause does not take, per-retry abort still drains and release proceeds.
    import requests as _real_requests

    fake_requests, _ = patched(
        flush_statuses=[200],
        post_raises={"pause_generation": _real_requests.exceptions.ConnectionError("boom")},
    )
    _make_engine().release_memory_occupation()

    assert "abort_request" in fake_requests.calls
    assert fake_requests.calls.count("release_memory_occupation") == 1


def test_full_resume_reopens_admission(patched):
    fake_requests, _ = patched()
    _make_engine().resume_memory_occupation(tags=None)

    assert "continue_generation" in fake_requests.calls
    assert fake_requests.calls.index("resume_memory_occupation") < fake_requests.calls.index("continue_generation")


def test_partial_resume_does_not_reopen_admission(patched):
    # A weights-only resume must not re-enable generation before KV cache is back.
    fake_requests, _ = patched()
    _make_engine().resume_memory_occupation(tags=["weights"])

    assert "continue_generation" not in fake_requests.calls

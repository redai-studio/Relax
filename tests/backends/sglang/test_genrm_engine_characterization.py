# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Characterization of the GenRM engine: how it initializes, and the edges of
its pre-offload drain that ``test_genrm_offload_drain.py`` does not cover.

These pin current behavior ahead of folding the GenRM subclass into the shared
engine. ``_make_genrm_engine`` is the only place that knows how a GenRM engine
is constructed; everything else must keep passing unmodified.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    import relax.backends.sglang.sglang_engine as m

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing sglang dependencies")


def _make_genrm_engine(args=None, *, rank: int = 0, base_gpu_id: int = 0):
    args = args or SimpleNamespace(rollout_external=False, fully_async=True)
    return m.GenRMEngine(args, rank=rank, worker_type="regular", base_gpu_id=base_gpu_id)


def _record_init(monkeypatch, engine, *, node_rank: int = 0) -> dict:
    """Replace everything ``init`` would launch with recorders."""
    calls: dict = {"genrm_args": None, "init_normal": None, "init_external": None, "dcs": 0, "router": 0}

    def fake_genrm_server_args(*args, **kwargs):
        calls["genrm_args"] = (args, kwargs)
        return {"node_rank": node_rank, "host": "10.0.0.1", "port": 16000}, ["port"]

    def fail_policy_server_args(*args, **kwargs):
        raise AssertionError("GenRM must not use the policy server-args builder")

    engine_cls = type(engine)
    monkeypatch.setattr(m, "_compute_genrm_server_args", fake_genrm_server_args)
    monkeypatch.setattr(m, "_compute_server_args", fail_policy_server_args)
    monkeypatch.setattr(m, "get_host_info", lambda: ("node", "10.0.0.1"))
    monkeypatch.setattr(
        engine_cls, "_init_normal", lambda self, server_args, **kw: calls.__setitem__("init_normal", (server_args, kw))
    )
    monkeypatch.setattr(
        engine_cls,
        "_init_external",
        lambda self, server_args, **kw: calls.__setitem__("init_external", (server_args, kw)),
    )
    monkeypatch.setattr(engine_cls, "register_dcs", lambda self: calls.__setitem__("dcs", calls["dcs"] + 1))
    monkeypatch.setattr(
        engine_cls, "register_to_router", lambda self, **kw: calls.__setitem__("router", calls["router"] + 1)
    )
    return calls


def test_genrm_engine_init_uses_genrm_server_args_and_skips_policy_wiring(monkeypatch):
    engine = _make_genrm_engine(rank=2, base_gpu_id=4)
    calls = _record_init(monkeypatch, engine)

    engine.init(dist_init_addr="10.0.0.1:17000", port=16000, nccl_port=16001)

    positional, keyword = calls["genrm_args"]
    assert positional == (engine.args, 2, "10.0.0.1:17000", 16001, "10.0.0.1", 16000, "regular", None)
    assert keyword == {"base_gpu_id": 4}
    # Launched without the policy load plan; never joins the router or weight sync.
    assert calls["init_normal"] == (
        {"node_rank": 0, "host": "10.0.0.1", "port": 16000},
        {"apply_policy_load_plan": False},
    )
    assert calls["init_external"] is None
    assert (calls["dcs"], calls["router"]) == (0, 0)
    assert engine._skip_router_registration is True
    assert (engine.node_rank, engine.server_host, engine.server_port) == (0, "10.0.0.1", 16000)


def test_genrm_engine_init_brackets_ipv6_addresses(monkeypatch):
    engine = _make_genrm_engine()
    calls = _record_init(monkeypatch, engine)

    engine.init(dist_init_addr="fe80::1:17000", port=16000, nccl_port=16001, host="fe80::2")

    positional, _keyword = calls["genrm_args"]
    assert positional[2] == "[fe80::1]:17000"
    assert positional[4] == "[fe80::2]"


def test_genrm_engine_init_external_checks_fields_instead_of_launching(monkeypatch):
    engine = _make_genrm_engine(SimpleNamespace(rollout_external=True, fully_async=True))
    calls = _record_init(monkeypatch, engine)

    engine.init(dist_init_addr="10.0.0.1:17000", port=16000, nccl_port=16001)

    assert calls["init_normal"] is None
    assert calls["init_external"] == (
        {"node_rank": 0, "host": "10.0.0.1", "port": 16000},
        {"external_engine_need_check_fields": ["port"]},
    )


class _Resp:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _Requests:
    """Stands in for the ``requests`` module inside the engine module."""

    def __init__(self, on_get):
        self.exceptions = m.requests.exceptions
        self._on_get = on_get
        self.gets: list[str] = []
        self.posts: list[str] = []

    def get(self, url, timeout=None):
        self.gets.append(url.rsplit("/", 1)[-1])
        return self._on_get()

    def post(self, url, json=None, timeout=None):
        self.posts.append(url.rsplit("/", 1)[-1])
        return _Resp()


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _running_engine(monkeypatch, on_get, *, node_rank: int = 0):
    engine = _make_genrm_engine()
    engine.node_rank = node_rank
    engine.server_host = "10.0.0.1"
    engine.server_port = 16000
    fake_requests = _Requests(on_get)
    monkeypatch.setattr(m, "requests", fake_requests)
    monkeypatch.setattr(m, "time", _Clock())
    return engine, fake_requests


def test_genrm_engine_release_fails_fast_when_engine_is_unreachable(monkeypatch):
    def refuse():
        raise m.requests.exceptions.ConnectionError("connection refused")

    engine, fake_requests = _running_engine(monkeypatch, refuse)

    with pytest.raises(ConnectionError, match="unreachable while draining"):
        engine.release_memory_occupation()

    assert len(fake_requests.gets) == m._MAX_CONSECUTIVE_CONNECT_ERRORS
    assert "release_memory_occupation" not in fake_requests.posts


def test_genrm_engine_follower_node_neither_drains_nor_reopens(monkeypatch):
    engine, fake_requests = _running_engine(monkeypatch, _Resp, node_rank=1)

    assert engine.release_memory_occupation() is None
    assert engine.resume_memory_occupation() is None

    assert (fake_requests.gets, fake_requests.posts) == ([], [])

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for HTTP timeouts on the actor's rollout/actor_fwd service probes.

Regression target (deadlock): the actor's synchronous ``requests.get`` probes to
the rollout service had no ``timeout``. If the rollout service stalled (e.g. an
engine died mid weight-update handshake), the actor blocked forever on the
socket read. Fix#2 bounds these handshake/probe requests with the configurable
``rollout_http_timeout`` argument.

Guard rail: ``/recover_rollout_engines`` is a long-running engine rebuild (large
MoE cold start can take minutes) and MUST NOT carry the configured probe
timeout, otherwise a legitimate recovery would be interrupted.
"""

import pytest
import requests


pytest.importorskip("megatron.core")

try:
    from relax.backends.megatron import actor as actor_mod
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {_exc}", allow_module_level=True)


MegatronTrainRayActor = actor_mod.MegatronTrainRayActor


class _Resp:
    def __init__(self, payload=False, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status={self.status_code}")
        return None

    def json(self):
        return self._payload


def _record_get(calls, *, payload=False):
    def _get(url, *args, **kwargs):
        calls.append((url, kwargs))
        return _Resp(payload)

    return _get


def _shell():
    from argparse import Namespace

    shell = object.__new__(MegatronTrainRayActor)
    shell.args = Namespace(rollout_http_timeout=120.0)
    return shell


# ---------------------------------------------------------------------------
# _run_step_evaluation: /evaluate + /end_update_weight
# ---------------------------------------------------------------------------
def test_run_step_evaluation_probes_carry_timeout(monkeypatch):
    from argparse import Namespace

    calls = []
    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _record_get(calls))
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_offline_mode", lambda _args: False)

    shell = _shell()
    shell.args = Namespace(rollout_http_timeout=17.0)
    shell.rollout_manager = object()  # non-None -> has_rollout

    shell._run_step_evaluation(3, end_update_weight=True)

    by_url = dict(calls)
    assert any(u.endswith("/evaluate") for u in by_url)
    assert any(u.endswith("/end_update_weight") for u in by_url)
    for url, kwargs in calls:
        if url.endswith("/evaluate"):
            assert "timeout" not in kwargs, url
        else:
            assert kwargs.get("timeout") == shell.args.rollout_http_timeout, url


@pytest.mark.parametrize("managed", [False, True])
def test_run_step_evaluation_ends_update_weight_even_if_evaluate_fails(monkeypatch, managed):
    # Regression: /evaluate failing must NOT skip the /end_update_weight cleanup,
    # otherwise rollout + health monitoring stay paused forever.

    calls = []

    def _get(url, *args, **kwargs):
        calls.append(url)
        if url.endswith("/evaluate"):
            raise requests.exceptions.HTTPError("eval boom")
        return _Resp()

    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _get)
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_offline_mode", lambda _args: False)

    shell = _shell()
    shell.rollout_manager = object()  # non-None -> has_rollout
    shell._managed_lora_publication = managed

    shell._run_step_evaluation(3, end_update_weight=True)

    assert any(u.endswith("/evaluate") for u in calls)
    assert any(u.endswith("/end_update_weight") for u in calls) is (not managed)


def test_run_step_evaluation_swallows_timeout(monkeypatch):

    def _timeout_get(*_a, **_k):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _timeout_get)
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_offline_mode", lambda _args: False)

    shell = _shell()
    shell.rollout_manager = object()

    # Timeout must be swallowed (existing ``except Exception``) -> no raise.
    shell._run_step_evaluation(3, end_update_weight=True)

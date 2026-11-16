# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP contracts at the group coordinator boundary; no training backend
needed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import requests

from relax.distributed.ray import inference_sync


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


def _patch_health_common(monkeypatch, get_fn):
    monkeypatch.setattr(inference_sync, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(inference_sync.requests, "get", get_fn)
    monkeypatch.setattr(inference_sync.time, "sleep", lambda *_a, **_k: None)


def test_check_services_health_can_do_and_get_step_carry_timeout(monkeypatch):
    from argparse import Namespace

    calls = []
    # payload truthy -> can_do returns 1 so the while loop proceeds to recover.
    _patch_health_common(monkeypatch, _record_get(calls, payload=True))

    shell = SimpleNamespace()
    shell.args = Namespace(true_on_policy_mode=False, hybrid=False, rollout_http_timeout=17.0)

    inference_sync.check_services_health(shell.args)

    can_do = [(u, k) for u, k in calls if u.endswith("/can_do_update_weight_for_async")]
    recover = [(u, k) for u, k in calls if u.endswith("/recover_rollout_engines")]
    get_step = [(u, k) for u, k in calls if u.endswith("/get_step")]

    assert can_do and all(k.get("timeout") == shell.args.rollout_http_timeout for _u, k in can_do)
    assert get_step and all(k.get("timeout") == shell.args.rollout_http_timeout for _u, k in get_step)

    # Regression guard: recovery is a long-running rebuild -> NO timeout.
    assert recover
    for _u, k in recover:
        assert "timeout" not in k


def test_check_services_health_swallows_timeout(monkeypatch):
    from argparse import Namespace

    def _timeout_get(*_a, **_k):
        raise requests.exceptions.Timeout("boom")

    _patch_health_common(monkeypatch, _timeout_get)

    shell = SimpleNamespace()
    shell.args = Namespace(true_on_policy_mode=False, hybrid=False, rollout_http_timeout=120.0)

    # Timeout swallowed by the two ``except Exception`` blocks -> degrades to
    # actor_fwd_only / rollout_only rather than crashing.
    rollout_only, actor_fwd_only = inference_sync.check_services_health(shell.args)
    assert actor_fwd_only is True
    assert rollout_only is True


def test_check_services_health_retries_scale_in_fence(monkeypatch):
    from argparse import Namespace

    calls = []
    attempts = iter([_Resp(status_code=503), _Resp(payload=True)])

    def _get(url, *args, **kwargs):
        calls.append(url)
        if url.endswith("/can_do_update_weight_for_async"):
            return next(attempts)
        return _Resp()

    _patch_health_common(monkeypatch, _get)
    shell = SimpleNamespace()
    shell.args = Namespace(true_on_policy_mode=True, hybrid=False, rollout_http_timeout=120.0)

    assert inference_sync.check_services_health(shell.args) == (True, False)
    assert sum(url.endswith("/can_do_update_weight_for_async") for url in calls) == 2


def test_check_services_health_resets_fence_deadline_after_non_503(monkeypatch):
    from argparse import Namespace

    calls = []
    attempts = iter(
        [
            _Resp(status_code=503),
            _Resp(payload=False),
            _Resp(payload=False),
            _Resp(payload=True),
        ]
    )

    def _get(url, *args, **kwargs):
        calls.append(url)
        if url.endswith("/can_do_update_weight_for_async"):
            return next(attempts)
        return _Resp()

    _patch_health_common(monkeypatch, _get)
    monkeypatch.setattr(inference_sync.time, "monotonic", MagicMock(side_effect=[0.0, 0.0, 1.0, 3.0]))
    shell = SimpleNamespace()
    shell.args = Namespace(true_on_policy_mode=True, hybrid=False, rollout_http_timeout=2.0)

    assert inference_sync.check_services_health(shell.args) == (True, False)
    assert sum(url.endswith("/can_do_update_weight_for_async") for url in calls) == 4

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP contracts at the group coordinator boundary; no training backend
needed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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


def test_run_step_evaluation_probes_carry_timeout(monkeypatch):
    from argparse import Namespace

    calls = []
    monkeypatch.setattr(inference_sync, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(inference_sync.requests, "get", _record_get(calls))

    shell = SimpleNamespace(args=SimpleNamespace(rollout_http_timeout=120.0))
    shell.args = Namespace(rollout_http_timeout=17.0)
    shell.rollout_manager = object()  # non-None -> has_rollout

    inference_sync._evaluate_rollout(
        shell.args, 3, asynchronous=True, managed=getattr(shell, "_managed_lora_publication", False)
    )

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

    monkeypatch.setattr(inference_sync, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(inference_sync.requests, "get", _get)

    shell = SimpleNamespace(args=SimpleNamespace(rollout_http_timeout=120.0))
    shell.rollout_manager = object()  # non-None -> has_rollout
    shell._managed_lora_publication = managed

    inference_sync._evaluate_rollout(
        shell.args, 3, asynchronous=True, managed=getattr(shell, "_managed_lora_publication", False)
    )

    assert any(u.endswith("/evaluate") for u in calls)
    assert any(u.endswith("/end_update_weight") for u in calls) is (not managed)


def test_run_step_evaluation_swallows_timeout(monkeypatch):

    def _timeout_get(*_a, **_k):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr(inference_sync, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(inference_sync.requests, "get", _timeout_get)

    shell = SimpleNamespace(args=SimpleNamespace(rollout_http_timeout=120.0))
    shell.rollout_manager = object()

    # Timeout must be swallowed (existing ``except Exception``) -> no raise.
    inference_sync._evaluate_rollout(
        shell.args, 3, asynchronous=True, managed=getattr(shell, "_managed_lora_publication", False)
    )


@pytest.mark.parametrize("actor_fwd_absent", [False, True])
def test_managed_health_check_never_enters_legacy_rollout_pause(monkeypatch, actor_fwd_absent):
    args = SimpleNamespace(true_on_policy_mode=actor_fwd_absent, hybrid=False, rollout_http_timeout=10)
    get_url = MagicMock(return_value="http://actor-fwd")
    request = MagicMock(return_value=_Resp())
    monkeypatch.setattr(inference_sync, "get_serve_url", get_url)
    monkeypatch.setattr(inference_sync.requests, "get", request)
    assert inference_sync.check_services_health(args, managed=True) == (actor_fwd_absent, False)
    if actor_fwd_absent:
        get_url.assert_not_called()
        request.assert_not_called()
    else:
        get_url.assert_called_once_with("actor_fwd")
        request.assert_called_once_with("http://actor-fwd/get_step", timeout=10)


@pytest.mark.parametrize("mode", ["sft", "dpo", "rm", "debug", "unbound"])
def test_offline_and_no_peer_training_never_enter_rl_sync(monkeypatch, mode):
    args = SimpleNamespace(loss_type=mode, debug_train_only=mode == "debug", debug_rollout_only=False)
    methods = []
    group = SimpleNamespace(async_inference_call=lambda method, step: methods.append((method, step)) or [])
    request = MagicMock(side_effect=AssertionError("unexpected RL peer request"))
    monkeypatch.setattr(inference_sync.requests, "get", request)
    inference_sync.complete_inference_step(args, group, 3, has_rollout=mode != "unbound", managed=False)
    assert methods == [("finish_training_metrics", 3)] + (
        [("evaluate_after_training", 3)] if mode in {"sft", "dpo", "rm"} else []
    )
    request.assert_not_called()

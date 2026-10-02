# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed Ray train-actor initialization contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
import ray

from relax.distributed.ray.actor_group import RayTrainGroup


def _group(*actors: Any) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = list(actors)
    return group


def _actor(probe: Any = None) -> Any:
    return SimpleNamespace(termination_probe=SimpleNamespace(remote=lambda: probe))


def _raise(error: BaseException) -> None:
    raise error


def test_init_and_wait_preserves_success_and_cleans_submission_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _actor()
    group = _group(actor)
    refs = [object(), object()]
    values = dict(zip(refs, ["rank-0", "rank-1"], strict=True))
    monkeypatch.setattr(group, "async_init", lambda *_args, **_kwargs: refs)
    monkeypatch.setattr(ray, "wait", lambda pending, **_kwargs: ([pending[0]], pending[1:]))
    monkeypatch.setattr(ray, "get", lambda ref: values[ref])
    assert group.init_and_wait(object(), "actor") == ["rank-0", "rank-1"]
    assert group._actor_handlers == [actor]

    monkeypatch.setattr(group, "async_init", lambda *_args, **_kwargs: _raise(RuntimeError("submission failed")))
    monkeypatch.setattr(group, "_terminate_failed_init", lambda: group._actor_handlers.clear())
    with pytest.raises(RuntimeError, match="submission failed"):
        group.init_and_wait(object(), "actor")
    assert group._actor_handlers == []


def test_first_rank_failure_kills_and_confirms_every_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    actors = [_actor("probe-0"), _actor("probe-1")]
    group = _group(*actors)
    monkeypatch.setattr(group, "async_init", lambda *_args, **_kwargs: ["init-0", "init-1"])

    def wait(refs: list[str], *, timeout: float | None = None, **_kwargs: Any):
        return (list(refs), []) if timeout is not None else (["init-1"], ["init-0"])

    def get(ref: str) -> None:
        if ref == "init-1":
            raise RuntimeError("rank initialization failed")
        raise ray.exceptions.RayActorError(error_msg="actor terminated")

    killed: list[Any] = []
    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append((actor, no_restart)))
    with pytest.raises(RuntimeError, match="rank initialization failed"):
        group.init_and_wait(object(), "actor")
    assert killed == [(actors[0], True), (actors[1], True)]
    assert group._actor_handlers == []


@pytest.mark.parametrize("failure", ["kill", "pending"])
def test_unconfirmed_cleanup_keeps_group_and_fails_closed(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    actors = [_actor("probe-0"), _actor("probe-1")]
    group = _group(*actors)
    if failure == "kill":
        monkeypatch.setattr(ray, "kill", lambda *_args, **_kwargs: _raise(RuntimeError("private detail")))
        monkeypatch.setattr(ray, "wait", lambda refs, **_kwargs: (list(refs), []))
        monkeypatch.setattr(
            ray,
            "get",
            lambda _ref: _raise(ray.exceptions.RayActorError(error_msg="terminated")),
        )
    else:
        monkeypatch.setattr(ray, "kill", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(ray, "wait", lambda refs, **_kwargs: ([], list(refs)))
    with pytest.raises(RuntimeError, match="Failed to confirm train actor cleanup") as excinfo:
        group._terminate_failed_init(timeout=0.1)
    assert "private detail" not in str(excinfo.value)
    assert group._actor_handlers == actors


def test_real_actor_failure_cannot_mutate_state_after_cleanup() -> None:
    env = os.environ.copy()
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    env.pop("RAY_ADDRESS", None)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with tempfile.TemporaryDirectory(prefix="train-init-ray-") as probe_dir:
        result = subprocess.run(
            [sys.executable, "-m", "tests.utils._train_actor_init_cleanup_probe", probe_dir],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    assert result.returncode == 0, f"probe stdout:\n{result.stdout}\nprobe stderr:\n{result.stderr}"

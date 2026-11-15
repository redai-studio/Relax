# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed Ray train-actor initialization contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

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


@pytest.fixture
def offload_launch(monkeypatch, tmp_path):
    from relax.distributed.ray import actor_group as module

    # Capture actual RayTrainGroup launch/init arguments without spawning workers.
    captured = {}
    actor = SimpleNamespace(init=Mock(), get_master_addr_and_port=SimpleNamespace(remote=lambda: ("host", 1234)))
    remote_class = Mock()
    remote_class.options.return_value = remote_class
    remote_class.remote.return_value = actor

    def remote(**kwargs):
        captured.update(kwargs["runtime_env"]["env_vars"])
        return lambda _cls: remote_class

    monkeypatch.setattr(module.ray, "remote", remote)
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module, "_resolve_train_actor_class", lambda args: object)
    monkeypatch.setattr(module, "Lock", SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=lambda: None)))
    monkeypatch.setattr(module, "with_control_plane_affinity", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "get_ray_accelerator_kwargs", lambda *args: {})
    monkeypatch.setattr(module, "PlacementGroupSchedulingStrategy", lambda **kwargs: None)
    log = Mock()
    monkeypatch.setattr(module, "logger", log)
    tms = ModuleType("torch_memory_saver")
    tms.__file__ = str(tmp_path / "torch_memory_saver" / "__init__.py")
    hook = tmp_path / "torch_memory_saver_hook_mode_preload.abi3.so"
    hook.touch()
    monkeypatch.setitem(sys.modules, "torch_memory_saver", tms)
    monkeypatch.delenv("RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP", raising=False)

    def launch(enabled, matching_world, selective, optimizer="adam", expert=False):
        config = dict(
            train_backend="megatron",
            colocate=True,
            use_shard_p2p_weight_sync=expert,
            offload_train=True,
            offload_rollout=True,
            enable_weights_backuper=True,
            selective_offload=selective,
            optimizer=optimizer,
            use_routing_replay=False,
            megatron_to_hf_mode="bridge",
            train_env_vars={},
            resource={"actor": (1, 1), "rollout": (1, 1 if matching_world else 2)},
            rollout_num_gpus_per_engine=1,
        )
        args = SimpleNamespace(**config)
        env = {"RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP": "1"} if enabled else {}
        if expert:
            env.update(RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP=str(int(enabled)))
        group = RayTrainGroup(args, 1, (object(), [0], [0]), role="actor", runtime_env={"env_vars": env})
        return group, args, actor, captured, log

    return launch, str(hook)


@pytest.mark.parametrize(
    "enabled,matching_world,selective,optimizer,expert",
    [
        (False, True, False, "adam", False),
        (True, True, False, "adam", False),
        (True, False, False, "adam", False),
        (True, False, True, "adam", False),
        (True, True, False, "sgd", False),
        (True, True, False, "muon", False),
        (False, True, False, "adam", True),
        (False, False, False, "adam", True),
    ],
)
def test_overlap_selects_offload_only_when_enabled_and_supported(
    monkeypatch, offload_launch, enabled, matching_world, selective, optimizer, expert
):
    launch, hook = offload_launch
    preload = "/user/first.so /user/last.so"
    monkeypatch.setenv("LD_PRELOAD", preload if selective else f"{preload}:{hook}")
    group, args, actor, env, log = launch(enabled, matching_world, selective, optimizer, expert=expert)
    group.async_init(args, "actor")
    sent = actor.init.remote.call_args.args[0]
    selected = (enabled or expert) and matching_world and optimizer == "adam"
    assert sent.selective_offload == (selected or selective)
    assert (sent is args) == (not selected)
    assert args.selective_offload == selective
    if expert:
        assert sent.use_shard_p2p_weight_sync is True
        assert env["RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP"] == str(int(enabled))
    if selected:
        assert env["LD_PRELOAD"] == preload
        assert env["TMS_INIT_ENABLE"] == env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"
        assert "automatically selecting selective offload" in log.info.call_args.args[0] % log.info.call_args.args[1:]
    else:
        log.info.assert_not_called()
        if not selective:
            assert env["LD_PRELOAD"] == hook and env["TMS_INIT_ENABLE"] == "1"


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
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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

# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import ast
import asyncio
import dataclasses
import enum
import time
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from relax.distributed.ray import inference_manager as core
from relax.utils.health_monitor import RolloutHealthMonitor
from relax.utils.logging_utils import get_logger


@pytest.fixture
def rollout_methods():
    """Run unchanged production method bodies without importing SGLang."""
    path = Path(__file__).parents[2] / "relax/distributed/ray/rollout.py"
    tree = ast.parse(path.read_text())
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    namespace = {
        "asyncio": asyncio,
        "dataclasses": dataclasses,
        "enum": enum,
        "time": time,
        "uuid": uuid,
        "Optional": Optional,
        "ray": core.ray,
        "logger": get_logger(__name__),
        "InferenceCleanupError": core.InferenceCleanupError,
        "InferenceRecoveryRequired": core.InferenceRecoveryRequired,
        "get_engine_shutdown_guard": core.get_engine_shutdown_guard,
    }
    nodes = [definitions[name] for name in ("ScaleOutStatus", "ScaleOutRequest", "EngineGroupLifecycle")]
    for class_name, names in {
        "EngineGroup": {"shutdown_engines"},
        "RolloutServer": {"recover"},
        "RolloutManager": {"_shutdown_engine_actors", "dispose", "execute_scale_out"},
    }.items():
        for node in definitions[class_name].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                node.decorator_list = []
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class _ObjectRef:
    def __init__(self, result):
        self.result = result

    def get(self):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def __await__(self):
        async def resolve():
            return self.get()

        return resolve().__await__()


class _Engine:
    def __init__(self, runtime, name):
        self.runtime = runtime
        self.name = name

    def __getattr__(self, method):
        def remote(**kwargs):
            self.runtime.calls.append((self.name, method))
            return _ObjectRef(self.runtime.results.get((self.name, method), True))

        return SimpleNamespace(remote=remote)


@pytest.fixture
def runtime(monkeypatch):
    runtime = SimpleNamespace(calls=[], results={}, killed=[], removed=[])
    monkeypatch.setattr(core.ray, "get", lambda ref, timeout=None: ref.get())
    monkeypatch.setattr(core.ray, "kill", runtime.killed.append)
    monkeypatch.setattr(core, "remove_placement_group", runtime.removed.append)
    return runtime


def _topology(methods, runtime, *, slots=1):
    engines = [_Engine(runtime, str(index)) for index in range(slots)]
    group = SimpleNamespace(
        all_engines=list(engines),
        nodes_per_engine=slots,
        num_gpus_per_engine=slots,
        num_new_engines=1,
        rank_offset=0,
        gpu_offset=0,
        pg=("owned-pg", [], []),
        pg_owned=True,
        is_scaled_out=False,
        lifecycle_status=methods.EngineGroupLifecycle.ACTIVE,
        start_engines=MagicMock(side_effect=AssertionError("must not rebuild fenced workers")),
    )
    group.shutdown_engines = MethodType(methods.shutdown_engines, group)
    server = SimpleNamespace(model_name="policy", engine_groups=[group])
    server.recover = MethodType(methods.recover, server)
    owner = core.InferenceManager.for_rollout(SimpleNamespace(offload_rollout=False), {"policy": server})
    manager = SimpleNamespace(
        _engine_lifecycle_lock=owner._lifecycle_lock,
        _get_inference_manager=lambda: owner,
        _scale_out_requests={},
        _stop_eviction_monitor=MagicMock(),
        _health_monitors=[SimpleNamespace(stop=MagicMock())],
        _shutdown_all_engines=MagicMock(),
        _scale_out_ray_native=AsyncMock(),
        _scale_out_external=AsyncMock(),
    )
    for name in ("_shutdown_engine_actors", "dispose", "execute_scale_out"):
        setattr(manager, name, MethodType(getattr(methods, name), manager))
    return manager, owner, server, group, engines


async def _cleanup(entry, manager, owner, group):
    if entry == "scale_in":
        await manager._shutdown_engine_actors(
            group, "replica-0", [(i, engine) for i, engine in enumerate(group.all_engines) if engine is not None], 1.0
        )
    elif entry == "group":
        group.shutdown_engines(set(range(len(group.all_engines))))
    else:
        owner.shutdown_rollout(timeout=1.0)


@pytest.mark.parametrize("first_entry", ["scale_in", "group", "core", "health"])
@pytest.mark.parametrize("slots", [1, 2])
async def test_terminal_cleanup_tombstone_fences_every_rollout_entry(rollout_methods, runtime, first_entry, slots):
    manager, owner, server, group, engines = _topology(rollout_methods, runtime, slots=slots)
    runtime.results["0", "shutdown"] = core.ray.exceptions.ActorDiedError()

    if first_entry == "health":
        runtime.results["0", "health_generate"] = core.ray.exceptions.ActorDiedError()
        monitor = object.__new__(RolloutHealthMonitor)
        monitor._engine_group = group
        monitor._intentionally_removed = set()
        monitor._consecutive_failures = {0: 3}
        monitor._check_timeout = 1.0
        monitor._check_engine_health(0, engines[0])
        monitor._check_engine_health(0, engines[0])
        assert monitor._consecutive_failures == {0: 3}
        assert runtime.calls.count(("0", "health_generate")) == 1
        assert ("0", "shutdown") not in runtime.calls
    else:
        with pytest.raises(core.InferenceRecoveryRequired, match="node/container"):
            await _cleanup(first_entry, manager, owner, group)
        assert runtime.calls.count(("0", "shutdown")) == 1

    calls_after_death = list(runtime.calls)
    runtime.results.clear()
    for _ in range(2):
        for entry in ("scale_in", "group", "core"):
            with pytest.raises(core.InferenceRecoveryRequired, match="node/container"):
                await _cleanup(entry, manager, owner, group)
        with pytest.raises(core.InferenceRecoveryRequired, match="node/container"):
            server.recover()
        with pytest.raises(core.InferenceRecoveryRequired):
            owner.recover_rollout()

    assert runtime.calls == calls_after_death
    assert group.all_engines == [engines[0]] + [None] * (slots - 1)
    assert runtime.killed == engines[1:]
    assert runtime.removed == []
    assert owner.servers == {"policy": server}
    assert server.engine_groups == [group]
    assert group.pg == ("owned-pg", [], [])
    group.start_engines.assert_not_called()


@pytest.mark.parametrize("entry", ["scale_in", "group", "core"])
@pytest.mark.parametrize("result", ["false", "unavailable", "timeout"])
async def test_unconfirmed_rollout_shutdown_retries_without_terminal_tombstone(
    rollout_methods, runtime, entry, result
):
    manager, owner, _server, group, engines = _topology(rollout_methods, runtime)
    runtime.results["0", "shutdown"] = {
        "false": False,
        "unavailable": core.ray.exceptions.ActorUnavailableError("restarting", None),
        "timeout": TimeoutError("shutdown acknowledgement pending"),
    }[result]

    with pytest.raises(core.InferenceCleanupError) as error:
        await _cleanup(entry, manager, owner, group)
    assert type(error.value) is core.InferenceCleanupError
    assert group.all_engines == engines
    assert runtime.killed == runtime.removed == []
    core.get_engine_shutdown_guard(group).require_recovery()

    runtime.results.clear()
    await _cleanup(entry, manager, owner, group)

    assert runtime.calls.count(("0", "shutdown")) == 2
    assert group.all_engines == [None]
    assert runtime.killed == engines
    assert runtime.removed == (["owned-pg"] if entry == "core" else [])


@pytest.mark.parametrize("mode", ["ray_native", "external"])
async def test_dispose_rejects_inflight_scale_out_and_fences_further_work(rollout_methods, runtime, mode):
    manager, owner, server, _group, _engines = _topology(rollout_methods, runtime)
    status = rollout_methods.ScaleOutStatus
    request = rollout_methods.ScaleOutRequest(
        "inflight",
        status.PENDING,
        num_replicas=1 if mode == "ray_native" else 0,
        engine_urls=[] if mode == "ray_native" else ["http://engine.example"],
    )
    manager._scale_out_requests[request.request_id] = request
    started, release = asyncio.Event(), asyncio.Event()

    async def provision(actual_request):
        assert actual_request is request
        assert request.status is status.CREATING
        started.set()
        await release.wait()

    provisioner = getattr(manager, f"_scale_out_{mode}")
    provisioner.side_effect = provision
    task = asyncio.create_task(manager.execute_scale_out(request.request_id))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        with pytest.raises(core.InferenceRecoveryRequired, match="in-flight provisioning.*inflight"):
            manager.dispose()
        assert manager._disposing is True
        assert owner._stopping is True
        assert owner.servers == {"policy": server}
        manager._shutdown_all_engines.assert_not_called()
        manager._stop_eviction_monitor.assert_not_called()
        manager._health_monitors[0].stop.assert_not_called()
        with pytest.raises(core.InferenceRecoveryRequired, match="shutdown has started"):
            owner.recover_rollout()

        new_request = rollout_methods.ScaleOutRequest("after-dispose", status.PENDING, num_replicas=1)
        manager._scale_out_requests[new_request.request_id] = new_request
        await manager.execute_scale_out(new_request.request_id)
        assert new_request.status is status.FAILED
        assert new_request.error_message == "Rollout shutdown has started"
        assert manager._scale_out_ray_native.await_count + manager._scale_out_external.await_count == 1
        assert runtime.calls == runtime.killed == runtime.removed == []
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)


async def test_dispose_rejects_cancelled_request_while_provisioning_is_still_active(rollout_methods, runtime):
    manager, owner, _server, _group, _engines = _topology(rollout_methods, runtime)
    status = rollout_methods.ScaleOutStatus
    request = rollout_methods.ScaleOutRequest("cancelled-inflight", status.PENDING, num_replicas=1)
    manager._scale_out_requests[request.request_id] = request
    started, release = asyncio.Event(), asyncio.Event()

    async def provision(actual_request):
        assert actual_request is request
        started.set()
        await release.wait()

    manager._scale_out_ray_native.side_effect = provision
    task = asyncio.create_task(manager.execute_scale_out(request.request_id))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        request.update_status(status.CANCELLED)
        assert request.is_terminal()
        assert manager._active_scale_out == 1
        assert not task.done()

        with pytest.raises(core.InferenceRecoveryRequired, match="in-flight provisioning.*1 active calls"):
            manager.dispose()

        assert manager._disposing is True
        assert owner._stopping is True
        manager._shutdown_all_engines.assert_not_called()
        manager._stop_eviction_monitor.assert_not_called()
        manager._health_monitors[0].stop.assert_not_called()
        assert runtime.calls == runtime.killed == runtime.removed == []
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert manager._active_scale_out == 0
    assert request.status is status.CANCELLED


def test_dispose_allows_terminal_requests_and_stops_recovery(rollout_methods, runtime):
    manager, owner, _server, _group, _engines = _topology(rollout_methods, runtime)
    status = rollout_methods.ScaleOutStatus
    for value in (status.ACTIVE, status.PARTIAL, status.FAILED, status.CANCELLED):
        request = rollout_methods.ScaleOutRequest(value.value, value)
        manager._scale_out_requests[request.request_id] = request

    manager.dispose()

    assert manager._disposing is True
    assert owner._stopping is True
    manager._stop_eviction_monitor.assert_called_once_with()
    manager._health_monitors[0].stop.assert_called_once_with()
    manager._shutdown_all_engines.assert_called_once_with()
    with pytest.raises(core.InferenceRecoveryRequired, match="shutdown has started"):
        owner.recover_rollout()

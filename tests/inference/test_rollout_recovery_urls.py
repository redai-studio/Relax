# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise recovery and removal method bodies without an SGLang
installation."""

import ast
import asyncio
import dataclasses
import enum
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from relax.distributed.ray import inference_manager as core
from relax.distributed.ray.rollout_validation import validate_server_group_gpu_indices
from relax.utils.logging_utils import get_logger


class _Ref:
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
    def __init__(self, runtime, name, url=None):
        self.runtime = runtime
        self.name = name
        self.url = url

    def __getattr__(self, method):
        def remote(**kwargs):
            self.runtime.calls.append((self.name, method, kwargs))
            default = self.url if method == "get_url" else True
            result = self.runtime.results.get((self.name, method), default)
            return result() if callable(result) else _Ref(result)

        return SimpleNamespace(remote=remote)


@pytest.fixture
def runtime(monkeypatch):
    runtime = SimpleNamespace(calls=[], results={}, killed=[], removed=[], replacements={}, options=[])

    def get(ref, timeout=None):
        if isinstance(ref, list):
            return [get(item, timeout) for item in ref]
        return ref.get()

    class ActorFactory:
        def options(self, **kwargs):
            runtime.options.append(kwargs)
            return self

        def remote(self, args, **kwargs):
            return runtime.replacements[kwargs["rank"]]

    monkeypatch.setattr(core.ray, "get", get)
    monkeypatch.setattr(core.ray, "remote", lambda cls: ActorFactory())
    monkeypatch.setattr(core.ray, "kill", runtime.killed.append)
    monkeypatch.setattr(core, "remove_placement_group", runtime.removed.append)
    return runtime


@pytest.fixture
def rollout_module(runtime):
    """Load production classes; replace only unavailable infrastructure."""
    path = Path(__file__).parents[2] / "relax/distributed/ray/rollout.py"
    tree = ast.parse(path.read_text())
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {
        "_normalize_engine_addr",
        "_scale_in",
        "_resolve_scale_in_url_candidates",
        "_select_engines_for_removal",
        "_remove_live_engines",
        "_drain_engines",
        "_get_live_engine_representatives",
        "_get_live_engine_actors",
        "_unregister_engine_dcs",
        "_shutdown_engine_actors",
        "_cleanup_engine_groups",
        "_handle_evictions",
    }
    manager = definitions["RolloutManager"]
    manager.bases = []
    manager.decorator_list = []
    manager.body = [
        node
        for node in manager.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods
    ]
    nodes = [
        definitions[name]
        for name in ("ScaleInStatus", "ScaleInRequest", "EngineGroupLifecycle", "EngineGroup", "RolloutServer")
    ]
    nodes.append(manager)

    def allocate(**kwargs):
        return {
            rank: {
                "host": "127.0.0.1",
                "port": 15000 + rank,
                "nccl_port": 16000 + rank,
                "dist_init_addr": "127.0.0.1:17000",
            }
            for rank, engine in kwargs["rollout_engines"]
        }, {}

    namespace = {
        "__name__": __name__,
        "Any": Any,
        "Optional": Optional,
        "asyncio": asyncio,
        "dataclasses": dataclasses,
        "enum": enum,
        "os": os,
        "time": time,
        "uuid": uuid,
        "ray": core.ray,
        "logger": get_logger(__name__),
        "EngineShutdownGuard": core.EngineShutdownGuard,
        "InferenceCleanupError": core.InferenceCleanupError,
        "get_engine_shutdown_guard": core.get_engine_shutdown_guard,
        "validate_server_group_gpu_indices": validate_server_group_gpu_indices,
        "_resolve_rollout_engine_class": lambda args: _Engine,
        "PlacementGroupSchedulingStrategy": lambda **kwargs: kwargs,
        "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": [],
        "build_runai_streamer_env_for_load": lambda *args: {},
        "get_ray_accelerator_kwargs": lambda count: {"num_gpus": count},
        "find_available_port": lambda port: port,
        "_allocate_rollout_engine_addr_and_ports_normal": allocate,
        "GPU_MEMORY_TYPE_WEIGHTS": "weights",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _topology(module, runtime, *, scaled=True):
    args = SimpleNamespace(
        debug_train_only=False,
        num_gpus_per_node=4,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=8,
        hf_checkpoint="unused",
        rollout_external=False,
        offload_rollout=False,
        scale_in_drain_timeout=0,
        scale_in_shutdown_timeout=1,
    )
    group = module.EngineGroup(
        args=args,
        pg=("owned-pg", list(range(8)), list(range(8))),
        all_engines=[None, None],
        num_gpus_per_engine=8,
        num_new_engines=0,
        rank_offset=2,
        is_scaled_out=scaled,
        pg_owned=scaled,
    )
    group.replica_urls[0] = "old:30000"
    head = _Engine(runtime, "new-head", "http://[::1]:30001")
    follower = _Engine(runtime, "new-follower")
    runtime.replacements.update({2: head, 3: follower})
    server = module.RolloutServer([group], model_name="policy")
    owner = core.InferenceManager.for_rollout(args, {"policy": server})
    manager = module.RolloutManager()
    manager.args = args
    manager.servers = owner.servers
    manager._engine_lifecycle_lock = owner._lifecycle_lock
    manager._get_inference_manager = lambda: owner
    manager._get_server = lambda name: owner.servers.get(name)
    manager._health_monitors = []
    manager._is_weight_updating = False
    manager._find_active_scale_request = lambda: None
    return manager, owner, server, group, head, follower


@pytest.mark.parametrize("removal", ["count", "eviction"])
def test_recovered_replica_url_retries_partial_cleanup_after_head_is_gone(rollout_module, runtime, removal):
    module = rollout_module
    manager, owner, server, group, head, follower = _topology(module, runtime)
    owner.recover_rollout("policy")
    assert group.replica_urls == {0: "[::1]:30001"}
    assert (head.name, "get_url", {}) in runtime.calls
    assert (follower.name, "get_url", {}) not in runtime.calls

    initial = module.EngineGroup(
        args=group.args,
        pg=None,
        all_engines=[_Engine(runtime, "initial")],
        num_gpus_per_engine=4,
        num_new_engines=0,
    )
    server.engine_groups.insert(0, initial)
    runtime.results[follower.name, "shutdown"] = TimeoutError("follower shutdown is not confirmed")
    if removal == "count":
        first = module.ScaleInRequest(
            request_id="count",
            status=module.ScaleInStatus.PENDING,
            model_name="policy",
            num_replicas=1,
            force=True,
        )
        asyncio.run(manager._scale_in(first))
        assert first.status is module.ScaleInStatus.FAILED
    else:
        manager._handle_evictions([("policy", group, 0)])

    assert group.all_engines == [None, follower]
    assert group.lifecycle_status is module.EngineGroupLifecycle.DRAINING
    assert group in server.engine_groups
    assert runtime.killed == [head]
    assert runtime.removed == []
    runtime.results.pop((follower.name, "shutdown"))
    retry = module.ScaleInRequest(
        request_id="retry-current-url",
        status=module.ScaleInStatus.PENDING,
        model_name="policy",
        engine_urls=["http://[::1]:30001"],
        force=True,
    )
    asyncio.run(manager._scale_in(retry))

    assert retry.status is module.ScaleInStatus.COMPLETED, retry.error_message
    assert retry.selected_engines == ["group_2_engine_0"]
    assert group.all_engines == [None, None]
    assert group not in server.engine_groups
    assert runtime.killed == [head, follower]
    assert runtime.removed == ["owned-pg"]
    assert runtime.calls.count((head.name, "shutdown", {})) == 1
    assert runtime.calls.count((follower.name, "shutdown", {})) == 2
    assert runtime.calls.count((head.name, "get_url", {})) == 1
    assert (follower.name, "get_url", {}) not in runtime.calls


@pytest.mark.parametrize("url_result", [None, "", "   ", 12, TimeoutError("URL probe timed out")])
def test_recovery_url_failure_rolls_back_and_never_keeps_previous_url(rollout_module, runtime, url_result):
    _manager, owner, _server, group, head, follower = _topology(rollout_module, runtime)
    runtime.results[head.name, "get_url"] = url_result

    with pytest.raises((RuntimeError, TimeoutError)):
        owner.recover_rollout("policy")

    assert group.replica_urls == {}
    assert group.all_engines == [None, None]
    assert runtime.killed == [head, follower]
    assert runtime.removed == []
    assert all(entry["state"] == "FAILED" for entry in owner._inference_observation.entries.values())


def test_recovery_init_failure_discards_old_url_before_initialization(rollout_module, runtime):
    _manager, owner, _server, group, head, follower = _topology(rollout_module, runtime)
    runtime.results[head.name, "init"] = RuntimeError("backend initialization failed")

    with pytest.raises(RuntimeError, match="initialization failed"):
        owner.recover_rollout("policy")

    assert group.replica_urls == {}
    assert group.all_engines == [None, None]
    assert runtime.killed == [head, follower]
    assert not any(method == "get_url" for name, method, kwargs in runtime.calls)


def test_recovery_url_rollback_preserves_terminal_actor_resource_fence(rollout_module, runtime):
    _manager, owner, server, group, head, follower = _topology(rollout_module, runtime)
    runtime.results[head.name, "get_url"] = TimeoutError("URL probe timed out")
    runtime.results[follower.name, "shutdown"] = core.ray.exceptions.ActorDiedError()

    with pytest.raises(core.InferenceRecoveryRequired):
        owner.recover_rollout("policy")

    calls_after_rollback = list(runtime.calls)
    runtime.results.clear()
    for _ in range(2):
        with pytest.raises(core.InferenceRecoveryRequired):
            owner.recover_rollout("policy")
    assert runtime.calls == calls_after_rollback
    assert group.replica_urls == {}
    assert group.all_engines == [None, follower]
    assert runtime.killed == [head]
    assert runtime.removed == []
    assert group in server.engine_groups


def test_recovery_refreshes_only_rebuilt_scaled_out_heads(rollout_module, runtime):
    _manager, owner, server, group, head, follower = _topology(rollout_module, runtime)
    unchanged_head = _Engine(runtime, "unchanged-head", "http://stable:30002")
    unchanged_follower = _Engine(runtime, "unchanged-follower")
    group.all_engines.extend([unchanged_head, unchanged_follower])
    group.pg = ("owned-pg", list(range(16)), list(range(16)))
    group.replica_urls[1] = "stable:30002"
    owner.recover_rollout("policy")

    assert group.replica_urls == {0: "[::1]:30001", 1: "stable:30002"}
    assert group.all_engines == [head, follower, unchanged_head, unchanged_follower]
    assert [name for name, method, kwargs in runtime.calls if method == "get_url"] == [head.name]


def test_initial_group_recovery_does_not_add_scale_in_url_probes(rollout_module, runtime):
    _manager, owner, _server, group, head, follower = _topology(rollout_module, runtime, scaled=False)
    owner.recover_rollout("policy")

    assert group.all_engines == [head, follower]
    assert not any(method == "get_url" for name, method, kwargs in runtime.calls)


async def test_late_old_head_url_probe_cannot_overwrite_recovered_url(rollout_module, runtime):
    module = rollout_module
    manager, owner, server, group, head, follower = _topology(module, runtime)
    old_head = _Engine(runtime, "old-head", "http://old:30000")
    old_follower = _Engine(runtime, "old-follower")
    group.all_engines[:] = [old_head, old_follower]
    started = asyncio.Event()
    release = asyncio.Event()

    async def old_url():
        started.set()
        await release.wait()
        return old_head.url

    runtime.results[old_head.name, "get_url"] = old_url
    request = module.ScaleInRequest(
        request_id="late-probe",
        status=module.ScaleInStatus.PENDING,
        model_name="policy",
        engine_urls=[old_head.url],
    )
    pending = asyncio.create_task(manager._resolve_scale_in_url_candidates(request, server))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        group.shutdown_engines({0, 1})
        owner.recover_rollout("policy")
        assert group.replica_urls == {0: "[::1]:30001"}
    finally:
        release.set()
    assert await asyncio.wait_for(pending, timeout=1) == []
    assert group.replica_urls == {0: "[::1]:30001"}
    assert group.all_engines == [head, follower]

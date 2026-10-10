# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real scale-out allocation and rollback without SGLang or GPUs."""

import ast
import asyncio
import dataclasses
import enum
import os
import time
import uuid
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from relax.distributed.ray import inference_manager as core
from relax.distributed.ray.rollout_validation import validate_server_group_gpu_indices
from relax.utils.logging_utils import get_logger
from relax.utils.s3_model_loader import build_runai_streamer_env_for_load, is_s3_uri
from relax.utils.scale_utils import ScaleOutFailure, ScaleOutFailureCategory


_ROOT = Path(__file__).parents[2]


def _load_production(runtime, *, legacy_slots=False):
    """Extract whole production definitions, replacing only remote
    boundaries."""
    path = _ROOT / "relax/distributed/ray/rollout.py"
    tree = ast.parse(path.read_text())
    definitions = {node.name: node for node in tree.body if hasattr(node, "name")}
    namespace = {
        "asyncio": asyncio,
        "dataclasses": dataclasses,
        "enum": enum,
        "os": os,
        "time": time,
        "uuid": uuid,
        "Any": Any,
        "Optional": Optional,
        "ray": runtime.ray,
        "logger": get_logger(__name__),
        "EngineShutdownGuard": core.EngineShutdownGuard,
        "InferenceCleanupError": core.InferenceCleanupError,
        "InferenceRecoveryRequired": core.InferenceRecoveryRequired,
        "get_engine_shutdown_guard": core.get_engine_shutdown_guard,
        "ScaleOutFailure": ScaleOutFailure,
        "ScaleOutFailureCategory": ScaleOutFailureCategory,
        "validate_server_group_gpu_indices": validate_server_group_gpu_indices,
        "build_runai_streamer_env_for_load": build_runai_streamer_env_for_load,
        "PlacementGroupSchedulingStrategy": SimpleNamespace,
        "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": [],
        "get_ray_accelerator_kwargs": lambda count: {"num_gpus": count},
        "SGLangEngine": _Engine,
        "RolloutServer": Any,
    }
    names = (
        "ScaleOutStatus",
        "ScaleOutRequest",
        "EngineGroupLifecycle",
        "EngineGroup",
        "EngineFinalizeResult",
        "ScaleResult",
        "_resolve_rollout_engine_class",
        "_allocate_rollout_engine_addr_and_ports_normal",
    )
    nodes = [definitions[name] for name in names]
    methods = {"_bring_up_single_replica", "_rollback_engines"}
    for node in definitions["RolloutManager"].body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in methods:
            node.decorator_list = []
            if legacy_slots and node.name == "_bring_up_single_replica":
                # Mutation control: reintroduce precisely the pre-PR singleton.
                changed = 0
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "EngineGroup"
                    ):
                        for keyword in call.keywords:
                            if keyword.arg == "all_engines":
                                keyword.value = ast.copy_location(
                                    ast.List(elts=[ast.Constant(None)], ctx=ast.Load()), keyword.value
                                )
                                changed += 1
                assert changed == 1
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _server_args_for(engine, init):
    """Use the actual backend argument builder for the node-rank contract."""
    path = _ROOT / "relax/backends/sglang/sglang_engine.py"
    names = {"_compute_server_args", "_to_local_gpu_id", "_enable_draft_weights_cpu_backup"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    peft = _ROOT / "relax/utils/megatron_peft_utils.py"
    nodes.extend(node for node in ast.parse(peft.read_text()).body if getattr(node, "name", None) == "is_lora_enabled")
    fields = "nnodes node_rank tp_size dp_size pp_size host port nccl_port dist_init_addr base_gpu_id".split()
    namespace = {
        "os": os,
        "dataclasses": dataclasses,
        "ServerArgs": dataclasses.make_dataclass("ServerArgs", fields),
        "is_s3_uri": is_s3_uri,
        "device_utils": SimpleNamespace(get_visible_devices_env_var=lambda: "RELAX_SCALE_TEST_VISIBLE_DEVICES"),
        "_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS": [],
        "logger": get_logger(__name__),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_compute_server_args"](
        engine.args,
        rank=engine.rank,
        base_gpu_id=engine.constructor["base_gpu_id"],
        num_gpus_per_engine=engine.constructor["num_gpus_per_engine"],
        **{key: init[key] for key in ("host", "port", "nccl_port", "dist_init_addr")},
    )[0]


class _ObjectRef:
    def __init__(self, result, resolved=None):
        self.result = result
        self.resolved = resolved

    def get(self):
        if self.resolved is not None:
            self.resolved()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def __await__(self):
        async def resolve():
            return self.get()

        return resolve().__await__()


class _Engine:
    def __init__(self, runtime, options, args, constructor):
        self.runtime, self.options, self.args, self.constructor = runtime, options, args, constructor
        self.rank = constructor["rank"]
        self.bundle = options["scheduling_strategy"].placement_group_bundle_index
        self.ip, _ = runtime.locations[self.bundle]

    def __getattr__(self, method):
        def remote(**kwargs):
            self.runtime.calls.append((self.rank, method, kwargs))
            if method == "_get_current_node_ip_and_free_port":
                return _ObjectRef((self.ip, kwargs.get("start_port", 15000)))
            result = self.runtime.results.get((self.rank, method), True)
            resolved = (lambda: self.runtime.initialized.append(self.rank)) if method == "init" else None
            return _ObjectRef(result, resolved)

        return SimpleNamespace(remote=remote)


def _runtime(gpus_per_node, replica_gpus):
    runtime = SimpleNamespace(calls=[], results={}, engines=[], initialized=[], killed=[])
    runtime.locations = [
        (f"192.0.2.{1 + index // gpus_per_node}", index % gpus_per_node) for index in range(replica_gpus)
    ]

    def engine_options(**options):
        def create(args, **constructor):
            engine = _Engine(runtime, options, args, constructor)
            runtime.engines.append(engine)
            return engine

        return SimpleNamespace(remote=create)

    def info_options(**options):
        bundle = options["scheduling_strategy"].placement_group_bundle_index
        actor = SimpleNamespace(
            get_ip_and_gpu_id=SimpleNamespace(remote=lambda: _ObjectRef(runtime.locations[bundle]))
        )
        return SimpleNamespace(remote=lambda: actor)

    runtime.ray = SimpleNamespace(
        remote=lambda cls: SimpleNamespace(options=engine_options),
        get=lambda ref, timeout=None: ref.get(),
        kill=runtime.killed.append,
    )
    runtime.InfoActor = SimpleNamespace(options=info_options)
    return runtime


def _setup(gpus_per_node=8, replica_gpus=16, *, legacy_slots=False):
    runtime = _runtime(gpus_per_node, replica_gpus)
    production = _load_production(runtime, legacy_slots=legacy_slots)
    args = SimpleNamespace(
        num_gpus_per_node=gpus_per_node,
        rollout_num_gpus=replica_gpus,
        rollout_num_gpus_per_engine=replica_gpus,
        debug_train_only=False,
        rollout_external=False,
        hf_checkpoint="model",
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        seed=1,
        offload_rollout=False,
        use_rollout_routing_replay=False,
        fp16=False,
    )
    server = SimpleNamespace(model_name="policy", router_ip="192.0.2.254", router_port=3000, engine_groups=[])
    owner = core.InferenceManager.for_rollout(args, {"policy": server})
    manager = SimpleNamespace(args=args, _port_cursors={0: 15000}, _get_inference_manager=lambda: owner)
    manager._bring_up_single_replica = MethodType(production._bring_up_single_replica, manager)
    manager._rollback_engines = MethodType(production._rollback_engines, manager)

    async def finalize(**kwargs):
        return production.EngineFinalizeResult(True, group=kwargs["engine_group"])

    manager._finalize_engine_group_registration = AsyncMock(side_effect=finalize)
    request = production.ScaleOutRequest("multinode", production.ScaleOutStatus.CREATING)
    return SimpleNamespace(
        runtime=runtime,
        production=production,
        manager=manager,
        owner=owner,
        server=server,
        request=request,
        pg=object(),
    )


async def _bring_up(case):
    return await case.manager._bring_up_single_replica(
        request=case.request,
        srv=case.server,
        pg=case.pg,
        replica_idx=0,
        num_gpus=case.manager.args.rollout_num_gpus_per_engine,
        gpus_per_engine=case.manager.args.rollout_num_gpus_per_engine,
        engine_offset=2,
        sort_key=lambda item: item,
        InfoActor=case.runtime.InfoActor,
    )


def _assert_multinode_created(case):
    assert [engine.rank for engine in case.runtime.engines] == [2, 3]
    assert [engine.bundle for engine in case.runtime.engines] == [0, 8]
    assert case.runtime.initialized == [2, 3]


async def test_multinode_scale_out_regression_rejects_pre_pr_singleton():
    case = _setup(legacy_slots=True)
    assert (await _bring_up(case)).success is True
    with pytest.raises(AssertionError):
        _assert_multinode_created(case)
    assert [engine.rank for engine in case.runtime.engines] == [2]


@pytest.mark.parametrize("replica_gpus, expected_bundles", [(2, [0]), (8, [0]), (16, [0, 8])])
async def test_scale_out_starts_every_physical_actor_and_finalizes_only_head(replica_gpus, expected_bundles):
    case = _setup(replica_gpus=replica_gpus)
    result = await _bring_up(case)
    assert result.success is True
    engines = case.runtime.engines
    expected_ranks = list(range(2, 2 + len(expected_bundles)))
    assert [engine.rank for engine in engines] == expected_ranks
    assert [engine.bundle for engine in engines] == expected_bundles
    assert case.runtime.initialized == expected_ranks
    inits = [kwargs for _, method, kwargs in case.runtime.calls if method == "init"]
    assert len(inits) == len(expected_bundles)
    assert len({init["dist_init_addr"] for init in inits}) == 1
    assert all(init["skip_dcs_registration"] and init["skip_router_registration"] for init in inits)
    assert [init["host"] for init in inits] == [engine.ip for engine in engines]
    finalizer = case.manager._finalize_engine_group_registration
    finalizer.assert_awaited_once()
    group = finalizer.await_args.kwargs["engine_group"]
    assert type(group) is case.production.EngineGroup
    assert group.all_engines == engines
    assert group.num_new_engines == len(expected_bundles)
    assert finalizer.await_args.kwargs["engines"] == [engines[0]]
    assert group.pg[0] is case.pg
    assert group.pg[1] == list(range(replica_gpus))
    assert not any(isinstance(actor, _Engine) for actor in case.runtime.killed)
    if replica_gpus == 16:
        _assert_multinode_created(case)
        server_args = [_server_args_for(engine, init) for engine, init in zip(engines, inits)]
        assert [args["node_rank"] for args in server_args] == [0, 1]
        assert [args["nnodes"] for args in server_args] == [2, 2]
        assert [args["tp_size"] for args in server_args] == [16, 16]
        assert server_args[0]["dist_init_addr"] == server_args[1]["dist_init_addr"]


@pytest.mark.parametrize("failed_rank", [2, 3])
async def test_multinode_scale_out_init_failure_rolls_back_both_workers(failed_rank):
    case = _setup()
    case.runtime.results[failed_rank, "init"] = RuntimeError("backend init failed")
    result = await _bring_up(case)
    assert result.success is False
    assert result.reason.category is ScaleOutFailureCategory.ENGINE_INIT_FAILED
    _assert_multinode_created(case)
    assert [rank for rank, method, _ in case.runtime.calls if method == "shutdown"] == [2, 3]
    assert [actor for actor in case.runtime.killed if isinstance(actor, _Engine)] == case.runtime.engines
    assert case.server.engine_groups == []
    case.manager._finalize_engine_group_registration.assert_not_awaited()


async def test_multinode_scale_out_terminal_follower_retains_group_and_fences_rebuild():
    case = _setup()
    case.runtime.results[3, "init"] = RuntimeError("follower init failed")
    case.runtime.results[3, "shutdown"] = core.ray.exceptions.ActorDiedError()
    with pytest.raises(core.InferenceRecoveryRequired, match="node/container"):
        await _bring_up(case)
    _assert_multinode_created(case)
    head, follower = case.runtime.engines
    assert len(case.server.engine_groups) == 1
    group = case.server.engine_groups[0]
    assert group.all_engines == [None, follower]
    assert group.pg[0] is case.pg
    assert group.pg_owned is True
    assert group.lifecycle_status is case.production.EngineGroupLifecycle.DRAINING
    assert [actor for actor in case.runtime.killed if isinstance(actor, _Engine)] == [head]
    calls = list(case.runtime.calls)
    for _ in range(2):
        with pytest.raises(core.InferenceRecoveryRequired):
            await case.manager._rollback_engines(group)
        with pytest.raises(core.InferenceRecoveryRequired):
            group.start_engines()
        with pytest.raises(core.InferenceRecoveryRequired):
            case.owner.recover_rollout("policy")
    assert case.runtime.calls == calls
    assert group.all_engines == [None, follower]
    assert case.owner.servers == {"policy": case.server}
    case.manager._finalize_engine_group_registration.assert_not_awaited()

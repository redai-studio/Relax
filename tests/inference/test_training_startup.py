# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run production startup methods with RPC boundaries stubbed, without Ray/GPU
imports."""

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_class(path: str, name: str, methods: tuple[str, ...], **namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    scope = {"asyncio": asyncio, **namespace}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


def _actor():
    cls = _load_class(
        "relax/components/actor.py",
        "Actor",
        ("set_rollout_manager", "update_weights_fully_async"),
        is_sft_mode=lambda config: config.sft,
    )
    actor = cls()
    actor.config = SimpleNamespace(fully_async=False, hybrid=False, sft=False)
    actor.actor_model = Mock()
    actor.rollout_manager = SimpleNamespace(set_policy_weights_ready=SimpleNamespace(remote=AsyncMock()))
    return actor


@pytest.mark.parametrize("phase", ["bind", "sync", "fully_async"])
async def test_actor_startup_remains_responsive_and_waits_for_rpc(phase):
    actor = _actor()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()

    def blocking_rpc(*args, **kwargs):
        assert threading.get_ident() != loop_thread, "Blocking Ray RPC ran on the Serve event loop"
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2), "Test did not release startup RPC"

    method = {"bind": "set_rollout_manager", "sync": "update_weights", "fully_async": "update_weights_fully_async"}[
        phase
    ]
    getattr(actor.actor_model, method).side_effect = blocking_rpc
    manager = object()
    request = (
        actor.update_weights_fully_async(rollout_only=True, actor_fwd_only=False)
        if phase == "fully_async"
        else actor.set_rollout_manager(manager)
    )
    task = asyncio.create_task(request)
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)  # a health probe can execute while the RPC is pending
        assert not task.done(), "Startup acknowledged before the RPC completed"
        if phase == "bind":
            actor.actor_model.update_weights.assert_not_called()
    finally:
        release.set()
        await task
    if phase == "fully_async":
        actor.actor_model.update_weights_fully_async.assert_called_once_with(0, True, False)
        actor.rollout_manager.set_policy_weights_ready.remote.assert_awaited_once_with(True)
    else:
        assert actor.rollout_manager is manager
        actor.actor_model.set_rollout_manager.assert_called_once_with(manager)
        actor.actor_model.update_weights.assert_called_once_with()


@pytest.mark.parametrize(
    "fully_async,hybrid,sft,expected", [(True, False, False, 0), (True, True, False, 1), (False, False, True, 0)]
)
async def test_actor_startup_preserves_weight_sync_modes(fully_async, hybrid, sft, expected):
    actor = _actor()
    actor.config = SimpleNamespace(fully_async=fully_async, hybrid=hybrid, sft=sft)
    await actor.set_rollout_manager(object())
    assert actor.actor_model.update_weights.call_count == expected


@pytest.mark.parametrize("phase", ["bind", "sync", "fully_async"])
async def test_actor_startup_propagates_rpc_failure(phase):
    actor = _actor()
    name = {"bind": "set_rollout_manager", "sync": "update_weights", "fully_async": "update_weights_fully_async"}[
        phase
    ]
    getattr(actor.actor_model, name).side_effect = RuntimeError("weight transaction failed")
    with pytest.raises(RuntimeError, match="weight transaction failed"):
        if phase == "fully_async":
            await actor.update_weights_fully_async()
        else:
            await actor.set_rollout_manager(object())
    if phase == "bind":
        actor.actor_model.update_weights.assert_not_called()
    elif phase == "fully_async":
        actor.rollout_manager.set_policy_weights_ready.remote.assert_not_awaited()


@pytest.mark.parametrize("rollout_only", [False, True])
async def test_actor_initial_sync_skipping_rollout_does_not_publish_readiness(rollout_only):
    actor = _actor()

    await actor.update_weights_fully_async(rollout_only=rollout_only, actor_fwd_only=True)

    actor.rollout_manager.set_policy_weights_ready.remote.assert_not_awaited()


def _service_class():
    return _load_class("relax/core/service.py", "Service", ("update_weights_fully_async", "recv_weight_fully_async"))


@pytest.mark.parametrize("method", ["update_weights_fully_async", "recv_weight_fully_async"])
@pytest.mark.parametrize("failed", [False, True])
async def test_service_waits_for_remote_weight_completion(method, failed):
    remote_result = asyncio.get_running_loop().create_future()
    service = _service_class()()
    service.handle = SimpleNamespace(**{method: SimpleNamespace(remote=Mock(return_value=remote_result))})
    task = asyncio.create_task(getattr(service, method)())
    try:
        await asyncio.sleep(0)
        assert not task.done(), "Service returned before the remote update completed"
        if failed:
            remote_result.set_exception(RuntimeError("remote update failed"))
            with pytest.raises(RuntimeError, match="remote update failed"):
                await task
        else:
            remote_result.set_result("committed")
            assert await task == "committed"
    finally:
        if not remote_result.done():
            remote_result.set_result(None)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failed", [False, True])
def test_controller_initial_sync_starts_all_peers_and_gates_training(failed):
    events = []
    roles = SimpleNamespace(actor="actor", actor_fwd="actor_fwd", reference="reference", rollout="rollout")
    service_cls = _service_class()

    async def bounded_run(coroutine):
        return await asyncio.wait_for(coroutine, 2)

    controller_cls = _load_class(
        "relax/core/controller.py",
        "Controller",
        ("training_loop",),
        ROLES=roles,
        GENRM_ROLE="genrm",
        set_managed_opd_teacher_on_actor_service=AsyncMock(),
        _needs_rollout_manager_setup=lambda services: False,
        run=lambda coroutine: asyncio.run(bounded_run(coroutine)),
        logger=Mock(),
    )
    controller = controller_cls()
    controller.config = SimpleNamespace(
        debug_train_only=False, debug_rollout_only=False, fully_async=True, hybrid=False
    )
    controller._teacher_manager = None
    controller._pending_task_refs_lock = threading.Lock()
    controller._restarting = False
    controller._report_error_to_metrics_service = Mock()
    controller.serve_dict = {}
    peers_started = set()

    async def exchange(role, **kwargs):
        peers_started.add(role)
        while len(peers_started) < 3:
            await asyncio.sleep(0)
        if failed and role == "actor":
            raise RuntimeError("initial sync failed")
        events.append("committed:" + role)

    for role in ("actor", "actor_fwd", "reference"):
        service = service_cls()
        method = "update_weights_fully_async" if role == "actor" else "recv_weight_fully_async"
        service.handle = SimpleNamespace(
            **{method: SimpleNamespace(remote=lambda role=role, **kw: asyncio.create_task(exchange(role, **kw)))}
        )
        service.get_step = AsyncMock(return_value=0)
        service.set_step = AsyncMock()
        service.run = lambda role=role: events.append("run:" + role)
        service.role = role
        controller.serve_dict[role] = service

    def cleanup():
        assert len([event for event in events if event.startswith("committed:")]) == 3
        events.append("cleanup")

    controller._cleanup_s3_model_weights_after_init = Mock(side_effect=cleanup)
    if failed:
        with pytest.raises(RuntimeError, match="initial sync failed"):
            controller.training_loop()
        controller._cleanup_s3_model_weights_after_init.assert_not_called()
        assert not any(event.startswith("run:") for event in events)
    else:
        controller.training_loop()
        assert events.index("cleanup") < events.index("run:actor")


def test_controller_decoupled_initial_sync_is_rollout_only():
    """A pure actor+rollout topology must not initialize absent forward
    consumers."""
    events = []
    roles = SimpleNamespace(actor="actor", actor_fwd="actor_fwd", reference="reference", rollout="rollout")
    service_cls = _service_class()

    async def bounded_run(coroutine):
        return await asyncio.wait_for(coroutine, 2)

    controller_cls = _load_class(
        "relax/core/controller.py",
        "Controller",
        ("training_loop",),
        ROLES=roles,
        GENRM_ROLE="genrm",
        set_managed_opd_teacher_on_actor_service=AsyncMock(),
        _needs_rollout_manager_setup=lambda services: False,
        run=lambda coroutine: asyncio.run(bounded_run(coroutine)),
        logger=Mock(),
    )
    controller = controller_cls()
    controller.config = SimpleNamespace(
        debug_train_only=False, debug_rollout_only=False, fully_async=True, hybrid=False
    )
    controller._teacher_manager = None
    controller._pending_task_refs_lock = threading.Lock()
    controller._restarting = False
    controller._report_error_to_metrics_service = Mock()
    controller._cleanup_s3_model_weights_after_init = Mock()

    actor = service_cls()
    actor.handle = SimpleNamespace(update_weights_fully_async=SimpleNamespace(remote=AsyncMock(return_value=None)))
    actor.get_step = AsyncMock(return_value=0)
    actor.set_step = AsyncMock()
    actor.run = lambda: events.append("run:actor")
    actor.role = "actor"

    rollout = service_cls()
    rollout.set_step = AsyncMock()
    rollout.run = lambda: events.append("run:rollout")
    rollout.role = "rollout"

    controller.serve_dict = {roles.actor: actor, roles.rollout: rollout}
    controller.training_loop()

    actor.handle.update_weights_fully_async.remote.assert_awaited_once_with(rollout_only=True, actor_fwd_only=False)
    assert events == ["run:actor", "run:rollout"]

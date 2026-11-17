# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute component methods without importing GPU/Serve startup
dependencies."""

import ast
import asyncio
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[3]


def component_methods(file: str, name: str, methods: list[str], **bindings):
    source = ast.parse((ROOT / "relax" / "components" / file).read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    env = {"asyncio": asyncio, "threading": threading, **bindings}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "relax" / "components" / file), "exec"), env)
    return env[name]()


def remote(fn):
    return SimpleNamespace(remote=fn)


class Gate:
    def __init__(self, events):
        self.state = object()
        self.events = events

    def admit(self, step, *args):
        self.events.append("admit")
        return step

    def fail(self, exc):
        self.events.append(("failure", str(exc)))

    def leave(self):
        self.events.append("leave")


@pytest.mark.parametrize("hybrid", [False, True])
def test_actor_commits_next_step_and_clears_partition_before_leaving_operation(hybrid):
    events = []
    actor = component_methods(
        "actor.py",
        "Actor",
        ["_background_run", "_run_training_step"],
        suppress=suppress,
        build_async_train_data_fields=lambda config: ["advantages"],
        build_data_fields=lambda config, **kwargs: ["tokens"],
        sft_partition_ids=lambda config, step: [f"train_{step}"],
        run=lambda value: value,
    )
    actor.config = SimpleNamespace(num_rollout=1, fully_async=True, hybrid=hybrid, use_health_check=True)
    actor.step = 0
    actor._lock = threading.RLock()
    actor._stop_event = threading.Event()
    actor._logger = Mock()
    actor._recovery = Gate(events)
    admit = actor._recovery.admit

    def check_fields(step, client, fields, stop):
        assert fields == (["tokens"] if hybrid else ["advantages"])
        return admit(step)

    actor._recovery.admit = check_fields
    actor._execute_training = lambda: events.append("train") or True
    actor.data_system_client = SimpleNamespace(
        async_clear_partition=lambda **kw: events.append(("clear", kw["partition_id"]))
    )
    actor.healthy = Mock()
    actor._recovery.leave = lambda: events.append(("leave", actor.step))

    actor._background_run()

    assert events == ["admit", "train", ("clear", "train_0"), ("leave", 1)]


def test_actor_failure_pauses_before_leaving_and_is_not_swallowed_by_health_check():
    events = []
    report_ack = Mock()
    actor = component_methods(
        "actor.py",
        "Actor",
        ["_background_run", "_run_training_step"],
        suppress=suppress,
        ray=SimpleNamespace(get=report_ack),
        build_async_train_data_fields=lambda config: ["advantages"],
    )
    actor.config = SimpleNamespace(num_rollout=1, fully_async=True, use_health_check=True)
    actor.step = 0
    actor._lock = threading.RLock()
    actor._stop_event = threading.Event()
    actor._stop_event.wait = Mock()
    actor._logger = Mock()
    actor._recovery = Gate(events)
    actor._execute_training = Mock(side_effect=ValueError("training-failed"))
    actor.data_system_client = Mock()
    actor.healthy = Mock()

    with pytest.raises(ValueError, match="training-failed"):
        actor._background_run()

    assert events == ["admit", ("failure", "training-failed"), "leave"]
    report_ack.assert_called_once()
    actor._stop_event.wait.assert_called_once_with()
    assert actor.step == 0
    actor.data_system_client.async_clear_partition.assert_not_called()


def test_actor_fwd_receive_stays_ready_until_all_receive_results_return():
    events = []
    ready = False

    def set_ready(value):
        nonlocal ready
        ready = value
        events.append(("ready", value))

    def wait_all(refs):
        events.append(("wait", tuple(refs), ready))

    worker = component_methods(
        "actor_fwd.py",
        "ActorFwd",
        ["compute_actor_log_prob"],
        ray=SimpleNamespace(get=lambda value: value),
        wait_all=wait_all,
    )
    worker._recovery = SimpleNamespace(state=SimpleNamespace(set_fwd_ready=remote(set_ready)))
    worker.actor_model = SimpleNamespace(
        async_compute_actor_log_prob=lambda step: ["compute-rank0", "compute-rank1"],
        async_inference_call=lambda method, step: ["receive-rank0", "receive-rank1"],
    )

    worker.compute_actor_log_prob(2)

    assert events == [
        ("wait", ("compute-rank0", "compute-rank1"), False),
        ("ready", True),
        ("wait", ("receive-rank0", "receive-rank1"), True),
        ("ready", False),
    ]


def test_actor_fwd_recovery_refuses_live_forward_thread():
    worker = component_methods("actor_fwd.py", "ActorFwd", ["stop_for_recovery"], wait_all=Mock())
    worker._stop_event = threading.Event()
    worker._loop_runner = SimpleNamespace(join=AsyncMock(return_value=False))
    worker.role = "actor_fwd"
    worker.actor_model = Mock()

    with pytest.raises(TimeoutError, match="still owns"):
        asyncio.run(worker.stop_for_recovery(timeout=0))

    worker.actor_model.async_inference_call.assert_not_called()


def test_actor_fwd_recovery_unregisters_before_returning_only_owned_children():
    events = []
    worker = component_methods(
        "actor_fwd.py", "ActorFwd", ["stop_for_recovery"], wait_all=lambda refs: events.append(refs)
    )
    worker._stop_event = threading.Event()
    worker._loop_runner = SimpleNamespace(join=AsyncMock(return_value=True))
    worker.actor_model = SimpleNamespace(
        async_inference_call=lambda method: method, _actor_handlers=["rank0", "rank1"], _actor_lock="lock"
    )

    assert asyncio.run(worker.stop_for_recovery()) == ["rank0", "rank1", "lock"]
    assert events == ["unregister_inference_transport"]


def test_advantages_keeps_complete_partition_write_and_step_inside_operation():
    events = []
    service = component_methods("advantages.py", "Advantages", ["_run_blocking"])
    service.config = SimpleNamespace(num_rollout=1)
    service.step = 0
    service._logger = Mock()
    service._recovery = Gate(events)
    service._recovery.leave = lambda: events.append(("leave", service.step))
    service._advantage_fields = lambda: ["log_probs"]
    service.data_system_client = Mock()
    service._compute_partition = lambda step, fields: events.append(("all-partition-writes-returned", step))

    service._run_blocking()

    assert events == ["admit", ("all-partition-writes-returned", 0), ("leave", 1)]


@pytest.mark.parametrize("failed_phase", ["compute", "recv"])
def test_forward_receive_error_requires_global_even_after_sender_finishes(failed_phase):
    class UnsafeInferenceError(RuntimeError):
        pass

    events = []
    failure = ValueError(f"{failed_phase} failed")

    def wait_all(refs):
        if refs == failed_phase:
            raise failure

    worker = component_methods(
        "actor_fwd.py",
        "ActorFwd",
        ["compute_actor_log_prob"],
        ray=SimpleNamespace(get=lambda value: value),
        wait_all=wait_all,
        UnsafeInferenceError=UnsafeInferenceError,
    )
    worker._recovery = SimpleNamespace(state=SimpleNamespace(set_fwd_ready=remote(events.append)))
    worker.actor_model = SimpleNamespace(
        async_compute_actor_log_prob=lambda step: "compute",
        async_inference_call=lambda method, step: "recv",
    )
    with pytest.raises(UnsafeInferenceError if failed_phase == "recv" else ValueError) as raised:
        worker.compute_actor_log_prob(2)
    if failed_phase == "recv":
        assert raised.value.__cause__ is failure
        assert events == [True, False]
    else:
        assert raised.value is failure and events == []


@pytest.mark.parametrize("enabled", [False, True])
def test_hybrid_training_waits_for_all_ranks_when_recovery_enabled(enabled):
    wait, get = Mock(), Mock()
    actor = component_methods("actor.py", "Actor", ["_execute_training"], wait_all=wait, ray=SimpleNamespace(get=get))
    actor.config = SimpleNamespace(hybrid=True, fully_async=True, num_critic_only_steps=0)
    actor.step = 4
    actor._recovery = SimpleNamespace(state=object() if enabled else None)
    refs = ["rank0", "rank1"]
    actor.actor_model = SimpleNamespace(train_hybrid=Mock(return_value=refs))
    assert actor._execute_training()
    (wait if enabled else get).assert_called_once_with(refs)
    (get if enabled else wait).assert_not_called()

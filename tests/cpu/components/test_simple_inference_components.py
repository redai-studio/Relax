# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute component methods without importing GPU/Serve startup
dependencies."""

import ast
import asyncio
import threading
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
def test_actor_commits_next_step_and_clears_partition_before_leaving_operation(hybrid, monkeypatch):
    from relax.components.actor import Actor

    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    monkeypatch.setattr("relax.utils.tq.lifecycle.detach_tq_client", Mock())
    events = []
    cls = Actor.func_or_class.__mro__[1]
    monkeypatch.setitem(
        cls._background_run.__globals__, "build_async_train_data_fields", lambda config: ["advantages"]
    )
    monkeypatch.setitem(cls._background_run.__globals__, "build_data_fields", lambda config, **kwargs: ["tokens"])
    actor = object.__new__(cls)
    actor.config = SimpleNamespace(num_rollout=1, fully_async=True, hybrid=hybrid, use_health_check=True)
    actor.step = 0
    actor._pending_sft_clear = None
    actor._lock = threading.RLock()
    actor._stop_event = threading.Event()
    actor._logger_instance = Mock()
    actor._recovery = Gate(events)
    admit = actor._recovery.admit

    def check_fields(step, client, fields, stop):
        assert fields == (["tokens"] if hybrid else ["advantages"])
        return admit(step)

    actor._recovery.admit = check_fields
    actor._execute_training = lambda: events.append("train") or True
    actor.data_system_client = SimpleNamespace(
        async_clear_partition=AsyncMock(side_effect=lambda **kw: events.append(("clear", kw["partition_id"])))
    )
    actor.healthy = Mock()
    actor._recovery.leave = lambda: events.append(("leave", actor.step))

    actor._background_run()

    assert events == ["admit", "train", ("clear", "train_0"), ("leave", 1)]


def test_actor_failure_pauses_before_leaving_and_is_not_swallowed_by_health_check(monkeypatch):
    from relax.components.actor import Actor

    events = []
    monkeypatch.setattr("relax.utils.tq.lifecycle.detach_tq_client", Mock())
    report_ack = Mock()
    cls = Actor.func_or_class.__mro__[1]
    monkeypatch.setitem(cls._background_run.__globals__, "ray", SimpleNamespace(get=report_ack))
    monkeypatch.setitem(
        cls._background_run.__globals__, "build_async_train_data_fields", lambda config: ["advantages"]
    )
    actor = object.__new__(cls)
    actor.config = SimpleNamespace(num_rollout=1, fully_async=True, use_health_check=True)
    actor.step = 0
    actor._pending_sft_clear = None
    actor._lock = threading.RLock()
    actor._stop_event = threading.Event()
    actor._stop_event.wait = Mock()
    actor._logger_instance = Mock()
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
        ["compute_actor_log_prob", "_receive_after_forward"],
        ray=SimpleNamespace(get=lambda value: value),
        wait_all=wait_all,
    )
    worker.role = "actor_fwd"
    worker._recovery = SimpleNamespace(state=SimpleNamespace(set_fwd_ready=remote(set_ready)), wait=wait_all)
    worker.actor_model = SimpleNamespace(
        async_compute_actor_log_prob=lambda step, **options: ["compute-rank0", "compute-rank1"],
        async_inference_call=lambda method, step: [f"{method}-rank0", f"{method}-rank1"],
    )

    worker.compute_actor_log_prob(2)

    assert events == [
        ("wait", ("compute-rank0", "compute-rank1"), False),
        ("ready", True),
        ("wait", ("recv_weight_fully_async-rank0", "recv_weight_fully_async-rank1"), True),
        ("ready", False),
        ("wait", ("finish_forward_metrics-rank0", "finish_forward_metrics-rank1"), False),
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
        async_inference_call=lambda method: method, get_recovery_children=lambda: ["rank0", "rank1", "lock"]
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
        ["compute_actor_log_prob", "_receive_after_forward"],
        ray=SimpleNamespace(get=lambda value: value),
        wait_all=wait_all,
        UnsafeInferenceError=UnsafeInferenceError,
    )
    worker.role = "actor_fwd"
    worker._recovery = SimpleNamespace(state=SimpleNamespace(set_fwd_ready=remote(events.append)), wait=wait_all)
    worker.actor_model = SimpleNamespace(
        async_compute_actor_log_prob=lambda step, **options: "compute",
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
def test_hybrid_training_uses_the_same_stages_with_or_without_recovery(enabled):
    wait, get = Mock(), Mock()
    actor = component_methods("actor.py", "Actor", ["_execute_training"], ray=SimpleNamespace(get=get))
    actor.config = SimpleNamespace(
        hybrid=True,
        fully_async=True,
        num_critic_only_steps=0,
        train_backend="megatron",
    )
    actor.step = 4
    actor._recovery = SimpleNamespace(
        state=object() if enabled else None, wait=wait, synchronize=Mock(return_value=True)
    )
    refs = ["rank0", "rank1"]
    actor.actor_model = SimpleNamespace(train_hybrid=Mock(return_value=refs))
    assert actor._execute_training()
    wait.assert_called_once_with(refs)
    actor._recovery.synchronize.assert_called_once()
    actor.actor_model.train_hybrid.assert_called_once_with(4)
    get.assert_not_called()


@pytest.mark.parametrize(
    "mode,outcome",
    [
        ("colocate", "success"),
        ("hybrid", "success"),
        ("fully_async", "success"),
        ("fully_async", "paused"),
        ("colocate", "tail_error"),
        ("hybrid", "training_error"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_component_owns_sync_after_all_training_ranks_settle(monkeypatch, mode, outcome, enabled):
    from relax.components import actor as actor_module
    from relax.components import recovery
    from relax.distributed.ray import inference_sync
    from relax.distributed.ray.actor_group import RayTrainGroup

    events = []
    state = SimpleNamespace(
        begin_sync=remote(lambda: events.append("reserve") or outcome != "paused"),
        snapshot=remote(lambda: {"paused": True}),
        end_sync=remote(lambda: events.append("release")),
    )
    if outcome == "paused" and not enabled:
        outcome = "success"
    gate = recovery.RecoveryGate(SimpleNamespace(inference_recovery_state=state if enabled else None), "actor")

    def settle(refs):
        (event,) = refs
        events.append(event)
        if (event == "tail" and outcome == "tail_error") or (event == "training" and outcome == "training_error"):
            raise RuntimeError("rank failed")

    monkeypatch.setattr(recovery.ray, "get", lambda value, **kw: settle(value) if isinstance(value, list) else value)
    monkeypatch.setattr(recovery, "wait_all", Mock())
    monkeypatch.setattr(inference_sync, "wait_all", settle)
    monkeypatch.setattr(inference_sync, "_evaluate_rollout", lambda *args, **kwargs: events.append("evaluate"))
    monkeypatch.setattr(inference_sync, "wait_for_previous_eval", Mock())
    monkeypatch.setattr(inference_sync, "check_services_health", Mock(return_value=(True, False)))
    execute = actor_module.Actor.func_or_class._execute_training
    rank = Mock()
    train = {"colocate": rank.train, "hybrid": rank.train_hybrid, "fully_async": rank.train_async}[mode]
    train.remote.return_value = "training"

    def send_weights(rollout_id, rollout_only=False, actor_fwd_only=False):
        assert (rollout_id, rollout_only, actor_fwd_only) == (4, True, False)
        return "tail"

    if mode == "fully_async":
        rank.update_weights_fully_async.remote.side_effect = send_weights
    else:
        rank.update_weights.remote.return_value = "tail"
    rank.record_inference_wait.remote.return_value = "wait_recorded"
    rank.finish_training_metrics.remote.return_value = "flushed"
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [rank]
    actor = SimpleNamespace(
        config=SimpleNamespace(
            hybrid=mode == "hybrid",
            fully_async=mode != "colocate",
            num_critic_only_steps=0,
            num_rollout=5,
            train_backend="megatron",
            inference_recovery=enabled,
            debug_train_only=False,
            debug_rollout_only=False,
        ),
        step=4,
        actor_model=group,
        rollout_manager=object(),
        _managed_lora_publication=False,
        _recovery=gate,
        _maybe_save_model=lambda: events.append("save"),
    )
    if outcome.endswith("error"):
        with pytest.raises(RuntimeError, match="rank failed"):
            execute(actor)
    else:
        assert execute(actor)
    expected = ["training"]
    if outcome != "training_error":
        if enabled:
            expected += ["reserve"]
        if outcome != "paused":
            if mode != "colocate":
                expected += ["wait_recorded"]
            expected += ["tail"]
            if outcome == "success":
                if mode == "colocate":
                    expected += ["flushed"]
                expected += ["evaluate"]
                if mode != "colocate":
                    expected += ["flushed"]
                if enabled:
                    expected += ["release"]
        else:
            expected += ["flushed"]
        if mode == "fully_async" and outcome in {"success", "paused"}:
            expected += ["save"]
    assert events == expected
    train.remote.assert_called_once_with(4)


@pytest.mark.parametrize(
    "failure", ["can_do_update_weight_for_async", "get_step", "end_update_weight", "pending_rank"]
)
def test_failed_weight_sync_does_not_commit_training_step(monkeypatch, failure):
    from requests.exceptions import ConnectionError

    from relax.components import actor as actor_module
    from relax.components import recovery
    from relax.distributed.ray import inference_sync
    from relax.distributed.ray.actor_group import RayTrainGroup

    state = recovery.InferenceRecoveryState.__ray_metadata__.modified_class(has_actor_fwd=True)
    state.set_fwd_ready(True)
    actor = Mock(step=4, _lock=threading.RLock())
    actor.config = SimpleNamespace(
        hybrid=False,
        fully_async=True,
        train_backend="megatron",
        num_critic_only_steps=0,
        num_rollout=6,
        debug_train_only=False,
        debug_rollout_only=False,
        inference_recovery=True,
        use_fault_tolerance=False,
        rollout_http_timeout=2,
        inference_recovery_state=SimpleNamespace(begin_sync=remote(state.begin_sync), end_sync=remote(state.end_sync)),
    )
    actor._managed_lora_publication = False
    actor._recovery = recovery.RecoveryGate(actor.config, "actor")
    actor_cls = actor_module.Actor.func_or_class
    actor._execute_training = lambda: actor_cls._execute_training(actor)
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "0")
    actor.data_system_client.async_clear_partition = AsyncMock()
    monkeypatch.setattr(recovery.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(inference_sync, "get_serve_url", lambda role: f"http://{role}")

    def wait(refs, *, num_returns, timeout):
        assert num_returns == len(refs) == 2 and 0 < timeout < float("inf")
        return (refs[:1], refs[1:]) if failure == "pending_rank" else (refs, [])

    actor.actor_model = object.__new__(RayTrainGroup)
    actor.actor_model._actor_handlers = [Mock(), Mock()]
    monkeypatch.setattr(recovery.ray, "wait", wait)

    def request(url, **kwargs):
        if url.endswith("/" + failure):
            raise ConnectionError(failure)
        return Mock(json=Mock(return_value=True))

    monkeypatch.setattr(inference_sync.requests, "get", request)
    monkeypatch.setattr(inference_sync, "wait_for_previous_eval", Mock())
    error, message = (
        (recovery.UnsafeInferenceError, "still running") if failure == "pending_rank" else (ConnectionError, failure)
    )
    with pytest.raises(error, match=message):
        actor_cls._run_training_step(actor, 4)
    assert actor.step == 4
    actor.data_system_client.async_clear_partition.assert_not_awaited()
    assert state.snapshot()["sync"]
    with pytest.raises(RuntimeError, match="requires global restart"):
        state.request_recovery("actor_fwd")


@pytest.mark.parametrize("paused", [False, True])
def test_rollout_replacement_reselects_router_endpoint(monkeypatch, paused):
    import copy

    from relax.components import rollout as rollout_module

    driver_config = SimpleNamespace(sglang_router_ip=None, sglang_router_port=31000, tq_config={})
    config = copy.deepcopy(driver_config)
    state = SimpleNamespace(snapshot=remote(lambda: {"paused": paused}))
    # Serve copies the ingress class; patch its external boundaries directly.
    component_type = rollout_module.Rollout.func_or_class.__mro__[1]
    bindings = component_type.__init__.__globals__
    monkeypatch.setitem(bindings, "RecoveryGate", lambda *_args: SimpleNamespace(state=state))
    monkeypatch.setitem(bindings, "ray", SimpleNamespace(get=lambda value, **kwargs: value))
    monkeypatch.setitem(bindings, "attach_tq_client", lambda *_args, **kwargs: None)
    manager = Mock(return_value=(object(), 1))
    monkeypatch.setitem(bindings, "create_rollout_manager", manager)
    component = component_type(None, None, config)

    assert manager.call_args.args[0].sglang_router_port == (None if paused else 31000)
    assert component.config.sglang_router_ip is None
    assert driver_config.sglang_router_port == 31000


@pytest.mark.asyncio
@pytest.mark.parametrize("discard_fails", [False, True])
async def test_rollout_discard_window_waits_for_its_data_source(monkeypatch, discard_fails):
    from relax.components import rollout as rollout_module

    discard, reset = AsyncMock(), AsyncMock()
    if discard_fails:
        discard.side_effect = RuntimeError("unsettled pending data")
    component = SimpleNamespace(
        data_source=SimpleNamespace(discard_pending=remote(discard)),
        rollout_manager=SimpleNamespace(reset_data_window=remote(reset)),
    )
    operation = rollout_module.Rollout.func_or_class.discard_recovery_window
    if discard_fails:
        with pytest.raises(RuntimeError, match="unsettled pending data"):
            await operation(component)
        reset.assert_not_awaited()
    else:
        await operation(component)
        discard.assert_awaited_once_with()
        reset.assert_awaited_once_with()


def test_fsdp_component_does_not_dispatch_megatron_tail():
    actor = component_methods("actor.py", "Actor", ["_execute_training"])
    actor.config = SimpleNamespace(
        train_backend="fsdp",
        hybrid=False,
        fully_async=False,
        num_critic_only_steps=0,
    )
    actor.step = 2
    actor.actor_model = SimpleNamespace(async_train=Mock(return_value=["fsdp-step"]))
    actor._recovery = SimpleNamespace(wait=Mock(), synchronize=Mock())
    assert actor._execute_training()
    actor.actor_model.async_train.assert_called_once_with(2)
    actor._recovery.wait.assert_called_once_with(["fsdp-step"])
    actor._recovery.synchronize.assert_not_called()


@pytest.mark.parametrize("role", ["actor_fwd", "reference"])
@pytest.mark.parametrize("enabled", [False, True])
def test_forward_roles_receive_once_and_only_actor_fwd_sets_readiness(role, enabled):
    events = []
    worker = component_methods(
        "actor_fwd.py",
        "ActorFwd",
        ["compute_actor_log_prob", "compute_ref_log_prob", "_receive_after_forward"],
        ray=SimpleNamespace(get=lambda value: value),
    )
    worker.role = role
    state = SimpleNamespace(set_fwd_ready=remote(lambda value: events.append(("ready", value)))) if enabled else None
    worker._recovery = SimpleNamespace(state=state, wait=lambda refs: events.append(refs))
    worker.actor_model = SimpleNamespace(
        async_compute_actor_log_prob=lambda step: ("compute", step),
        async_compute_ref_log_prob=lambda step: ("compute", step),
        async_inference_call=lambda method, step: (method, step),
    )
    method = worker.compute_actor_log_prob if role == "actor_fwd" else worker.compute_ref_log_prob
    method(5)
    expected = [("compute", 5)]
    if enabled and role == "actor_fwd":
        expected += [("ready", True)]
    expected += [("recv_weight_fully_async", 5)]
    if enabled and role == "actor_fwd":
        expected += [("ready", False)]
    assert events == expected + [("finish_forward_metrics", 5)]


def test_healthy_step_can_outlast_recovery_deadline(monkeypatch):
    from relax.components import recovery

    gate = recovery.RecoveryGate(SimpleNamespace(inference_recovery_state=object()), "actor")
    refs = ["fast", "slow"]

    # The slow rank finishes after the recovery settlement deadline.
    def wait(refs, *, num_returns, timeout=None):
        return (refs, []) if timeout is None else (refs[:1], refs[1:])

    def get(ref):
        return [f"{rank} result" for rank in ref] if isinstance(ref, list) else f"{ref} result"

    monkeypatch.setattr(recovery.ray, "wait", wait)
    monkeypatch.setattr(recovery.ray, "get", get)
    assert gate.wait(refs) == ["fast result", "slow result"]
    with pytest.raises(recovery.UnsafeInferenceError, match="still running"):
        gate.wait_for_maintenance(refs)


def test_failed_step_with_unsettled_peer_requires_global_restart(monkeypatch):
    from relax.components import recovery

    gate = recovery.RecoveryGate(SimpleNamespace(inference_recovery_state=object()), "actor")
    monkeypatch.setattr(recovery.ray, "get", Mock(side_effect=ValueError("rank failed")))

    def wait(refs, *, num_returns, timeout):
        assert 0 < timeout < float("inf")
        return refs[:1], refs[1:]

    monkeypatch.setattr(recovery.ray, "wait", wait)
    with pytest.raises(recovery.UnsafeInferenceError, match="still running"):
        gate.wait(["failed", "pending"])


@pytest.mark.parametrize("sync_step", [0, 7])
def test_initial_and_recovery_sync_use_one_explicit_step_for_sender_and_receiver(sync_step):
    sender = component_methods("actor.py", "Actor", ["update_weights_fully_async"])
    receiver = component_methods("actor_fwd.py", "ActorFwd", ["recv_weight_fully_async"])
    sender.step, receiver.step = 5, 0
    receiver.role = "reference"
    for service in (sender, receiver):
        service.actor_model = SimpleNamespace(async_inference_call=Mock(return_value=[]))
        service._recovery = SimpleNamespace(wait_for_maintenance=lambda refs: refs)
    options = {"sync_step": sync_step} if sync_step else {}
    sender.update_weights_fully_async(**options)
    receiver.recv_weight_fully_async(**options)
    assert sender.actor_model.async_inference_call.call_args.args[:2] == ("update_weights_fully_async", sync_step)
    receiver.actor_model.async_inference_call.assert_called_once_with("recv_weight_fully_async", sync_step)

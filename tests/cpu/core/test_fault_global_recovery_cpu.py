# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Production startup/restart control flow; models and transports are CPU
doubles."""

import ast
import asyncio
import socket
import sys
import threading
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils import health_system
from relax.utils.fault_events.models import EventDraft, new_id
from relax.utils.fault_events.query_server import FaultQueryServer
from relax.utils.fault_events.receipts import ReceiptInbox
from relax.utils.fault_events.recovery import GlobalStartup, expected_evidence, global_plan
from relax.utils.fault_events.runtime import FaultEventRuntime
from tests.cpu.core.test_fault_local_recovery_cpu import ROOT, Roles, make_controller, make_service, observe, records
from tests.cpu.core.test_fault_local_recovery_cpu import cpu_algorithm_registry as _cpu_registry
from tests.cpu.utils.test_fault_health_flow import HealthStatusImpl, LocalStatusRPC


cpu_algorithm_registry = _cpu_registry


def test_sparse_configuration_uses_the_same_plan_defaults():
    roles = ("actor", "actor_fwd")
    assert global_plan(Namespace(), roles).expected == expected_evidence("global", Namespace(), roles) == 7


@pytest.mark.parametrize("stage,reason", [("weight_sync", None), ("step_restore", "step_restore_failed")])
def test_cancelled_startup_stage_keeps_classification_and_resets_on_retry(runtime, stage, reason):
    controller = startup_controller(runtime, (Roles.actor,), [])
    recovery = prepare_handoff(controller, runtime)
    startup = GlobalStartup(runtime, recovery, initialized=True)
    with pytest.raises(asyncio.CancelledError):
        with getattr(startup, stage)():
            raise asyncio.CancelledError()
    startup.fail(False, recovery)
    assert recovery.outcome == ("recovery_failed" if reason else None)
    if reason:
        assert records(runtime)[-1]["reason_code"] == reason
    else:
        assert recovery.observed == 1 and not recovery.failed

    # A later successful stage must not retain the cancellation category.
    with startup.weight_sync():
        pass
    startup.fail(False, recovery)
    if not reason:
        assert records(runtime)[-1]["reason_code"] == "handoff_exception"


@pytest.fixture
def runtime():
    value = FaultEventRuntime()
    value.bind_source()
    yield value
    value.close()
    value._consumer.join(3)
    assert not value._consumer.is_alive()


class StartupService:
    """Replace model/RPC work, leaving Controller's full startup body
    intact."""

    def __init__(self, role, trace, *, fail=None):
        self.role, self.trace, self.fail = role, trace, fail
        self._task_ref = None
        self.handle = object()

    async def set_rollout_manager(self, manager):
        self.trace.append((self.role, "rollout_manager"))
        if self.fail == "wiring":
            raise RuntimeError("injected manager failure")

    async def get_rollout_manager(self):
        return object()

    async def set_barriers(self, **kwargs):
        self.trace.append((self.role, "barriers", tuple(kwargs)))

    async def update_weights_fully_async(self, **kwargs):
        self.trace.append((self.role, "send"))
        if self.fail == "weight":
            raise RuntimeError("injected weight response failure")
        return object()

    async def recv_weight_fully_async(self, **kwargs):
        self.trace.append((self.role, "recv"))
        return object()

    async def get_step(self):
        self.trace.append((self.role, "get_step"))
        return 11

    async def set_step(self, step):
        self.trace.append((self.role, "set_step", step))
        if self.fail == "step":
            raise RuntimeError("injected step failure")

    def run(self):
        self.trace.append((self.role, "run"))
        if self.fail == "submit":
            raise RuntimeError("injected task submission failure")
        if self.fail == "no_task":
            return None

        async def completed_cpu_task():
            return None

        self._task_ref = completed_cpu_task()
        return self._task_ref


def startup_controller(
    runtime, roles, trace, *, fully_async=True, hybrid=False, debug=None, loss_type=None, failure=None
):
    services = {role: StartupService(role, trace, fail=failure if role == Roles.actor else None) for role in roles}
    controller, _ = make_controller(runtime, services, trace, fully_async=fully_async, debug=debug == "train")
    controller.config.hybrid = hybrid
    controller.config.debug_rollout_only = debug == "rollout"
    controller.config.loss_type = loss_type
    controller._teacher_manager = None
    controller._cleanup_s3_model_weights_after_init = lambda: None
    namespace = type(controller).training_loop.__globals__

    async def no_teacher(*args):
        return None

    namespace.update(
        asyncio=asyncio,
        run=asyncio.run,
        _needs_rollout_manager_setup=lambda services: Roles.rollout in services,
        _is_colocate=lambda config: not config.fully_async and not config.hybrid,
        set_managed_opd_teacher_on_actor_service=no_teacher,
        RolloutOffloadBarrier=lambda *a, **k: object(),
        PeerStepBarrier=lambda *a, **k: object(),
    )
    return controller


def prepare_handoff(controller, runtime):
    observe(runtime, (next(iter(controller.serve_dict)),))
    plan = global_plan(controller.config, controller.serve_dict)
    recovery = runtime.start_recovery(next(iter(controller.serve_dict)), "global", plan.expected, plan=plan)
    runtime.bind_source(recovery)
    controller._fault_recovery = recovery
    controller._fault_rebuild_ready = recovery  # explicit CPU initialization seam
    controller._restart_mode = "global"
    controller._restart_done_event.set()
    controller._restarting = True
    namespace = type(controller).training_loop.__globals__
    calls = 0

    def run(coroutine):
        nonlocal calls
        calls += 1
        if calls == 1:
            coroutine.close()
            raise RuntimeError("old tasks cancelled by global restart")
        return asyncio.run(coroutine)

    namespace["run"] = run
    return recovery


# Explicit RFC matrix. Each role graph executes the real startup method;
# expectations below are not obtained from the production plan calculator.
MATRIX = [
    (False, False, None, (Roles.actor,), 7),
    (False, False, None, (Roles.actor, Roles.rollout), 7),
    (False, False, None, (Roles.actor, Roles.rollout, Roles.critic), 7),
    (True, True, None, (Roles.actor,), 7),
    (True, True, None, (Roles.actor, Roles.rollout, Roles.critic), 7),
    (True, False, None, (Roles.actor,), 15),
    (True, False, None, (Roles.actor, Roles.actor_fwd), 31),
    (True, False, None, (Roles.actor, Roles.reference), 47),
    (True, False, None, (Roles.actor, Roles.actor_fwd, Roles.reference), 63),
    (True, False, None, (Roles.actor_fwd, Roles.reference), 5),
    (False, False, None, (Roles.rollout,), 5),
    (True, True, None, (Roles.actor_fwd,), 5),
    *[
        (a, h, debug, (Roles.actor, Roles.actor_fwd, Roles.reference), 5)
        for a, h in ((False, False), (True, True), (True, False))
        for debug in ("train", "rollout")
    ],
]


@pytest.mark.parametrize("fully_async,hybrid,debug,roles,expected", MATRIX)
def test_actual_startup_mode_and_role_matrix(runtime, fully_async, hybrid, debug, roles, expected):
    trace, baseline_trace = [], []
    controller = startup_controller(runtime, roles, trace, fully_async=fully_async, hybrid=hybrid, debug=debug)
    baseline = startup_controller(None, roles, baseline_trace, fully_async=fully_async, hybrid=hybrid, debug=debug)
    baseline.training_loop()
    recovery = prepare_handoff(controller, runtime)
    controller.training_loop()
    assert trace == baseline_trace
    assert recovery.expected == expected
    assert recovery.observed == (5 if debug or Roles.actor not in roles else 7)
    assert recovery.handoff
    assert recovery.outcome == (None if expected & 56 else "recovery_succeeded")
    steps = [(role, "set_step", 11) for role in roles]
    assert [item for item in trace if item[1] == "set_step"] == ([] if debug or Roles.actor not in roles else steps)
    weight_ops = [item for item in trace if item[1] in ("send", "recv")]
    if fully_async and not hybrid and not debug and Roles.actor in roles:
        assert weight_ops == [(Roles.actor, "send")] + [
            (r, "recv") for r in roles if r in (Roles.actor_fwd, Roles.reference)
        ]
    else:
        assert not weight_ops
    if expected & 56:
        inbox = ReceiptInbox(runtime.token(recovery))
        for operation, bit in (("actor_send", 8), ("actor_fwd_recv", 16), ("reference_recv", 32)):
            if expected & bit:
                inbox.receive(runtime.token(recovery), operation, "succeeded")
        runtime.observe_snapshot(
            {
                "roles": [],
                "source_generation": 1,
                "faults": [],
                "observation_error": False,
                "receipts": inbox.snapshot(),
            }
        )
        assert recovery.outcome == "recovery_succeeded"
    assert records(runtime)[-1]["observed_evidence"] == expected


def test_failed_weight_response_waits_for_operation_receipt(runtime):
    controller = startup_controller(runtime, (Roles.actor,), [], failure="weight")
    recovery = prepare_handoff(controller, runtime)
    with pytest.raises(RuntimeError, match="weight response failure"):
        controller.training_loop()
    assert recovery.outcome is None and not recovery.handoff
    inbox = ReceiptInbox(runtime.token(recovery))
    inbox.receive(runtime.token(recovery), "actor_send", "failed")
    runtime.observe_snapshot(
        {
            "roles": [],
            "source_generation": runtime.generation,
            "faults": [],
            "observation_error": False,
            "receipts": inbox.snapshot(),
        }
    )
    terminal = records(runtime)[-1]
    assert terminal["event_type"] == "recovery_failed" and terminal["reason_code"] == "operation_failed"
    assert terminal["failed_evidence"] == 8 and terminal["observed_evidence"] == 1


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("wiring", "handoff_exception"),
        ("step", "step_restore_failed"),
        ("submit", "handoff_exception"),
        ("no_task", None),
    ],
)
def test_actual_startup_failure_never_becomes_success(runtime, failure, reason):
    controller = startup_controller(runtime, (Roles.actor, Roles.rollout), [], fully_async=False, failure=failure)
    recovery = prepare_handoff(controller, runtime)
    if reason:
        with pytest.raises(RuntimeError):
            controller.training_loop()
        terminal = records(runtime)[-1]
        assert terminal["event_type"] == "recovery_failed" and terminal["reason_code"] == reason
        assert terminal["observed_evidence"] == (3 if failure == "submit" else 1)
        if failure == "step":
            assert terminal["failed_evidence"] == 2
    else:
        # rollout submitted, actor did not; the partial task set cannot prove bit 4.
        controller.training_loop()
        assert recovery.observed == 3 and recovery.outcome is None
        assert "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]


def test_actual_startup_same_mask_role_drift_is_unverified(runtime):
    controller = startup_controller(runtime, (Roles.actor, Roles.critic), [], fully_async=False)
    recovery = prepare_handoff(controller, runtime)
    controller.serve_dict.pop(Roles.critic)
    controller.serve_dict[Roles.rollout] = StartupService(Roles.rollout, [])
    controller.training_loop()
    assert not recovery.handoff and recovery.outcome is None and recovery.expected == 7
    assert "OBSERVATION_ERROR" in runtime.diagnostics()["unknown_reasons"]


def test_training_thread_copies_attempt_before_ack_can_replace_shared_fields(runtime):
    controller = startup_controller(runtime, (Roles.actor,), [], fully_async=False)
    recovery = prepare_handoff(controller, runtime)

    class Ack:
        def set(self):
            controller._fault_recovery = None
            controller._fault_rebuild_ready = None

    controller._restart_consumed_event = Ack()
    controller.training_loop()
    assert recovery.outcome == "recovery_succeeded"


@pytest.mark.parametrize("stage", ["wiring", "step"])
def test_actual_startup_cancelled_by_next_restart_is_interrupted_not_failed(runtime, stage):
    controller = startup_controller(runtime, (Roles.actor, Roles.rollout), [], fully_async=False)
    first = prepare_handoff(controller, runtime)

    async def replaced(*args):
        observe(runtime, (Roles.actor,))
        second = runtime.start_recovery("actor", "global", 7, plan=first.plan)
        assert first.outcome is None  # R2 starting alone must not end R1.
        controller._fault_recovery = second
        controller._restarting = True
        controller._restart_mode = "global"
        controller._restart_error = RuntimeError("second rebuild failed")
        controller._restart_done_event.set()
        raise RuntimeError("old replica was deleted by R2")

    service = controller.serve_dict[Roles.actor]
    if stage == "wiring":
        service.set_rollout_manager = replaced
    else:
        service.get_step = replaced
    with pytest.raises(RuntimeError, match="global restart failed"):
        controller.training_loop()
    terminal = [e for e in records(runtime) if e["recovery_id"] == first.recovery_id][-1]
    assert terminal["event_type"] == "recovery_interrupted" and terminal["reason_code"] == "superseded"


def test_actual_error_return_flows_through_restart_consume_and_training_loop(runtime, monkeypatch):
    controller = startup_controller(runtime, (Roles.actor,), [])
    observe(runtime, (Roles.actor,))
    controller.runtime_env = {}
    controller.config.use_agentic_rollout = False
    controller._autoscaler_config = None
    controller._health_manager.stop = lambda **kwargs: None
    controller._cancel_pending_tasks = lambda: None
    controller._dispose_rollout_manager = lambda: True
    namespace = type(controller).training_loop.__globals__
    namespace.update(
        ray=SimpleNamespace(shutdown=lambda: None, init=lambda **kwargs: None),
        serve=SimpleNamespace(delete=lambda role: None, shutdown=lambda: None, start=lambda **kwargs: None),
        recovery_load_path=lambda config: None,
        tq=SimpleNamespace(close=lambda: None),
        shutdown_async_loop=lambda: None,
        time=SimpleNamespace(sleep=lambda seconds: None),
    )
    router = ModuleType("relax.distributed.ray.rollout")
    router.stop_launched_routers = lambda: 0
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", router)
    original = RuntimeError("real initializer failed inside production restart")

    def fail_init(*args, **kwargs):
        raise original

    controller.__init__ = fail_init

    def interrupted(coroutine):
        coroutine.close()
        # Executes actual restart_serve and _run_global_restart. The latter
        # catches initializer failure, stores the error, and returns normally.
        controller.restart_serve(Roles.actor)
        raise RuntimeError("old task failed")

    namespace["run"] = interrupted
    with pytest.raises(RuntimeError, match="global restart failed") as exc:
        controller.training_loop()
    assert exc.value.__cause__ is original and controller._restart_error is original
    assert not controller._restarting and controller._restart_consumed_event.is_set()
    events = records(runtime)
    assert [e["event_type"] for e in events] == ["fault_detected", "handling_started", "recovery_failed"]
    assert events[-1]["reason_code"] == "restart_exception"


@pytest.mark.parametrize("debug", [False, True])
def test_actual_local_debug_still_requires_all_three_bits(runtime, debug):
    observe(runtime)
    trace = []
    controller, _ = make_controller(
        runtime, {Roles.advantages: make_service(trace, failure="step")}, trace, debug=debug
    )
    controller.restart_serve(Roles.advantages)
    terminal = records(runtime)[-1]
    assert terminal["expected_evidence"] == 7 and terminal["failed_evidence"] == 2


@pytest.mark.parametrize("loss_type,sync", [(None, True), ("sft", False), ("dpo", False), ("rm", False)])
def test_plan_tracks_only_actual_conditional_sync_and_ignores_unrelated_config(runtime, loss_type, sync):
    controller = startup_controller(runtime, (Roles.actor, Roles.rollout), [], fully_async=False, loss_type=loss_type)
    plan = global_plan(controller.config, controller.serve_dict)
    assert plan.sync_weights is sync
    controller.config.coordinator_url = "changed-by-rebuild"
    controller.config.start_rollout_id = 100
    assert global_plan(controller.config, controller.serve_dict) == plan


@pytest.mark.parametrize("failure", ["admission_lock", "queue_full", "store_uncertain", "http_port"])
def test_actual_startup_business_trace_survives_observation_failures(runtime, failure, monkeypatch):
    trace, baseline_trace = [], []
    controller = startup_controller(runtime, (Roles.actor,), trace, fully_async=False)
    baseline = startup_controller(None, (Roles.actor,), baseline_trace, fully_async=False)
    baseline.training_loop()
    recovery = prepare_handoff(controller, runtime)
    release, entered = threading.Event(), threading.Event()
    append = runtime.store._append_committed
    occupied = None

    def filler():
        return EventDraft(
            runtime.run_id, new_id(), new_id(), "critic", runtime.generation, "fault_detected", "reported_error"
        )

    def blocked(draft):
        entered.set()
        assert release.wait(5)
        if failure == "store_uncertain":
            raise RuntimeError("uncertain commit with sensitive payload")
        return append(draft)

    try:
        if failure in ("queue_full", "store_uncertain"):
            records(runtime)
            monkeypatch.setattr(runtime.store, "_append_committed", blocked)
            assert runtime.record(filler())
            assert entered.wait(2)
            if failure == "queue_full":
                runtime.queue_capacity = 2
                assert runtime.record(filler()) and runtime.record(filler())
            else:
                release.set()
                assert runtime._disabled.wait(2)
        elif failure == "http_port":
            occupied = socket.socket()
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            unavailable = FaultQueryServer(runtime, occupied.getsockname()[1])
            assert not unavailable.available and unavailable.url is None
            unavailable.close()
        if failure == "admission_lock":
            with runtime._lock:
                controller.training_loop()
        else:
            controller.training_loop()
        assert trace == baseline_trace and controller._restart_error is None
        assert not controller._pending_task_refs
        if failure == "admission_lock":
            assert not recovery.handoff and runtime.diagnostics()["admission_rejections"] > 0
        elif failure == "queue_full":
            assert recovery.outcome == "recovery_succeeded"
            assert runtime.diagnostics()["known_lost_events"] >= 2
        elif failure == "store_uncertain":
            with pytest.raises(RuntimeError, match="UNAVAILABLE"):
                runtime.query()
            assert "STORE_COMMIT_UNCERTAIN" in runtime.diagnostics()["unknown_reasons"]
        else:
            assert recovery.outcome == "recovery_succeeded" and not runtime.diagnostics()["query_available"]
    finally:
        release.set()
        if occupied is not None:
            occupied.close()


def test_actual_checker_local_then_global_does_not_readopt_local_fault(runtime, monkeypatch):
    status = HealthStatusImpl(fault_generation=0)
    status.report_error("advantages", "first fault")
    status.report_error("actor", "second fault")
    rpc = LocalStatusRPC(status)
    monkeypatch.setattr(health_system.ray, "get", lambda result: result)
    trace = []
    controller, _ = make_controller(runtime, {Roles.advantages: make_service(trace), Roles.actor: object()}, trace)
    controller._health_manager.mark_healthy = status.mark_healthy
    controller._run_global_restart = lambda: None
    adopted = []

    def callback(role):
        controller.restart_serve(role)
        adopted.append(controller._fault_recovery)
        controller._consume_restart_cycle()
        if role == "actor":
            checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, callback, check_interval=0, fault_events=runtime)
    asyncio.run(checker._check_loop())
    local, global_ = adopted
    assert local.outcome == "recovery_succeeded"
    assert [fault.role for fault in local.members] == ["advantages"]
    assert [fault.role for fault in global_.members] == ["actor"]
    events = records(runtime)
    assert sum(e["event_type"] == "fault_detected" for e in events) == 2
    assert [e["role"] for e in events if e["recovery_id"] == global_.recovery_id] == ["actor"]


def test_actual_checker_merges_receipts_before_callback_destroys_receiver(runtime, monkeypatch):
    controller = startup_controller(runtime, (Roles.actor,), [])
    first = prepare_handoff(controller, runtime)
    runtime.complete_handoff(first, first.plan, 7)
    status = HealthStatusImpl(fault_generation=1, fault_seed=runtime.token(first))
    status.record_recovery_receipt(runtime.token(first), "actor_send", "succeeded")
    status.report_error("actor", "next fault")
    rpc = LocalStatusRPC(status)
    monkeypatch.setattr(health_system.ray, "get", lambda result: result)
    controller._restarting = False
    controller._restart_consumed_event.set()
    controller._run_global_restart = lambda: runtime.retire_source(1)

    def callback(role):
        assert first.outcome == "recovery_succeeded"
        controller.restart_serve(role)
        assert first.outcome == "recovery_succeeded"
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, callback, check_interval=0, fault_events=runtime)
    asyncio.run(checker._check_loop())
    assert controller._fault_recovery.receiver_generation == 2
    assert first.outcome == "recovery_succeeded"


@pytest.mark.parametrize("fully_async,hybrid", [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("loss_type", [None, "sft", "dpo", "rm"])
def test_actual_actor_setter_sync_matches_plan_condition(fully_async, hybrid, loss_type):
    from relax.engine.sft.runtime import is_offline_mode

    tree = ast.parse((ROOT / "relax/components/actor.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Actor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_rollout_manager")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [fn], type_ignores=[])
    namespace = {"is_offline_mode": is_offline_mode}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "relax/components/actor.py"), "exec"), namespace)
    calls = []
    worker = SimpleNamespace(
        set_rollout_manager=lambda m: calls.append("manager"), update_weights=lambda: calls.append("sync")
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(fully_async=fully_async, hybrid=hybrid, loss_type=loss_type), actor_model=worker
    )
    namespace[fn.name](owner, object())
    assert calls == (["manager", "sync"] if (not fully_async or hybrid) and loss_type is None else ["manager"])


def test_actual_empty_task_graph_keeps_submission_required_and_unverified(runtime):
    controller = startup_controller(runtime, (), [], fully_async=False)
    observe(runtime, (Roles.actor,))
    plan = global_plan(controller.config, controller.serve_dict)
    recovery = runtime.start_recovery("actor", "global", plan.expected, plan=plan)
    runtime.bind_source(recovery)
    controller._fault_recovery = controller._fault_rebuild_ready = recovery
    controller._restart_mode, controller._restarting = "global", True
    controller._restart_done_event.set()
    calls = 0

    def old_cancelled(coroutine):
        nonlocal calls
        calls += 1
        if calls == 1:
            coroutine.close()
            raise RuntimeError("old generation stopped")
        return asyncio.run(coroutine)

    type(controller).training_loop.__globals__["run"] = old_cancelled
    controller.training_loop()
    assert recovery.expected == 5 and recovery.observed == 1 and recovery.handoff
    assert recovery.outcome is None and "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]

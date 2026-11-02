# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import importlib
import threading
from argparse import Namespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from relax.components.background_loop import BackgroundLoopRunner


@pytest.fixture(params=[("actor", "Actor"), ("actor_fwd", "ActorFwd"), ("critic", "Critic")])
def component(request, monkeypatch):
    monkeypatch.setattr("relax.utils.tq.lifecycle.detach_tq_client", Mock())
    module_name, class_name = request.param
    module = importlib.import_module(f"relax.components.{module_name}")
    deployment_cls = getattr(module, class_name).func_or_class
    # Exercise the real component class beneath Serve's ASGI lifecycle wrapper.
    cls = next(cls for cls in deployment_cls.__mro__ if "run" in cls.__dict__)
    obj = object.__new__(cls)
    obj._loop_runner = BackgroundLoopRunner()
    obj._stop_event = threading.Event()
    obj._lock = threading.RLock()
    obj._logger_instance = Mock()
    obj.role = module_name
    obj.step = 0
    obj.config = Namespace(num_rollout=1, fully_async=True, loss_type="policy_loss", use_health_check=False)
    obj.healthy = Mock()
    obj.data_system_client = Mock()
    yield obj
    # Detach while this fixture still owns the process-global lifecycle mock.
    obj.__del__()


@pytest.mark.parametrize("health_check", [False, True])
@pytest.mark.parametrize("report_fails", [False, True])
async def test_component_concurrent_failure_preserves_health_policy(
    component, health_check, report_fails, monkeypatch
):
    component.config.use_health_check = health_check
    report_ack = threading.Event()
    if component.role == "actor":
        namespace = type(component)._background_run.__globals__
        monkeypatch.setitem(namespace, "ray", Mock(get=lambda ref, timeout: report_ack.set()))
    release = threading.Event()
    error = ValueError("compute failed")
    report_error = RuntimeError("health report failed")
    if report_fails:
        component.healthy.report_error.remote.side_effect = report_error

    def fail(*args):
        assert release.wait(5)
        raise error

    method = {"actor": "_execute_training", "actor_fwd": "compute_actor_log_prob", "critic": "train"}[component.role]
    worker = Mock(side_effect=fail)
    setattr(component, method, worker)
    tasks = [asyncio.create_task(component.run()) for _ in range(2)]
    try:
        await asyncio.sleep(0)
    finally:
        release.set()
    if component.role == "actor" and health_check and not report_fails:
        assert await asyncio.to_thread(report_ack.wait, 5)
        assert all(not task.done() for task in tasks)
        component._stop_event.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    expected = None if health_check else error
    if component.role == "actor":
        expected = error
    elif report_fails and component.role != "critic":
        expected = report_error
    assert all(result is expected for result in results)
    worker.assert_called_once()
    component.healthy.report_error.remote.assert_called_once()
    component.healthy.update_heartbeat.remote.assert_not_called()
    assert component.step == 0
    assert component.data_system_client.reset_consumption.call_count == (component.role != "critic")
    assert await component._loop_runner.join(1)


async def test_component_success_preserves_step_heartbeat_and_reset(component):
    if component.role == "actor":
        component._execute_training = Mock(return_value=False)
    elif component.role == "actor_fwd":
        component.compute_actor_log_prob = Mock()
    else:
        component.config.advantage_estimator = "grpo"
        component.critic_model = Mock()
    await component.run()
    assert component.step == 1
    component.healthy.update_heartbeat.remote.assert_called_once_with(component.role, 1)
    component.healthy.report_error.remote.assert_not_called()
    if component.role != "critic":
        component.data_system_client.reset_consumption.assert_called_once_with(
            partition_id="train_0", task_name="train_actor" if component.role == "actor" else "actor_log_probs"
        )
    assert await component._loop_runner.join(1)


@pytest.mark.parametrize("joins", [[True], [False, True], [False, False]])
async def test_component_stop_preserves_timeouts_and_logging(component, joins):
    if component.role == "critic":
        await component.stop()
        assert not component._stop_event.is_set()
        return  # Critic inherits the existing no-op stop contract.
    component._loop_runner.join = AsyncMock(side_effect=joins)
    await component.stop()
    assert component._stop_event.is_set()
    expected_calls = [call(timeout=5)]
    if component.role == "actor_fwd" and not joins[0]:
        expected_calls.append(call(timeout=600))
    assert component._loop_runner.join.call_args_list == expected_calls
    assert component._logger.warning.call_count == (component.role == "actor_fwd" and not joins[0])
    assert component._logger.error.call_count == (component.role == "actor_fwd" and joins == [False, False])


@pytest.mark.parametrize(
    "loss_type,partition,task",
    [
        ("policy_loss", "train_7", "train_actor"),
        ("sft", "sft_7", "sft_train"),
        ("dpo", "sft_7", "sft_train"),
        ("rm", "sft_7", "sft_train"),
    ],
)
@pytest.mark.parametrize("component", [("actor", "Actor")], indirect=True)
async def test_actor_run_preserves_partition_reset(component, loss_type, partition, task):
    component.config.loss_type = loss_type
    component.step = 7
    component.config.num_rollout = 7
    await component.run()
    component.data_system_client.reset_consumption.assert_called_once_with(partition_id=partition, task_name=task)
    assert await component._loop_runner.join(1)


@pytest.mark.parametrize("component", [("actor_fwd", "ActorFwd")], indirect=True)
async def test_reference_forward_run_keeps_reference_task_and_heartbeat(component):
    component.role = "ref"
    component.compute_ref_log_prob = Mock()
    await component.run()
    component.compute_ref_log_prob.assert_called_once_with(0)
    component.data_system_client.reset_consumption.assert_called_once_with(
        partition_id="train_0", task_name="ref_log_probs"
    )
    component.healthy.update_heartbeat.remote.assert_called_once_with("ref", 1)
    assert await component._loop_runner.join(1)


async def test_component_constructor_initializes_shared_runner(component, monkeypatch):
    # Serve ingress serializes the class; its functions can own copied globals.
    # Patch the bindings the deployed constructor actually resolves.
    namespace = type(component).__init__.__globals__
    train_group = Mock()
    train_group.init_and_wait.return_value = [0]
    monkeypatch.setitem(namespace, "allocate_train_group", Mock(return_value=train_group))
    monkeypatch.setitem(namespace, "attach_tq_client", Mock(return_value=Mock()))
    config = Namespace(
        tq_config={},
        kl_coef=0,
        use_kl_loss=False,
        opd_teacher_load=None,
        start_rollout_id=0,
        loss_type="policy_loss",
        num_rollout=0,
    )
    initialized = type(component)(healthy=Mock(), pgs=None, num_gpus=1, config=config, role=component.role)
    try:
        assert isinstance(initialized._loop_runner, BackgroundLoopRunner)
        await initialized.run()
        assert await initialized._loop_runner.join(1)
        assert initialized.step == 0
    finally:
        initialized.__del__()

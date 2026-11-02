# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only regression for the fully async startup weight synchronization
gate."""

import ast
import asyncio
import copy
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


ROOT = Path(__file__).resolve().parents[2]


def extract_method(relative_path, class_name, name, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[name]


class FakeDeploymentResponse:
    """A second awaitable level, matching Ray Serve DeploymentResponse."""

    def __init__(self, completion):
        self.completion = completion
        self.await_count = 0

    def __await__(self):
        self.await_count += 1
        return self.completion().__await__()


class InitialWeightSyncTests(unittest.TestCase):
    def setUp(self):
        self.update = extract_method("relax/core/service.py", "Service", "update_weights_fully_async", {})
        self.receive = extract_method("relax/core/service.py", "Service", "recv_weight_fully_async", {})

    def _startup(self, fail_role=None, include_receivers=True):
        roles = SimpleNamespace(
            actor="actor", rollout="rollout", actor_fwd="actor_fwd", reference="reference", critic="critic"
        )
        started, completed, responses = [], [], {}
        sender_started, receivers_started = asyncio.Event(), asyncio.Event()
        required_receivers = {"actor_fwd", "reference"} if include_receivers else set()

        async def sender():
            started.append("actor")
            sender_started.set()
            if required_receivers:
                await asyncio.wait_for(receivers_started.wait(), timeout=0.5)
            if fail_role == "actor":
                raise RuntimeError("sender partial-weight failure")
            completed.append("actor")

        async def receiver(role):
            started.append(role)
            if required_receivers.issubset(started):
                receivers_started.set()
            await asyncio.wait_for(sender_started.wait(), timeout=0.5)
            if fail_role == role:
                raise RuntimeError(f"{role} receive failure")
            completed.append(role)

        services = {}
        for role in ["actor", "rollout", *sorted(required_receivers)]:
            service = SimpleNamespace(role=role, run=MagicMock(return_value=None), set_step=AsyncMock())
            if role == "actor":
                response = FakeDeploymentResponse(sender)
                service.handle = SimpleNamespace(
                    update_weights_fully_async=SimpleNamespace(remote=lambda response=response, **_kwargs: response)
                )
                service.update_weights_fully_async = lambda service=service: self.update(service)
                service.get_step = AsyncMock(return_value=0)
                responses[role] = response
            elif role in required_receivers:
                response = FakeDeploymentResponse(lambda role=role: receiver(role))
                service.handle = SimpleNamespace(
                    recv_weight_fully_async=SimpleNamespace(remote=lambda response=response: response)
                )
                service.recv_weight_fully_async = lambda service=service: self.receive(service)
                responses[role] = response
            services[role] = service

        namespace = {
            "asyncio": asyncio,
            "ROLES": roles,
            "GENRM_ROLE": "genrm",
            "logger": MagicMock(),
            "run": asyncio.run,
            "set_managed_opd_teacher_on_actor_service": AsyncMock(),
            "_needs_rollout_manager_setup": lambda _: False,
        }
        training_loop = extract_method("relax/core/controller.py", "Controller", "training_loop", namespace)
        controller = SimpleNamespace(
            config=SimpleNamespace(debug_train_only=False, debug_rollout_only=False, fully_async=True, hybrid=False),
            serve_dict=services,
            _teacher_manager=None,
            _restarting=False,
            _cleanup_s3_model_weights_after_init=MagicMock(),
            _report_error_to_metrics_service=MagicMock(),
            _pending_task_refs_lock=threading.Lock(),
            _pending_task_refs=[],
        )
        error = None
        try:
            training_loop(controller)
        except RuntimeError as exc:
            error = exc
        return controller, services, started, completed, responses, error

    def test_sender_and_all_receivers_start_concurrently_and_are_awaited(self):
        controller, services, started, completed, responses, error = self._startup()
        self.assertIsNone(error)
        self.assertEqual(set(started), {"actor", "actor_fwd", "reference"})
        self.assertEqual(set(completed), set(started))
        self.assertTrue(all(response.await_count == 1 for response in responses.values()))
        for service in services.values():
            service.set_step.assert_awaited_once_with(0)
            service.run.assert_called_once()
        controller._cleanup_s3_model_weights_after_init.assert_called_once()

    def test_any_startup_weight_failure_prevents_set_step_cleanup_and_run(self):
        for role in ("actor", "actor_fwd", "reference"):
            with self.subTest(role=role):
                controller, services, _, _, _, error = self._startup(fail_role=role)
                self.assertIsNotNone(error)
                services["actor"].get_step.assert_not_awaited()
                for service in services.values():
                    service.set_step.assert_not_awaited()
                    service.run.assert_not_called()
                controller._cleanup_s3_model_weights_after_init.assert_not_called()
                controller._report_error_to_metrics_service.assert_called_once()

    def test_on_policy_without_actor_fwd_or_reference_still_awaits_sender(self):
        _, services, _, completed, responses, error = self._startup(include_receivers=False)
        self.assertIsNone(error)
        self.assertEqual(completed, ["actor"])
        self.assertEqual(responses["actor"].await_count, 1)
        services["rollout"].run.assert_called_once()


if __name__ == "__main__":
    unittest.main()

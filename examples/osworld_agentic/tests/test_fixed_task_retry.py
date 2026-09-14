# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contracts for retries around a resident preparation runtime."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from examples.osworld_agentic.fixed_task_retry import install_prepare_retry


@dataclass
class _Sample:
    class Status:
        ABORTED = "aborted"

    index: int
    group_index: int
    metadata: dict[str, Any]
    tokens: list[int] = field(default_factory=lambda: [10, 20])
    session_id: str | None = None
    status: str = "pending"


@dataclass(frozen=True)
class _GroupInput:
    group_id: str
    samples: list[_Sample]


def _group(task_id: str = "task-a", group_index: int = 0) -> _GroupInput:
    return _GroupInput(
        f"train:prepare_group_{group_index}",
        [_Sample(group_index * 8 + i, group_index, {"task_id": task_id}) for i in range(8)],
    )


def _decision(value: bool = True) -> asyncio.Future[bool]:
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


class TestFixedTaskRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.logger = Mock()
        logging_module = SimpleNamespace(get_logger=lambda _name: self.logger)
        module_patch = patch.dict(sys.modules, {"relax.utils.logging_utils": logging_module})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def _real_prepare_domain(self):
        configured = os.environ.get("OSWORLD_TEST_PREPARE_SOURCE")
        source = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[3] / "relax/agentic/pipeline/prepare.py"
        )
        if not source.is_file():
            if configured:
                self.fail(f"OSWORLD_TEST_PREPARE_SOURCE does not exist: {source}")
            self.skipTest("Real PrepareDomain source is absent; set OSWORLD_TEST_PREPARE_SOURCE to run integration")
        pipeline = ModuleType("relax.agentic.pipeline")
        pipeline.GroupInput = _GroupInput
        runtime = ModuleType("relax.agentic.pipeline.runtime")
        runtime.RuntimeDomain = object
        runtime.RuntimeGroupStream = object
        sample_types = ModuleType("relax.utils.types")
        sample_types.Sample = _Sample
        spec = importlib.util.spec_from_file_location("fixed_task_real_prepare_under_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                spec.name: module,
                "relax.agentic.pipeline": pipeline,
                "relax.agentic.pipeline.runtime": runtime,
                "relax.utils.types": sample_types,
            },
        ):
            # Execute the repository's complete, unmodified PrepareDomain source.
            # Only its model/runtime type imports are substituted for this CPU test.
            spec.loader.exec_module(module)
        return module.PrepareDomain

    @staticmethod
    async def _advance_prepare(domain, completed, until) -> None:
        for _ in range(200):
            domain.collect()
            completed.extend(domain.take_ready(74 - len(completed)))
            if until():
                return
            domain.fill(74 - len(completed))
            await asyncio.sleep(0)
        raise AssertionError("PrepareDomain did not reach the expected state within 200 event-loop turns")

    async def test_first_success_preserves_objects_and_return_value(self) -> None:
        group = _group()
        decision = _decision()
        stream = object()
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append((group_input, launch_decision))
            return stream

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        self.assertIs(await runtime.prepare_group(group, decision), stream)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], group)
        self.assertIs(calls[0][1], decision)

    async def test_failed_attempt_finishes_cleanup_before_retry_with_clean_samples(self) -> None:
        group = _group()
        calls = []
        events = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            attempt = len(calls)
            events.append((attempt, "start"))
            for sample in group_input.samples:
                self.assertIsNone(sample.session_id)
                self.assertEqual(sample.tokens, [10, 20])
                self.assertNotIn("attempt_mutation", sample.metadata)
                sample.session_id = f"attempt-{attempt}-{sample.index}"
                sample.tokens.append(99)
                sample.metadata["attempt_mutation"] = attempt
            if attempt == 1:
                events.append((attempt, "cleanup_start"))
                await asyncio.sleep(0)
                events.append((attempt, "cleanup_finished"))
                return None
            return group_input

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        result = await runtime.prepare_group(group, _decision())
        self.assertEqual(events, [(1, "start"), (1, "cleanup_start"), (1, "cleanup_finished"), (2, "start")])
        self.assertIs(calls[0], group)
        self.assertEqual(result.group_id, "train:prepare_group_0:retry_2")
        self.assertEqual([sample.index for sample in result.samples], list(range(8)))
        self.assertEqual({sample.group_index for sample in result.samples}, {0})
        self.assertEqual({sample.metadata["task_id"] for sample in result.samples}, {"task-a"})
        self.assertTrue(all(first is not second for first, second in zip(group.samples, result.samples)))
        self.assertTrue(all(sample.session_id.startswith("attempt-2-") for sample in result.samples))
        self.logger.warning.assert_called_once_with(
            "Fixed OSWorld prepare retry task_id=%s group_id=%s retry_group_id=%s attempt=%s/%s",
            "task-a",
            "train:prepare_group_0",
            "train:prepare_group_0:retry_2",
            2,
            3,
        )

    async def test_each_retry_uses_a_distinct_runtime_group_id(self) -> None:
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input.group_id)
            return None if len(calls) < 3 else group_input

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        result = await runtime.prepare_group(_group(), _decision())
        self.assertEqual(
            calls,
            ["train:prepare_group_0", "train:prepare_group_0:retry_2", "train:prepare_group_0:retry_3"],
        )
        self.assertEqual(result.group_id, calls[-1])

    async def test_late_old_future_cannot_complete_the_retry(self) -> None:
        loop = asyncio.get_running_loop()
        old_future = loop.create_future()
        new_future = loop.create_future()
        streams = {}
        retry_started = asyncio.Event()
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            if len(calls) == 1:
                streams[group_input.group_id] = old_future
                return None  # The real runtime has already dropped this stream.
            current_future = streams.setdefault(group_input.group_id, new_future)
            retry_started.set()
            return await current_future

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        task = asyncio.create_task(runtime.prepare_group(_group(), _decision()))
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        old_future.set_result("late old result")
        calls[0].samples[0].metadata["late_event"] = True
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertNotIn("late_event", calls[1].samples[0].metadata)
        self.assertNotEqual(calls[0].group_id, calls[1].group_id)
        new_future.set_result("current result")
        self.assertEqual(await task, "current result")

    async def test_false_none_and_pending_launch_decisions_do_not_retry(self) -> None:
        pending = asyncio.get_running_loop().create_future()
        for decision in (_decision(False), None, pending):
            with self.subTest(decision=decision):
                calls = []

                async def prepare(group_input, launch_decision=None):
                    calls.append(group_input)
                    return None

                runtime = SimpleNamespace(prepare_group=prepare)
                install_prepare_retry(runtime)
                result = await asyncio.wait_for(runtime.prepare_group(_group(), decision), timeout=1)
                self.assertIsNone(result)
                self.assertEqual(len(calls), 1)
        self.assertFalse(pending.done())

    async def test_original_exception_propagates_without_retry(self) -> None:
        error = OSError("sandbox unavailable")
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            raise error

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        with self.assertRaises(OSError) as caught:
            await runtime.prepare_group(_group(), _decision())
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)

    async def test_cancellation_propagates_and_original_cleanup_runs(self) -> None:
        started = asyncio.Event()
        calls = []
        cleanup = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.append(group_input.group_id)

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        task = asyncio.create_task(runtime.prepare_group(_group(), _decision()))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(calls), 1)
        self.assertEqual(cleanup, ["train:prepare_group_0"])

    async def test_cancelled_or_failed_decision_is_not_swallowed(self) -> None:
        async def prepare(group_input, launch_decision=None):
            return None

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await runtime.prepare_group(_group(), cancelled)
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(ValueError("admission failed"))
        with self.assertRaisesRegex(ValueError, "admission failed"):
            await runtime.prepare_group(_group(), failed)

    async def test_exhaustion_is_bounded_and_identifies_the_task(self) -> None:
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            return None

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime, max_attempts=3)
        with self.assertRaisesRegex(RuntimeError, "exhausted 3 attempts.*task-a.*prepare_group_0"):
            await runtime.prepare_group(_group(), _decision())
        self.assertEqual(len(calls), 3)

    async def test_different_tasks_retry_concurrently_without_sharing_state(self) -> None:
        first_attempts_started = asyncio.Event()
        counts = {}
        calls = {}

        async def prepare(group_input, launch_decision=None):
            task_id = group_input.samples[0].metadata["task_id"]
            counts[task_id] = counts.get(task_id, 0) + 1
            calls.setdefault(task_id, []).append(group_input)
            if counts[task_id] == 1:
                if len(counts) == 2:
                    first_attempts_started.set()
                await asyncio.wait_for(first_attempts_started.wait(), timeout=1)
                return None
            return group_input

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        result_a, result_b = await asyncio.gather(
            runtime.prepare_group(_group("task-a", 0), _decision()),
            runtime.prepare_group(_group("task-b", 1), _decision()),
        )
        self.assertEqual(counts, {"task-a": 2, "task-b": 2})
        self.assertEqual(result_a.group_id, "train:prepare_group_0:retry_2")
        self.assertEqual(result_b.group_id, "train:prepare_group_1:retry_2")
        self.assertTrue(
            {sample.index for sample in result_a.samples}.isdisjoint(sample.index for sample in result_b.samples)
        )

    async def test_install_is_idempotent_for_resident_runtime(self) -> None:
        calls = []

        async def prepare(group_input, launch_decision=None):
            calls.append(group_input)
            return None

        runtime = SimpleNamespace(prepare_group=prepare)
        install_prepare_retry(runtime)
        wrapper = runtime.prepare_group
        install_prepare_retry(runtime)
        self.assertIs(runtime.prepare_group, wrapper)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_prepare_retry(runtime, max_attempts=2)
        with self.assertRaises(RuntimeError):
            await runtime.prepare_group(_group(), _decision())
        self.assertEqual(len(calls), 3)

    async def test_invalid_attempt_limits_are_rejected_before_install(self) -> None:
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                runtime = SimpleNamespace(prepare_group=object())
                original = runtime.prepare_group
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    install_prepare_retry(runtime, max_attempts=value)
                self.assertIs(runtime.prepare_group, original)

    async def test_real_prepare_collect_keeps_retry_warming_without_a_75th_group(self) -> None:
        prepare_domain_class = self._real_prepare_domain()
        source = _FiniteSource()
        retry_started = asyncio.Event()
        release_retry = asyncio.Event()
        attempts = {}
        events = []
        failed_session_ids = set()

        async def prepare(group_input, launch_decision=None):
            if not launch_decision.done():
                launch_decision.set_result(True)
            task_id = group_input.samples[0].metadata["task_id"]
            attempts[task_id] = attempts.get(task_id, 0) + 1
            for sample in group_input.samples:
                sample.session_id = f"{group_input.group_id}:session_{sample.index}"
            if task_id == "task-0":
                if attempts[task_id] == 1:
                    failed_session_ids.update(sample.session_id for sample in group_input.samples)
                    await asyncio.sleep(0)
                    events.append("old_cleanup_complete")
                    return None
                events.append("retry_started")
                retry_started.set()
                await release_retry.wait()
            return group_input

        runtime = SimpleNamespace(prepare_group=prepare, concurrency=74)
        # Construct the actual resident domain before installation, as production
        # does, so this also detects a cached, unwrapped prepare_group reference.
        domain = prepare_domain_class("train", source, runtime)
        install_prepare_retry(runtime)
        domain.open_step(prelaunch_next_step=False)
        completed = []
        try:
            await self._advance_prepare(domain, completed, lambda: retry_started.is_set() and len(completed) == 73)
            self.assertEqual(source.requests, [74])
            self.assertEqual(tuple(domain.counts), (1, 0))
            self.assertEqual(events, ["old_cleanup_complete", "retry_started"])
            release_retry.set()
            await self._advance_prepare(domain, completed, lambda: len(completed) == 74)
            self.assertEqual(source.requests, [74])
            self.assertEqual(len({group.samples[0].metadata["task_id"] for group in completed}), 74)
            samples = [sample for group in completed for sample in group.samples]
            self.assertEqual(len(samples), 592)
            self.assertEqual({sample.index for sample in samples}, set(range(592)))
            self.assertEqual(len({sample.session_id for sample in samples}), 592)
            self.assertTrue(failed_session_ids.isdisjoint(sample.session_id for sample in samples))
            self.assertEqual(attempts["task-0"], 2)
            self.assertTrue(all(value == 1 for key, value in attempts.items() if key != "task-0"))
            from examples.osworld_agentic.fixed_task_rollout import _StepValidation

            validation = _StepValidation(0, tuple(f"task-{i}" for i in range(74)), 8)
            for start in range(0, len(completed), 9):
                batch = [sample for group in completed[start : start + 9] for sample in group.samples]
                validation.accept_batch(batch)
            validation.assert_complete()
            validation.validate_output([group.samples for group in completed])
        finally:
            await domain.shutdown()

    async def test_real_prepare_without_retry_reproduces_exhausted_source_request(self) -> None:
        prepare_domain_class = self._real_prepare_domain()
        source = _FiniteSource()

        async def prepare(group_input, launch_decision=None):
            launch_decision.set_result(True)
            return None if group_input.samples[0].metadata["task_id"] == "task-0" else group_input

        domain = prepare_domain_class("train", source, SimpleNamespace(prepare_group=prepare, concurrency=74))
        domain.open_step(prelaunch_next_step=False)
        completed = []
        try:
            with self.assertRaisesRegex(RuntimeError, "requested replacement beyond 74"):
                await self._advance_prepare(domain, completed, lambda: len(completed) == 74)
            self.assertEqual(source.requests, [74, 1])
        finally:
            await domain.shutdown()

    async def test_real_prepare_exhausted_retries_raise_without_refilling(self) -> None:
        prepare_domain_class = self._real_prepare_domain()
        source = _FiniteSource()
        failed_attempts = []

        async def prepare(group_input, launch_decision=None):
            if not launch_decision.done():
                launch_decision.set_result(True)
            if group_input.samples[0].metadata["task_id"] == "task-0":
                failed_attempts.append(group_input.group_id)
                await asyncio.sleep(0)
                return None
            return group_input

        runtime = SimpleNamespace(prepare_group=prepare, concurrency=74)
        domain = prepare_domain_class("train", source, runtime)
        install_prepare_retry(runtime)
        domain.open_step(prelaunch_next_step=False)
        completed = []
        try:
            with self.assertRaisesRegex(RuntimeError, "exhausted 3 attempts.*task-0"):
                await self._advance_prepare(domain, completed, lambda: len(completed) == 74)
            self.assertEqual(source.requests, [74])
            self.assertEqual(len(failed_attempts), 3)
        finally:
            await domain.shutdown()


class _FiniteSource:
    """Replace only Ray transport and dataset I/O at the real prepare
    boundary."""

    def __init__(self) -> None:
        self.requests = []
        self.groups = [_group(f"task-{i}", i).samples for i in range(74)]
        self.get_samples = SimpleNamespace(remote=self._get_samples)

    async def _get_samples(self, count):
        self.requests.append(count)
        if count > len(self.groups):
            raise RuntimeError("requested replacement beyond 74")
        result, self.groups = self.groups[:count], self.groups[count:]
        return result


if __name__ == "__main__":
    unittest.main()

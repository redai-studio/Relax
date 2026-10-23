# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Test session methods from the complete patched SGLang source."""

from __future__ import annotations

import ast
import asyncio
import unittest
from contextlib import nullcontext
from functools import lru_cache
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

from tests.tools.sglang_patch_source import PATCH_PATH, patched_functions


CONTROL_PATH = "python/sglang/srt/managers/tokenizer_control_mixin.py"
MANAGER_PATH = "python/sglang/srt/managers/tokenizer_manager.py"
SCHEDULER_PATH = "python/sglang/srt/managers/scheduler_components/weight_updater.py"
LORA_PATH = "python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py"
CONTROL_METHODS = {
    "_weight_update_session_call",
    "begin_weight_update",
    "end_weight_update",
    "_ensure_relax_weight_update_session_locked",
    "_finish_relax_weight_update_session_locked",
    "_relax_update_weights_locked",
    "update_weights_from_distributed",
    "update_weights_from_tensor",
    "post_process_weights",
    "release_memory_occupation",
    "resume_memory_occupation",
}


class _FanOut:
    @staticmethod
    def merge_results(results: list[SimpleNamespace]) -> tuple[bool, str]:
        return all(result.success for result in results), "; ".join(result.message for result in results)


class _Request(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(selector=kwargs.pop("selector", "all"), **kwargs)


@lru_cache(maxsize=1)
def _manager_class() -> type[Any]:
    methods = patched_functions(CONTROL_PATH, CONTROL_METHODS)
    methods += patched_functions(MANAGER_PATH, {"continue_generation"})
    module = ast.parse("from __future__ import annotations\nclass Manager:\n    pass\n")
    manager_class = module.body[1]
    assert isinstance(manager_class, ast.ClassDef)
    manager_class.body = methods
    ast.fix_missing_locations(module)
    namespace = {
        "FanOutCommunicator": _FanOut,
        "BeginWeightUpdateReqInput": _Request,
        "EndWeightUpdateReqInput": _Request,
        "get_parallel": lambda: SimpleNamespace(dp_size=1, enable_dp_attention=False),
        "normalize_serialized_named_tensor_payloads": lambda values: values,
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace["Manager"]


def _manager() -> Any:
    manager = _manager_class()()
    manager.is_pause = True
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = SimpleNamespace(writer_lock=asyncio.Lock())
    manager._relax_weight_update_session = None
    manager._relax_weight_update_selector = None
    manager._relax_weight_update_error = None
    manager._relax_weight_update_has_bucket = False
    manager.mm_processor = None
    manager.events = []
    manager.server_open = False
    manager.auto_create_handle_loop = lambda: None
    manager.abort_request = lambda **kwargs: None
    manager._update_weight_version_if_provided = lambda value: manager.events.append(("version", value))

    async def begin(request: _Request) -> list[SimpleNamespace]:
        manager.events.append(("begin", request.selector))
        if manager.server_open:
            return [SimpleNamespace(success=False, message="already open")]
        manager.server_open = True
        return [SimpleNamespace(success=True, message="Success")]

    async def end(request: _Request) -> list[SimpleNamespace]:
        manager.events.append(("end", None))
        if not manager.server_open:
            return [SimpleNamespace(success=False, message="not open")]
        manager.server_open = False
        return [SimpleNamespace(success=True, message="Success")]

    async def update(request: _Request) -> list[SimpleNamespace]:
        manager.events.append(("bucket", request.selector))
        return [
            SimpleNamespace(
                success=manager.server_open,
                message="Success" if manager.server_open else "must run between begin/end",
            )
        ]

    async def dispatch(request: _Request) -> None:
        manager.events.append(("continue", None))

    async def release(request: _Request) -> None:
        manager.events.append(("release", None))

    async def resume(request: _Request) -> None:
        manager.events.append(("resume-memory", None))

    manager.begin_weight_update_communicator = begin
    manager.end_weight_update_communicator = end
    manager.update_weights_from_tensor_communicator = update
    manager.update_weights_from_distributed_communicator = update
    manager.release_memory_occupation_communicator = release
    manager.resume_memory_occupation_communicator = resume
    manager._async_dispatch_to_scheduler = dispatch
    return manager


def _bucket(**kwargs: Any) -> _Request:
    fields = {
        "abort_all_requests": False,
        "flush_cache": False,
        "weight_version": None,
        "serialized_named_tensors": ["x"],
        "names": ["x"],
    }
    fields.update(kwargs)
    return _Request(**fields)


class _CPUFlag:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _CPUCollectives:
    """Model CPU MIN using declared TP peers; no torch or distributed
    runtime."""

    def __init__(self, peer_empty_flags: list[int]) -> None:
        self.peer_empty_flags = peer_empty_flags
        self.group = object()
        self.min_operation = object()
        self.reduced_inputs: list[int] = []
        self.barrier_count = 0

    def tensor(self, value: int, *, dtype: str, device: str) -> _CPUFlag:
        assert dtype == "int32"
        assert device == "cpu"
        return _CPUFlag(value)

    def all_reduce(self, flag: _CPUFlag, *, op: object, group: object) -> None:
        assert op is self.min_operation
        assert group is self.group
        self.reduced_inputs.append(flag.value)
        flag.value = min(flag.value, *self.peer_empty_flags)

    def barrier(self, *, group: object) -> None:
        assert group is self.group
        self.barrier_count += 1


def _scheduler_manager(peer_empty_flags: list[int], *, legacy: bool = True) -> tuple[Any, _CPUCollectives]:
    collectives = _CPUCollectives(peer_empty_flags)
    methods = patched_functions(
        SCHEDULER_PATH,
        {"_legacy_tensor_update_is_empty", "update_weights_from_tensor", "end_weight_update"},
    )
    module = ast.parse("from __future__ import annotations\nclass Scheduler:\n    pass\n")
    scheduler_class = module.body[1]
    assert isinstance(scheduler_class, ast.ClassDef)
    scheduler_class.body = methods
    ast.fix_missing_locations(module)
    namespace = {
        "torch": SimpleNamespace(
            tensor=collectives.tensor,
            int32="int32",
            distributed=SimpleNamespace(
                all_reduce=collectives.all_reduce,
                barrier=collectives.barrier,
                ReduceOp=SimpleNamespace(MIN=collectives.min_operation),
            ),
        ),
        "msgspec": SimpleNamespace(
            structs=SimpleNamespace(replace=lambda session, **changes: SimpleNamespace(**{**vars(session), **changes}))
        ),
        "UpdateWeightsFromTensorReqOutput": SimpleNamespace,
        "EndWeightUpdateReqOutput": SimpleNamespace,
        "logger": SimpleNamespace(error=lambda message: None),
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    scheduler = namespace["Scheduler"]()
    scheduler._session = SimpleNamespace(selector="all", loaded_weights=False, relax_legacy=legacy)
    scheduler.tp_cpu_group = collectives.group
    scheduler.tp_worker = SimpleNamespace(deserialize_own_rank=lambda payload: payload)
    scheduler._observe_weight_load = lambda source: nullcontext()
    scheduler.events = []

    def update(*, named_tensors: Any, load_format: str) -> tuple[bool, str]:
        scheduler.events.append(("load", named_tensors))
        return True, "Success"

    def end(*, run_post_load: bool) -> None:
        scheduler.events.append(("repack", run_post_load))

    runner = SimpleNamespace(weight_updater=SimpleNamespace(update_weights_from_tensor=update, end_weight_update=end))
    scheduler._select_runners = lambda selector: [("target", runner)]
    scheduler.flush_cache_after_weight_update = lambda request: scheduler.events.append(("flush", None))
    scheduler.record_weight_version_after_update = lambda version: scheduler.events.append(("version", version))
    return scheduler, collectives


def _scheduler_bucket(payload: Any) -> _Request:
    return _Request(serialized_named_tensors=payload, load_format="flattened_bucket", weight_version=None)


@lru_cache(maxsize=1)
def _tensor_lora_setter() -> Any:
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(patched_functions(LORA_PATH, {"set_lora_from_tensors"}))
    ast.fix_missing_locations(module)
    namespace = {
        "dist": SimpleNamespace(get_rank=lambda: 0),
        "logger": SimpleNamespace(info=lambda *args: None),
        "normalize_lora_state_dict": lambda weights, *, logger: weights,
        "nullcontext": nullcontext,
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace["set_lora_from_tensors"]


def _tensor_lora_pipeline(*, layerwise: bool = False, needs_context: bool = True) -> Any:
    pipeline = Mock()
    pipeline.VALID_TARGETS = {"transformer"}
    pipeline.lora_initialized = False
    pipeline._has_layerwise_offload.return_value = layerwise
    pipeline._get_target_lora_layers.return_value = ([("transformer", {"layer": object()})], None)
    pipeline._resolve_lora_merge_mode.return_value = "merge" if needs_context else "dynamic"
    pipeline._should_merge_lora_for_layers.return_value = needs_context
    pipeline._needs_lora_weight_update_context.return_value = needs_context
    pipeline._weight_update_context.return_value = nullcontext() if layerwise else MagicMock()
    pipeline._temporarily_disable_offload.side_effect = AssertionError("Must use the upstream update context")
    pipeline._apply_lora_to_layers.return_value = 1
    for name in (
        "loaded_adapter_paths",
        "cur_adapter_name",
        "cur_adapter_path",
        "is_lora_merged",
        "cur_adapter_strength",
        "cur_adapter_config",
    ):
        setattr(pipeline, name, {})

    def convert(*, snapshot_base: bool) -> None:
        pipeline.lora_initialized = True

    pipeline.convert_to_lora_layers.side_effect = convert
    return pipeline


class TestTensorLoraCompatibility(unittest.TestCase):
    def test_initialization_and_update_follow_upstream_offload_contract(self) -> None:
        for layerwise in (False, True):
            with self.subTest(layerwise=layerwise):
                pipeline = _tensor_lora_pipeline(layerwise=layerwise)
                self.assertEqual(_tensor_lora_setter()(pipeline, "rl", {"lora_A": object()}), 1)
                pipeline.convert_to_lora_layers.assert_called_once_with(snapshot_base=not layerwise)
                pipeline._weight_update_context.assert_called_once_with(
                    pipeline._get_target_lora_layers.return_value[0]
                )
                pipeline._temporarily_disable_offload.assert_not_called()
                self.assertEqual(pipeline.mock_calls[0], call._invalidate_lora_conditioning())
                calls = [item[0] for item in pipeline.mock_calls]
                self.assertLess(calls.index("_invalidate_lora_conditioning"), calls.index("_ingest_lora_state_dict"))
                self.assertLess(calls.index("_ingest_lora_state_dict"), calls.index("_apply_lora_to_layers"))
                if not layerwise:
                    context = pipeline._weight_update_context.return_value
                    context.__enter__.assert_called_once_with()
                    context.__exit__.assert_called_once_with(None, None, None)

    def test_dynamic_update_without_backing_weight_changes_keeps_null_context(self) -> None:
        pipeline = _tensor_lora_pipeline(needs_context=False)
        pipeline.lora_initialized = True
        self.assertEqual(_tensor_lora_setter()(pipeline, "rl", {"lora_A": object()}), 1)
        pipeline.convert_to_lora_layers.assert_not_called()
        pipeline._weight_update_context.assert_not_called()
        pipeline._temporarily_disable_offload.assert_not_called()
        self.assertFalse(pipeline._apply_lora_to_layers.call_args.kwargs["merge_weights"])
        pipeline._invalidate_lora_conditioning.assert_called_once_with()

    def test_same_adapter_name_always_installs_fresh_weights_and_invalidates_cache(self) -> None:
        pipeline = _tensor_lora_pipeline(layerwise=True)
        payloads = [{"lora_A": object()}, {"lora_A": object()}]
        for payload in payloads:
            pipeline.reset_mock()
            self.assertEqual(_tensor_lora_setter()(pipeline, "rl", payload), 1)
            self.assertEqual(pipeline.mock_calls[0], call._invalidate_lora_conditioning())
            pipeline._ingest_lora_state_dict.assert_called_once_with(payload, "rl")
            pipeline._apply_lora_to_layers.assert_called_once_with(
                pipeline._get_target_lora_layers.return_value[0][0][1],
                ["rl"],
                ["<tensors:rl>"],
                0,
                [1.0],
                clear_existing=True,
                merge_weights=True,
            )
        pipeline.convert_to_lora_layers.assert_not_called()
        self.assertEqual(pipeline.cur_adapter_config["transformer"], (["rl"], [1.0]))

    def test_failed_apply_still_invalidates_conditioning_before_weight_changes(self) -> None:
        pipeline = _tensor_lora_pipeline()
        pipeline._apply_lora_to_layers.side_effect = RuntimeError("partial write")
        with self.assertRaisesRegex(RuntimeError, "partial write"):
            _tensor_lora_setter()(pipeline, "rl", {"lora_A": object()})
        self.assertEqual(pipeline.mock_calls[0], call._invalidate_lora_conditioning())
        self.assertEqual(pipeline.cur_adapter_config, {})

    def test_invalid_target_does_not_modify_adapter_or_cache_state(self) -> None:
        pipeline = _tensor_lora_pipeline()
        with self.assertRaisesRegex(ValueError, "Invalid target"):
            _tensor_lora_setter()(pipeline, "rl", {}, target="unknown")
        self.assertEqual(pipeline.mock_calls, [])


class TestLegacyWeightUpdateSessions(unittest.IsolatedAsyncioTestCase):
    async def test_tensor_and_distributed_multibucket_finalize_before_continue(self) -> None:
        manager = _manager()
        for update in (
            manager.update_weights_from_tensor,
            manager.update_weights_from_tensor,
            manager.update_weights_from_distributed,
        ):
            self.assertTrue((await update(_bucket()))[0])
        self.assertEqual([event[0] for event in manager.events], ["begin", "bucket", "bucket", "bucket"])
        await manager.continue_generation(_Request())
        self.assertEqual([event[0] for event in manager.events][-2:], ["end", "continue"])
        self.assertFalse(manager.is_pause)
        self.assertFalse(manager.server_open)

    async def test_compressed_prepost_share_native_session(self) -> None:
        manager = _manager()
        restore = _Request(restore_weights_before_load=True, post_process_quantization=False)
        repack = _Request(restore_weights_before_load=False, post_process_quantization=True)
        self.assertTrue((await manager.post_process_weights(restore))[0])
        self.assertTrue((await manager.post_process_weights(restore))[0])
        self.assertTrue((await manager.update_weights_from_tensor(_bucket()))[0])
        self.assertTrue((await manager.post_process_weights(repack))[0])
        self.assertTrue((await manager.post_process_weights(repack))[0])
        await manager.continue_generation(_Request())
        self.assertEqual([event[0] for event in manager.events], ["begin", "bucket", "end", "continue"])

    async def test_postprocess_without_session_is_idempotent_noop(self) -> None:
        manager = _manager()
        repack = _Request(restore_weights_before_load=False, post_process_quantization=True)
        self.assertTrue((await manager.post_process_weights(repack))[0])
        self.assertTrue((await manager.post_process_weights(repack))[0])
        self.assertEqual(manager.events, [])
        self.assertIsNone(manager._relax_weight_update_session)
        self.assertIsNone(manager._relax_weight_update_error)

    async def test_restore_and_repack_without_bucket_fails_closed(self) -> None:
        manager = _manager()
        self.assertFalse(
            (
                await manager.post_process_weights(
                    _Request(restore_weights_before_load=True, post_process_quantization=True)
                )
            )[0]
        )
        with self.assertRaises(ValueError):
            await manager.continue_generation(_Request())
        self.assertEqual(manager._relax_weight_update_session, "legacy")
        self.assertIsNotNone(manager._relax_weight_update_error)
        self.assertTrue(manager.is_pause)
        self.assertEqual([event[0] for event in manager.events], ["begin"])

    async def test_native_explicit_session_not_auto_closed(self) -> None:
        manager = _manager()
        manager.is_pause = False
        self.assertTrue((await manager.begin_weight_update(_Request(selector="target")))[0])
        self.assertTrue((await manager.update_weights_from_tensor(_bucket(selector="target")))[0])
        manager.is_pause = True
        self.assertTrue(
            (
                await manager.post_process_weights(
                    _Request(restore_weights_before_load=True, post_process_quantization=True)
                )
            )[0]
        )
        await manager.continue_generation(_Request())
        self.assertTrue(manager.server_open)
        self.assertTrue((await manager.end_weight_update(_Request()))[0])
        self.assertEqual([event[0] for event in manager.events], ["begin", "bucket", "continue", "end"])

    async def test_unpaused_without_explicit_session_stays_rejected(self) -> None:
        for transport in ("tensor", "distributed"):
            with self.subTest(transport=transport):
                manager = _manager()
                manager.is_pause = False
                update = getattr(manager, "update_weights_from_" + transport)
                self.assertFalse((await update(_bucket()))[0])
                self.assertNotIn("begin", [event[0] for event in manager.events])

    async def test_failed_bucket_does_not_unpause_or_release(self) -> None:
        manager = _manager()

        async def fail(request: _Request) -> list[SimpleNamespace]:
            return [SimpleNamespace(success=False, message="partial load")]

        manager.update_weights_from_tensor_communicator = fail
        self.assertFalse((await manager.update_weights_from_tensor(_bucket()))[0])
        with self.assertRaises(ValueError):
            await manager.continue_generation(_Request())
        with self.assertRaises(ValueError):
            await manager.release_memory_occupation(_Request())
        self.assertTrue(manager.is_pause)
        self.assertNotIn("continue", [event[0] for event in manager.events])

    async def test_cancelled_bucket_fails_closed(self) -> None:
        manager = _manager()

        async def cancel(request: _Request) -> list[SimpleNamespace]:
            raise asyncio.CancelledError()

        manager.update_weights_from_distributed_communicator = cancel
        with self.assertRaises(asyncio.CancelledError):
            await manager.update_weights_from_distributed(_bucket())
        with self.assertRaises(ValueError):
            await manager.continue_generation(_Request())
        self.assertTrue(manager.is_pause)

    async def test_failed_begin_and_end_do_not_unpause(self) -> None:
        for failing_stage in ("begin", "end"):
            with self.subTest(failing_stage=failing_stage):
                manager = _manager()

                async def fail(request: _Request) -> list[SimpleNamespace]:
                    return [SimpleNamespace(success=False, message="fanout failed")]

                if failing_stage == "begin":
                    manager.begin_weight_update_communicator = fail
                    self.assertFalse((await manager.update_weights_from_tensor(_bucket()))[0])
                else:
                    self.assertTrue((await manager.update_weights_from_tensor(_bucket()))[0])
                    manager.end_weight_update_communicator = fail
                with self.assertRaises(ValueError):
                    await manager.continue_generation(_Request())
                self.assertTrue(manager.is_pause)

    async def test_empty_restore_session_cannot_finalize_or_resume(self) -> None:
        for first_finalizer in ("continue", "end", "repack"):
            with self.subTest(first_finalizer=first_finalizer):
                manager = _manager()
                self.assertTrue(
                    (
                        await manager.post_process_weights(
                            _Request(restore_weights_before_load=True, post_process_quantization=False)
                        )
                    )[0]
                )
                repack = _Request(restore_weights_before_load=False, post_process_quantization=True)
                if first_finalizer == "continue":
                    with self.assertRaises(ValueError):
                        await manager.continue_generation(_Request())
                elif first_finalizer == "end":
                    self.assertFalse((await manager.end_weight_update(_Request()))[0])
                else:
                    self.assertFalse((await manager.post_process_weights(repack))[0])
                with self.assertRaises(ValueError):
                    await manager.continue_generation(_Request())
                self.assertFalse((await manager.end_weight_update(_Request()))[0])
                self.assertFalse((await manager.post_process_weights(repack))[0])
                with self.assertRaises(ValueError):
                    await manager.release_memory_occupation(_Request())
                with self.assertRaises(ValueError):
                    await manager.resume_memory_occupation(_Request())
                self.assertTrue(manager.is_pause)
                self.assertIsNotNone(manager._relax_weight_update_error)
                self.assertEqual([event[0] for event in manager.events], ["begin"])

    async def test_completed_update_can_release_and_resume_memory(self) -> None:
        manager = _manager()
        self.assertTrue((await manager.update_weights_from_tensor(_bucket()))[0])
        await manager.continue_generation(_Request())
        await manager.release_memory_occupation(_Request())
        await manager.resume_memory_occupation(_Request())
        self.assertEqual(
            [event[0] for event in manager.events], ["begin", "bucket", "end", "continue", "release", "resume-memory"]
        )

    async def test_empty_tensor_and_distributed_updates_do_not_auto_begin(self) -> None:
        manager = _manager()
        self.assertTrue((await manager.update_weights_from_tensor(_bucket(serialized_named_tensors=[])))[0])
        self.assertTrue((await manager.update_weights_from_distributed(_bucket(names=[])))[0])
        self.assertEqual(manager.events, [])
        self.assertIsNone(manager._relax_weight_update_session)
        self.assertFalse(manager._relax_weight_update_has_bucket)
        await manager.continue_generation(_Request())
        self.assertEqual([event[0] for event in manager.events], ["continue"])

    async def test_normalized_empty_tensor_update_does_not_auto_begin(self) -> None:
        manager = _manager()
        namespace = manager.update_weights_from_tensor.__globals__
        with patch.dict(namespace, {"normalize_serialized_named_tensor_payloads": lambda values: []}):
            self.assertTrue((await manager.update_weights_from_tensor(_bucket()))[0])
        self.assertEqual(manager.events, [])
        self.assertIsNone(manager._relax_weight_update_session)
        self.assertFalse(manager._relax_weight_update_has_bucket)

    async def test_empty_updates_do_not_bypass_sticky_legacy_failure(self) -> None:
        manager = _manager()

        async def fail(request: _Request) -> list[SimpleNamespace]:
            return [SimpleNamespace(success=False, message="partial load")]

        manager.update_weights_from_tensor_communicator = fail
        self.assertFalse((await manager.update_weights_from_tensor(_bucket()))[0])
        error = manager._relax_weight_update_error
        self.assertIsNotNone(error)
        self.assertFalse((await manager.update_weights_from_tensor(_bucket(serialized_named_tensors=[])))[0])
        self.assertFalse((await manager.update_weights_from_distributed(_bucket(names=[])))[0])
        self.assertEqual(manager._relax_weight_update_error, error)
        self.assertTrue(manager.is_pause)
        self.assertEqual([event[0] for event in manager.events], ["begin"])

    async def test_explicit_begin_does_not_clear_sticky_legacy_failure(self) -> None:
        manager = _manager()

        async def fail(request: _Request) -> list[SimpleNamespace]:
            return [SimpleNamespace(success=False, message="partial load")]

        manager.update_weights_from_tensor_communicator = fail
        self.assertFalse((await manager.update_weights_from_tensor(_bucket()))[0])
        error = manager._relax_weight_update_error
        self.assertFalse((await manager.begin_weight_update(_Request()))[0])
        self.assertEqual(manager._relax_weight_update_error, error)
        self.assertEqual(manager._relax_weight_update_session, "legacy")
        self.assertTrue(manager.is_pause)
        self.assertEqual([event[0] for event in manager.events], ["begin"])

    async def test_selector_change_is_rejected(self) -> None:
        manager = _manager()
        self.assertTrue((await manager.update_weights_from_tensor(_bucket(selector="target")))[0])
        self.assertFalse((await manager.update_weights_from_tensor(_bucket(selector="all")))[0])
        self.assertEqual([event[0] for event in manager.events], ["begin", "bucket"])

    async def test_resume_memory_rejects_active_and_failed_legacy_sessions(self) -> None:
        for failed in (False, True):
            with self.subTest(failed=failed):
                manager = _manager()

                async def fail(request: _Request) -> list[SimpleNamespace]:
                    return [SimpleNamespace(success=False, message="partial load")]

                if failed:
                    manager.update_weights_from_tensor_communicator = fail
                self.assertEqual((await manager.update_weights_from_tensor(_bucket()))[0], not failed)
                with self.assertRaises(ValueError):
                    await manager.resume_memory_occupation(_Request())
                self.assertTrue(manager.is_pause)
                self.assertNotIn("resume-memory", [event[0] for event in manager.events])


class TestSchedulerLegacyPadding(unittest.TestCase):
    def test_all_tp_empty_padding_is_noop_and_cannot_finalize_empty_session(self) -> None:
        for payload in ([], {"metadata": [], "flattened_tensor": object()}):
            with self.subTest(payload_type=type(payload).__name__):
                scheduler, collectives = _scheduler_manager([1, 1])
                self.assertTrue(scheduler.update_weights_from_tensor(_scheduler_bucket(payload)).success)
                self.assertFalse(scheduler._session.loaded_weights)
                self.assertFalse(scheduler.end_weight_update(_Request()).success)
                self.assertIsNotNone(scheduler._session)
                self.assertEqual(scheduler.events, [])
                self.assertEqual(collectives.reduced_inputs, [1])
                self.assertEqual(collectives.barrier_count, 0)

    def test_mixed_tp_ranks_uniformly_load_legitimate_rank_local_padding(self) -> None:
        for payloads in (
            ([], ["weight"]),
            (
                {"metadata": [], "flattened_tensor": object()},
                {"metadata": ["weight"], "flattened_tensor": object()},
            ),
        ):
            for rank, payload in enumerate(payloads):
                with self.subTest(payload_type=type(payload).__name__, rank=rank):
                    scheduler, collectives = _scheduler_manager([1, 0])
                    self.assertTrue(scheduler.update_weights_from_tensor(_scheduler_bucket(payload)).success)
                    self.assertTrue(scheduler._session.loaded_weights)
                    self.assertEqual([event[0] for event in scheduler.events], ["load", "flush", "version"])
                    self.assertEqual(collectives.reduced_inputs, [1 if rank == 0 else 0])
                    self.assertEqual(collectives.barrier_count, 1)
                    self.assertTrue(scheduler.end_weight_update(_Request()).success)
                    self.assertEqual(scheduler.events[-1], ("repack", False))
                    self.assertIsNone(scheduler._session)

    def test_padding_after_real_weights_preserves_valid_session_and_single_repack(self) -> None:
        scheduler, collectives = _scheduler_manager([0, 0])
        real_bucket = {"metadata": ["weight"], "flattened_tensor": object()}
        padding_bucket = {"metadata": [], "flattened_tensor": object()}
        self.assertTrue(scheduler.update_weights_from_tensor(_scheduler_bucket(real_bucket)).success)
        self.assertTrue(scheduler._session.loaded_weights)
        collectives.peer_empty_flags = [1, 1]
        self.assertTrue(scheduler.update_weights_from_tensor(_scheduler_bucket(padding_bucket)).success)
        self.assertTrue(scheduler._session.loaded_weights)
        self.assertTrue(scheduler.end_weight_update(_Request()).success)
        self.assertEqual([event[0] for event in scheduler.events], ["load", "flush", "version", "repack"])
        self.assertEqual(scheduler.events[-1], ("repack", False))
        self.assertEqual(collectives.reduced_inputs, [0, 1])
        self.assertEqual(collectives.barrier_count, 2)
        self.assertIsNone(scheduler._session)

    def test_explicit_native_sessions_keep_original_empty_update_and_finalize_behavior(self) -> None:
        scheduler, collectives = _scheduler_manager([1, 1], legacy=False)
        self.assertTrue(scheduler.update_weights_from_tensor(_scheduler_bucket([])).success)
        self.assertTrue(scheduler._session.loaded_weights)
        self.assertTrue(scheduler.end_weight_update(_Request()).success)
        self.assertEqual([event[0] for event in scheduler.events], ["load", "flush", "version", "repack"])
        self.assertEqual(scheduler.events[-1], ("repack", False))
        self.assertEqual(collectives.reduced_inputs, [])
        self.assertEqual(collectives.barrier_count, 2)

        scheduler, collectives = _scheduler_manager([1, 1], legacy=False)
        self.assertTrue(scheduler.end_weight_update(_Request()).success)
        self.assertEqual(scheduler.events, [("repack", True)])
        self.assertEqual(collectives.reduced_inputs, [])
        self.assertEqual(collectives.barrier_count, 1)

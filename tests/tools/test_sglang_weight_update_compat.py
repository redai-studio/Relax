# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise tensor LoRA and weight-update completion from the shipped patch."""

from __future__ import annotations

import ast
import unittest
from contextlib import nullcontext
from functools import lru_cache
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, call

from tests.tools.sglang_patch_source import PATCH_PATH, patched_functions


LORA_PATH = "python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py"
UPDATER_PATH = "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"


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


def _native_updater(model: Any, loader: Any, device: Any) -> Any:
    """Run the actual updater; isolate model and device I/O at its boundary."""

    class Bucket:
        def __init__(self, *, flattened_tensor, metadata):
            self.tensor = flattened_tensor
            self.metadata = metadata

        def reconstruct_tensors(self):
            return [(metadata.name, self.tensor) for metadata in self.metadata]

    names = {
        "begin_weight_update",
        "end_weight_update",
        "update_weights_from_tensor",
        "_update_weights_from_flattened_bucket",
    }
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(patched_functions(UPDATER_PATH, names))
    namespace = {
        "torch": SimpleNamespace(device=lambda value: value, get_device_module=lambda _: device),
        "DefaultModelLoader": loader,
        "post_load_weights": loader.post_load_weights,
        "_unsupported_derived_weight_cache_error": lambda _: None,
        "monkey_patch_torch_reductions": lambda: None,
        "_unwrap_tensor": lambda value, **_: value,
        "FlattenedTensorMetadata": SimpleNamespace,
        "FlattenedTensorBucket": Bucket,
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    updater = SimpleNamespace(
        get_model=lambda: model,
        device="cuda",
        tp_rank=0,
        custom_weight_loaders={},
        _assert_weight_cache_inactive=lambda _: None,
    )
    for name in names:
        setattr(updater, name, MethodType(namespace[name], updater))
    return updater


class TestNativeWeightUpdateCompletion(unittest.TestCase):
    def test_model_owned_cache_hooks_survive_native_updates(self) -> None:
        for run_post_load in (False, True):
            with self.subTest(run_post_load=run_post_load):
                model = SimpleNamespace(cache="stale")

                def begin():
                    model.cache = None

                def end():
                    model.cache = "fresh"

                def restore(actual, device):
                    self.assertIs(actual, model)
                    self.assertIsNone(model.cache)

                def postprocess(actual, device):
                    self.assertIs(actual, model)
                    self.assertEqual(model.cache, "fresh")

                model.begin_weight_update = begin
                model.end_weight_update = end
                loader = SimpleNamespace(
                    restore_weights_before_loading=Mock(side_effect=restore),
                    postprocess_weights=Mock(side_effect=postprocess),
                    post_load_weights=Mock(),
                )
                updater = _native_updater(model, loader, SimpleNamespace())
                updater.begin_weight_update()
                updater.end_weight_update(run_post_load=run_post_load)
                self.assertEqual(model.cache, "fresh")
                loader.restore_weights_before_loading.assert_called_once()
                loader.postprocess_weights.assert_called_once()
                loader.post_load_weights.assert_not_called()

    def test_models_without_hooks_keep_default_postload_policy(self) -> None:
        for run_post_load in (False, True):
            with self.subTest(run_post_load=run_post_load):
                model = SimpleNamespace()
                loader = SimpleNamespace(
                    restore_weights_before_loading=Mock(), postprocess_weights=Mock(), post_load_weights=Mock()
                )
                updater = _native_updater(model, loader, SimpleNamespace())
                updater.begin_weight_update()
                updater.end_weight_update(run_post_load=run_post_load)
                loader.restore_weights_before_loading.assert_called_once_with(model, "cuda")
                loader.postprocess_weights.assert_called_once_with(model, "cuda")
                self.assertEqual(loader.post_load_weights.call_args_list, [call(model)] if run_post_load else [])

    def test_tensor_rpc_finishes_copies_before_sender_reuses_storage(self) -> None:
        for load_format in (None, "flattened_bucket"):
            with self.subTest(load_format=load_format):
                source, received, pending = [1, 2, 3], {}, []

                def load_weights(named_tensors):
                    for name, value in named_tensors:
                        pending.append(lambda name=name, value=value: received.update({name: list(value)}))

                def synchronize():
                    for copy in pending:
                        copy()
                    pending.clear()

                model = SimpleNamespace(load_weights=load_weights)
                device = SimpleNamespace(current_device=lambda: 0, synchronize=synchronize)
                updater = _native_updater(model, SimpleNamespace(post_load_weights=Mock()), device)
                payload = [("weight", source)]
                if load_format == "flattened_bucket":
                    metadata = SimpleNamespace(
                        name="weight", shape=(3,), dtype="float32", start_idx=0, end_idx=3, numel=3
                    )
                    payload = {"flattened_tensor": source, "metadata": [metadata]}
                success, _ = updater.update_weights_from_tensor(payload, load_format=load_format)
                self.assertTrue(success)
                source[:] = [9, 9, 9]
                self.assertEqual(received, {"weight": [1, 2, 3]})
                self.assertEqual(pending, [])

    def test_async_copy_failure_cannot_return_success(self) -> None:
        for load_format in (None, "flattened_bucket"):
            with self.subTest(load_format=load_format):
                device = SimpleNamespace(
                    current_device=lambda: 0, synchronize=Mock(side_effect=RuntimeError("async write failed"))
                )
                updater = _native_updater(
                    SimpleNamespace(load_weights=Mock()), SimpleNamespace(post_load_weights=Mock()), device
                )
                payload = [] if load_format is None else {"flattened_tensor": [], "metadata": []}
                with self.assertRaisesRegex(RuntimeError, "async write failed"):
                    updater.update_weights_from_tensor(payload, load_format=load_format)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check V4.1 patch control flow without importing Torch or SGLang."""

from __future__ import annotations

import ast
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from tests.tools.sglang_patch_source import PATCH_PATH, patched_functions


SRT = "python/sglang/srt/"


def _method(path: str, name: str, namespace: dict[str, Any]) -> Any:
    node = patched_functions(SRT + path, {name})[0]
    node.decorator_list = []
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace[name]


def _capture_helper(namespace: dict[str, Any]) -> Any:
    return _method("layers/moe/topk.py", "capture_routed_experts_if_allowed", namespace)


class _Rows:
    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.shape = (len(values),)

    def __getitem__(self, index: slice) -> _Rows:
        return _Rows(self.values[index])

    def clone(self) -> _Rows:
        return _Rows(self.values.copy())

    def cpu(self) -> _Rows:
        return self


class TestDeepseekV41Routing(unittest.TestCase):
    def test_capture_masks_a_copy_and_respects_draft_opt_out(self) -> None:
        capturer = Mock()
        mask = Mock()
        capture = _capture_helper(
            {"get_global_experts_capturer": lambda: capturer, "_mask_topk_ids_padded_region": mask}
        )
        ids = _Rows([3, 7, 9, 11])
        config = SimpleNamespace(allow_routed_experts_capture=True)
        capture(config, 2, ids, 2)
        copied = capturer.capture.call_args.kwargs["topk_indices"]
        self.assertIsNot(copied, ids)
        mask.assert_called_once_with(copied, 2)
        self.assertEqual(capturer.capture.call_args.kwargs["layer_id"], 2)
        config.allow_routed_experts_capture = False
        capture(config, 2, ids, 2)
        self.assertEqual(capturer.capture.call_count, 1)

    def test_dp_local_routing_does_not_apply_dp_concatenation_offset(self) -> None:
        for scattered in (False, True):
            with self.subTest(scattered=scattered):
                dp_slice = Mock(return_value=(8, 2))
                method = _method(
                    "state_capturer/routed_experts.py",
                    "_get_local_slice",
                    {
                        "is_dp_attention_enabled": lambda: True,
                        "is_moe_input_scattered_across_dp_ranks": lambda: scattered,
                        "get_dp_local_slice_cpu": dp_slice,
                    },
                )
                buffer = Mock()
                buffer.__getitem__ = Mock(return_value="rows")
                capturer = SimpleNamespace(device_cache=SimpleNamespace(buffer=buffer), topk_size=6)
                batch = SimpleNamespace(out_cache_loc=_Rows([1, 2]))
                self.assertEqual(method(capturer, batch, True, 8), "rows")
                row_slice = buffer.__getitem__.call_args.args[0][0]
                self.assertEqual((row_slice.start, row_slice.stop), (0, 2) if scattered else (8, 10))
                self.assertEqual(dp_slice.call_count, 0 if scattered else 1)

    def test_short_capture_trims_cache_locations_for_both_copy_paths(self) -> None:
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                method = _method("state_capturer/base.py", "on_forward_end", {"TopkCaptureOutput": SimpleNamespace})
                host = Mock()
                host.__setitem__ = Mock()
                capturer = SimpleNamespace(
                    _get_local_slice=lambda *args: _Rows([10, 11]), host_cache=SimpleNamespace(buffer=host)
                )
                batch = SimpleNamespace(out_cache_loc=_Rows([0, 1, 2, 3]), out_cache_loc_virtual=None)
                result = method(capturer, batch, True, 8, no_copy_to_cpu=overlap)
                if overlap:
                    self.assertEqual(result.out_cache_loc.values, [0, 1])
                    self.assertEqual(result.topk.values, [10, 11])
                else:
                    locations, rows = host.__setitem__.call_args.args
                    self.assertEqual(len(locations.values), len(rows.values))


class TestDeepseekV41GraphStride(unittest.TestCase):
    def test_decode_prefill_and_eager_report_their_own_token_stride(self) -> None:
        class EagerRunner:
            pass

        method = _method(
            "model_executor/model_runner.py",
            "_forward_raw",
            {
                "has_forward_context": lambda: True,
                "contextlib": SimpleNamespace(nullcontext=nullcontext),
                "get_global_dwdp_manager": lambda: None,
                "_prefill_cuda_graph_allows_context_parallel": lambda *args: True,
                "device_timer_ctx": lambda *args: nullcontext(),
                "EagerRunner": EagerRunner,
                "ModelRunnerOutput": SimpleNamespace,
            },
        )
        for mode, expected_stride in (("decode", 24), ("prefill", 64), ("eager", None)):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_cuda_graph=lambda: mode == "decode",
                        is_decode=lambda: False,
                        is_split_prefill=lambda: False,
                        is_extend=lambda **kwargs: mode == "prefill",
                        is_target_verify=lambda: False,
                    ),
                    token_indices_to_pool=None,
                    global_num_tokens_cpu=None,
                )
                runner = SimpleNamespace(
                    device="cuda",
                    decode_cuda_graph_runner=SimpleNamespace(
                        bs=8,
                        captured_req_width=3,
                        can_run_graph=lambda batch: True,
                        execute=lambda *args, **kwargs: mode,
                    ),
                    prefill_cuda_graph_runner=SimpleNamespace(
                        last_replay_num_tokens=64,
                        can_run_graph=lambda batch: True,
                        execute=lambda *args, **kwargs: mode,
                    ),
                    eager_runner=SimpleNamespace(execute=lambda *args, **kwargs: mode),
                    hisparse_coordinator=None,
                    device_timer=None,
                    _prepare_eager_forward_batch=lambda batch: None,
                    _maybe_execute_deferred_mamba_cow_and_clear=lambda batch: None,
                    _extend_forward_kwargs=lambda *args: {},
                )
                result = method(runner, batch, None)
                self.assertEqual(getattr(result, "graph_num_tokens", None), expected_stride)
                self.assertEqual(result.logits_output, mode)
                self.assertEqual(result.can_run_graph, mode != "eager")


class TestDeepseekV41WeightUpdate(unittest.TestCase):
    def test_tensor_update_finishes_ipc_copies_before_reply(self) -> None:
        for load_format in (None, "direct", "flattened_bucket", "custom"):
            with self.subTest(load_format=load_format):
                events: list[str] = []
                device = SimpleNamespace(current_device=lambda: 0, synchronize=lambda: events.append("synchronize"))
                load = lambda *args, **kwargs: events.append("load")
                method = _method(
                    "model_executor/model_runner_components/weight_updater.py",
                    "update_weights_from_tensor",
                    {
                        "_unsupported_derived_weight_cache_error": lambda model: None,
                        "monkey_patch_torch_reductions": lambda: None,
                        "torch": SimpleNamespace(get_device_module=lambda target: device),
                        "_unwrap_tensor": lambda tensor, **kwargs: tensor,
                        "_model_load_weights_direct": load,
                        "dynamic_import": lambda name: load,
                    },
                )
                updater = SimpleNamespace(
                    device="cuda:0",
                    tp_rank=0,
                    get_model=lambda: SimpleNamespace(load_weights=load),
                    _assert_weight_cache_inactive=lambda operation: None,
                    custom_weight_loaders={"custom"},
                    _update_weights_from_flattened_bucket=lambda **kwargs: (load(), (True, "Success"))[1],
                )
                payload = {} if load_format == "flattened_bucket" else [("weight", object())]
                self.assertEqual(method(updater, payload, load_format), (True, "Success"))
                self.assertEqual(events, ["load", "synchronize"])

    def test_kv_cache_clear_finishes_before_memory_unmap(self) -> None:
        events: list[str] = []
        method = _method(
            "managers/scheduler_components/weight_updater.py",
            "release_memory_occupation",
            {
                "GPU_MEMORY_TYPE_KV_CACHE": "kv",
                "GPU_MEMORY_TYPE_WEIGHTS": "weights",
                "GPU_MEMORY_TYPE_CUDA_GRAPH": "graph",
                "torch": SimpleNamespace(
                    get_device_module=lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))
                ),
                "ReleaseMemoryOccupationReqOutput": SimpleNamespace,
            },
        )
        manager = SimpleNamespace(
            scheduler=None,
            offload_tags=set(),
            is_fully_idle=lambda **kwargs: True,
            flush_cache=lambda: (events.append("flush"), True)[1],
            memory_saver_adapter=SimpleNamespace(pause=lambda tag: events.append("unmap")),
        )
        method(manager, SimpleNamespace(tags=["kv"]))
        self.assertEqual(events, ["flush", "sync", "unmap", "sync"])

    def test_loader_calls_duck_typed_marlin_restore_hook(self) -> None:
        restore = Mock()
        layer = object()
        method = _method(
            "model_loader/loader.py",
            "restore_weights_before_loading",
            {
                "_modules_with_quant_method": lambda model: [
                    (layer, SimpleNamespace(restore_weights_before_loading=restore)),
                    (object(), object()),
                    (object(), SimpleNamespace(restore_weights_before_loading=None)),
                ],
                "device_loading_context": lambda *args: nullcontext(),
            },
        )
        method(object(), "cuda:0")
        restore.assert_called_once_with(layer)

    def test_repeated_marlin_restore_does_not_clear_loaded_buckets(self) -> None:
        method = _method("layers/quantization/mxfp4_marlin_moe.py", "restore_weights_before_loading", {})
        layer = SimpleNamespace(_mxfp4_marlin_reload_in_place=True, w13_weight=Mock())
        method(SimpleNamespace(), layer)
        layer.w13_weight.data.zero_.assert_not_called()

    def test_repeated_marlin_postprocess_does_not_repack_runtime_bytes(self) -> None:
        method = _method("layers/quantization/mxfp4_marlin_moe.py", "process_weights_after_loading", {})
        fp8 = Mock()
        layer = SimpleNamespace(_mxfp4_marlin_runtime_parameter_specs={}, _mxfp4_marlin_reload_in_place=False)
        method(SimpleNamespace(_fp8=fp8), layer)
        fp8.process_weights_after_loading.assert_not_called()


if __name__ == "__main__":
    unittest.main()

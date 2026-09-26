# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.deepeyes.processor_patch_utils import (
    SUPPORTED_SGLANG_VERSION,
    collapse_consecutive_image_tokens,
    merge_mrope_grid_items,
    process_preexpanded_inputs,
    validate_sglang_contract,
)


@dataclass
class _ProcessorOutput:
    input_ids: list[int]
    mm_items: object
    mrope_positions: object
    mrope_position_delta: object


def _processor_class(calls):
    class FakeQwenVLImageProcessor:
        def __init__(self):
            self.mm_tokens = SimpleNamespace(image_token_id=151655)

        def compute_mrope_positions(self, input_ids, mm_items):
            calls.append(("mrope", input_ids, mm_items))
            return "restored positions", "restored delta"

        async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
            calls.append(input_text)
            return _ProcessorOutput(
                input_ids=[90, 91],
                mm_items=image_data,
                mrope_positions="temporary positions",
                mrope_position_delta="temporary delta",
            )

    return FakeQwenVLImageProcessor


def test_collapse_consecutive_image_tokens_preserves_boundaries():
    input_ids = [1, 151655, 151655, 2, 151655, 151655, 151655, 3]

    assert collapse_consecutive_image_tokens(input_ids, 151655) == [1, 151655, 2, 151655, 3]
    assert collapse_consecutive_image_tokens("raw prompt", 151655) == "raw prompt"


def test_deepeyes_does_not_overwrite_sglang_installation():
    repo_root = Path(__file__).resolve().parents[2]
    run_script = (repo_root / "examples/deepeyes/run_deepeyes_r3.sh").read_text()

    assert "--sglang-external-model-package examples.deepeyes.sglang_patch" in run_script
    assert "/sgl-workspace" not in run_script
    assert "cp " not in run_script
    assert not (repo_root / "examples/deepeyes/qwen_vl.py").exists()


def test_patch_preserves_upstream_outputs_and_recomputes_mrope():
    calls = []
    processor_cls = _processor_class(calls)
    validate_sglang_contract(processor_cls, str(SUPPORTED_SGLANG_VERSION))
    processor = processor_cls()
    input_ids = [1, 151655, 151655, 2]

    output = asyncio.run(
        process_preexpanded_inputs(
            processor,
            processor.process_mm_data_async,
            ["image"],
            input_ids,
            SimpleNamespace(),
        )
    )

    assert calls == [[1, 151655, 2], ("mrope", input_ids, ["image"])]
    assert output.input_ids == input_ids
    assert output.mm_items == ["image"]
    assert output.mrope_positions == "restored positions"
    assert output.mrope_position_delta == "restored delta"


def test_merge_mrope_grid_items_combines_multi_turn_image_grids():
    first = SimpleNamespace(model_specific_data={"image_grid_thw": [[1, 32, 24]]})
    second = SimpleNamespace(model_specific_data={"image_grid_thw": [[1, 16, 12]]})

    merged = merge_mrope_grid_items([first, second])

    assert len(merged) == 1
    assert merged[0].model_specific_data["image_grid_thw"] == [[1, 32, 24], [1, 16, 12]]


def test_patch_is_stateless_and_leaves_text_prompts_unchanged():
    calls = []
    processor = _processor_class(calls)()

    first = asyncio.run(
        process_preexpanded_inputs(
            processor,
            processor.process_mm_data_async,
            [],
            "raw prompt",
            SimpleNamespace(),
        )
    )
    second = asyncio.run(
        process_preexpanded_inputs(
            processor,
            processor.process_mm_data_async,
            [],
            "raw prompt",
            SimpleNamespace(),
        )
    )

    assert calls == ["raw prompt", "raw prompt"]
    assert first == second


def test_patch_fails_fast_for_unsupported_sglang_version():
    processor_cls = _processor_class([])

    with pytest.raises(RuntimeError, match="supports SGLang"):
        validate_sglang_contract(processor_cls, "0.5.13")


def test_patch_fails_fast_for_incompatible_method_signature():
    class IncompatibleProcessor:
        def compute_mrope_positions(self, input_ids, mm_items):
            return input_ids, mm_items

        async def process_mm_data_async(self, request):
            return request

    with pytest.raises(RuntimeError, match="incompatible signature"):
        validate_sglang_contract(IncompatibleProcessor, str(SUPPORTED_SGLANG_VERSION))

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise installed Bridge text and multimodal RoPE behavior on CPU."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
VISION, IMAGE, VIDEO = 248053, 248056, 248057


@pytest.fixture
def runtime(monkeypatch):
    # Bridge's package initializer registers unrelated models. Execute its installed
    # RoPE functions with the same CPU metadata boundary as the original test.
    bridge = Path(importlib.util.find_spec("megatron.bridge").origin).parent
    target = bridge / "models/qwen_vl/modelling_qwen3_vl/rope.py"
    spec = importlib.util.spec_from_file_location("text_positions_recipe", ROOT / "relax/models/qwen3_5/recipe.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    monkeypatch.setitem(sys.modules, "relax.models.qwen3_5.recipe", recipe)

    def load_functions(text):
        names = {
            "_get_flat_packed_ranges",
            "get_packed_seq_attention_mask",
            "_build_llm_rope_positions",
            "get_rope_index",
        }
        functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(functions) == len(names)
        scope = dict(
            torch=torch,
            sys=sys,
            Optional=Optional,
            PackedSeqParams=SimpleNamespace,
            get_packed_seq_q_cu_seqlens=lambda params: (params.cu_seqlens_q, params.cu_seqlens_q_padded),
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(target), "exec"), scope)
        return SimpleNamespace(**scope)

    return recipe, load_functions(target.read_text())


def kwargs(**changes):
    result = dict(
        spatial_merge_size=2,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VISION,
        image_grid_thw=None,
        video_grid_thw=None,
        image_index=0,
        video_index=0,
    )
    result.update(changes)
    return result


@pytest.mark.parametrize("tokens", [[7, 8, 9], [VISION, IMAGE, 9], [VISION, VIDEO, 9], [7, VISION]])
def test_text_markers_are_ordinary_positions(runtime, tokens):
    recipe, patched = runtime
    if tokens != [7, 8, 9]:
        with pytest.raises((TypeError, IndexError)):
            patched._build_llm_rope_positions(torch.tensor(tokens), **kwargs())
    recipe._TEXT_ONLY_CPT = True
    actual, image_index, video_index = patched._build_llm_rope_positions(
        torch.tensor(tokens), **kwargs(image_index=3, video_index=4)
    )
    torch.testing.assert_close(actual, torch.arange(len(tokens)).expand(3, -1))
    assert actual.dtype == torch.int64 and actual.device.type == "cpu"
    assert (image_index, video_index) == (3, 4)


def test_flat_packed_documents_reset_and_preserve_padding_and_deltas(runtime):
    recipe, patched = runtime
    params = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 5]), cu_seqlens_q_padded=torch.tensor([0, 4, 8]))
    arguments = kwargs()
    arguments.pop("image_index")
    arguments.pop("video_index")
    plain = torch.tensor([[7, 8, 9, 0, 10, 11, 0, 0]])
    expected, expected_deltas = patched.get_rope_index(input_ids=plain, packed_seq_params=params, **arguments)
    marked = torch.tensor([[VISION, IMAGE, 9, 0, 10, VISION, 0, 0]])
    with pytest.raises(TypeError):
        patched.get_rope_index(input_ids=marked, packed_seq_params=params, **arguments)
    recipe._TEXT_ONLY_CPT = True
    actual, deltas = patched.get_rope_index(input_ids=marked, packed_seq_params=params, **arguments)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(deltas, expected_deltas)
    torch.testing.assert_close(actual[:, 0], torch.tensor([[0, 1, 2, 1, 0, 1, 1, 1]]).expand(3, -1))
    torch.testing.assert_close(deltas, torch.tensor([[-1], [-2]]))


@pytest.mark.parametrize("kind", ["image", "video"])
def test_real_grid_uses_original_multimodal_positions(runtime, kind):
    recipe, patched = runtime
    arguments = kwargs(**{f"{kind}_grid_thw": torch.tensor([[1, 4, 4]])})
    token = IMAGE if kind == "image" else VIDEO
    tokens = torch.tensor([7, VISION, token, token, token, token, 8])
    expected = patched._build_llm_rope_positions(tokens, **arguments)
    recipe._TEXT_ONLY_CPT = True
    actual = patched._build_llm_rope_positions(tokens, **arguments)
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("loaded", [False, True])
def test_disabled_recipe_preserves_native_behavior(runtime, monkeypatch, loaded):
    recipe, patched = runtime
    assert recipe._TEXT_ONLY_CPT is False
    if not loaded:
        monkeypatch.delitem(sys.modules, "relax.models.qwen3_5.recipe")
    tokens = torch.tensor([7, 8, 9])
    torch.testing.assert_close(
        patched._build_llm_rope_positions(tokens, **kwargs())[0],
        torch.arange(3).expand(3, -1),
    )
    with pytest.raises(TypeError):
        patched._build_llm_rope_positions(torch.tensor([VISION, IMAGE, 9]), **kwargs())

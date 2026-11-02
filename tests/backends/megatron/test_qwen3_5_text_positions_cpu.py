# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the static RoPE patch on pinned Bridge functions without CUDA.

Set QWEN_BRIDGE_ROPE_SOURCE to rope.py from Bridge
6862ee170a2569607e8ebc5dde91f119779093ca.
"""

import ast
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
VISION, IMAGE, VIDEO = 248053, 248056, 248057


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = os.environ.get("QWEN_BRIDGE_ROPE_SOURCE")
    if not source:
        pytest.skip("Requires pinned Bridge rope.py; no GPU is required")
    source = Path(source).read_bytes()
    assert hashlib.sha256(source).hexdigest() == "3f0fc02e8d8bee962be3c66bb8f37c0ba124a7510ce4f323c4edef0f579d9818"
    name = "megatron/bridge/models/qwen_vl/modelling_qwen3_vl/rope.py"
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    patch = (ROOT / "docker/patch/megatron/20261020-model-operators.patch").read_text()
    section = next(part for part in patch.split("diff --git ") if part.startswith(f"a/{name} "))
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1"],
        input="diff --git " + section,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "offset" not in result.stdout and "fuzz" not in result.stdout
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

    return recipe, load_functions(source), load_functions(target.read_text())


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
    recipe, native, patched = runtime
    if tokens != [7, 8, 9]:
        with pytest.raises((TypeError, IndexError)):
            native._build_llm_rope_positions(torch.tensor(tokens), **kwargs())
    recipe._TEXT_ONLY_CPT = True
    actual, image_index, video_index = patched._build_llm_rope_positions(
        torch.tensor(tokens), **kwargs(image_index=3, video_index=4)
    )
    torch.testing.assert_close(actual, torch.arange(len(tokens)).expand(3, -1))
    assert actual.dtype == torch.int64 and actual.device.type == "cpu"
    assert (image_index, video_index) == (3, 4)


def test_flat_packed_documents_reset_and_preserve_padding_and_deltas(runtime):
    recipe, native, patched = runtime
    params = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 5]), cu_seqlens_q_padded=torch.tensor([0, 4, 8]))
    arguments = kwargs()
    arguments.pop("image_index")
    arguments.pop("video_index")
    plain = torch.tensor([[7, 8, 9, 0, 10, 11, 0, 0]])
    expected, expected_deltas = native.get_rope_index(input_ids=plain, packed_seq_params=params, **arguments)
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
@pytest.mark.parametrize("cpt", [False, True])
def test_real_grid_uses_original_multimodal_positions(runtime, kind, cpt):
    recipe, native, patched = runtime
    recipe._TEXT_ONLY_CPT = cpt
    arguments = kwargs(**{f"{kind}_grid_thw": torch.tensor([[1, 4, 4]])})
    token = IMAGE if kind == "image" else VIDEO
    tokens = torch.tensor([7, VISION, token, token, token, token, 8])
    expected = native._build_llm_rope_positions(tokens, **arguments)
    actual = patched._build_llm_rope_positions(tokens, **arguments)
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("loaded", [False, True])
def test_disabled_recipe_preserves_native_behavior(runtime, monkeypatch, loaded):
    recipe, native, patched = runtime
    assert recipe._TEXT_ONLY_CPT is False
    if not loaded:
        monkeypatch.delitem(sys.modules, "relax.models.qwen3_5.recipe")
    tokens = torch.tensor([7, 8, 9])
    torch.testing.assert_close(
        patched._build_llm_rope_positions(tokens, **kwargs())[0],
        native._build_llm_rope_positions(tokens, **kwargs())[0],
    )
    with pytest.raises(TypeError):
        patched._build_llm_rope_positions(torch.tensor([VISION, IMAGE, 9]), **kwargs())

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the installed Qwen PLE shard copier without importing SGLang."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch


@pytest.fixture
def copy_ple_rows() -> Callable:
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang checkout")
    path = Path(source) / "python/sglang/srt/models/qwen4_exp.py"
    tree = ast.parse(path.read_text())
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpForConditionalGeneration"
    )
    loader = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    copier = next(
        node
        for node in loader.body
        if isinstance(node, ast.FunctionDef) and node.name == "copy_ple_rows_to_tp_embedding"
    )
    module = ast.Module(body=[copier], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[copier.name]


def _embedding(weight: torch.Tensor, start: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        shard_indices=SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=start + weight.shape[0]),
    )


@pytest.mark.parametrize("row_start,row_end", [(2, 7), (6, 11), (0, 3), (9, 12)])
def test_qwen38_ple_copy_updates_only_overlapping_rows(copy_ple_rows: Callable, row_start: int, row_end: int) -> None:
    emb = _embedding(torch.full((5, 3), -1.0))
    source = torch.arange((row_end - row_start) * 3, dtype=torch.float32).reshape(-1, 3)
    expected = emb.weight.detach().clone()
    for global_row in range(max(4, row_start), min(9, row_end)):
        expected[global_row - 4] = source[global_row - row_start]
    pointer = emb.weight.data_ptr()
    copy_ple_rows(emb, source, row_start, row_end)
    torch.testing.assert_close(emb.weight, expected, rtol=0, atol=0)
    assert emb.weight.data_ptr() == pointer


def test_qwen38_ple_copy_reloads_noncontiguous_source_in_place(copy_ple_rows: Callable) -> None:
    emb = _embedding(torch.zeros(5, 3, dtype=torch.bfloat16))
    pointer = emb.weight.data_ptr()
    for offset in (1, 10):
        source = (torch.arange(15, dtype=torch.bfloat16).reshape(3, 5) + offset).t()
        assert not source.is_contiguous()
        copy_ple_rows(emb, source, 4, 9)
        torch.testing.assert_close(emb.weight, source, rtol=0, atol=0)
        assert emb.weight.data_ptr() == pointer


@pytest.mark.parametrize(
    "source_dtype,target_dtype", [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)]
)
def test_qwen38_ple_copy_converts_dtype(
    copy_ple_rows: Callable, source_dtype: torch.dtype, target_dtype: torch.dtype
) -> None:
    source = torch.linspace(-2.1, 3.7, 15, dtype=source_dtype).reshape(5, 3)
    emb = _embedding(torch.zeros(5, 3, dtype=target_dtype))
    pointer = emb.weight.data_ptr()
    copy_ple_rows(emb, source, 4, 9)
    torch.testing.assert_close(emb.weight, source.to(target_dtype), rtol=0, atol=0)
    assert emb.weight.data_ptr() == pointer


@pytest.mark.parametrize("to_fp8", [False, True])
def test_qwen38_ple_copy_preserves_fp8_cast(copy_ple_rows: Callable, to_fp8: bool) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("This PyTorch build has no float8_e4m3fn dtype")
    fp8 = torch.float8_e4m3fn
    source_dtype, target_dtype = (torch.bfloat16, fp8) if to_fp8 else (fp8, torch.bfloat16)
    try:
        source = torch.linspace(-2.1, 3.7, 15).reshape(5, 3).to(source_dtype)
        expected = source.to(target_dtype)
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"This PyTorch build does not support the required FP8 cast: {error}")
    emb = _embedding(torch.empty(5, 3, dtype=target_dtype))
    pointer = emb.weight.data_ptr()
    copy_ple_rows(emb, source, 4, 9)
    torch.testing.assert_close(emb.weight.float(), expected.float(), rtol=0, atol=0)
    assert emb.weight.data_ptr() == pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-to-pinned-host PLE copy requires a GPU")
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_qwen38_ple_copy_cuda_to_pinned_host_is_ready_on_return(copy_ple_rows: Callable, noncontiguous: bool) -> None:
    shape = (257, 32)
    emb = _embedding(torch.empty(shape, dtype=torch.bfloat16, pin_memory=True))
    pointer = emb.weight.data_ptr()
    for offset in (1, 2):
        expected = (torch.arange(shape[0] * shape[1]).reshape(shape) % 97 + offset).to(torch.bfloat16)
        source = expected.cuda()
        if noncontiguous:
            source = source.t().contiguous().t()
            assert not source.is_contiguous()
        # Queue the write immediately before copying. Do not synchronize before
        # the CPU assertion: successful return must make host rows readable.
        source.add_(1)
        copy_ple_rows(emb, source, 4, 4 + shape[0])
        torch.testing.assert_close(emb.weight, expected + 1, rtol=0, atol=0)
        assert emb.weight.data_ptr() == pointer
        assert emb.weight.is_pinned()

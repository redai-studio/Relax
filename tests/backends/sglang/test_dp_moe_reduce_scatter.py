# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check cancellation, buffer dtypes and owner layout of the shipped SGLang
patch."""

import ast
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")


def _combine(rank: int, dp_size: int, dtype, *, variable: bool = False):
    calls = []
    allocations = []
    rows, width, tp_size = 8, 4, 4
    contribution = (256, 1, -256, 0.5)
    source = torch.full((rows, width), contribution[rank], dtype=dtype)
    sizes = [0, 1, 3, 4] if variable else [rows // dp_size] * dp_size
    output = torch.empty((sizes[rank] if variable else rows // dp_size, width), dtype=dtype)

    def reduce_scatter(destination, value):
        expected_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        assert destination.dtype == value.dtype == expected_dtype
        assert value.shape == (rows, width)
        assert destination.shape == (rows // tp_size, width)
        assert value.numel() == destination.numel() * tp_size
        torch.testing.assert_close(
            value, torch.full((rows, width), contribution[rank], dtype=expected_dtype), rtol=0, atol=0
        )
        # Emulate destination-dependent low-precision cancellation. Accumulator
        # precision comes from the actual tensor supplied to the collective.
        order = [(rank + offset + 1) % tp_size for offset in range(tp_size)]
        total = torch.zeros((), dtype=value.dtype)
        for peer in order:
            total.add_(contribution[peer])
        destination.fill_(total)
        calls.append(("reduce_scatter", value.dtype, destination.dtype))

    def all_gather(destination, value):
        assert destination.dtype == value.dtype
        assert value.shape == (rows // tp_size, width)
        assert destination.shape == (rows // dp_size, width)
        assert destination.numel() == value.numel() * (tp_size // dp_size)
        destination.copy_(value.repeat(2, 1))
        calls.append(("all_gather", value.dtype, destination.dtype))

    def reduce_scatterv(value, *, output, sizes):
        assert sizes == [0, 1, 3, 4] and value is source
        output.fill_(10)
        calls.append(("reduce_scatterv", value.dtype, output.dtype))

    tp = SimpleNamespace(reduce_scatter_tensor=reduce_scatter, reduce_scatterv=reduce_scatterv)
    attn = SimpleNamespace(all_gather_into_tensor=all_gather)

    @contextmanager
    def symmetric_memory(group, *, disabled):
        assert not disabled
        allocations.append(group)
        yield

    namespace = {
        "torch": torch,
        "is_dp_gatherv_active": lambda: variable,
        "get_dp_global_num_tokens": lambda: sizes,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "get_tensor_model_parallel_rank": lambda: rank,
        "get_attention_dp_size": lambda: dp_size,
        "get_tp_group": lambda: tp,
        "get_attn_tp_group": lambda: attn,
        "use_symmetric_memory": symmetric_memory,
        "is_dp_max_padding": lambda: True,
    }
    path = sglang_source_path("srt/layers/dp_attention.py")
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "dp_reduce_scatter_tensor"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    pointer = output.data_ptr()
    result = namespace["dp_reduce_scatter_tensor"](output, source)
    assert result is None and output.data_ptr() == pointer and output.dtype == dtype
    return output, calls, allocations, tp, attn


@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_bf16_combine_preserves_cancellation_across_owners(rank: int, dp_size: int) -> None:
    output, calls, allocations, tp, attn = _combine(rank, dp_size, torch.bfloat16)
    torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
    assert calls[0] == ("reduce_scatter", torch.float32, torch.float32)
    assert allocations[0] is tp
    if dp_size == 2:
        assert calls[1] == ("all_gather", torch.bfloat16, torch.bfloat16)
        assert allocations[1] is attn
    else:
        assert allocations[1] is tp and len(calls) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dp_size", [2, 4])
def test_non_bf16_combine_preserves_collective_dtype(dtype, dp_size: int) -> None:
    output, calls, allocations, _, _ = _combine(0, dp_size, dtype)
    torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
    assert all(call[1:] == (dtype, dtype) for call in calls)
    assert not allocations


@pytest.mark.parametrize("rank", range(4))
def test_variable_combine_keeps_existing_sizes_and_empty_owner(rank: int) -> None:
    output, calls, allocations, _, _ = _combine(rank, 4, torch.bfloat16, variable=True)
    assert output.shape[0] == [0, 1, 3, 4][rank]
    torch.testing.assert_close(output, torch.full_like(output, 10), rtol=0, atol=0)
    assert calls == [("reduce_scatterv", torch.bfloat16, torch.bfloat16)]
    assert not allocations

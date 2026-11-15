# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check cancellation, buffer dtypes and owner layout of the shipped SGLang
patch."""

import ast
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import MethodType, SimpleNamespace
from typing import Optional

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")


def _load_reduce_scatterv(name: str = "reduce_scatterv", *, deterministic: bool = False) -> Callable[..., object]:
    path = sglang_source_path("srt/distributed/parallel_state.py")
    group = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "GroupCoordinator"
    )
    function = next(node for node in group.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "List": list,
        "envs": SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(get=lambda: deterministic)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _check_deterministic_reduce_scatterv(rank: int, init_method: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=4, init_method=init_method, timeout=timedelta(seconds=60)
    )
    try:

        def all_reduce(value):
            torch.distributed.all_reduce(value, group=torch.distributed.group.WORLD)
            return value

        group = SimpleNamespace(world_size=4, rank_in_group=rank, pynccl_comm=None, all_reduce=all_reduce)
        group._deterministic_reduce_scatterv = MethodType(
            _load_reduce_scatterv("_deterministic_reduce_scatterv"), group
        )
        reduce_scatter = MethodType(_load_reduce_scatterv(deterministic=True), group)
        # Check real cancellation, an empty owner, caller-owned storage and aliasing.
        for sizes, output_mode in [([0, 1, 3, 4], "provided"), (None, "alias"), (None, "allocated")]:
            counts = sizes if sizes is not None else [2] * 4
            count, offset = counts[rank], sum(counts[:rank])
            source = torch.full((sum(counts), 4), (256, 1, -256, 0.5)[rank], dtype=torch.bfloat16)
            original = source.clone()
            output = None
            if output_mode == "provided":
                output = torch.empty((count, 4), dtype=source.dtype)
            elif output_mode == "alias":
                output = source.narrow(0, offset, count)
            result = reduce_scatter(source, output=output, sizes=sizes)
            assert result.dtype == source.dtype
            if output is not None:
                assert result is output
            torch.testing.assert_close(result, torch.full((count, 4), 1.5, dtype=source.dtype), rtol=0, atol=0)
            if output_mode == "alias":
                original.narrow(0, offset, count).fill_(1.5)
            torch.testing.assert_close(source, original, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_deterministic_bf16_reduce_scatterv_uses_fp32_accumulation(tmp_path) -> None:
    source = ast.parse(sglang_source_path("srt/distributed/parallel_state.py").read_text())
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "_deterministic_reduce_scatterv"
        for node in ast.walk(source)
    ):
        pytest.skip("This SGLang version has no deterministic reduce_scatterv path")
    torch.multiprocessing.spawn(
        _check_deterministic_reduce_scatterv, args=(f"file://{tmp_path / 'gloo-init'}",), nprocs=4, join=True
    )


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
        assert destination.shape == (sizes[rank] if variable else rows // tp_size, width)
        if variable:
            assert value.numel() == sum(sizes) * width
        else:
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
        calls.append(("reduce_scatterv" if variable else "reduce_scatter", value.dtype, destination.dtype))

    def all_gather(destination, value):
        assert destination.dtype == value.dtype
        assert value.shape == (rows // tp_size, width)
        assert destination.shape == (rows // dp_size, width)
        assert destination.numel() == value.numel() * (tp_size // dp_size)
        destination.copy_(value.repeat(2, 1))
        calls.append(("all_gather", value.dtype, destination.dtype))

    def reduce_scatterv(destination, value, *, sizes):
        assert sizes == [0, 1, 3, 4]
        reduce_scatter(destination, value)

    tp = SimpleNamespace(reduce_scatter_tensor=reduce_scatter, world_size=tp_size, rank_in_group=rank)
    if variable:
        tp.pynccl_comm = SimpleNamespace(
            disabled=False, change_state=lambda **_: nullcontext(), reduce_scatter=reduce_scatterv
        )
        tp.reduce_scatterv = MethodType(_load_reduce_scatterv(), tp)
    attn = SimpleNamespace(all_gather_into_tensor=all_gather)

    @contextmanager
    def symmetric_memory(group, *, disabled):
        assert not disabled
        allocations.append(group)
        yield

    namespace = {
        "torch": torch,
        "_note_dp_gather_in_prefill_graph": lambda: None,
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
    namespace["_note_dp_gather_in_prefill_graph"] = lambda: None
    namespace["get_parallel"] = lambda: SimpleNamespace(
        tp_size=tp_size, tp_rank=rank, attn_dp_size=dp_size, tp_group=tp, attn_tp_group=attn
    )
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
def test_variable_combine_preserves_cancellation_and_empty_owner(rank: int) -> None:
    output, calls, allocations, _, _ = _combine(rank, 4, torch.bfloat16, variable=True)
    assert output.shape[0] == [0, 1, 3, 4][rank]
    torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
    assert calls == [("reduce_scatterv", torch.float32, torch.float32)]
    assert not allocations


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("sizes", [[0, 1, 3, 4], None, [0, 0, 0, 0]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("output_mode", ["allocated", "provided", "alias"])
def test_reduce_scatterv_preserves_owner_values_and_buffers(
    rank: int, sizes: list[int] | None, dtype: torch.dtype, output_mode: str
) -> None:
    rows, width = (sum(sizes) if sizes is not None else 8), 4
    counts = sizes if sizes is not None else [rows // 4] * 4
    count, offset = counts[rank], sum(counts[:rank])
    row_values = 2 * torch.arange(rows).reshape(-1, 1) + torch.arange(width) / 4

    def contribution(peer: int) -> torch.Tensor:
        if peer == 1:
            return (row_values + 1).to(dtype)
        return torch.full((rows, width), (256, 0, -256, 0.5)[peer], dtype=dtype)

    source = contribution(rank)
    original = source.clone()
    output = None
    if output_mode == "provided":
        output = torch.empty((count, width), dtype=dtype)
    elif output_mode == "alias":
        output = source.narrow(0, offset, count)
    calls = []
    comm = SimpleNamespace(disabled=True)
    expected_sizes = sizes

    @contextmanager
    def change_state(*, enable: bool) -> Iterator[None]:
        assert enable and comm.disabled
        comm.disabled = False
        try:
            yield
        finally:
            comm.disabled = True

    def reduce_scatter(destination: torch.Tensor, value: torch.Tensor, *, sizes: list[int] | None) -> None:
        assert not comm.disabled
        assert sizes == expected_sizes
        accumulator_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        assert value.dtype == destination.dtype == accumulator_dtype
        assert value.shape == (rows, width) and destination.shape == (count, width)
        assert value.numel() == sum(counts) * width
        torch.testing.assert_close(value, original.to(accumulator_dtype), rtol=0, atol=0)
        if dtype != torch.bfloat16:
            assert value is source
            if output is not None:
                assert destination is output
        total = torch.zeros((count, width), dtype=value.dtype)
        for peer in [(rank + step + 1) % 4 for step in range(4)]:
            total.add_(contribution(peer).narrow(0, offset, count))
        destination.copy_(total)
        calls.append(True)

    comm.change_state = change_state
    comm.reduce_scatter = reduce_scatter
    group = SimpleNamespace(world_size=4, rank_in_group=rank, pynccl_comm=comm)
    result = _load_reduce_scatterv()(group, source, output=output, sizes=sizes)
    expected = (row_values + 1.5).to(dtype).narrow(0, offset, count)
    assert calls == [True] and comm.disabled
    assert result.dtype == dtype and result.shape == (count, width)
    if output is not None:
        assert result is output
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if output_mode == "alias":
        original.narrow(0, offset, count).copy_(expected)
    torch.testing.assert_close(source, original, rtol=0, atol=0)

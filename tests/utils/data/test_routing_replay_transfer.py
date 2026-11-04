"""Routing replay transfer regressions without importing Megatron or Ray.

Run the NCCL cases explicitly on two otherwise available GPUs with
``RELAX_ROUTING_REPLAY_GPU_TEST=1 pytest -q tests/utils/data/test_routing_replay_transfer.py``.
"""

import ast
import gc
import os
import sys
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_ROOT = Path(__file__).resolve().parents[3]
_CHUNK_BYTES = 64 * 1024 * 1024
_GPU_TESTS = os.environ.get("RELAX_ROUTING_REPLAY_GPU_TEST") == "1"


def _load_functions(*names: str, **namespace):
    """Execute the production functions, leaving optional runtime imports
    out."""
    source = _ROOT / "relax/utils/data/stream_dataloader.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    namespace = {"torch": torch, **namespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return tuple(namespace[name] for name in names)


class _ParallelGroups:
    """Real process groups in Megatron's CP/PP/TP fanout order."""

    def __init__(self, rank: int, cp_size: int, pp_size: int, tp_size: int):
        self.cp_size, self.pp_size, self.tp_size = cp_size, pp_size, tp_size
        self.tp_rank = rank % tp_size
        self.cp_rank = rank // tp_size % cp_size
        self.pp_rank = rank // (tp_size * cp_size)
        self.names = {}
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = [(pp * cp_size + cp) * tp_size + tp for cp in range(cp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self.cp_group = group
                    self.names[group] = "cp"
        for cp in range(cp_size):
            for tp in range(tp_size):
                ranks = [(pp * cp_size + cp) * tp_size + tp for pp in range(pp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self.pp_group = group
                    self.names[group] = "pp"
        for pp in range(pp_size):
            for cp in range(cp_size):
                ranks = [(pp * cp_size + cp) * tp_size + tp for tp in range(tp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self.tp_group = group
                    self.names[group] = "tp"

    def get_context_parallel_world_size(self):
        return self.cp_size

    def get_tensor_model_parallel_world_size(self):
        return self.tp_size

    def get_pipeline_model_parallel_world_size(self):
        return self.pp_size

    def get_context_parallel_rank(self):
        return self.cp_rank

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def get_pipeline_model_parallel_rank(self):
        return self.pp_rank

    def get_context_parallel_group(self):
        return self.cp_group

    def get_tensor_model_parallel_group(self):
        return self.tp_group

    def get_pipeline_model_parallel_group(self):
        return self.pp_group


def _assert_pattern(values: torch.Tensor, rows: int, width: int) -> None:
    assert values.shape == (rows, width)
    assert values.dtype == torch.int32
    assert values.device.type == "cpu"
    flat = values.reshape(-1)
    # Check all elements while keeping the expected-data allocation bounded.
    for start in range(0, flat.numel(), 262_144):
        end = min(start + 262_144, flat.numel())
        expected = torch.arange(start, end, dtype=torch.int64).remainder(251).to(torch.int32)
        assert torch.equal(flat[start:end], expected)


def _transfer_worker(rank: int, init_file: str, topology: tuple[int, int, int], backend: str) -> None:
    torch.set_num_threads(1)
    cp_size, pp_size, tp_size = topology
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=cp_size * pp_size * tp_size,
        timeout=timedelta(seconds=180),
    )
    try:
        groups = _ParallelGroups(rank, *topology)
        calls = []

        def observed_broadcast(tensor, src, group):
            size = tensor.numel() * tensor.element_size()
            assert size <= _CHUNK_BYTES, f"full-batch transfer allocated {size} bytes"
            calls.append((groups.names[group], size))
            return dist.broadcast(tensor, src=src, group=group)

        (broadcast,) = _load_functions(
            "_broadcast_routed_experts",
            mpu=groups,
            dist=SimpleNamespace(broadcast=observed_broadcast, get_global_rank=dist.get_global_rank),
        )
        # The second payload is larger than two staging buffers. Both must have
        # the same bounded GPU peak; no complete payload may reside on GPU.
        widths = [17, 0, 17]
        rows_by_case = [13, 7, _CHUNK_BYTES // (17 * 4) + 19]
        if backend == "nccl":
            rows_by_case.append(2 * _CHUNK_BYTES // (17 * 4) + 19)
            widths.append(17)
        peaks = []
        for rows, width in zip(rows_by_case, widths, strict=True):
            calls.clear()
            source = None
            offsets = None
            if rank == 0:
                # Strided input exercises dtype conversion and contiguity as
                # well as payload order across chunk and sample boundaries.
                storage = torch.empty(rows, width * 2, dtype=torch.int64)
                source = storage[:, ::2]
                source.copy_(torch.arange(rows * width).remainder(251).reshape(rows, width))
                offsets = torch.tensor([0, 0, rows // 3, rows], dtype=torch.int32)
            if backend == "nccl":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
            actual, actual_offsets = broadcast(source, offsets, rank == 0, device, True)
            if backend == "nccl":
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated() - before)
                assert peaks[-1] <= _CHUNK_BYTES + 2 * 1024 * 1024
                assert torch.cuda.memory_allocated() - before < 1024 * 1024
            _assert_pattern(actual, rows, width)
            assert actual.is_contiguous()
            assert actual_offsets.device.type == "cpu"
            assert actual_offsets.dtype == torch.int64
            assert actual_offsets.tolist() == [0, 0, rows // 3, rows]
            assert [actual[int(a) : int(b)].shape[0] for a, b in zip(actual_offsets, actual_offsets[1:])] == [
                0,
                rows // 3,
                rows - rows // 3,
            ]
            expected_axes = []
            if cp_size > 1 and groups.tp_rank == 0 and groups.pp_rank == 0:
                expected_axes.append("cp")
            if pp_size > 1 and groups.tp_rank == 0:
                expected_axes.append("pp")
            if tp_size > 1:
                expected_axes.append("tp")
            # Values finish CP->PP->TP before the offsets repeat that order.
            observed_axes = [axis for i, (axis, _) in enumerate(calls) if i == 0 or axis != calls[i - 1][0]]
            expected_axes = expected_axes * 2 if len(expected_axes) > 1 else expected_axes
            assert observed_axes == expected_axes
            if rows * width * 4 > _CHUNK_BYTES:
                assert any(size == _CHUNK_BYTES for _, size in calls)
                assert sum(size > 24 for _, size in calls) >= 2
            del actual, actual_offsets, source, offsets
            gc.collect()
        if backend == "nccl":
            assert abs(peaks[-1] - peaks[-2]) < 1024 * 1024
            # The indexer replay caller still requests and receives GPU data.
            source = torch.tensor([[9, 2], [5, 0]], dtype=torch.int32) if rank == 0 else None
            offsets = torch.tensor([0, 2]) if rank == 0 else None
            actual, actual_offsets = broadcast(source, offsets, rank == 0, device, True, keep_on_gpu=True)
            assert actual.is_cuda and actual_offsets.is_cuda
            assert torch.equal(actual.cpu(), torch.tensor([[9, 2], [5, 0]], dtype=torch.int32))
    finally:
        dist.destroy_process_group()


def test_routing_replay_cp_pp_tp_broadcast_is_chunked_and_cpu_resident(tmp_path: Path) -> None:
    mp.spawn(_transfer_worker, args=(str(tmp_path / "gloo_init"), (2, 2, 2), "gloo"), nprocs=8, join=True)


@pytest.mark.skipif(not _GPU_TESTS, reason="set RELAX_ROUTING_REPLAY_GPU_TEST=1 on two available GPUs")
def test_routing_replay_nccl_staging_peak_is_independent_of_batch_size(tmp_path: Path) -> None:
    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("NCCL and two available GPUs are required")
    mp.spawn(_transfer_worker, args=(str(tmp_path / "nccl_init"), (2, 1, 1), "nccl"), nprocs=2, join=True)


def test_routing_replay_single_rank_preserves_ragged_offsets_without_collectives() -> None:
    groups = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_world_size=lambda: 1,
    )
    (broadcast,) = _load_functions("_broadcast_routed_experts", mpu=groups)
    source = torch.tensor([[8, 2], [5, 7], [3, 1]], dtype=torch.int64).T
    offsets = torch.tensor([0, 0, 1, 2], dtype=torch.int32)

    actual, actual_offsets = broadcast(source, offsets, True, torch.device("cpu"), True)

    assert torch.equal(actual, source.to(torch.int32))
    assert actual.dtype == torch.int32 and actual.device.type == "cpu" and actual.is_contiguous()
    assert torch.equal(actual_offsets, offsets.to(torch.int64))


@pytest.mark.parametrize(
    "use_gpu", [False, pytest.param(True, marks=pytest.mark.skipif(not _GPU_TESTS, reason="GPU opt-in"))]
)
def test_routing_replay_postprocess_keeps_full_batch_on_cpu(monkeypatch, use_gpu: bool) -> None:
    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", 0) if use_gpu else torch.device("cpu")
    cp_utils = ModuleType("relax.backends.megatron.cp_utils")
    cp_utils.maybe_padded_total_lengths = lambda *args, **kwargs: None
    cp_utils.slice_log_prob_with_cp = lambda values, *args, **kwargs: values
    monkeypatch.setitem(sys.modules, cp_utils.__name__, cp_utils)
    tensorclass = ModuleType("tensordict.tensorclass")
    tensorclass.NonTensorData = type("NonTensorData", (), {})
    monkeypatch.setitem(sys.modules, tensorclass.__name__, tensorclass)
    (post_process,) = _load_functions(
        "post_process_rollout_data",
        Envs=SimpleNamespace(RELAX_SFT_SELECTIVE_LOGITS=False),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: device),
        _add_sft_token_lengths=lambda *args: None,
        iter_opd_cp_float_fields=lambda: (),
    )
    routes = [torch.tensor([[4, 1], [2, 9]], dtype=torch.int64), torch.tensor([[3, 7]], dtype=torch.int64)]
    data = {
        "tokens": [[1, 2, 3], [4, 5]],
        "loss_masks": [[0, 1, 1], [0, 1]],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "rollout_routed_experts": routes,
    }
    args = Namespace(qkv_format="thd", use_opd=False, optimize_routing_replay=True)

    post_process(args, data)

    assert all(value.device == device for value in data["tokens"])
    for actual, expected in zip(data["rollout_routed_experts"], routes, strict=True):
        assert actual.device.type == "cpu"
        assert actual.dtype == torch.int32
        assert torch.equal(actual, expected.to(torch.int32))

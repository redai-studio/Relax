# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-rank Gloo coverage for native drain and Dense callback failure gates."""

from contextlib import ExitStack
from datetime import timedelta
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _info(name: str, rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        dtype=torch.float32,
        shape=torch.Size((2, 1)),
        attrs={},
        size=8,
        src_rank=rank,
    )


def _make_iterator(names: list[str], rank: int, conversion_failure: bool = False, *, failure_rank: int = 0):
    infos = [_info(name, rank) for name in names]
    iterator = bridge.HfWeightIteratorBridge.__new__(bridge.HfWeightIteratorBridge)
    iterator.lora_merge_mode = False
    iterator.lora_adapter_mode = False
    iterator.quantization_config = None
    iterator._expert_buckets = []
    iterator._non_expert_buckets = [infos]
    iterator._vanilla_key_map = {name: name for name in names}
    iterator._gdn_split_sizes = {}
    iterator.converted_names = []

    def convert(name: str, parameter: torch.Tensor):
        iterator.converted_names.append(name)
        if conversion_failure and rank == failure_rank:
            raise ValueError("injected Bridge convert failure")
        return [(f"hf.{name}", parameter.detach().clone())]

    iterator._bridge_converter = SimpleNamespace(convert=convert, broadcast_and_apply_configs=lambda: None)
    return iterator


def _rank_worker(rank: int, rendezvous: str, diagnostics: str) -> None:
    global bridge, common
    from relax.backends.megatron.weight_update import common
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:

        def mark(stage: str) -> None:
            with open(f"{diagnostics}.{rank}", "a", encoding="utf-8") as stream:
                stream.write(f"{stage}\n")

        with ExitStack() as patches:
            patches.enter_context(patch.object(bridge.mpu, "get_pipeline_model_parallel_world_size", return_value=1))
            patches.enter_context(patch.object(bridge.mpu, "get_context_parallel_world_size", return_value=1))
            patches.enter_context(patch.object(bridge.mpu, "get_expert_model_parallel_world_size", return_value=1))
            patches.enter_context(
                patch.object(bridge.mpu, "get_virtual_pipeline_model_parallel_world_size", return_value=None)
            )
            patches.enter_context(
                patch.object(bridge.mpu, "get_tensor_model_parallel_group", return_value=dist.group.WORLD)
            )
            patches.enter_context(
                patch.object(bridge.device_utils, "make_current_torch_device", return_value=torch.device("cpu"))
            )

            def load(infos, weights, key_map, _device, _rank):
                parameters = []
                for info in infos:
                    parameter = torch.nn.Parameter(weights[key_map[info.name]])
                    parameter.tensor_model_parallel = True
                    parameter.parallel_mode = "column"
                    parameter.partition_dim = 0
                    parameter.partition_stride = 1
                    parameters.append(parameter)
                return parameters

            patches.enter_context(patch.object(bridge, "_load_to_gpu", side_effect=load))

            def gated(error: str | None) -> None:
                errors = [None, None]
                dist.all_gather_object(errors, error, group=dist.group.WORLD)
                if any(errors):
                    raise RuntimeError(f"test gate failed: {errors}")

            name = "decoder.layers.0.self_attention.q_proj.weight"

            # One rank lacks the owner-local source. Both ranks must fail at
            # the pre-gather gate, then still be able to use the Gloo group.
            iterator = _make_iterator([name], rank)
            weights = {} if rank == 0 else {name: torch.ones((2, 1))}
            mark("missing-source")
            _assert_converged_error(
                lambda: list(iterator.iter_gated_nonexpert_tensors(weights, error_gate=gated)), "KeyError"
            )

            # A local empty_like allocation failure must be gated before the
            # peer enters all_gather.
            iterator = _make_iterator([name], rank)
            weights = {name: torch.full((2, 1), rank, dtype=torch.float32)}
            mark("allocation-failure")
            original_empty_like = torch.empty_like
            allocation_failure_injected = False

            def failing_empty_like(tensor, *args, **kwargs):
                nonlocal allocation_failure_injected
                if rank == 0 and not allocation_failure_injected:
                    allocation_failure_injected = True
                    raise MemoryError("injected gather allocation failure")
                return original_empty_like(tensor, *args, **kwargs)

            with patch.object(common.torch, "empty_like", side_effect=failing_empty_like):
                _assert_converged_error(
                    lambda: list(iterator.iter_gated_nonexpert_tensors(weights, error_gate=gated)), "MemoryError"
                )

            # Bridge conversion errors are reported after the data gather,
            # before another rank can continue to the next converted bucket.
            iterator = _make_iterator([name], rank, conversion_failure=True)
            mark("conversion-failure")
            _assert_converged_error(
                lambda: list(iterator.iter_gated_nonexpert_tensors(weights, error_gate=gated)),
                "injected Bridge convert failure",
            )

            # Successful TP gathering restores rank-major row concatenation.
            iterator = _make_iterator([name], rank)
            weights = {name: torch.arange(rank * 2, rank * 2 + 2, dtype=torch.float32).reshape(2, 1)}
            mark("success")
            converted = list(iterator.iter_gated_nonexpert_tensors(weights, error_gate=gated))
            assert len(converted) == 1
            assert converted[0][0] == f"hf.{name}"
            torch.testing.assert_close(converted[0][1], torch.arange(4, dtype=torch.float32).reshape(4, 1))

            # Exercise the updater's external drain contract: rank 0's local
            # encoder failure drains the iterator while rank 1 consumes it;
            # both then meet at the same final gate.
            names = [name, "decoder.layers.0.self_attention.k_proj.weight"]
            iterator = _make_iterator(names, rank)
            mark("consumer-drain")
            weights = {key: torch.full((2, 1), rank + index, dtype=torch.float32) for index, key in enumerate(names)}
            named = iterator.iter_gated_nonexpert_tensors(weights, error_gate=gated)
            consumer_error = None
            added = 0
            try:
                for output_name, tensor in named:
                    if rank == 0:
                        raise OSError("injected encoder.add failure")
                    assert output_name.startswith("hf.") and tensor.numel() > 0
                    added += 1
            except Exception as exc:
                consumer_error = f"{type(exc).__name__}: {exc}"
                try:
                    for _ in named:
                        pass
                except Exception:
                    pass
            with pytest.raises(RuntimeError, match="injected encoder.add failure"):
                gated(consumer_error)
            assert added == (0 if rank == 0 else 2)

            _exercise_dense_callback(rank, gated, mark, name, tp_size=2)
            # DP2/TP1 still shares the all-source error gates. Create both
            # singleton TP groups on both ranks in the same order.
            singleton_groups = [dist.new_group(ranks=[index], backend="gloo") for index in range(2)]
            with patch.object(bridge.mpu, "get_tensor_model_parallel_group", return_value=singleton_groups[rank]):
                _exercise_dense_callback(rank, gated, mark, name, tp_size=1)

            # This collective proves each earlier scenario left no unmatched
            # collective operation pending on either process.
            dist.barrier(group=dist.group.WORLD)
            mark("done")
    finally:
        dist.destroy_process_group()


def _assert_converged_error(function, expected: str) -> None:
    with pytest.raises(RuntimeError, match=expected) as caught:
        function()
    errors = [None, None]
    dist.all_gather_object(errors, str(caught.value), group=dist.group.WORLD)
    assert errors[0] == errors[1]
    dist.barrier(group=dist.group.WORLD)


def _exercise_dense_callback(rank, gate, mark, name, *, tp_size: int) -> None:
    from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory
    from relax.distributed.weight_sync.config import MemorySyncConfig
    from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder

    names = [name, "decoder.layers.0.self_attention.k_proj.weight"]
    weights = {
        key: torch.arange((rank if tp_size == 2 else 0) * 2, (rank if tp_size == 2 else 0) * 2 + 2).float().view(2, 1)
        for key in names
    }
    iterator = _make_iterator(names, rank)
    captured = []
    mark(f"callback-tp{tp_size}-success")
    iterator.capture_gated_dense_tensors(
        weights,
        emit=lambda key, value: captured.append((key, value.clone())) if rank == 0 else None,
        error_gate=gate,
        metadata_group=dist.group.WORLD,
    )
    assert iterator.converted_names == names  # The nonpublisher also converts.
    assert len(captured) == (2 if rank == 0 else 0)
    if rank == 0:
        for key, value in captured:
            assert key in [f"hf.{item}" for item in names]
            torch.testing.assert_close(value, torch.arange(2 * tp_size, dtype=torch.float32).view(-1, 1))

    for failure in ("prepare", "schema", "convert", "emit", "begin"):
        mark(f"callback-tp{tp_size}-{failure}")
        iterator = _make_iterator(names, rank, conversion_failure=failure == "convert", failure_rank=1)
        if failure == "schema" and rank == 1:
            iterator._non_expert_buckets[0].reverse()
        local_weights = {} if failure == "prepare" and rank == 1 else weights
        updater = object.__new__(UpdateWeightFromMemory)
        updater.config = MemorySyncConfig(stream="callback-test", bucket_bytes=16, gpu_budget_bytes=1 << 20)
        updater.weights_getter = lambda: local_weights
        updater.iterator = iterator
        updater.canonical_hf = updater.capture_dense_bridge = True
        updater.encoder = StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=16)
        updater.source = updater.members = None
        updater.engines = ()
        updater.failed = False
        installs = []

        def forbidden_install(*args):
            installs.append(args)
            raise AssertionError("An invalid capture reached publication/installation")

        updater._install = forbidden_install
        updater._close_generation = lambda _manager: []
        with ExitStack() as patches:
            patches.enter_context(patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD))
            if failure in ("emit", "begin") and rank == 0:
                patches.enter_context(
                    patch.object(
                        updater.encoder,
                        "add" if failure == "emit" else "begin",
                        side_effect=MemoryError(f"injected publisher {failure} failure"),
                    )
                )
            expected = {
                "prepare": "KeyError",
                "schema": "Dense source bucket schemas disagree",
                "convert": "injected Bridge convert failure",
                "emit": "injected publisher emit failure",
                "begin": "injected publisher begin failure",
            }[failure]
            try:
                _assert_converged_error(lambda: updater.update_weights(None), expected)
                assert not installs and updater.source is None
                assert updater.failed and updater.encoder.version == -1
                assert updater.encoder._pending is None and updater.encoder.budget.used_bytes == 0
                # Failure at the first fixed parameter boundary must prevent
                # either rank from converting the following parameter.
                if failure in ("convert", "emit"):
                    assert iterator.converted_names == names[:1]
            finally:
                updater.encoder.close()


def test_gated_nonexpert_conversion_and_consumer_drain_converge(tmp_path) -> None:
    pytest.importorskip("megatron.core", reason="Gated bridge iterator requires Megatron Core", exc_type=ImportError)
    context = mp.start_processes(
        _rank_worker,
        args=(f"file://{tmp_path / 'gloo-rendezvous'}", str(tmp_path / "stage")),
        nprocs=2,
        join=False,
        start_method="fork",
    )
    deadline = monotonic() + 40
    try:
        while not context.join(timeout=1):
            if monotonic() > deadline:
                stages = {
                    rank: (tmp_path / f"stage.{rank}").read_text(encoding="utf-8")
                    if (tmp_path / f"stage.{rank}").exists()
                    else "no worker stage reached"
                    for rank in range(2)
                }
                pytest.fail(f"Gated conversion failure stranded a Gloo rank past 40 seconds: {stages}")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)

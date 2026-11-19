# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-rank Gloo coverage for gated nonexpert preparation and consumer
drain."""

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


def _make_iterator(names: list[str], rank: int, conversion_failure: bool = False):
    infos = [_info(name, rank) for name in names]
    iterator = bridge.HfWeightIteratorBridge.__new__(bridge.HfWeightIteratorBridge)
    iterator.lora_merge_mode = False
    iterator.lora_adapter_mode = False
    iterator._non_expert_buckets = [infos]
    iterator._vanilla_key_map = {name: name for name in names}
    iterator._gdn_split_sizes = {}

    def convert(name: str, parameter: torch.Tensor):
        if conversion_failure and rank == 0:
            raise ValueError("injected Bridge convert failure")
        return [(f"hf.{name}", parameter.detach().clone())]

    iterator._bridge_converter = SimpleNamespace(convert=convert)
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

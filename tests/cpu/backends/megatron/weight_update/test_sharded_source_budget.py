# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relax.backends.megatron.weight_update.sharded_source import BridgeShardSource


def test_native_full_seed_refreshes_the_sparse_directory_when_geometry_changes():
    source = BridgeShardSource(SimpleNamespace(get_hf_weight_chunks=lambda weights: iter([list(weights.items())])))
    original = {"hf.weight": torch.zeros((2, 4), dtype=torch.bfloat16)}
    assert list(source.seed(original)) == [list(original.items())]
    assert source.schema == {"hf.weight": ((2, 4), torch.bfloat16)}
    existing_index = [object()]
    source.index = existing_index

    list(source.seed({"hf.weight": torch.ones((2, 4), dtype=torch.bfloat16)}))
    assert source.index is existing_index
    list(source.seed({"hf.weight": torch.ones((4, 2), dtype=torch.bfloat16)}))
    assert source.index is None
    assert source.schema == {"hf.weight": ((4, 2), torch.bfloat16)}


def test_native_full_seed_rejects_conflicting_slots():
    native = [("hf.weight", torch.zeros((2, 4))), ("hf.weight", torch.zeros((4, 2)))]
    source = BridgeShardSource(SimpleNamespace(get_hf_weight_chunks=lambda _: iter([native])))
    with pytest.raises(ValueError, match="Conflicting native FULL slots"):
        list(source.seed({}))


def _streamed_rows(rank, rendezvous):
    from relax.backends.megatron.weight_update.shard_snapshot import CudaShardInputs
    from relax.backends.megatron.weight_update.streamed_row_projection import StreamedRowTable
    from relax.backends.megatron.weight_update.verl_projection import McoreParamExport, ShardSpec

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        group = dist.group.WORLD
        slots = [("a", (5, 3)), ("b", (5, 3)), ("c", (2, 3))]
        table = StreamedRowTable(tuple(slots), (6, 3), group, 2, rank)
        old = torch.arange(18, dtype=torch.bfloat16).reshape(3, 6).t()
        new = old.clone()
        # Native SUM with zeros normalizes signed zero. A genuine new NaN is a verl mask sentinel.
        new[rank, 2] = -0.0
        new[2, 1] = float("nan")
        new[5, 1] = 11 + rank
        source = BridgeShardSource(None)
        source.index = [
            McoreParamExport(
                "table", new, ShardSpec((6, 3), gather_group=group), None, None, slots, streamed_rows=table
            )
        ]
        source.keys, source.dtypes, source.output_dtypes = ["table"], [new.dtype], [new.dtype]
        source.schema = {name: (shape, new.dtype) for name, shape in slots}

        def native(tensor):
            full = tensor.new_zeros(12, 3)
            full[rank * 6 : (rank + 1) * 6].copy_(tensor)
            dist.all_reduce(full, group=group)
            return dict(zip((n for n, _ in slots), full.split([5, 5, 2]), strict=True))

        baseline, oracle = native(old), native(new)
        # Two words per window, smaller than one native row. Both inputs have non-contiguous strides.
        wire = source.chunks(
            CudaShardInputs({"table": new}, torch.device("cpu")),
            CudaShardInputs({"table": old}, torch.device("cpu")),
            full_names=set(),
            metadata_group=group,
            device="cpu",
            max_round_bytes=128,
            staging_budget_bytes=128,
        )
        for batch in wire:
            for entry in batch:
                baseline[entry.name].view(-1)[entry.positions.long()] = entry.values
        if rank == 0:
            old_native = torch.arange(18, dtype=old.dtype).reshape(3, 6).t()
            for name, expected in oracle.items():
                nan = torch.isnan(expected)
                assert torch.equal(baseline[name][~nan].view(torch.int16), expected[~nan].view(torch.int16))
            assert baseline["a"][2, 1] == old_native[2, 1]
            assert baseline["b"][3, 1] == old_native[2, 1]
        stats = source.metrics["streamed_rows"]
        assert 0 < stats["tp_sum_input_bytes"] < 2 * new.nbytes and 0 < stats["changed_windows"] < stats["windows"]
    finally:
        dist.destroy_process_group()


def test_streamed_row_projection_matches_native_tp_sum_across_uneven_hf_and_owner_boundaries(tmp_path):
    torch.multiprocessing.start_processes(
        _streamed_rows, args=((tmp_path / "rows").as_uri(),), nprocs=2, join=True, start_method="fork"
    )


def _projection_failures(rank, rendezvous):
    from relax.backends.megatron.weight_update.verl_projection import McoreParamExport, ShardSpec

    timeout = timedelta(seconds=10)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timeout)
    try:
        meta = dist.group.WORLD
        data = dist.new_group([0, 1], timeout=timeout)
        source = BridgeShardSource(None)
        source.index = []
        keys = [f"weight_{i}" for i in range(32)]
        previous = {}
        current = {}

        def projection(name):
            def project(tensor, _):
                full = tensor.new_full((2,), float("nan"))
                full[rank] = tensor[0]
                return {name: full}

            return project

        for i, key in enumerate(keys):
            # One row is owned exclusively by rank 0 (a PP placeholder on rank 1).
            owned = rank == 0 or i != 0
            param = torch.zeros(1, dtype=torch.bfloat16) if owned else None
            if owned:
                current[key] = torch.tensor([1 + rank], dtype=torch.bfloat16)
                previous[key] = param
            source.index.append(
                McoreParamExport(
                    key,
                    param,
                    ShardSpec((1,), gather_group=data, contributes=owned),
                    SimpleNamespace(megatron_to_hf=projection(key)),
                    None,
                    [(key, (2,))],
                )
            )
        source.keys = [key if (rank == 0 or i != 0) else None for i, key in enumerate(keys)]
        source.dtypes = source.output_dtypes = [torch.bfloat16] * len(keys)

        def publish(shards):
            return list(
                source.chunks(
                    shards,
                    previous,
                    full_names=set(),
                    metadata_group=meta,
                    device="cpu",
                    max_round_bytes=48,
                    staging_budget_bytes=48,
                )
            )

        missing = dict(current)
        if rank == 1:
            del missing[keys[-1]]
        with pytest.raises(RuntimeError, match="projection failed before sparse gather.*weight_31"):
            publish(missing)

        def broken(*args):
            raise ValueError("rejected native projection")

        if rank == 1:
            source.index[-1].probe.megatron_to_hf = broken
        with pytest.raises(RuntimeError, match="projection failed before sparse gather.*rejected native projection"):
            publish(current)
        if rank == 1:
            source.index[-1].probe.megatron_to_hf = projection(keys[-1])
        chunks = publish(current)
        stats = source.metrics["groups"][0]
        assert stats["rows"] == 32 and stats["ranks"] == [0, 1] and stats["peak_root_padded_bytes"] <= 48
        assert stats["local_gather_index_bytes"] == stats["local_index_bytes"] == (32 - rank) * 4
        if rank == 0:
            rows = [
                (entry.name, i, v)
                for chunk in chunks
                for entry in chunk
                for i, v in zip(entry.positions.tolist(), entry.values.tolist(), strict=True)
            ]
            assert rows == [(k, r, r + 1) for n, k in enumerate(keys) for r in range(1 if n == 0 else 2)]
        else:
            assert all(chunk == [] for chunk in chunks)
    finally:
        dist.destroy_process_group()


def test_one_bad_rank_votes_before_any_sparse_gather_and_peers_can_publish_again(tmp_path):
    torch.multiprocessing.start_processes(
        _projection_failures, args=((tmp_path / "projection-vote").as_uri(),), nprocs=2, start_method="fork"
    )

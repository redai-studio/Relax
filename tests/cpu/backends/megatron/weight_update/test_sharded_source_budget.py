# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two real Gloo contributors, staging admission and native loader
boundaries."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relax.backends.megatron.weight_update.shard_snapshot import CudaShardInputs
from relax.backends.megatron.weight_update.sharded_source import BridgeShardSource, _Route, _Row
from relax.distributed.weight_sync.codec import encode


def _admission_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    group = dist.group.WORLD
    try:
        full_words = {
            "a": torch.tensor([0x7FC2, -32768, 5, 0, -59, 7, 0, 0], dtype=torch.int16),
            "b": torch.tensor([0, 0, 0, 0, 0, 0, -32768, 0], dtype=torch.int16),
        }
        expected = {name: words.view(torch.bfloat16) for name, words in full_words.items()}
        shards = {name: tensor.chunk(2)[rank] for name, tensor in expected.items()}
        baseline = {name: torch.zeros_like(tensor) for name, tensor in shards.items()}
        reads = []
        fail = False

        def native(current, *, selected):
            # The existing native iterator is the dependency under test here.
            # Its exact outputs are supplied independently of the sparse data.
            expert, ordinary = selected
            assert not expert
            for bucket in ordinary:
                for info in bucket:
                    reads.append(info.name)
                    yield info.name, expected[info.name]
                    if fail and rank == 1:
                        raise RuntimeError("native current read failed after its last result")

        source = BridgeShardSource(SimpleNamespace(_vanilla_key_map={"a": "a", "b": "b"}, _iter_hf_params=native))
        source._routes = lambda _: (rank, {False: _Route(group, (0, 1), rank, 2, True, True)})
        for name, tensor in expected.items():
            slot = (name, tuple(tensor.shape), tensor.dtype)
            layout = SimpleNamespace(local_slices=((0, 4, rank * 4),))
            row = _Row(SimpleNamespace(name=name), (slot,), False, (0, 1), layout, None)
            source.by_slot[name] = source.rows[name] = row
        source.seed_chunks = [tuple(source.rows[n].slots[0] for n in ("a", "b"))]

        def chunks(budget, previous=baseline):
            return list(
                source.chunks(
                    shards,
                    previous,
                    full_names=frozenset(),
                    metadata_group=group,
                    device="cpu",
                    max_round_bytes=96,
                    staging_budget_bytes=budget,
                )
            )

        # 5 * (4 + 2) bytes in A, aligned to 32, plus 6 bytes in B.
        # At exactly 38 bytes both sparse entries must survive unchanged.
        [entries] = chunks(38)
        assert not reads
        if rank == 0:
            assert all(not e.full for e in entries)
            assert [e.positions.tolist() for e in entries] == [[0, 1, 2, 4, 5], [6]]
            args = [(e.name, e.shape, e.dtype, e.positions, e.values) for e in entries]
            assert encode(args, max_bytes=38)[0]["bytes"] == 38

        # When the actual staging window is 24 bytes, only A needs the native
        # read. The immutable group is 16 bytes in A and 6 bytes in B.
        [entries] = chunks(24)
        assert reads == ["a"]
        if rank == 0:
            assert [(e.name, e.full) for e in entries] == [("a", True), ("b", False)]
            args = [(e.name, e.shape, e.dtype, e.positions, e.values) for e in entries]
            assert encode(args, max_bytes=24)[0]["bytes"] == 22
            assert torch.equal(entries[0].values.view(torch.int16), full_words["a"])
            assert entries[1].positions.tolist() == [6]
            assert entries[1].values.view(torch.int16).tolist() == [-32768]
            assert source.metrics["ordinary_tp"]["index_bytes"] == 6 * 4
            assert source.metrics["ordinary_tp"]["value_bytes"] == 6 * 2
            assert source.metrics["native_fallback_rows"] == 1
            assert (
                source.metrics["native_fallback_wire_bytes"]
                == source.metrics["native_fallback_details"]["a"]["full_bytes"]
                == 16
            )
            assert "38 bytes" in source.metrics["native_fallback_details"]["a"]["reason"]

        reads.clear()
        with pytest.raises(ValueError, match="at least 22 bytes"):
            chunks(21)
        assert not reads

        fail = True
        with pytest.raises(RuntimeError, match="native current read failed"):
            chunks(24)
        assert reads == ["a"]

        class FailingPrefetch(CudaShardInputs):
            def prefetch(self, buckets, vanilla, rank):
                if rank == 1:
                    raise RuntimeError("prefetch failed before projection")
                yield from buckets

        previous = FailingPrefetch(baseline, torch.device("cpu"), 0)
        with pytest.raises(RuntimeError, match="prefetch failed before projection"):
            chunks(38, previous)

        # All ranks must finish each injected error in the same collective
        # order and retain a functioning group for the caller's abort/ACK.
        alive = torch.ones(1)
        dist.all_reduce(alive, group=group)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


def test_sharded_source_honors_staging_window_and_agrees_on_local_failures(tmp_path):
    torch.multiprocessing.start_processes(
        _admission_worker, args=((tmp_path / "store").as_uri(),), nprocs=2, join=True, start_method="fork"
    )

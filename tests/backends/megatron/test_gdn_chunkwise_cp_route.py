# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for the prebuilt THD CP layout route (Task 32, phase 3).

Phase 1 derived the zigzag<->contiguous all-to-all plan inside every conversion,
from device tensors, which cost a device-host synchronisation per CP rank per
call. Phase 3 backports NVIDIA/Megatron-LM#5664's idea instead: derive the plan
once per micro-batch on CPU and hand the same route to every GDN layer.

Two things therefore need proving on CPU, with no process group:

1. the segment-based route describes *exactly* the permutation the phase-1
   index-based partition described -- otherwise chunkwise CP silently reorders
   tokens;
2. a cached route is only ever reused for the micro-batch, CP geometry and
   direction it was built for.

The real all-to-all round trip over NCCL stays in
``test_gdn_chunkwise_cp_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch


cpl = pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402


DIRECTIONS = [("zigzag", "contiguous"), ("contiguous", "zigzag")]

# Packed boundary shapes worth covering: single sequence, uneven multi-sequence,
# and a duplicated boundary (an empty padding slot), which the compaction step
# has to drop before the segments line up.
LENGTH_CASES = [
    [1],
    [3, 1, 2],
    [2, 0, 1, 3],
]


def _cu(lengths: list[int], unit: int) -> torch.Tensor:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n * unit)
    return torch.tensor(cu, dtype=torch.int64)


def _packed_seq_params(cu: torch.Tensor) -> PackedSeqParams:
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=int(cu[-1]),
        max_seqlen_kv=int(cu[-1]),
    )


def _apply_route_across_ranks(
    x: torch.Tensor, cu: torch.Tensor, cp_size: int, source: str, target: str
) -> list[torch.Tensor]:
    """Run the route-driven swap for every rank, emulating the all-to-all
    locally."""
    source_by_rank = [cpl.get_thd_context_parallel_rank_indices(cu, cp_size, r, source) for r in range(cp_size)]
    routes = [cpl.build_thd_cp_partition_route(cu, cp_size, r, source, target) for r in range(cp_size)]

    send_bufs = []
    for rank, route in enumerate(routes):
        local = x[source_by_rank[rank]]
        assert local.size(0) == route.local_source_length
        send_bufs.append(local if route.send_rows is None else local.index_select(0, route.send_rows))

    outputs = []
    for dst, route in enumerate(routes):
        parts = []
        for src in range(cp_size):
            offset = sum(routes[src].input_split_sizes[:dst])
            length = routes[src].input_split_sizes[dst]
            assert length == route.output_split_sizes[src], "split sizes disagree between peers"
            parts.append(send_bufs[src][offset : offset + length])
        recv = torch.cat(parts, dim=0)
        if route.recv_rows is None:
            outputs.append(recv)
        else:
            out = recv.new_empty((route.local_target_length,) + tuple(x.shape[1:]))
            out.index_copy_(0, route.recv_rows, recv)
            outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# The route is the same permutation phase 1 computed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source,target", DIRECTIONS)
@pytest.mark.parametrize("lengths", LENGTH_CASES)
def test_route_reproduces_index_based_partition(cp_size, source, target, lengths):
    cu = _cu(lengths, unit=2 * cp_size)
    total = int(cu[-1])
    x = torch.arange(total * 3, dtype=torch.float64).reshape(total, 3)

    got = _apply_route_across_ranks(x, cu, cp_size, source, target)
    for rank in range(cp_size):
        want = x[cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, target)]
        assert torch.equal(got[rank], want), f"cp_size={cp_size} rank={rank} {source}->{target}"


# ---------------------------------------------------------------------------
# Fail-fast parity with the index-based builder
# ---------------------------------------------------------------------------
def test_route_rejects_lengths_not_divisible_by_two_cp():
    cu = torch.tensor([0, 12], dtype=torch.int64)  # 12 % (2 * 4) != 0
    with pytest.raises(ValueError, match="divisible by"):
        cpl.get_thd_context_parallel_rank_indices(cu, 4, 0, "zigzag")
    with pytest.raises(ValueError, match="divisible by"):
        cpl.build_thd_cp_partition_route(cu, 4, 0, "zigzag", "contiguous")


def test_route_rejects_malformed_cu_seqlens():
    with pytest.raises(ValueError, match="must start at 0"):
        cpl.build_thd_cp_partition_route(torch.tensor([8, 16], dtype=torch.int64), 2, 0, "zigzag", "contiguous")
    with pytest.raises(ValueError, match="nondecreasing"):
        cpl.build_thd_cp_partition_route(torch.tensor([0, 16, 8], dtype=torch.int64), 2, 0, "zigzag", "contiguous")


def test_route_rejects_unknown_layout():
    cu = _cu([1], unit=4)
    with pytest.raises(ValueError, match="Unsupported CP layout conversion"):
        cpl.build_thd_cp_partition_route(cu, 2, 0, "zigzag", "interleaved")


# ---------------------------------------------------------------------------
# Caching: reuse only within the micro-batch it was built for
# ---------------------------------------------------------------------------
def test_route_is_cached_per_packed_seq_params():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    second = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    assert second is first, "a second layer of the same micro-batch must reuse the route"
    assert psp.cp_partition_route_zigzag_to_contiguous is first


def test_both_directions_are_cached_separately():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    to_contiguous = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    to_zigzag = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "contiguous", "zigzag")
    assert to_contiguous is not to_zigzag
    assert psp.cp_partition_route_zigzag_to_contiguous is to_contiguous
    assert psp.cp_partition_route_contiguous_to_zigzag is to_zigzag


def test_route_is_rebuilt_for_new_packed_boundaries():
    """Packed boundaries move every micro-batch; a stale route would corrupt
    tokens."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    next_cu = _cu([2, 2], unit=8)
    psp.cu_seqlens_q = next_cu
    rebuilt = cpl.get_thd_cp_partition_route(psp, next_cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt is not first
    assert rebuilt.cu_seqlens is next_cu

    want = cpl.build_thd_cp_partition_route(next_cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt.input_split_sizes == want.input_split_sizes
    assert rebuilt.output_split_sizes == want.output_split_sizes
    for field in ("send_rows", "recv_rows"):
        got_rows, want_rows = getattr(rebuilt, field), getattr(want, field)
        assert (got_rows is None) == (want_rows is None)
        if want_rows is not None:
            assert torch.equal(got_rows, want_rows)


def test_route_is_rebuilt_when_cu_seqlens_is_mutated_in_place():
    """Identity alone would miss a caller refilling a preallocated boundary
    buffer.

    Megatron already has that pattern elsewhere (persistent ``_cu_seqlens_buffer``
    written with ``buf[0] = 0``), so the cache also fingerprints autograd's version
    counter, which every in-place write bumps.
    """
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    # Same tensor object, refilled with different boundaries.
    cu.copy_(_cu([2, 2], unit=8))
    rebuilt = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt is not first, "an in-place refill must invalidate the cached route"

    want = cpl.build_thd_cp_partition_route(cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt.input_split_sizes == want.input_split_sizes
    assert rebuilt.output_split_sizes == want.output_split_sizes


def test_route_is_rebuilt_when_a_view_of_cu_seqlens_is_mutated():
    """Views share the version counter with their base, so writes through one
    count."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    cu[1:] = _cu([2, 2], unit=8)[1:]
    assert cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous") is not first


def test_route_is_rebuilt_when_the_dynamic_cp_geometry_changes():
    """Dynamic CP varies cp_size/cp_rank across micro-batches on one module."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    cp4 = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    cp2 = cpl.get_thd_cp_partition_route(psp, cu, 2, 1, "zigzag", "contiguous")
    assert cp2 is not cp4
    assert (cp2.cp_size, cp2.cp_rank) == (2, 1)

    other_rank = cpl.get_thd_cp_partition_route(psp, cu, 2, 0, "zigzag", "contiguous")
    assert other_rank is not cp2
    assert other_rank.cp_rank == 0


def test_prebuild_populates_both_directions():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    class _FakeGroup:
        def size(self):
            return 4

        def rank(self):
            return 2

    psp.cp_group = _FakeGroup()
    psp.local_cp_size = 4
    cpl.prebuild_thd_cp_partition_routes(psp)

    for attr in ("cp_partition_route_zigzag_to_contiguous", "cp_partition_route_contiguous_to_zigzag"):
        route = getattr(psp, attr)
        assert route is not None
        assert (route.cp_size, route.cp_rank) == (4, 2)


def test_prebuild_is_a_noop_without_context_parallelism():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    cpl.prebuild_thd_cp_partition_routes(psp)
    assert getattr(psp, "cp_partition_route_zigzag_to_contiguous", None) is None

    non_thd = PackedSeqParams(qkv_format="sbhd")
    cpl.prebuild_thd_cp_partition_routes(non_thd)
    assert getattr(non_thd, "cp_partition_route_zigzag_to_contiguous", None) is None

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for GDN CP partitions, layout routes, and boundary caches.

Checks token ownership, agreement with Relax's sharding, route equivalence, and
cache reuse/invalidation. Real NCCL layout communication is covered in
``test_gdn_cp_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch


cpl = pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

from relax.backends.megatron.cp_utils import gdn_cp_slice, slice_with_cp  # noqa: E402


DIRECTIONS = [("zigzag", "contiguous"), ("contiguous", "zigzag")]

# Packed boundary shapes worth covering: single sequence, uneven multi-sequence,
# and a duplicated boundary (an empty padding slot), which the compaction step
# has to drop before the segments line up.
LENGTH_CASES = [
    [1],
    [3, 1, 2],
    [2, 0, 1, 3],
]


def _cu(lengths: list[int], unit: int = 1) -> torch.Tensor:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n * unit)
    return torch.tensor(cu, dtype=torch.int64)


def _tagged_tokens(total: int, width: int = 3) -> torch.Tensor:
    """[total, width] where row t is (t, t+1e6, t+2e6): token identity is
    unambiguous."""
    base = torch.arange(total, dtype=torch.float64).unsqueeze(1)
    return base + torch.arange(width, dtype=torch.float64).unsqueeze(0) * 1e6


# ---------------------------------------------------------------------------
# Partition definitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("lengths_factor", [[1], [1, 2, 3], [3, 1, 1, 2]])
def test_thd_rank_indices_partition_all_tokens_exactly_once(cp_size, layout, lengths_factor):
    lengths = [2 * cp_size * f for f in lengths_factor]
    cu = _cu(lengths)
    owned = torch.cat([cpl.get_thd_context_parallel_rank_indices(cu, cp_size, r, layout) for r in range(cp_size)])
    assert owned.numel() == int(cu[-1])
    assert torch.equal(torch.sort(owned).values, torch.arange(int(cu[-1])))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_rank_indices_match_relax_data_sharding(cp_size):
    """MCore's zigzag partition must be token-for-token what Relax's data path
    produces.

    If these ever disagree, chunkwise CP would silently permute tokens relative
    to the all-gather fallback and the attention layers.
    """
    lengths = [2 * cp_size * f for f in (1, 3, 2)]
    cu = _cu(lengths)
    full = _tagged_tokens(int(cu[-1])).reshape(-1, 1, 3)  # [s, b=1, C]

    for rank in range(cp_size):
        mcore_idx = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "zigzag")
        mcore_shard = full[mcore_idx]

        # Relax data.py: per-sample slice_with_cp then concat.
        relax_shard = torch.cat(
            [
                slice_with_cp(
                    full[cu[i] : cu[i + 1]],
                    pad_value=0.0,
                    qkv_format="thd",
                    dynamic_cp_size=cp_size,
                    dynamic_cp_rank=rank,
                )
                for i in range(len(lengths))
            ],
            dim=0,
        )
        assert torch.equal(mcore_shard, relax_shard)

        # Relax model.py (all-gather fallback) re-slices with gdn_cp_slice.
        assert torch.equal(mcore_shard, gdn_cp_slice(full, cu, cp_size, rank))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("lengths_factor", [[1], [1, 2, 3], [3, 1, 1, 2]])
def test_both_layouts_are_permutations_of_each_other(cp_size, lengths_factor):
    """The two partitions must describe the same token set with the same per-
    rank size.

    That is the precondition for the all-to-all between them to be a pure
    permutation -- no token invented, dropped, or duplicated. The real collective
    round trip is asserted in ``test_gdn_cp_gpu.py``.
    """
    lengths = [2 * cp_size * f for f in lengths_factor]
    cu = _cu(lengths)
    total = int(cu[-1])
    zig_by_rank = []
    con_by_rank = []
    for rank in range(cp_size):
        zig = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "zigzag")
        con = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "contiguous")
        zig_by_rank.append(zig)
        con_by_rank.append(con)
        assert zig.numel() == con.numel() == total // cp_size
        # contiguous is exactly this rank's span of the flattened buffer
        assert torch.equal(con, torch.arange(rank * (total // cp_size), (rank + 1) * (total // cp_size)))

    # Across the whole CP group, both layouts are permutations of exactly the
    # same global token rows.
    assert torch.equal(
        torch.cat(zig_by_rank).sort().values,
        torch.cat(con_by_rank).sort().values,
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_rank_indices_reject_lengths_not_divisible_by_two_cp(cp_size):
    bad = _cu([2 * cp_size, 2 * cp_size + 1])
    with pytest.raises(ValueError, match="divisible by"):
        cpl.get_thd_context_parallel_rank_indices(bad, cp_size, 0, "zigzag")


def test_gdn_rejects_packed_lengths_not_divisible_by_cp():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    cu = _cu([8, 6])
    with pytest.raises(ValueError, match="divisible by cp_size=4"):
        GatedDeltaNet._resolve_cu_seqlens(None, None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=4)


def test_rank_indices_reject_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported context-parallel layout"):
        cpl.get_thd_context_parallel_rank_indices(_cu([16, 16]), 2, 0, "contiguous_ish")


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_rank_indices_ignore_duplicate_boundaries(layout):
    compact = torch.tensor([0, 16, 40], dtype=torch.int64)
    padded = torch.tensor([0, 16, 40, 40, 40], dtype=torch.int64)
    for rank in range(2):
        assert torch.equal(
            cpl.get_thd_context_parallel_rank_indices(compact, 2, rank, layout),
            cpl.get_thd_context_parallel_rank_indices(padded, 2, rank, layout),
        )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_rank_indices_reject_decreasing_boundaries(layout):
    with pytest.raises(ValueError, match="nondecreasing"):
        cpl.get_thd_context_parallel_rank_indices(torch.tensor([0, 16, 8]), 2, 0, layout)


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


def test_route_does_not_cache_unversioned_inference_boundaries():
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        psp = _packed_seq_params(cu)
        first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
        cu.copy_(_cu([2, 2], unit=8))
        second = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
        assert second is not first
        expected = cpl.build_thd_cp_partition_route(cu, 4, 1, "zigzag", "contiguous")
        assert second.input_split_sizes == expected.input_split_sizes


def _boundary_resolver():
    from types import SimpleNamespace

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    calls = []
    module = SimpleNamespace(cp_size=8)

    def resolve(*args, **kwargs):
        calls.append(kwargs["cp_size"])
        return GatedDeltaNet._resolve_cu_seqlens(module, *args, **kwargs)

    module._resolve_cu_seqlens = resolve
    return module, calls, GatedDeltaNet._resolve_thd_cu_seqlens


def test_boundary_validation_is_shared_across_layers_and_recompute():
    module, calls, resolve = _boundary_resolver()
    # Static max CP=8 would reject a length of 12; runtime CP=2 is legal.
    cu = _cu([3, 1], unit=4)
    psp = _packed_seq_params(cu)
    first = resolve(module, psp, 16, 2)
    assert calls == [2, 2]
    assert resolve(module, psp, 16, 2) is first
    assert calls == [2, 2]

    other_module, other_calls, _ = _boundary_resolver()
    assert resolve(other_module, psp, 16, 2) is first
    assert other_calls == []


@pytest.mark.parametrize("change", ["replace", "inplace", "view", "new_pack", "runtime_cp", "global_length", "padded"])
def test_boundary_cache_is_invalidated_when_inputs_change(change):
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    cp_size, total = 2, 32
    if change == "replace":
        psp.cu_seqlens_q = cu.clone()
        psp.cu_seqlens_kv = psp.cu_seqlens_q
    elif change == "inplace":
        cu.copy_(_cu([2, 2], unit=8))
    elif change == "view":
        cu[1:2].fill_(16)
    elif change == "new_pack":
        psp = _packed_seq_params(cu)
    elif change == "runtime_cp":
        cp_size = 4
    elif change == "global_length":
        total = 64
    elif change == "padded":
        psp.cu_seqlens_q_padded = _cu([2, 2], unit=8)
        psp.cu_seqlens_kv_padded = psp.cu_seqlens_q_padded
    if change == "global_length":
        with pytest.raises(ValueError, match="total_sequence_length"):
            resolve(module, psp, total, cp_size)
        assert len(calls) == 3
    else:
        resolve(module, psp, total, cp_size)
        assert len(calls) == 4


def test_boundary_validation_rechecks_inference_tensors():
    module, calls, resolve = _boundary_resolver()
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        psp = _packed_seq_params(cu)
        resolve(module, psp, 32, 2)
        cu[1] = 16
        resolve(module, psp, 32, 2)
        assert len(calls) == 4


def test_boundary_cache_does_not_hide_q_kv_mismatch():
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    psp.cu_seqlens_kv = _cu([2, 2], unit=8)
    with pytest.raises(AssertionError, match="cu_seqlens_q equals"):
        resolve(module, psp, 32, 2)

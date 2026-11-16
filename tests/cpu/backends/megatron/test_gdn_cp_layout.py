# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for GDN CP partitions, layout routes, and boundary caches.

Checks token ownership, agreement with Relax's sharding, route equivalence, and
explicit route refresh and version-keyed boundary-cache invalidation. These
tests require the image's pinned native Megatron API; a missing dependency/API
must fail, not skip. Real NCCL layout communication is covered in
``test_gdn_cp_gpu.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from megatron.core.context_parallel_layout import routes
from megatron.core.packed_seq_params import PackedSeqParams

from relax.backends.megatron.cp_utils import gdn_cp_slice, slice_with_cp


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


def _reference_rank_indices(cu: torch.Tensor, cp_size: int, rank: int, layout: str) -> torch.Tensor:
    """Independent partition oracle; all ownership tests execute native
    routes."""
    if layout == "contiguous":
        local = int(cu[-1]) // cp_size
        return torch.arange(rank * local, (rank + 1) * local)
    assert layout == "zigzag"
    parts = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        chunk = (end - start) // (2 * cp_size)
        parts.extend(
            torch.arange(start + owner * chunk, start + (owner + 1) * chunk)
            for owner in (rank, 2 * cp_size - rank - 1)
        )
    return torch.cat(parts)


def _exchange_routes(
    local_by_rank: list[torch.Tensor], plans: list[routes.ThdCpRoute], source: str, target: str
) -> list[torch.Tensor]:
    """Execute native send/split/receive metadata with CPU tensors, not
    NCCL."""
    send = []
    for local, plan in zip(local_by_rank, plans, strict=True):
        splits = getattr(plan, f"{source}_split_sizes")
        assert len(splits) == len(plans)
        assert sum(splits) == local.size(0)
        index = getattr(plan, f"{source}_index")
        if index is not None:
            assert torch.equal(index.sort().values, torch.arange(local.size(0)))
        send.append(local if index is None else local.index_select(0, index))
    outputs = []
    for rank, plan in enumerate(plans):
        parts = []
        for peer, peer_plan in enumerate(plans):
            splits = getattr(peer_plan, f"{source}_split_sizes")
            assert splits[rank] == getattr(plan, f"{target}_split_sizes")[peer]
            parts.append(send[peer].narrow(0, sum(splits[:rank]), splits[rank]))
        received = torch.cat(parts)
        index = getattr(plan, f"{target}_index")
        actual = received
        if index is not None:
            assert torch.equal(index.sort().values, torch.arange(received.size(0)))
            actual = torch.empty_like(received)
            actual.index_copy_(0, index, received)
        outputs.append(actual)
    return outputs


def _apply_route_across_ranks(
    x: torch.Tensor,
    cu: torch.Tensor,
    cp_size: int,
    source: str,
    target: str,
    plans: list[routes.ThdCpRoute] | None = None,
) -> list[torch.Tensor]:
    if plans is None:
        plans = [routes.build_thd_cp_partition_route(cu, cp_size, rank) for rank in range(cp_size)]
    local = [x[_reference_rank_indices(cu, cp_size, rank, source)] for rank in range(cp_size)]
    return _exchange_routes(local, plans, source, target)


def _routed_rank_indices(cu: torch.Tensor, cp_size: int, layout: str) -> list[torch.Tensor]:
    source = "contiguous" if layout == "zigzag" else "zigzag"
    return _apply_route_across_ranks(torch.arange(int(cu[-1])), cu, cp_size, source, layout)


# ---------------------------------------------------------------------------
# Partition definitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("lengths_factor", [[1], [1, 2, 3], [3, 1, 1, 2]])
def test_thd_routes_partition_all_tokens_exactly_once(cp_size, layout, lengths_factor):
    lengths = [2 * cp_size * f for f in lengths_factor]
    cu = _cu(lengths)
    by_rank = _routed_rank_indices(cu, cp_size, layout)
    for rank, actual in enumerate(by_rank):
        assert torch.equal(actual, _reference_rank_indices(cu, cp_size, rank, layout))
    owned = torch.cat(by_rank)
    assert owned.numel() == int(cu[-1])
    assert torch.equal(torch.sort(owned).values, torch.arange(int(cu[-1])))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_routes_match_relax_data_sharding(cp_size):
    """MCore's zigzag partition must be token-for-token what Relax's data path
    produces.

    If these ever disagree, chunkwise CP would silently permute tokens relative
    to the all-gather fallback and the attention layers.
    """
    lengths = [2 * cp_size * f for f in (1, 3, 2)]
    cu = _cu(lengths)
    full = _tagged_tokens(int(cu[-1])).reshape(-1, 1, 3)  # [s, b=1, C]

    for rank, mcore_idx in enumerate(_routed_rank_indices(cu, cp_size, "zigzag")):
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
    zigzag = _routed_rank_indices(cu, cp_size, "zigzag")
    contiguous = _routed_rank_indices(cu, cp_size, "contiguous")
    for rank, (zig, con) in enumerate(zip(zigzag, contiguous, strict=True)):
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
def test_routes_reject_lengths_not_divisible_by_two_cp(cp_size):
    bad = _cu([2 * cp_size, 2 * cp_size + 1])
    with pytest.raises(ValueError, match="divisible by"):
        routes.build_thd_cp_partition_route(bad, cp_size, 0)


def test_gdn_rejects_packed_lengths_not_divisible_by_cp():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    cu = _cu([8, 6])
    with pytest.raises(ValueError, match="divisible by cp_size=4"):
        GatedDeltaNet._resolve_cu_seqlens(None, None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=4)


def test_routes_reject_unknown_source_layout():
    with pytest.raises(ValueError, match="Unsupported CP partition mode conversion"):
        routes.get_thd_cp_partition_route(_packed_seq_params(_cu([16, 16])), "contiguous_ish", "zigzag")


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_routes_ignore_duplicate_boundaries(layout):
    compact = torch.tensor([0, 16, 40], dtype=torch.int64)
    padded = torch.tensor([0, 16, 40, 40, 40], dtype=torch.int64)
    compact_shards = _routed_rank_indices(compact, 2, layout)
    padded_shards = _routed_rank_indices(padded, 2, layout)
    for rank in range(2):
        assert torch.equal(compact_shards[rank], padded_shards[rank])
        assert torch.equal(padded_shards[rank], _reference_rank_indices(compact, 2, rank, layout))


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_routes_reject_decreasing_boundaries(layout):
    with pytest.raises(ValueError, match="nondecreasing"):
        _routed_rank_indices(torch.tensor([0, 16, 8]), 2, layout)


def _packed_seq_params(cu: torch.Tensor) -> PackedSeqParams:
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=int(cu[-1]),
        max_seqlen_kv=int(cu[-1]),
    )


# ---------------------------------------------------------------------------
# The route is the same permutation the independent partition formula defines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source,target", DIRECTIONS)
@pytest.mark.parametrize("lengths", LENGTH_CASES)
def test_route_reproduces_reference_partition(cp_size, source, target, lengths):
    cu = _cu(lengths, unit=2 * cp_size)
    total = int(cu[-1])
    x = torch.arange(total * 3, dtype=torch.float64).reshape(total, 3)

    got = _apply_route_across_ranks(x, cu, cp_size, source, target)
    for rank in range(cp_size):
        want = x[_reference_rank_indices(cu, cp_size, rank, target)]
        assert torch.equal(got[rank], want), f"cp_size={cp_size} rank={rank} {source}->{target}"


# ---------------------------------------------------------------------------
# Fail-fast validation in the native builder and directional lookup
# ---------------------------------------------------------------------------
def test_route_rejects_lengths_not_divisible_by_two_cp():
    cu = torch.tensor([0, 12], dtype=torch.int64)  # 12 % (2 * 4) != 0
    with pytest.raises(ValueError, match="divisible by"):
        routes.build_thd_cp_partition_route(cu, 4, 0)


def test_route_rejects_malformed_cu_seqlens():
    with pytest.raises(ValueError, match="must start at 0"):
        routes.build_thd_cp_partition_route(torch.tensor([8, 16], dtype=torch.int64), 2, 0)
    with pytest.raises(ValueError, match="nondecreasing"):
        routes.build_thd_cp_partition_route(torch.tensor([0, 16, 8], dtype=torch.int64), 2, 0)


def test_route_rejects_unknown_layout():
    cu = _cu([1], unit=4)
    with pytest.raises(ValueError, match="Unsupported CP partition mode conversion"):
        routes.get_thd_cp_partition_route(_packed_seq_params(cu), "zigzag", "interleaved")


# Native routes are batch-owned, not version-keyed lookup caches. Rebuild at
# batch preparation after changing boundaries or CP geometry; forward/recompute
# lookups share one bidirectional plan. Boundary-validation caching below remains
# version-keyed and is a separate contract.
def _prebuild(packed: PackedSeqParams, cp_size: int = 4, rank: int = 1) -> routes.ThdCpRoute:
    packed.cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: rank)
    routes.prebuild_thd_cp_partition_routes(packed)
    return packed.cp_partition_route


def _assert_cached_route(packed: PackedSeqParams, cu: torch.Tensor, cp_size: int = 4, rank: int = 1) -> None:
    """Check BOTH views and execute the cached plan against tagged-token
    oracles."""
    expected = routes.build_thd_cp_partition_route(cu, cp_size, rank)
    actual = packed.cp_partition_route
    for layout in ("zigzag", "contiguous"):
        assert getattr(actual, f"{layout}_split_sizes") == getattr(expected, f"{layout}_split_sizes")
        got, want = getattr(actual, f"{layout}_index"), getattr(expected, f"{layout}_index")
        assert (got is None) == (want is None)
        if want is not None:
            assert torch.equal(got, want)
    full = _tagged_tokens(int(cu[-1]))
    plans = [routes.build_thd_cp_partition_route(cu, cp_size, r) for r in range(cp_size)]
    plans[rank] = actual
    for source, target in DIRECTIONS:
        assert routes.get_thd_cp_partition_route(packed, source, target) is actual
        outputs = _apply_route_across_ranks(full, cu, cp_size, source, target, plans)
        for r, output in enumerate(outputs):
            assert torch.equal(output, full[_reference_rank_indices(cu, cp_size, r, target)])


def test_route_is_cached_per_packed_seq_params():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    packed.cp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    # Native lookup retains a compatibility fallback for packs not prebuilt yet.
    with pytest.warns(FutureWarning, match="missing precomputed"):
        first = routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous")
    for _ in range(2):  # another layer / recompute of the same microbatch
        assert routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous") is first
    _assert_cached_route(packed, cu)
    other_pack = _packed_seq_params(cu)
    assert _prebuild(other_pack) is not first


def test_both_directions_share_one_cached_route():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    first = _prebuild(packed)
    assert routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous") is first
    assert routes.get_thd_cp_partition_route(packed, "contiguous", "zigzag") is first
    _assert_cached_route(packed, cu)


def test_prebuild_refreshes_for_new_packed_boundaries():
    packed = _packed_seq_params(_cu([3, 1], unit=8))
    first = _prebuild(packed)
    next_cu = _cu([2, 2], unit=8)
    packed.cu_seqlens_q = packed.cu_seqlens_kv = next_cu
    assert _prebuild(packed) is not first
    _assert_cached_route(packed, next_cu)


def test_prebuild_refreshes_when_cu_seqlens_is_mutated_in_place():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    first = _prebuild(packed)
    cu.copy_(_cu([2, 2], unit=8))
    assert _prebuild(packed) is not first
    _assert_cached_route(packed, cu)


def test_prebuild_refreshes_when_a_view_of_cu_seqlens_is_mutated():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    first = _prebuild(packed)
    cu[1:2].fill_(16)
    assert _prebuild(packed) is not first
    _assert_cached_route(packed, cu)


def test_prebuild_refreshes_when_the_dynamic_cp_geometry_changes():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    cp4 = _prebuild(packed)
    cp2 = _prebuild(packed, cp_size=2)
    assert cp2 is not cp4
    _assert_cached_route(packed, cu, cp_size=2)
    assert _prebuild(packed, cp_size=2, rank=0) is not cp2
    _assert_cached_route(packed, cu, cp_size=2, rank=0)


def test_prebuild_populates_both_directions():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    _prebuild(packed, rank=2)
    _assert_cached_route(packed, cu, rank=2)


def test_prebuild_is_a_noop_without_context_parallelism():
    for packed in (
        None,
        PackedSeqParams(qkv_format="sbhd"),
        _packed_seq_params(_cu([3, 1], unit=8)),
    ):
        # Exercise both absent group and an explicit size-one group.
        for group in (None, SimpleNamespace(size=lambda: 1, rank=lambda: 0)):
            routes.prebuild_thd_cp_partition_routes(packed, cp_group=group)
            assert getattr(packed, "cp_partition_route", None) is None


def test_prebuild_refreshes_unversioned_inference_boundaries():
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        packed = _packed_seq_params(cu)
        first = _prebuild(packed)
        # Unlike the legacy cache, native prebuild does not read tensor._version.
        assert routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous") is first
        cu.copy_(_cu([2, 2], unit=8))
        assert _prebuild(packed) is not first
        _assert_cached_route(packed, cu)


def _boundary_resolver():
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


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source,target", DIRECTIONS)
@pytest.mark.parametrize("lengths", LENGTH_CASES)
def test_native_route_round_trip(cp_size, source, target, lengths):
    cu = _cu(lengths, unit=2 * cp_size)
    full = _tagged_tokens(int(cu[-1]))
    plans = [routes.build_thd_cp_partition_route(cu, cp_size, rank) for rank in range(cp_size)]
    original = [full[_reference_rank_indices(cu, cp_size, rank, source)] for rank in range(cp_size)]
    converted = _exchange_routes(original, plans, source, target)
    for rank, actual in enumerate(converted):
        assert torch.equal(actual, full[_reference_rank_indices(cu, cp_size, rank, target)])
    restored = _exchange_routes(converted, plans, target, source)
    for actual, expected in zip(restored, original, strict=True):
        assert torch.equal(actual, expected)


def test_native_prebuild_reuses_both_directions_and_refreshes_for_next_batch():
    packed = _packed_seq_params(_cu([3, 1], unit=8))
    packed.cp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    routes.prebuild_thd_cp_partition_routes(packed)
    first = routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous")
    assert routes.get_thd_cp_partition_route(packed, "contiguous", "zigzag") is first

    packed.cu_seqlens_q = packed.cu_seqlens_kv = _cu([2, 2], unit=8)
    packed.cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    routes.prebuild_thd_cp_partition_routes(packed)
    refreshed = routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous")
    assert refreshed is not first
    expected = routes.build_thd_cp_partition_route(packed.cu_seqlens_q, 2, 0)
    assert refreshed.zigzag_split_sizes == expected.zigzag_split_sizes
    assert refreshed.contiguous_split_sizes == expected.contiguous_split_sizes
    for name in ("zigzag_index", "contiguous_index"):
        actual, wanted = getattr(refreshed, name), getattr(expected, name)
        assert (actual is None) == (wanted is None)
        if wanted is not None:
            assert torch.equal(actual, wanted)


def test_native_routes_reject_invalid_packed_boundaries():
    for cu, match in (
        (torch.tensor([8, 16]), "must start at 0"),
        (torch.tensor([0, 16, 8]), "nondecreasing"),
        (torch.tensor([0, 12]), "divisible by"),
    ):
        with pytest.raises(ValueError, match=match):
            routes.build_thd_cp_partition_route(cu, 4, 0)

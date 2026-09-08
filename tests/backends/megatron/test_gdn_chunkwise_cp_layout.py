# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for the GDN chunkwise-CP layout backport (Task 32, phase 1).

Covers the pure-tensor half of the backported MCore capability: the two THD CP
partitions (``zigzag`` / ``contiguous``), their agreement with Relax's existing
zigzag sharding, dynamic group resolution, and the construction-time
``linear_cp_mode`` gate.

Everything here runs on CPU with no process group. It validates partition
definitions only; the actual all-to-all round trip is exercised with NCCL in
``test_gdn_chunkwise_cp_gpu.py``.

The real-kernel / real-collective half lives in
``test_gdn_chunkwise_cp_gpu.py``.
"""

from __future__ import annotations

import inspect

import pytest
import torch


cpl = pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group  # noqa: E402

from relax.backends.megatron.cp_utils import gdn_cp_slice, slice_with_cp  # noqa: E402


def _cu(lengths: list[int]) -> torch.Tensor:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
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
    round trip is asserted in ``test_gdn_chunkwise_cp_gpu.py``.
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


# ---------------------------------------------------------------------------
# Dynamic CP group resolution
# ---------------------------------------------------------------------------
def test_resolve_cp_group_prefers_packed_seq_params():
    static = object()
    dynamic = object()
    assert resolve_cp_group(static, None) is static
    assert resolve_cp_group(static, PackedSeqParams(qkv_format="thd")) is static
    assert resolve_cp_group(static, PackedSeqParams(qkv_format="thd", cp_group=dynamic)) is dynamic


# ---------------------------------------------------------------------------
# Construction-time capability gate
# ---------------------------------------------------------------------------
def _gdn_config(**overrides):
    import torch.nn.functional as F
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        hidden_size=2048,
        num_layers=1,
        num_attention_heads=16,
        num_query_groups=2,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_config_default_mode_is_headwise():
    """Upgrading the image must not silently reroute an existing recipe."""
    assert _gdn_config().linear_cp_mode == "headwise"


def test_headwise_config_requires_heads_divisible_by_tp_times_cp():
    # 16 key heads, tp=2, cp=4 -> 16 % 8 == 0: fine.
    _gdn_config(tensor_model_parallel_size=2, context_parallel_size=4)
    # tp=2, cp=16 -> 16 % 32 != 0: the geometry headwise cannot express.
    with pytest.raises(AssertionError, match="linear_num_key_heads"):
        _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16)


def test_chunkwise_config_only_requires_heads_divisible_by_tp():
    """This is what replaces Relax's temporary head-count rewrite hack."""
    cfg = _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16, linear_cp_mode="chunkwise")
    assert cfg.linear_num_key_heads == 16 and cfg.linear_num_value_heads == 32
    # ... but TP divisibility is still enforced: GDN weights stay TP-sharded.
    with pytest.raises(AssertionError, match="linear_num_key_heads"):
        _gdn_config(
            tensor_model_parallel_size=8,
            context_parallel_size=2,
            linear_cp_mode="chunkwise",
            num_query_groups=8,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )


def test_all_gather_config_uses_the_tp_only_head_rule():
    """`--gdn-cp-mode=all_gather` must be constructible on a non-divisible
    geometry.

    Relax's all-gather fallback keeps GDN weights TP-only, so declaring it
    should relax the head check exactly as chunkwise does.
    """
    cfg = _gdn_config(tensor_model_parallel_size=2, context_parallel_size=16, linear_cp_mode="all_gather")
    assert cfg.linear_num_key_heads == 16 and cfg.linear_num_value_heads == 32


def test_config_rejects_unresolved_and_unknown_linear_cp_mode():
    """`auto` is resolved before construction; MCore must never see it."""
    for bad in ("auto", "allgather", "chunk", ""):
        with pytest.raises(AssertionError, match="linear_cp_mode"):
            _gdn_config(context_parallel_size=2, linear_cp_mode=bad)
        with pytest.raises(AssertionError, match="linear_cp_mode"):
            _gdn_config(context_parallel_size=4, tensor_model_parallel_size=2, linear_cp_mode=bad)


def test_gdn_forward_has_no_per_call_mode_override():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    assert "linear_cp_mode" not in inspect.signature(GatedDeltaNet.forward).parameters

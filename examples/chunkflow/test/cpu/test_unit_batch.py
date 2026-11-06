# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit slicing, loss normalization and sequence-local RoPE positions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


try:
    from megatron.core.packed_seq_params import PackedSeqParams

    from examples.chunkflow import rope as cf_rope
    from examples.chunkflow.scheduler import build_chunks
    from examples.chunkflow.unit_batch import _rope_offsets, _rows_for_unit, slice_parent_batch
    from examples.chunkflow.units import build_micro_batches
    from relax.backends.megatron.cp_utils import get_cp_local_num_tokens
except ImportError as exc:
    pytest.skip(f"ChunkFlow batch dependencies unavailable: {exc}", allow_module_level=True)


PAD = 0


def _parent(lengths: list[int], device="cpu") -> dict:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    total = cu[-1]

    tokens = torch.arange(1, total + 1, device=device).unsqueeze(0)
    mask = torch.zeros(total, dtype=torch.long, device=device)
    for lo, hi in zip(cu[:-1], cu[1:], strict=True):
        lo, hi = int(lo), int(hi)
        mask[max(hi - (hi - lo) // 4, lo + 1) : hi - 1] = 1

    packed = PackedSeqParams(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor(cu, dtype=torch.int32, device=device),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        qkv_format="thd",
    )
    packed.cu_seqlens_q_cpu = cu

    return {
        "tokens": tokens,
        "unconcat_tokens": [tokens[0][lo:hi].clone() for lo, hi in zip(cu[:-1], cu[1:], strict=True)],
        "full_loss_masks": mask.unsqueeze(0),
        "loss_masks": [mask[lo:hi].clone() for lo, hi in zip(cu[:-1], cu[1:], strict=True)],
        "total_lengths": list(lengths),
        "response_lengths": list(lengths),
        "max_seq_lens": list(lengths),
        "packed_seq_params": packed,
    }


def _units(lengths, chunk_size, policy="staggered"):
    chunks = build_chunks(lengths, chunk_size=chunk_size)
    return build_micro_batches(chunks, policy), chunks


@pytest.mark.parametrize("policy", ["staggered", "lockstep"])
@pytest.mark.parametrize("lengths,chunk_size", [([40, 12, 16], 16), ([301, 12, 16], 128)])
def test_units_preserve_tokens_targets_masks_and_loss(policy, lengths, chunk_size):
    parent = _parent(lengths)
    original_tokens, original_mask = parent["tokens"].clone(), parent["full_loss_masks"].clone()
    units, _ = _units(lengths, chunk_size, policy)
    tokens, mask = parent["tokens"][0], parent["full_loss_masks"][0]
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    seen, counts, total_loss = [], [], torch.zeros(())
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        real = batch["response_lengths"][0]
        rows = _rows_for_unit(cu, unit)
        expected_tokens = torch.cat([tokens[start:end] for start, end in rows])
        expected_mask = torch.cat([mask[start:end] for start, end in rows])
        torch.testing.assert_close(batch["tokens"][0][:real], expected_tokens)
        torch.testing.assert_close(batch["full_loss_masks"][0][:real], expected_mask)
        targets = []
        segments = [seg for chunk in unit.chunks for seg in chunk.segments]
        for (start, end), seg in zip(rows, segments, strict=True):
            part = tokens[start + 1 : min(end + 1, cu[seg.seq_id + 1])]
            if part.numel() < end - start:
                part = torch.cat([part, tokens.new_full((end - start - part.numel(),), PAD)])
            targets.append(part)
        torch.testing.assert_close(batch["unconcat_tokens"][0][:real], torch.cat(targets))
        assert batch["full_loss_masks"].shape == batch["tokens"].shape
        assert (batch["tokens"][0][real:] == PAD).all()
        assert (batch["full_loss_masks"][0][real:] == 0).all()
        counts.append(
            get_cp_local_num_tokens(
                batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], dynamic_cp_size=1
            )
        )
        seen.extend(batch["tokens"][0][:real].tolist())
        total_loss += (batch["full_loss_masks"][0].float() * batch["tokens"][0].float()).sum()
    assert any(count.item() == 0 for count in counts), "fixture must include a prompt-only unit"
    torch.testing.assert_close(torch.stack(counts).sum(), parent["full_loss_masks"].sum())
    assert sorted(seen) == tokens.tolist()
    assert len(seen) == len(set(seen))
    torch.testing.assert_close(total_loss, (mask.float() * tokens.float()).sum())
    torch.testing.assert_close(parent["tokens"], original_tokens)
    torch.testing.assert_close(parent["full_loss_masks"], original_mask)


def test_attention_lengths_and_rope_offsets_are_sequence_local():
    parent = _parent([40, 40])
    units, _ = _units([40, 40], 16)
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD)
        packed = batch["packed_seq_params"]
        real = batch["response_lengths"][0]
        prefix = unit.chunks[0].segments[0].prefix
        assert _rope_offsets(unit) == [prefix]
        assert packed.cu_seqlens_q_cpu == [0, real]
        if prefix:
            assert getattr(packed, cf_rope.HEADROOM_ATTR, 0) > 0
            assert packed.cu_seqlens_q_padded.tolist() == [prefix, prefix + real]
            assert packed.max_seqlen_kv == prefix + real
        else:
            assert packed.cu_seqlens_q_padded is None


def test_lockstep_dependent_units_fail_closed_until_per_segment_rope_exists():
    units, _ = _units([40, 40], 16, "lockstep")
    merged = next(unit for unit in units if unit.is_dependent and len(unit.chunks) > 1)

    with pytest.raises(NotImplementedError, match="per-segment RoPE offsets"):
        _rope_offsets(merged)


def test_bridge_chunk_positions_and_padding_match_full_sequence():
    bridge_rope = pytest.importorskip("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope")
    from examples.chunkflow.schedules import _forward_kwargs

    parent = _parent([301, 301, 11, 13])
    units, _ = _units(parent["total_lengths"], 128)
    full_positions, _ = bridge_rope.get_rope_index(
        1, -1, -2, -3, parent["tokens"], packed_seq_params=parent["packed_seq_params"]
    )
    rope = bridge_rope.Qwen3VLMultimodalRotaryEmbedding.__new__(bridge_rope.Qwen3VLMultimodalRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.inv_freq = 1 / (1_000_000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    rope.seq_len_interpolation_factor = None
    rope.cp_group = SimpleNamespace(size=lambda: 1)
    rope.is_thd_format = True
    config = SimpleNamespace(apply_rope_fusion=False, apply_rotary_pos_emb_in_fp32=True, rotary_interleaved=False)
    full_input = torch.randn(parent["tokens"].numel(), 2, 64, requires_grad=True)
    full_output = bridge_rope.apply_rotary_pos_emb_absolute(
        full_input, rope(full_positions, [11, 11, 10]), config, parent["packed_seq_params"].cu_seqlens_q
    )
    full_output.sum().backward()
    saw_packed = saw_prefix = saw_padding = False
    for unit in units:
        batch = slice_parent_batch(parent, unit, pad_token_id=PAD, args=SimpleNamespace(uses_unsplit_forward=True))
        packed = batch["packed_seq_params"]
        rows = _rows_for_unit(parent["packed_seq_params"].cu_seqlens_q_cpu, unit)
        real = sum(end - start for start, end in rows)
        physical = batch["tokens"].numel()
        expected = torch.cat([full_positions[..., start:end] for start, end in rows], dim=-1)
        torch.testing.assert_close(batch["position_ids"][..., :real], expected)
        assert batch["position_ids"].shape == (3, 1, physical)
        assert (batch["position_ids"][..., real:] == 1).all()
        assert _forward_kwargs(batch)["position_ids"] is batch["position_ids"]
        physical_cu = packed.cu_seqlens_q_padded.tolist()
        assert physical_cu == [*packed.cu_seqlens_q_cpu[:-1], physical]
        assert packed.max_seqlen_q == max(b - a for a, b in zip(physical_cu[:-1], physical_cu[1:]))
        mask = bridge_rope.get_packed_seq_attention_mask(batch["tokens"], packed)
        assert mask.sum() == real
        assert not mask[..., real:].any()
        chunk_input = torch.cat([full_input.detach()[start:end] for start, end in rows])
        chunk_input = torch.cat([chunk_input, chunk_input.new_zeros(physical - real, 2, 64)]).requires_grad_()
        output = bridge_rope.apply_rotary_pos_emb_absolute(
            chunk_input, rope(batch["position_ids"], [11, 11, 10], packed), config, packed.cu_seqlens_q_padded
        )
        torch.testing.assert_close(output[:real], torch.cat([full_output[start:end] for start, end in rows]))
        output[:real].sum().backward()
        torch.testing.assert_close(
            chunk_input.grad[:real], torch.cat([full_input.grad[start:end] for start, end in rows])
        )
        saw_packed |= len(rows) > 1
        saw_prefix |= any(_rope_offsets(unit))
        saw_padding |= physical > real
    assert saw_packed and saw_prefix and saw_padding

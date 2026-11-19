# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed text RoPE regression: actual MCore operations, independent doc
positions.

No process group or CUDA allocation is needed: the unfused RoPE operator only
reads the CP group's size/rank. This tests local CP arithmetic, not
collectives.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture(scope="module")
def provider_type() -> type:
    pytest.importorskip("megatron.bridge")
    from relax.models.keye_vl2.provider import KeyeModelProvider

    return KeyeModelProvider


def _provider(provider_type: type, cp_size: int, position_type: str = "rope") -> Any:
    return provider_type(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=1,
        context_parallel_size=cp_size,
        position_embedding_type=position_type,
        mrope_section=[16, 24, 24],
    )


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("mode", ["text", "multimodal", "explicit_mrope"])
def test_keye_runtime_rope_contract(provider_type: type, cp_size: int, mode: str) -> None:
    provider = _provider(provider_type, cp_size, "mrope" if mode == "explicit_mrope" else "rope")
    args = SimpleNamespace(
        multimodal_keys=["image"] if mode == "multimodal" else None,
        cp_partition_mode="zigzag",
        is_vl_model=True,
        uses_unsplit_forward=True,
    )
    provider.configure_runtime(args)
    assert provider.apply_rope_fusion is False
    assert provider.position_embedding_type == ("rope" if mode == "text" else "mrope")
    assert provider.mrope_section == (None if mode == "text" else [16, 24, 24])
    assert args.is_vl_model is (mode != "text")
    assert args.uses_unsplit_forward is (mode != "text")
    if mode == "multimodal":
        assert args.uses_unsplit_forward is True
        assert args.multimodal_input_key_map == {"image_grid_thw": "grid_thws"}


def _doc_positions(lengths: tuple[int, ...], cp_size: int, cp_rank: int) -> list[int]:
    positions = []
    for length in lengths:
        if cp_size == 1:
            positions.extend(range(length))
        else:
            # Independent explicit per-document zigzag slices, not MCore's
            # searchsorted/local-cumulative-length mapping implementation.
            width = length // (2 * cp_size)
            positions.extend(range(cp_rank * width, (cp_rank + 1) * width))
            positions.extend(range(length - (cp_rank + 1) * width, length - cp_rank * width))
    return positions


def _reference(x: torch.Tensor, dy: torch.Tensor, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (10_000_000 ** (torch.arange(0, x.shape[-1], 2).float() / x.shape[-1]))
    angles = torch.tensor(positions).float()[:, None] * inv_freq[None, :]
    cos = angles.cos().to(x.dtype)[:, None, :]
    sin = angles.sin().to(x.dtype)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    da, db = dy[..., :half], dy[..., half:]
    # Keep the real BF16 rounding at each multiply/add. The VJP is analytic,
    # independent of MCore's backward and the reference forward's autograd.
    output = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    gradient = torch.cat((da * cos + db * sin, db * cos - da * sin), dim=-1)
    return output, gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,cp_size,cp_rank",
    [((8,), 1, 0), ((8, 12), 1, 0), ((8, 8), 2, 0), ((8, 8), 2, 1), ((8, 12), 2, 0), ((8, 12), 2, 1)],
)
def test_text_packed_rope_forward_and_backward(
    provider_type: type, lengths: tuple[int, ...], cp_size: int, cp_rank: int, dtype: torch.dtype
) -> None:
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    config = _provider(provider_type, cp_size)
    config.configure_runtime(SimpleNamespace(multimodal_keys=None, cp_partition_mode="zigzag"))
    group = SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank)
    positions = _doc_positions(lengths, cp_size, cp_rank)
    generator = torch.Generator().manual_seed(827)
    # A non-contiguous THD input catches hidden layout assumptions.
    x = torch.randn(len(positions), 128, 3, generator=generator, dtype=dtype).transpose(1, 2).requires_grad_()
    dy = torch.randn(x.shape, generator=generator, dtype=dtype)
    rotary = RotaryEmbedding(128, 1.0, rotary_base=10_000_000, use_cpu_initialization=True, cp_group=group)
    # get_emb moves CPU buffers to CUDA; use the real CPU frequency generator
    # and its documented [max_seq, 1, 1, head_dim] table representation instead.
    frequencies = rotary.get_freqs_non_repeated(max(lengths))
    frequencies = torch.cat((frequencies, frequencies), dim=-1)[:, None, None, :]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    actual = apply_rotary_pos_emb(
        x, frequencies, config, cu_seqlens=cu_seqlens, cp_group=group, max_seqlen=max(lengths)
    )
    expected, expected_gradient = _reference(x.detach(), dy, positions)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.autograd.grad(actual, x, dy)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)

    if cp_size == 2 and max(lengths) == len(positions):
        # Bridge-derived VL sections used to misclassify this text position
        # table as already-local mRoPE, skipping both doc resets and CP mapping.
        config.mrope_section = [16, 24, 24]
        stale = apply_rotary_pos_emb(
            x.detach(), frequencies, config, cu_seqlens=cu_seqlens, cp_group=group, max_seqlen=max(lengths)
        )
        assert not torch.equal(stale, expected)


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("lengths", [(8, 8), (7, 11)])
def test_text_runtime_uses_real_packed_batch(
    provider_type: type, monkeypatch: pytest.MonkeyPatch, cp_rank: int, lengths: tuple[int, ...]
) -> None:
    from relax.backends.megatron import data

    config = _provider(provider_type, 2)
    # Actor's HF processor detection sets this even for a pure-text run.
    args = SimpleNamespace(
        multimodal_keys=None, cp_partition_mode="zigzag", is_vl_model=True, uses_unsplit_forward=True
    )
    config.configure_runtime(args)
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 1)

    def pack() -> dict:
        batch = {
            "tokens": [torch.arange(length) + 100 * (i + 1) for i, length in enumerate(lengths)],
            "loss_masks": [torch.ones(3, dtype=torch.int) for _ in lengths],
            "total_lengths": list(lengths),
            "response_lengths": [3] * len(lengths),
        }
        return data.get_batch(
            iter([(batch, None)]),
            list(batch),
            pad_multiplier=1,
            is_vl_model=args.is_vl_model,
            pack_device=torch.device("cpu"),
        )

    actual = pack()
    assert "unsplit_tokens" not in actual
    assert "vlm_packed_seq_params" not in actual
    expected = []
    padded_lengths = []
    for i, length in enumerate(lengths):
        padded_length = (length + 3) // 4 * 4
        padded_lengths.append(padded_length)
        row = list(range(100 * (i + 1), 100 * (i + 1) + length)) + [0] * (padded_length - length)
        expected.extend(row[position] for position in _doc_positions((padded_length,), 2, cp_rank))
    assert actual["tokens"].shape == (1, len(expected))
    assert actual["tokens"].tolist() == [expected]
    packed = actual["packed_seq_params"]
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q_cpu == [0, padded_lengths[0], sum(padded_lengths)]
    assert packed.max_seqlen_q == max(padded_lengths)
    assert tuple(packed.valid_seqlens_q_cpu) == lengths
    assert packed.num_valid_tokens == sum(lengths)

    # Reintroduce the stale actor detection: the real packer now emits B>1
    # unsplit input, which plain GPT cannot consume as packed THD input.
    args.is_vl_model = True
    args.uses_unsplit_forward = True
    stale = pack()
    assert stale["unsplit_tokens"].shape == (2, max(padded_lengths))
    assert stale["vlm_packed_seq_params"].qkv_format == "thd"

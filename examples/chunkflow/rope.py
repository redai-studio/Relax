# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from relax.utils.logging_utils import get_logger

from .metadata import execution_optimizations_enabled, int32_metadata


logger = get_logger(__name__)

HEADROOM_ATTR = "chunkflow_rope_headroom"

_ORIGINAL_GET_ROTARY_SEQ_LEN = None


def apply_rope_offset(
    packed_seq_params,
    offsets: list[int],
    lengths: list[int],
    headroom: int = 1,
) -> None:
    absolute = [0]
    for length in lengths:
        absolute.append(absolute[-1] + length)
    padded = [offsets[0]]
    for offset, length in zip(offsets, lengths, strict=True):
        padded.append(offset + length)

    if not execution_optimizations_enabled() or getattr(packed_seq_params, "cu_seqlens_q_cpu", None) != absolute:
        packed_seq_params.cu_seqlens_q = int32_metadata(absolute, packed_seq_params.cu_seqlens_q.device)
    packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_q
    packed_seq_params.cu_seqlens_q_padded = int32_metadata(padded, packed_seq_params.cu_seqlens_q.device)
    packed_seq_params.cu_seqlens_kv_padded = packed_seq_params.cu_seqlens_q_padded
    packed_seq_params.cu_seqlens_q_cpu = absolute
    packed_seq_params.max_seqlen_q = max(lengths)
    packed_seq_params.max_seqlen_kv = max(padded[i + 1] for i in range(len(lengths)))

    setattr(packed_seq_params, HEADROOM_ATTR, headroom)


def install() -> None:
    global _ORIGINAL_GET_ROTARY_SEQ_LEN
    if _ORIGINAL_GET_ROTARY_SEQ_LEN is not None:
        return

    try:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    except ImportError as exc:  # pragma: no cover - megatron always present in training
        logger.warning("chunkflow RoPE patch not applied -- import failed (%r).", exc)
        return

    _ORIGINAL_GET_ROTARY_SEQ_LEN = RotaryEmbedding.get_rotary_seq_len

    def get_rotary_seq_len(
        self,
        inference_context,
        transformer,
        transformer_input,
        transformer_config,
        packed_seq_params=None,
        *,
        inference_params=None,
    ):
        length = _ORIGINAL_GET_ROTARY_SEQ_LEN(
            self,
            inference_context,
            transformer,
            transformer_input,
            transformer_config,
            packed_seq_params,
            inference_params=inference_params,
        )
        headroom = getattr(packed_seq_params, HEADROOM_ATTR, 0)
        return length + headroom if headroom else length

    RotaryEmbedding.get_rotary_seq_len = get_rotary_seq_len
    logger.info(
        "chunkflow RoPE offset patch applied to RotaryEmbedding.get_rotary_seq_len (no-op unless a batch carries %s).",
        HEADROOM_ATTR,
    )


def uninstall() -> None:
    global _ORIGINAL_GET_ROTARY_SEQ_LEN
    if _ORIGINAL_GET_ROTARY_SEQ_LEN is None:
        return
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    RotaryEmbedding.get_rotary_seq_len = _ORIGINAL_GET_ROTARY_SEQ_LEN
    _ORIGINAL_GET_ROTARY_SEQ_LEN = None

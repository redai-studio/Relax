# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from typing import List

import torch
from torch import Tensor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextRotaryEmbedding


class Qwen3OmniMoeThinkerTextRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding):
    """Qwen3-Omni MoE text rotary position embedding."""

    def forward(
        self, position_ids: torch.Tensor, mrope_section: List[int], packed_seq_params=None, **kwargs
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Raw frequency embeddings for Megatron Core (shape: [seq_length, bs, 1, dim]).
                    Megatron Core will compute cos/sin internally and apply attention_scaling.
        """
        # Use fp32 for position indices to avoid precision loss when inv_freq is bf16.
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)

        # if self.seq_len_interpolation_factor is not None:
        #     seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        return emb

    def apply_interleaved_mrope(self, freqs: Tensor, mrope_section: List[int]) -> Tensor:
        """Interleave T/H/W frequencies; transformers>=5.19 no longer provides
        this method."""
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

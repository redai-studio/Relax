# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Team. All rights reserved.
# Adapted from diffusers 0.37.0 QwenDoubleStreamAttnProcessor2_0.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


class QwenImageFlashAttnProcessor:
    """Joint image-text FlashAttention-2 processor for QwenImage transformer
    blocks."""

    def __init__(self, deterministic: bool = False) -> None:
        self.deterministic = deterministic

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        from flash_attn import flash_attn_func

        if encoder_hidden_states is None:
            raise ValueError("Qwen joint attention requires text hidden states")
        if attention_mask is not None or encoder_hidden_states_mask is not None:
            raise ValueError("Qwen FlashAttention training requires unpadded inputs without a mask")
        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        query = torch.cat([txt_query, img_query], dim=1)
        key = torch.cat([txt_key, img_key], dim=1)
        value = torch.cat([txt_value, img_value], dim=1)
        joint = flash_attn_func(query, key, value, dropout_p=0.0, causal=False, deterministic=self.deterministic)
        joint = joint.flatten(2, 3).to(query.dtype)
        img_output = attn.to_out[0](joint[:, seq_txt:, :].contiguous())
        if len(attn.to_out) > 1:
            img_output = attn.to_out[1](img_output)
        txt_output = attn.to_add_out(joint[:, :seq_txt, :].contiguous())
        return img_output, txt_output

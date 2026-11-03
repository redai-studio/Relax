# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2023 Zhongjie Duan.
# Adapted and modified by Relax from DiffSynth-Studio, revision f7b7db96.
# Source: https://github.com/modelscope/DiffSynth-Studio
# SPDX-License-Identifier: Apache-2.0
# See the repository root LICENSE for the full license text.
from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from einops import repeat
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from .processing import build_abs_positions_from_grid_hw


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern: str = "b n s d",
    k_pattern: str = "b n s d",
    v_pattern: str = "b n s d",
    out_pattern: str = "b s n d",
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    if (q_pattern, k_pattern, v_pattern, out_pattern) != ("b n s d", "b n s d", "b n s d", "b s n d"):
        raise ValueError("Unsupported SenseNova attention layout")
    if attn_mask is not None or os.environ.get("RELAX_ATTENTION_IMPLEMENTATION") == "torch":
        # Match the audited Torch2.11 reference: its signature probe cannot
        # introspect the builtin SDPA, so GQA uses explicit K/V repetition.
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            k = repeat(k, "b n s d -> b (n m) s d", m=q.shape[1] // k.shape[1])
            v = repeat(v, "b n s d -> b (n m) s d", m=q.shape[1] // v.shape[1])
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=False)
        return output.transpose(1, 2)
    from flash_attn import flash_attn_func

    return flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=scale,
        causal=False,
        window_size=(-1, -1),
        deterministic=os.environ.get("RELAX_DETERMINISTIC_ATTENTION", "0") == "1",
    )


def gradient_checkpoint_forward(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
        return model(*args, **kwargs)
    context = torch.autograd.graph.save_on_cpu() if use_gradient_checkpointing_offload else nullcontext()
    with context:
        return checkpoint(model, *args, **kwargs, use_reentrant=False)


def rotate_half(x: Any) -> Any:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Any, k: Any, cos: Any, sin: Any, unsqueeze_dim: Any = 1) -> Any:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SenseNovaU1RMSNorm(nn.Module):
    def __init__(self, hidden_size: Any, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> Any:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SenseNovaU1RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: Any, rope_theta: Any, max_position_embeddings: Any, device: Any = None) -> None:
        super().__init__()
        self.rope_head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.attention_scaling = 1.0
        self.inv_freq = None

    @staticmethod
    def _compute_inv_freq(base: Any, dim: Any, device: Any = None) -> Any:
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))

    def _ensure_inv_freq(self, device: Any) -> Any:
        if self.inv_freq is not None and self.inv_freq.device == device:
            return
        inv_freq_full = self._compute_inv_freq(self.rope_theta, self.rope_head_dim * 2, device)
        self.inv_freq = inv_freq_full[::2]

    @torch.no_grad()
    def forward(self, x: Any, position_ids: Any) -> Any:
        self._ensure_inv_freq(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SenseNovaU1MLP(nn.Module):
    def __init__(self, hidden_size: Any, intermediate_size: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Any) -> Any:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SenseNovaU1Attention(nn.Module):
    def __init__(
        self,
        hidden_size: Any = 4096,
        num_attention_heads: Any = 32,
        num_key_value_heads: Any = 8,
        head_dim: Any = 128,
        attention_bias: Any = False,
        attention_dropout: Any = 0.0,
        rms_norm_eps: Any = 1e-06,
        rope_theta: Any = 5000000.0,
        rope_theta_hw: Any = 10000.0,
        max_position_embeddings: Any = 262144,
        max_position_embeddings_hw: Any = 10000,
        layer_idx: Any = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.q_proj_mot_gen = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.k_proj_mot_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj_mot_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.o_proj_mot_gen = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_hw = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_hw_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)

        self.k_norm = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_hw = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_hw_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)

        self.rotary_emb = SenseNovaU1RotaryEmbedding(head_dim // 2, rope_theta, max_position_embeddings)
        self.rotary_emb_hw = SenseNovaU1RotaryEmbedding(head_dim // 4, rope_theta_hw, max_position_embeddings_hw)

    def _project_qkv(
        self,
        hidden_states: Any,
        q_proj: Any,
        k_proj: Any,
        v_proj: Any,
        q_norm: Any,
        q_norm_hw: Any,
        k_norm: Any,
        k_norm_hw: Any,
        indexes: Any,
    ) -> Any:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = q_proj(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = q_norm(query_states_t).transpose(1, 2)
        query_states_hw = q_norm_hw(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = k_proj(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = k_norm(key_states_t).transpose(1, 2)
        key_states_hw = k_norm_hw(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)
        return query_states, key_states, value_states, input_shape

    def _merge_cache(self, key_states: Any, value_states: Any, past_key_value: Any) -> Any:
        if past_key_value is None:
            return key_states, value_states
        past_k, past_v = past_key_value
        return torch.cat([past_k, key_states], dim=2), torch.cat([past_v, value_states], dim=2)

    def forward_und(
        self,
        hidden_states: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        query_states, key_states, value_states, input_shape = self._project_qkv(
            hidden_states,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.q_norm,
            self.q_norm_hw,
            self.k_norm,
            self.k_norm_hw,
            indexes,
        )
        current_key_states, current_value_states = key_states, value_states
        key_states, value_states = self._merge_cache(key_states, value_states, past_key_values)

        attn_output = attention_forward(
            query_states,
            key_states,
            value_states,
            q_pattern="b n s d",
            k_pattern="b n s d",
            v_pattern="b n s d",
            out_pattern="b s n d",
            attn_mask=None
            if attention_mask is None
            else attention_mask[:, :, :, : key_states.shape[-2]].to(query_states.dtype),
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        return (output, current_key_states, current_value_states) if update_cache else output

    def forward_gen(
        self,
        hidden_states: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        query_states, key_states, value_states, input_shape = self._project_qkv(
            hidden_states,
            self.q_proj_mot_gen,
            self.k_proj_mot_gen,
            self.v_proj_mot_gen,
            self.q_norm_mot_gen,
            self.q_norm_hw_mot_gen,
            self.k_norm_mot_gen,
            self.k_norm_hw_mot_gen,
            indexes,
        )
        current_key_states, current_value_states = key_states, value_states
        key_states, value_states = self._merge_cache(key_states, value_states, past_key_values)

        attn_output = attention_forward(
            query_states,
            key_states,
            value_states,
            q_pattern="b n s d",
            k_pattern="b n s d",
            v_pattern="b n s d",
            out_pattern="b s n d",
            attn_mask=None
            if attention_mask is None
            else attention_mask[:, :, :, : key_states.shape[-2]].to(query_states.dtype),
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj_mot_gen(attn_output)
        return (output, current_key_states, current_value_states) if update_cache else output

    def forward(
        self,
        hidden_states: Any,
        exist_non_image_gen_tokens: Any,
        exist_image_gen_tokens: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, update_cache)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, indexes, attention_mask, past_key_values, update_cache)
        raise NotImplementedError(
            "Mixed understanding/generation attention is not supported. Split the sequence at "
            "token-type boundaries and run the understanding and generation paths separately."
        )


class SenseNovaU1DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: Any = 4096,
        intermediate_size: Any = 12288,
        num_attention_heads: Any = 32,
        num_key_value_heads: Any = 8,
        head_dim: Any = 128,
        attention_bias: Any = False,
        attention_dropout: Any = 0.0,
        rms_norm_eps: Any = 1e-06,
        rope_theta: Any = 5000000.0,
        rope_theta_hw: Any = 10000.0,
        max_position_embeddings: Any = 262144,
        max_position_embeddings_hw: Any = 10000,
        layer_idx: Any = 0,
    ) -> None:
        super().__init__()
        self.self_attn = SenseNovaU1Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position_embeddings=max_position_embeddings,
            max_position_embeddings_hw=max_position_embeddings_hw,
            layer_idx=layer_idx,
        )
        self.mlp = SenseNovaU1MLP(hidden_size, intermediate_size)
        self.mlp_mot_gen = SenseNovaU1MLP(hidden_size, intermediate_size)
        self.input_layernorm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward_und(
        self,
        hidden_states: Any,
        exist_non_image_gen_tokens: Any,
        exist_image_gen_tokens: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=update_cache,
        )
        if update_cache:
            attention_output, key_states, value_states = attention_output
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, key_states, value_states) if update_cache else hidden_states

    def forward_gen(
        self,
        hidden_states: Any,
        exist_non_image_gen_tokens: Any,
        exist_image_gen_tokens: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=update_cache,
        )
        if update_cache:
            attention_output, key_states, value_states = attention_output
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, key_states, value_states) if update_cache else hidden_states

    def forward(
        self,
        hidden_states: Any,
        exist_non_image_gen_tokens: Any,
        exist_image_gen_tokens: Any,
        indexes: Any,
        attention_mask: Any,
        past_key_values: Any = None,
        update_cache: Any = True,
    ) -> Any:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                past_key_values,
                update_cache,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                past_key_values,
                update_cache,
            )
        raise NotImplementedError(
            "Mixed understanding/generation decoder layer is not supported. Split the sequence at "
            "token-type boundaries and run the understanding and generation paths separately."
        )


class SenseNovaU1Model(nn.Module):
    def __init__(
        self,
        vocab_size: Any = 151936,
        hidden_size: Any = 4096,
        intermediate_size: Any = 12288,
        num_hidden_layers: Any = 42,
        num_attention_heads: Any = 32,
        num_key_value_heads: Any = 8,
        head_dim: Any = 128,
        attention_bias: Any = False,
        attention_dropout: Any = 0.0,
        rms_norm_eps: Any = 1e-06,
        rope_theta: Any = 5000000.0,
        rope_theta_hw: Any = 10000.0,
        max_position_embeddings: Any = 262144,
        max_position_embeddings_hw: Any = 10000,
        pad_token_id: Any = 151643,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.layers = nn.ModuleList(
            [
                SenseNovaU1DecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    attention_bias=attention_bias,
                    attention_dropout=attention_dropout,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    rope_theta_hw=rope_theta_hw,
                    max_position_embeddings=max_position_embeddings,
                    max_position_embeddings_hw=max_position_embeddings_hw,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: Any = None,
        inputs_embeds: Any = None,
        image_gen_indicators: Any = None,
        indexes: Any = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: Any = True,
        update_cache: Any = True,
        use_gradient_checkpointing: Any = False,
        use_gradient_checkpointing_offload: Any = False,
    ) -> Any:
        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            past_key_value = None
            if past_key_values is not None and layer_idx < len(past_key_values.layers):
                cache_layer = past_key_values.layers[layer_idx]
                if cache_layer.keys is not None:
                    past_key_value = cache_layer.keys, cache_layer.values
            write_cache = update_cache and past_key_values is not None
            layer_output = gradient_checkpoint_forward(
                decoder_layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=past_key_value,
                update_cache=write_cache,
            )
            if write_cache:
                hidden_states, key_states, value_states = layer_output
                past_key_values.update(key_states, value_states, layer_idx, cache_kwargs=None)
            else:
                hidden_states = layer_output

        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm_mot_gen(hidden_states)
        return hidden_states, past_key_values


class SenseNovaU1ForCausalLM(nn.Module):
    def __init__(self, vocab_size: Any = 151936, hidden_size: Any = 4096, **kwargs: Any) -> None:
        super().__init__()
        self.model = SenseNovaU1Model(vocab_size=vocab_size, hidden_size=hidden_size, **kwargs)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, *args: Any, logits_to_keep: Any = 0, **kwargs: Any) -> Any:
        hidden_states, past_key_values = self.model(*args, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits, hidden_states, past_key_values


def precompute_rope_freqs_sincos(dim: int, max_position: int, base: float = 10000.0, device: Any = None) -> Any:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb_1d(x: Any, cos_cached: Any, sin_cached: Any, positions: Any) -> Any:
    cos = cos_cached[positions]
    sin = sin_cached[positions]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(
    x: Any,
    cos_cached_x: Any,
    sin_cached_x: Any,
    cos_cached_y: Any,
    sin_cached_y: Any,
    abs_positions_x: Any,
    abs_positions_y: Any,
) -> Any:
    dim_half = x.shape[-1] // 2
    rotated_part_1 = apply_rotary_emb_1d(x[..., :dim_half], cos_cached_x, sin_cached_x, abs_positions_x)
    rotated_part_2 = apply_rotary_emb_1d(x[..., dim_half:], cos_cached_y, sin_cached_y, abs_positions_y)
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class SenseNovaU1VisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: Any = 1024,
        llm_hidden_size: Any = 4096,
        patch_size: Any = 16,
        num_channels: Any = 3,
        downsample_ratio: Any = 0.5,
        rope_theta_vision: Any = 10000.0,
        max_position_embeddings_vision: Any = 10000,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.llm_embed_dim = llm_hidden_size
        self.downsample_factor = int(1 / downsample_ratio)
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

        self.rope_dim_part = self.embed_dim // 2
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.rope_theta_vision = rope_theta_vision

        self.register_buffer("cos_cached_x", None, persistent=False)
        self.register_buffer("sin_cached_x", None, persistent=False)
        self.register_buffer("cos_cached_y", None, persistent=False)
        self.register_buffer("sin_cached_y", None, persistent=False)

    def _ensure_rope_cache(self, device: torch.device) -> None:
        if self.cos_cached_x is not None and self.cos_cached_x.device == device:
            return
        cos, sin = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            self.max_position_embeddings_vision,
            base=self.rope_theta_vision,
            device=device,
        )
        self.cos_cached_x = cos
        self.sin_cached_x = sin
        self.cos_cached_y = cos.clone()
        self.sin_cached_y = sin.clone()

    def _apply_2d_rotary_pos_emb(self, patch_embeds: Any, grid_hw: Any) -> Any:
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(self, pixel_values: torch.Tensor, grid_hw: Any = None) -> torch.Tensor:
        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)
        self._ensure_rope_cache(patch_embeds.device)
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)

        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor**2)

        return embeddings


class SenseNovaU1VisionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: Any = 1024,
        llm_hidden_size: Any = 4096,
        patch_size: Any = 16,
        num_channels: Any = 3,
        downsample_ratio: Any = 0.5,
        rope_theta_vision: Any = 10000.0,
        max_position_embeddings_vision: Any = 10000,
    ) -> None:
        super().__init__()
        self.embeddings = SenseNovaU1VisionEmbeddings(
            hidden_size=hidden_size,
            llm_hidden_size=llm_hidden_size,
            patch_size=patch_size,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            rope_theta_vision=rope_theta_vision,
            max_position_embeddings_vision=max_position_embeddings_vision,
        )

    def forward(self, pixel_values: Any = None, grid_hw: Any = None, pixel_embeds: Any = None) -> Any:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
        if pixel_embeds is not None:
            return pixel_embeds
        assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
        return self.embeddings(pixel_values, grid_hw=grid_hw)


class SenseNovaU1TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: Any, frequency_embedding_size: Any = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> Any:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Any) -> Any:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class SenseNovaU1ConvDecoder(nn.Module):
    def __init__(self, input_dim: Any = 4096, hidden_dim: Any = 1024) -> None:
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)

        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x: Any) -> Any:
        x = self.act1(self.conv1(self.ps1(x)))
        x = self.ps3(self.conv2(self.ps2(x)))
        return x


class SenseNovaU1DiT(nn.Module):
    _repeated_blocks = ["SenseNovaU1DecoderLayer"]

    def __init__(
        self,
        vocab_size: Any = 151936,
        hidden_size: Any = 4096,
        intermediate_size: Any = 12288,
        num_hidden_layers: Any = 42,
        num_attention_heads: Any = 32,
        num_key_value_heads: Any = 8,
        head_dim: Any = 128,
        rms_norm_eps: Any = 1e-06,
        attention_bias: Any = False,
        attention_dropout: Any = 0.0,
        rope_theta: Any = 5000000.0,
        rope_theta_hw: Any = 10000.0,
        max_position_embeddings: Any = 262144,
        max_position_embeddings_hw: Any = 10000,
        pad_token_id: Any = 151643,
        vision_hidden_size: Any = 1024,
        patch_size: Any = 16,
        num_channels: Any = 3,
        rope_theta_vision: Any = 10000.0,
        max_position_embeddings_vision: Any = 10000,
        downsample_ratio: Any = 0.5,
        use_pixel_head: Any = True,
        fm_head_dim: Any = 1536,
        fm_head_layers: Any = 2,
        fm_head_mlp_ratio: Any = 1,
        add_noise_scale_embedding: Any = True,
        noise_scale: Any = 1.0,
        noise_scale_mode: Any = "resolution",
        noise_scale_base_image_seq_len: Any = 64,
        noise_scale_max_value: Any = 16.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.merge_size = int(1 / downsample_ratio)
        self.hidden_size = hidden_size
        self.use_pixel_head = use_pixel_head
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.noise_scale_max_value = noise_scale_max_value

        self.language_model = SenseNovaU1ForCausalLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position_embeddings=max_position_embeddings,
            max_position_embeddings_hw=max_position_embeddings_hw,
            pad_token_id=pad_token_id,
        )
        self.vision_model = SenseNovaU1VisionEncoder(
            hidden_size=vision_hidden_size,
            llm_hidden_size=hidden_size,
            patch_size=patch_size,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            rope_theta_vision=rope_theta_vision,
            max_position_embeddings_vision=max_position_embeddings_vision,
        )

        fm_modules = {
            "vision_model_mot_gen": SenseNovaU1VisionEncoder(
                hidden_size=vision_hidden_size,
                llm_hidden_size=hidden_size,
                patch_size=patch_size,
                num_channels=num_channels,
                downsample_ratio=downsample_ratio,
                rope_theta_vision=rope_theta_vision,
                max_position_embeddings_vision=max_position_embeddings_vision,
            ),
            "timestep_embedder": SenseNovaU1TimestepEmbedder(hidden_size),
        }
        if add_noise_scale_embedding:
            fm_modules["noise_scale_embedder"] = SenseNovaU1TimestepEmbedder(hidden_size)
        fm_modules["fm_head"] = SenseNovaU1ConvDecoder(hidden_size)
        self.fm_modules = nn.ModuleDict(fm_modules)

    def get_input_embeddings(self) -> Any:
        return self.language_model.model.embed_tokens

    def embed_tokens(self, input_ids: Any) -> Any:
        return self.language_model.model.embed_tokens(input_ids)

    def compute_noise_scale(self, height: Any, width: Any) -> Any:
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (self.merge_size**2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        return min(noise_scale, self.noise_scale_max_value)

    def extract_und_feature(self, pixel_values: Any, grid_hw: Any) -> Any:
        return self.vision_model(pixel_values=pixel_values, grid_hw=grid_hw)

    def extract_gen_feature(self, pixel_values: Any, grid_hw: Any) -> Any:
        return self.fm_modules["vision_model_mot_gen"](pixel_values=pixel_values, grid_hw=grid_hw)

    def embed_timestep(self, timestep: Any, noise_scale: Any, num_tokens: Any, batch_size: Any = 1) -> Any:
        t_expanded = timestep.expand(batch_size * num_tokens)
        embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(batch_size, num_tokens, -1)
        if self.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
            noise_embeddings = self.fm_modules["noise_scale_embedder"](noise_scale_tensor)
            embeddings = embeddings + noise_embeddings.view(batch_size, num_tokens, -1)
        return embeddings

    def encode_prefix(
        self,
        input_ids: Any = None,
        inputs_embeds: Any = None,
        indexes: Any = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_gradient_checkpointing: Any = False,
        use_gradient_checkpointing_offload: Any = False,
    ) -> Any:
        hidden_states, past_key_values = self.language_model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            update_cache=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return past_key_values, hidden_states

    def _append_tokens_to_cache(self, past_key_values: Any, t_idx: Any, input_ids: Any) -> Any:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
        zeros = torch.zeros(seq_len, dtype=torch.long, device=device)
        indexes = torch.stack([t_indexes, zeros, zeros], dim=0)

        past_len = past_key_values.get_seq_length()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        attention_mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
        attention_mask[:, :, :, past_len:] = torch.where(
            causal == 1, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device)
        )

        self.language_model.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return t_idx + seq_len

    def _decode_one_token(self, token: Any, t_idx: Any, past_key_values: Any) -> Any:
        indexes = torch.tensor([[t_idx + 1], [0], [0]], dtype=torch.long, device=token.device)
        return self.language_model(
            input_ids=token.view(1, 1),
            indexes=indexes,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

    @torch.no_grad()
    def generate_think(
        self,
        input_ids: Any = None,
        inputs_embeds: Any = None,
        indexes: Any = None,
        attention_mask: Any = None,
        eos_token_id: Any = None,
        think_end_token_id: Any = None,
        append_ids: Any = None,
        max_think_tokens: Any = 1024,
    ) -> Any:
        logits, _, past_key_values = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        t_idx = int(indexes[0].max().item())
        token_ids = []
        next_token = logits[:, -1, :].argmax(dim=-1)

        for _ in range(max_think_tokens):
            token_id = int(next_token.item())
            if token_id == eos_token_id:
                break
            token_ids.append(token_id)
            logits, _, past_key_values = self._decode_one_token(next_token, t_idx, past_key_values)
            t_idx += 1
            if token_id == think_end_token_id:
                break
            next_token = logits[:, -1, :].argmax(dim=-1)

        if append_ids is not None and append_ids.shape[1] > 0:
            t_idx = self._append_tokens_to_cache(past_key_values, t_idx, append_ids)
        return past_key_values, t_idx, token_ids

    def forward(
        self,
        image_embeds: Any,
        indexes_image: Any,
        past_key_values: Any,
        image_size: Any,
        use_gradient_checkpointing: Any = False,
        use_gradient_checkpointing_offload: Any = False,
    ) -> Any:
        batch_size, num_tokens = image_embeds.shape[0], image_embeds.shape[1]
        image_gen_indicators = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=image_embeds.device)

        hidden_states, _ = self.language_model.model(
            inputs_embeds=image_embeds,
            image_gen_indicators=image_gen_indicators,
            indexes=indexes_image,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            update_cache=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        merged_patch = self.patch_size * self.merge_size
        token_h = image_size[1] // merged_patch
        token_w = image_size[0] // merged_patch

        img_reshaped = hidden_states[:, -num_tokens:].view(batch_size, token_h, token_w, -1)
        img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
        img_2d = img_2d.contiguous().view(batch_size, -1, token_h, token_w)

        smoothed_img_2d = self.fm_modules["fm_head"](img_2d)

        smoothed = smoothed_img_2d.view(batch_size, 3, token_h, merged_patch, token_w, merged_patch)
        smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
        return smoothed.contiguous().view(batch_size, num_tokens, merged_patch * merged_patch * 3)

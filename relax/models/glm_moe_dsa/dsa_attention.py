# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
DSAMLASelfAttention: Fused Dynamic Sparse Attention with Multi-Latent Attention.

Ported from https://github.com/THUDM/slime (slime_plugins/models/glm5/glm5.py).

Key improvements over Megatron Core's native DSAttention:
  1. Context Parallelism (CP) support via gather/scatter on sequence parallel regions
  2. Fused SparseMLA kernel (tilelang) — never materializes O(n^2) attention matrix
  3. Query-blocked indexer kernel and GLM-5.2/5.3 cross-layer index sharing
  4. Absorbed MLA — absorbs KV up-projection into query for memory efficiency
"""

import copy
import math
from dataclasses import dataclass
from typing import NoReturn

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding, _yarn_get_mscale
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

# use fp32 for index weight
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction as WeightLinearFunction
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from .index_sharing import shared_topk, validate_stage
from .lora import absorbed_kv_weight, fused_norm_weight
from .ops.indexer import generate_varlen_mask_params, lighting_indexer
from .ops.sparse_mla import SparseMLA


@dataclass
class DSASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_down_proj: ModuleSpec | type = None
    linear_q_up_proj: ModuleSpec | type = None
    linear_kv_down_proj: ModuleSpec | type = None
    linear_kv_up_proj: ModuleSpec | type = None
    linear_v_up_proj: ModuleSpec | type = None
    core_attention: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None
    q_layernorm: ModuleSpec | type = None
    kv_layernorm: ModuleSpec | type = None
    # added for indexer
    wq_b: ModuleSpec | type = None
    wk: ModuleSpec | type = None
    k_norm: ModuleSpec | type = None
    weights_proj: ModuleSpec | type = None


class DSAMultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == "selective" and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(f"Unsupported RoPE type: {self.config.rope_type}, supported types are 'rope' and 'yarn'")

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        self.index_topk = getattr(config, "dsa_indexer_topk", 2048)
        sources = getattr(config, "glm_dsa_index_sources", ())
        # MTP uses local layer numbers and its own indexer weights. It must not
        # accidentally consume a decoder entry with the same layer number.
        self.index_share = not is_mtp_layer and any(i != source for i, source in enumerate(sources, 1))
        self.index_source_layer = sources[layer_number - 1] if self.index_share else layer_number
        self.skip_topk = self.index_source_layer != layer_number

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        """Forward pass for multi-latent attention."""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query_absorbed: [96, 16, 576], kv: [96, 1, 576], wv: [16, 128, 512]
        q, kv, wv, index_query, index_key, head_weights = self.get_absorb_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        @torch.no_grad()
        def fused_select_topk(index_q, index_k, w, starts, ends, block_size=1024):
            seq_len = index_q.shape[0]
            topk_indices = []
            for start in range(0, seq_len, block_size):
                end = min(start + block_size, seq_len)
                index_q_block = index_q[start:end]
                w_block = w[start:end]
                starts_block = starts[start:end]
                ends_block = ends[start:end]
                _, topk_indices_block = lighting_indexer(
                    index_q_block,
                    index_k,
                    w_block,
                    starts_block.to(torch.int32),
                    ends_block.to(torch.int32),
                    self.index_topk,
                    topk_indices=None,
                )
                topk_indices.append(topk_indices_block)
            return torch.cat(topk_indices, dim=0).unsqueeze(1)

        def compute_topk():
            # CP uses contiguous allgather-CP query shards and globally gathered K.
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
            if cu_seqlens is None:
                cu_seqlens = packed_seq_params.cu_seqlens_q
            starts, ends = generate_varlen_mask_params(cu_seqlens, seq_len=kv.shape[0])
            starts = scatter_to_sequence_parallel_region(starts, group=self.pg_collection.cp)
            ends = scatter_to_sequence_parallel_region(ends, group=self.pg_collection.cp)
            return fused_select_topk(index_query, index_key.squeeze(1), head_weights.unsqueeze(-1), starts, ends)

        topk_indices = (
            shared_topk(packed_seq_params, self.layer_number, self.index_source_layer, compute_topk)
            if self.index_share
            else compute_topk()
        )

        core_attn_out, _ = SparseMLA.apply(q, kv, topk_indices, self.softmax_scale)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, wv)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class DSAMLASelfAttention(DSAMultiLatentAttention):
    """MLA Self-attention layer class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            is_mtp_layer=is_mtp_layer,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        q_down_proj_kwargs = {}
        if submodules.linear_q_down_proj in [TELinear]:
            q_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_q_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            q_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_down_proj",
            skip_weight_param_allocation=False,
            **q_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up_proj",
        )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_down_proj",
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # Shared layers have no indexer tensors in the HF checkpoint.
        if self.skip_topk:
            return

        # added for indexer
        indexer_linear_kwargs = dict(
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
        )

        self.wq_b = build_module(
            submodules.wq_b,
            input_size=self.config.q_lora_rank,
            output_size=self.config.index_num_attention_heads * self.config.index_head_dim,
            tp_comm_buffer_name="wq_b",
            **indexer_linear_kwargs,
        )
        self.wq_b.weight._skip_gather = True

        # Build key projection
        self.wk = build_module(
            submodules.wk,
            input_size=self.config.hidden_size,
            output_size=self.config.index_head_dim,
            tp_comm_buffer_name="wk",
            **indexer_linear_kwargs,
        )

        # Build key normalization
        old_value = self.config.normalization
        assert config.normalization == "RMSNorm"
        self.config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            hidden_size=self.config.index_head_dim,
            config=self.config,
            # The layernorm eps is hardcoded at the moment
            eps=1e-6,
        )
        self.config.normalization = old_value

        # Build attention weight projection (per-head gating)
        # not sharded weights
        self.weights_proj = build_module(
            submodules.weights_proj,
            input_size=self.config.hidden_size,
            output_size=self.config.index_num_attention_heads,
            tp_comm_buffer_name="weights_proj",
            **indexer_linear_kwargs,
        )
        self.weights_proj.weight._skip_gather = True

    def get_absorb_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Derives `query`, `key` and `value` tensors from `hidden_states`."""
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        assert packed_seq_params is not None

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )
        # TODO: support apply_rope_fusion
        thd_packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        mscale = 1.0
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=thd_packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=thd_packed_seq)

        # Allgather CP keeps real lengths separate from the physical buffer.
        # Apex writes only the rows covered by cu_seqlens; using real lengths
        # leaves the padded RoPE tail uninitialized and poisons backward.
        cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
        if cu_seqlens_q is None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
        if cu_seqlens_kv is None:
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # down proj are `TELinear`s, so the output is gathered and not TP-partitioned.`
        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if self.config.sequence_parallel:
            kv_combined = gather_from_sequence_parallel_region(kv_combined)

        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # absorb
        # =========================================
        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

        w_kc, w_vc = (
            absorbed_kv_weight(self.linear_kv_up_proj)
            .unflatten(
                0,
                (-1, self.config.qk_head_dim + self.config.v_head_dim),
            )
            .split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)
        )

        # absorb
        q_no_pe = torch.einsum("thd,hdm->thm", q_no_pe, w_kc)

        # use scatter and gather here, to make the kv grad all reduce in tp
        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=fused_norm_weight(self.linear_kv_up_proj).float(),
            eps=self.config.layernorm_epsilon,
        ).to(kv_compressed.dtype)

        k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=parallel_state.get_context_parallel_group())
        kv_compressed = gather_from_sequence_parallel_region(
            kv_compressed, group=parallel_state.get_context_parallel_group()
        )

        def fuse_rope(q, cu_seqlens, gathered=False):
            # worse precision than apex.
            # from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
            from apex.transformer.functional import fused_apply_rotary_pos_emb_thd

            # mla use rope interleave
            x1 = q[..., 0::2]
            x2 = q[..., 1::2]
            t = torch.cat((x1, x2), dim=-1)
            # TODO remove copy here
            # fuse rope not support this way rope (diff with cp)
            if gathered:
                return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
            else:
                seq_len = q.shape[0]
                cp_size = parallel_state.get_context_parallel_world_size()
                cp_rank = parallel_state.get_context_parallel_rank()
                t = t.repeat(cp_size, 1, 1)
                out = fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
                return out[cp_rank * seq_len : (cp_rank + 1) * seq_len]

        q_pos_emb = fuse_rope(q_pos_emb, cu_seqlens_q, gathered=False)
        k_pos_emb = fuse_rope(k_pos_emb, cu_seqlens_kv, gathered=True)

        query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
        key = torch.cat([kv_compressed, k_pos_emb], dim=-1)

        query = query.contiguous()
        key = key.contiguous()

        if self.skip_topk:
            return query, key, w_vc, None, None, None

        # =========================================
        # Indexer
        # =========================================
        # HF reference (transformers.GlmMoeDsaIndexer.forward):
        #   q_resid = q_a_layernorm(q_a_proj(x))   ← POST-layernorm
        #   q = wq_b(q_resid).view(..., n_heads, head_dim)
        #   q_pe, q_nope = split(q, [rope_dim, head_dim - rope_dim], dim=-1)   ← pe FIRST
        #   q_pe = apply_rotary_pos_emb(q_pe)
        #   q = cat([q_pe, q_nope], dim=-1)
        #
        # In Megatron, q_layernorm is IdentityOp (the actual layernorm is fused
        # inside linear_q_up_proj.layer_norm_weight), so q_compressed is still
        # PRE-layernorm here.  Apply the same RMSNorm explicitly before wq_b.
        q_compressed = torch.nn.functional.rms_norm(
            q_compressed.float(),
            normalized_shape=(q_compressed.shape[-1],),
            weight=fused_norm_weight(self.linear_q_up_proj).float(),
            eps=self.config.layernorm_epsilon,
        ).to(q_compressed.dtype)
        q_compressed = q_compressed.detach()
        hidden_states = hidden_states.detach()
        rotary_pos_emb = rotary_pos_emb.detach()

        index_q, _ = self.wq_b(q_compressed)
        index_q = index_q.view(*index_q.size()[:-1], self.config.index_num_attention_heads, self.config.index_head_dim)
        # [total_tokens, index_num_attention_heads_per_partition, index_head_dim]
        if self.config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k.squeeze(1).float()).bfloat16()
        if self.config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())
        index_k = index_k.unsqueeze(1)  # [total_tokens, 1, head_dim]

        # head_weights, _ = self.weights_proj(hidden_states.float())
        head_weights = WeightLinearFunction.apply(hidden_states, self.weights_proj.weight, None, torch.float32)
        head_weights = head_weights.squeeze(1) * (
            (self.config.index_num_attention_heads**-0.5) * (self.config.index_head_dim**-0.5)
        )
        # [total_tokens, index_num_attention_heads_per_partition]
        if self.config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        # Indexer split order is [pe, nope] per HF reference (NOT [nope, pe] like MLA).
        index_q_pe, index_q_no_pe = torch.split(
            index_q,
            [self.config.qk_pos_emb_head_dim, self.config.index_head_dim - self.config.qk_pos_emb_head_dim],
            dim=-1,
        )
        index_q_pe = fuse_rope(index_q_pe, cu_seqlens_q, gathered=False)
        index_query = torch.cat([index_q_pe, index_q_no_pe], dim=-1)

        index_k_pe, index_k_no_pe = torch.split(
            index_k,
            [self.config.qk_pos_emb_head_dim, self.config.index_head_dim - self.config.qk_pos_emb_head_dim],
            dim=-1,
        )
        index_k_pe = fuse_rope(index_k_pe, cu_seqlens_kv, gathered=True)
        index_key = torch.cat([index_k_pe, index_k_no_pe], dim=-1)

        return query, key, w_vc, index_query, index_key, head_weights

    def get_query_key_value_tensors(self):
        pass

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations."""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Update weights for KV projection layers."""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Update weights for Q projection layers."""
        self.linear_q_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Update weights for output projection layer."""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm.

        Only needed for fp8.
        """
        if self.config.q_lora_rank is not None:
            if hasattr(self.linear_q_down_proj, "save_original_input"):
                self.linear_q_down_proj.save_original_input = True
            else:
                raise ValueError(
                    "layernorm recompute for fp8 with MLASelfAttention needs transformer-engine>=2.6.0dev0."
                )
        if hasattr(self.linear_kv_down_proj, "save_original_input"):
            self.linear_kv_down_proj.save_original_input = True
        else:
            raise ValueError("layernorm recompute for fp8 with MLASelfAttention needs transformer-engine>=2.6.0dev0.")
        if hasattr(self.linear_proj, "save_original_input"):
            self.linear_proj.save_original_input = True
        else:
            raise ValueError("layernorm recompute for fp8 with MLASelfAttention needs transformer-engine>=2.6.0dev0.")


def get_glm5_dsa_spec(config, vp_stage=None):
    """Build a transformer block spec with DSAMLASelfAttention replacing
    standard attention.

    This is the bridge-mode equivalent of slime's get_glm5_spec(). It generates
    the standard GPT decoder block spec (with per-layer dense/MoE support) and
    then replaces the self_attention submodule in every layer with
    DSAMLASelfAttention + fused tilelang kernels.

    Args:
        config: TransformerConfig (or GPTModelProvider) with index_num_attention_heads
                and index_head_dim already set.
        vp_stage: Virtual pipeline stage (optional).

    Returns:
        TransformerBlockSubmodules with DSAMLASelfAttention in every layer.
    """
    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage

    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    sources = getattr(config, "glm_dsa_index_sources", ())
    if sources:
        validate_stage(sources, get_transformer_layer_offset(config, vp_stage=vp_stage), num_layers_to_build)

    for layer_id in range(num_layers_to_build):
        transformer_layer_spec.layer_specs[layer_id] = transform_glm5_layer_spec(
            config, transformer_layer_spec.layer_specs[layer_id]
        )

    return transformer_layer_spec


def transform_glm5_layer_spec(config: MLATransformerConfig, spec: ModuleSpec) -> ModuleSpec:
    """Apply GLM attention to decoder specs and standalone MTP pipeline
    specs."""
    backend = TESpecProvider()
    layer_spec = copy.deepcopy(spec)
    layer_spec.submodules.self_attention = ModuleSpec(
        module=DSAMLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            linear_v_up_proj=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )

    return layer_spec

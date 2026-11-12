# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3.1's frozen NoPE indexer and sparse MLA, using Megatron Core DSA."""

import copy
from typing import Any

import torch
import torch.nn.functional as F
from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, _linear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from torch import Tensor, nn


class KimiK31Indexer(nn.Module):
    """Replicated checkpoint indexer; policy training has no indexer
    objective."""

    def __init__(self, config: Any, pg_collection: Any = None) -> None:
        super().__init__()
        del pg_collection
        self.num_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.eps = config.dsa_indexer_k_norm_epsilon
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        kwargs = {"device": device, "dtype": config.params_dtype}
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, **kwargs)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, **kwargs)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=self.eps, **kwargs)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, **kwargs)
        self.requires_grad_(False)

    def forward_before_topk(
        self, x: Tensor, qr: Tensor, packed_seq_params: PackedSeqParams | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        del packed_seq_params
        query = self.wq_b(qr).unflatten(-1, (self.num_heads, self.head_dim))
        key = F.layer_norm(
            self.wk(x).float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.eps,
        ).to(x.dtype)
        # Match SGLang's BF16 head gate: the first scaling rounds in BF16.
        weights = (self.weights_proj(x) * self.num_heads**-0.5).float() * self.head_dim**-0.5
        return query, key, weights

    def backward_dw(self) -> None:
        """There are no deferred weight gradients for the frozen indexer."""


class KimiK31DSAAttention(KimiK3Attention):
    """K3 MLA with sparse selection and its unrotated shared key channels."""

    def _init_mla(self, config: Any) -> None:
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention, DSAttentionSubmodules

        super()._init_mla(config)
        dsa = config.kimi_dsa_config
        core_config = copy.copy(config)
        # K3Attention.forward already gathers the sequence-parallel input.
        core_config.sequence_parallel = False
        core_config.dsa_indexer_n_heads = dsa["index_num_heads"]
        core_config.dsa_indexer_head_dim = dsa["index_head_dim"]
        core_config.dsa_indexer_topk = dsa["index_topk"]
        core_config.dsa_indexer_topk_freq = 1
        core_config.dsa_indexer_skip_topk_offset = 0
        core_config.dsa_indexer_loss_coeff = 0.0
        core_config.dsa_indexer_k_norm_epsilon = dsa.get("index_k_norm_eps", 1e-5)
        core_config.dsa_indexer_k_norm_fp32 = True
        core_config.dsa_indexer_scoring_relu = True
        core_config.dsa_indexer_rotate_activation = False
        if core_config.dsa_kernel_backend == "none":
            core_config.dsa_kernel_backend = "tilelang"
        from relax.utils.env import Envs

        if Envs.RELAX_KIMI_K31_EXPANDED_DSA_BACKEND not in (None, "", "core"):
            from relax.models.kimi_k3.expanded_dsa import configure_expanded_attention

            configure_expanded_attention(core_config, Envs.RELAX_KIMI_K31_EXPANDED_DSA_BACKEND)
        self.core_attention = DSAttention(
            config=core_config,
            submodules=DSAttentionSubmodules(indexer=KimiK31Indexer),
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            # Core owns the CP key gather and the packed zigzag permutations.
            cp_comm_type="allgather",
            pg_collection=self.pg_collection,
        )

    def _forward_mla(self, hidden_states: Tensor, packed_seq_params: PackedSeqParams | None) -> Tensor:
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.utils import ensure_params_ready

        x = hidden_states
        q_lora = self.q_a_layernorm(_linear(self.q_a_proj, x))
        query = _linear(self.q_b_proj, q_lora).unflatten(-1, (self.local_num_heads, self.q_head_dim))
        kv_latent, key_extra = _linear(self.kv_a_proj_with_mqa, x).split(
            (self.kv_lora_rank, self.qk_extra_head_dim), dim=-1
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        up_v_weight = None
        if hasattr(self.kv_b_proj, "adapter"):
            # Adapter forward preserves dropout and TP gradient communication;
            # directly merging its weights would lose both contracts.
            # This expanded path uses more memory for long-context LoRA.
            key_value = _linear(self.kv_b_proj, kv_latent).unflatten(
                -1, (self.local_num_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            key_nope, value = key_value.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
            key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
            key = torch.cat((key_nope, key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)), dim=-1)
            value = value.contiguous()
        else:
            # Reading weights directly bypasses their module's param-gather
            # hook. Autograd produces ordinary grads; Core DDP accumulates
            # them into main_grad, since no TE linear backward ran here.
            ensure_params_ready(self.kv_b_proj.parameters())
            up_k_weight, up_v_weight = self.kv_b_proj.weight.unflatten(
                0, (self.local_num_heads, self.qk_nope_head_dim + self.v_head_dim)
            ).split((self.qk_nope_head_dim, self.v_head_dim), dim=1)
            query_nope, query_extra = query.split((self.qk_nope_head_dim, self.qk_extra_head_dim), dim=-1)
            query = torch.cat((torch.einsum("sbhd,hdc->sbhc", query_nope, up_k_weight), query_extra), dim=-1)
            # kv_b_proj's column-parallel input reduction is bypassed too.
            key = copy_to_tensor_model_parallel_region(
                torch.cat((kv_latent, key_extra), dim=-1), group=self.tp_group
            ).unsqueeze(-2)
            value = None
            if self.core_attention.config.dsa_kernel_backend == "tilelang":
                # Native SparseMLA requires power-of-two head tiles. K3's
                # 96 heads (or 48/24 with TP) otherwise silently fall back to
                # quadratic attention. Discard only the added heads below.
                # TODO(agent): validate fused BF16 forward/backward on multi-node GPUs.
                kernel_heads = max(16, 1 << (self.local_num_heads - 1).bit_length())
                query = F.pad(query, (0, 0, 0, kernel_heads - self.local_num_heads))
                up_v_weight = F.pad(up_v_weight, (0, 0, 0, 0, 0, kernel_heads - self.local_num_heads))
        output = self.core_attention(
            query,
            key,
            value,
            None,
            x=x,
            qr=q_lora,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
            up_v_weight=up_v_weight,
        )
        output = output[..., : self.local_num_heads * self.v_head_dim]
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)


def build_kimi_k31_spec(config: Any, vp_stage: int | None = None) -> Any:
    """Retain Bridge's KDA, MoE and AttnRes specs; replace only MLA layers."""
    from megatron.bridge.models.kimi.kimi_k3_spec import build_kimi_k3_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    block_spec = build_kimi_k3_spec(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage)
    for local_idx, layer_spec in enumerate(block_spec.layer_specs):
        if offset + local_idx + 1 in config.kimi_dsa_config["dsa_layers"]:
            layer_spec.submodules.self_attention = ModuleSpec(module=KimiK31DSAAttention)
    return block_spec

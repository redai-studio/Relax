# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Modified for Relax integration; see NOTICE.

import copy
from typing import Any

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TENorm, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch import nn

from .ops.compressor import DeepSeekV41Compressor
from .ops.indexer import DeepSeekV41Indexer
from .runtime import V41Runtime


class DeepSeekV41Attention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        runtime: V41Runtime,
        submodules: Any = None,
        layer_number: int = 1,
        attn_mask_type: Any = None,
        attention_type: str | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        self.runtime = runtime
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp if hasattr(pg_collection, "cp") else None
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1

        layer_id = layer_number - 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // config.tensor_model_parallel_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.kv_lora_rank
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // config.tensor_model_parallel_size
        self.window_size = config.csa_window_size
        self.compress_ratio = config.csa_compress_ratios[layer_id]
        self.eps = config.layernorm_epsilon
        self.softmax_scale = self.head_dim**-0.5
        self.sequence_parallel = config.sequence_parallel
        assert self.compress_ratio in (0, 1, 2)
        assert self.head_dim == 512 and self.rope_head_dim == 64 and self.window_size == 128

        self.is_kv_source = layer_id in config.v41_kv_source_layer_ids
        self.is_index_source = layer_id in config.v41_index_source_layer_ids
        self.kv_source_layer = None
        if self.compress_ratio:
            sources = [
                layer
                for layer in config.v41_kv_source_layer_ids
                if layer <= layer_id and config.csa_compress_ratios[layer] == self.compress_ratio
            ]
            assert sources, f"layer {layer_id} has no kv_source layer"
            self.kv_source_layer = max(sources)

        config_no_sp = copy.copy(config)
        config_no_sp.sequence_parallel = False

        self.core_attention = nn.Module()
        self.core_attention.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        mark_keep_in_fp32(self.core_attention.attn_sink)
        set_tensor_model_parallel_attributes(self.core_attention.attn_sink, is_parallel=True, dim=0, stride=1)

        self.linear_q_down_proj = TELinear(
            self.dim,
            self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_layernorm = TENorm(config_no_sp, self.q_lora_rank, eps=self.eps)
        self.linear_q_up_proj = TEColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.linear_kv_proj = TELinear(
            self.dim,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.kv_layernorm = TENorm(config_no_sp, self.head_dim, eps=self.eps)
        for p in list(self.linear_q_down_proj.parameters()) + list(self.linear_kv_proj.parameters()):
            p.sequence_parallel = False

        o_group_proj = torch.empty(
            self.n_local_groups * self.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        config.init_method(o_group_proj)
        self.linear_o_group_proj = nn.Parameter(o_group_proj)
        set_tensor_model_parallel_attributes(self.linear_o_group_proj, is_parallel=True, dim=0, stride=1)
        self.linear_proj = TERowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

        if self.is_kv_source:
            self.core_attention.compressor = DeepSeekV41Compressor(config, self.head_dim, self.compress_ratio)
        if self.is_index_source:
            self.core_attention.indexer = DeepSeekV41Indexer(config, layer_id=layer_id, owns_k=self.is_kv_source)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        ans = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ans.update(
            make_sharded_tensors_for_checkpoint(
                state_dict={
                    "core_attention.attn_sink": self.core_attention.attn_sink,
                    "linear_o_group_proj": self.linear_o_group_proj,
                },
                prefix=prefix,
                tensor_parallel_layers_axis_map={"core_attention.attn_sink": 0, "linear_o_group_proj": 0},
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=(metadata or {}).get("dp_cp_group"),
            )
        )
        return ans

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inference_context: Any = None,
        rotary_pos_emb: Any = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        sequence_len_offset: Any = None,
    ) -> tuple[torch.Tensor, None]:
        if attention_bias is not None or inference_context is not None:
            raise NotImplementedError("DeepSeek V4.1 training attention does not support attention bias or inference.")
        if packed_seq_params is not None:
            if getattr(packed_seq_params, "qkv_format", None) != "thd":
                raise ValueError("DeepSeek V4.1 packed attention requires THD metadata.")
            from .thd.attention import forward_attention

            return forward_attention(self, hidden_states, packed_seq_params)
        from .bshd.attention import forward_attention

        return forward_attention(self, hidden_states)

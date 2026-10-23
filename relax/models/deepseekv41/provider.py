# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from copy import copy
from dataclasses import dataclass, field
from typing import Any

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt import GPTModel


@dataclass
class DeepSeekV41ModelProvider(MLAModelProvider):
    """V4.1 text architecture and execution provider."""

    experimental_attention_variant: str | None = None
    mtp_num_layers: int | None = None
    moe_n_hash_layers: int = 0
    o_groups: int = 8
    o_lora_rank: int = 1024
    v41_hf_config: dict[str, Any] = field(default_factory=dict)
    v41_kv_source_layer_ids: tuple[int, ...] = ()
    v41_index_source_layer_ids: tuple[int, ...] = ()
    v41_candidate_source_layer_id: int = -1
    v41_candidate_topk_blocks: int = 0
    v41_candidate_block_size: int = 0
    v41_engram_layer_ids: tuple[int, ...] = ()
    v41_engram_num_embeddings: tuple[int, ...] = ()
    v41_engram_max_ngram_size: int = 0
    v41_engram_vocab_size: int = 0
    v41_engram_n_heads: int = 0
    v41_engram_head_dim: int = 0
    v41_engram_pad_token_id: int = 0
    v41_engram_compressed_vocab_size: int = 0
    v41_hc_eps: float = 1e-6
    v41_hf_checkpoint: str | None = None
    v41_qkv_format: str = "bshd"
    v41_allgather_cp: bool = False

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> GPTModel:
        import torch.distributed as dist
        from megatron.bridge.models.gpt_provider import calculate_padded_vocab_size, logit_dtype_kwarg
        from megatron.core.process_groups_config import ProcessGroupCollection

        from .checkpoint import validate_bf16_checkpoint
        from .execution_config import validate_execution_config
        from .model import DeepSeekV41GPTModel

        validate_execution_config(self)
        if self.init_model_with_meta_device:
            raise NotImplementedError(
                "DeepSeek V4.1 host-backed Engram loading does not support meta-device construction."
            )
        validate_bf16_checkpoint(self.v41_hf_checkpoint, require_v41=True)
        groups = getattr(self, "_pg_collection", None)
        if groups is None:
            groups = ProcessGroupCollection.use_mpu_process_groups()
        pp_rank = dist.get_rank(groups.pp)
        pp_size = dist.get_world_size(groups.pp)
        pre_process = pp_rank == 0 if pre_process is None else pre_process
        post_process = pp_rank == pp_size - 1 if post_process is None else post_process
        vocab_size = self.vocab_size
        if vocab_size is None:
            raise ValueError("DeepSeek V4.1 vocab_size must be configured before model construction.")
        if self.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        # Separate configuration objects too; actor/reference instances must not
        # share transient model state when a provider is invoked more than once.
        config = copy(self)
        config._vp_stage = vp_stage
        return DeepSeekV41GPTModel(
            config,
            vocab_size=vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            **logit_dtype_kwarg(GPTModel, self.logit_dtype),
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rope_scaling=self.rope_scaling,
            rope_scaling_factor=self.rope_scaling_factor,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            pg_collection=groups,
            vp_stage=vp_stage,
        )

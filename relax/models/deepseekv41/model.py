# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Modified for Relax integration; see NOTICE.

"""V4.1 text execution with model-instance-owned cross-layer state."""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.hyper_connection import BroadcastTensorFused, HyperConnectionModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor

from .engram import DeepSeekV41Engram
from .runtime import (
    V41CarryPlan,
    V41IdsOnlyPlan,
    V41Runtime,
    v41_aggregate,
    v41_pack_carry,
    v41_unpack_carry,
)


def _uses_carry(config: TransformerConfig) -> bool:
    return config.pipeline_model_parallel_size > 1 or config.recompute_granularity == "full"


class V41HyperConnection(HyperConnectionModule):
    """Consume the previous sublayer's pre; produce the next sublayer's pre."""

    def __init__(self, config: TransformerConfig, layer_number: int, runtime: V41Runtime) -> None:
        super().__init__(config, layer_number)
        self.v41_runtime = runtime
        # Use the checkpoint's RMS epsilon for projection normalization,
        # and the mHC epsilon for the sigmoid floor and Sinkhorn iterations.
        self.norm_eps = config.layernorm_epsilon
        self.compute_h_eps = config.v41_hc_eps
        self.sinkhorn_eps = config.v41_hc_eps

    def forward(
        self, hidden_states: torch.Tensor, mhc_recompute_manager: Any = None, output_slot: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError("DeepSeek V4.1 does not support selective mHC recompute.")
        return self._forward_normal(hidden_states)

    def _forward_normal(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s, b, _ = hidden_states.shape
        mappings, aggregate, residual = BroadcastTensorFused.apply(hidden_states, self._fused_add_3_op)
        projection, norm = self._projection_and_get_norm(mappings)
        next_pre, post, residual_logits = self._compute_h(projection, norm)
        mixing = self._sinkhorn_op(
            residual_logits.view(s, b, self.n, self.n), self.sinkhorn_iterations, self.sinkhorn_eps
        )
        output = v41_aggregate(aggregate, self.v41_runtime.pre, self.n)
        self.v41_runtime.pre = next_pre
        return output, mixing, post, residual

    def fused_h_res_h_post_bda(
        self,
        h_res: torch.Tensor,
        original_residual: torch.Tensor,
        h_post: torch.Tensor,
        layer_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Any = None,
    ) -> torch.Tensor:
        output, bias = layer_output_with_bias
        if bias is not None or dropout_prob != 0 or manager is not None:
            raise NotImplementedError("DeepSeek V4.1 mHC supports no bias, zero hidden dropout and no mHC manager.")
        s, b, _ = original_residual.shape
        residual = original_residual.view(s, b, self.n, self.hidden_size).float()
        mixed = torch.einsum("sbij,sbid->sbjd", h_res.float(), residual)
        output = h_post.float().unsqueeze(-1) * output.float().unsqueeze(2) + mixed
        return output.view(s, b, self.n * self.hidden_size).to(original_residual.dtype)


class V41TransformerLayer(HyperConnectionTransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Any,
        runtime: V41Runtime,
        layer_number: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, submodules, layer_number=layer_number, **kwargs)
        self.v41_runtime = runtime
        layer_id = self.layer_number - 1
        self.v41_engram = None
        if layer_id in config.v41_engram_layer_ids:
            self.v41_engram = DeepSeekV41Engram(
                config, layer_id, self.pg_collection.tp, getattr(self.pg_collection, "cp", None)
            )
        self.v41_is_last = self.layer_number == config.num_layers
        self.v41_carry = _uses_carry(config)
        self.v41_input_plan = V41CarryPlan(config, layer_id - 1) if self.v41_carry else None
        self.v41_output_plan = V41CarryPlan(config, layer_id) if self.v41_carry and not self.v41_is_last else None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        if self.v41_input_plan is not None:
            hidden_states = v41_unpack_carry(hidden_states, self.v41_runtime, self.v41_input_plan)
        if self.v41_engram is not None:
            hidden_states = self.v41_engram(hidden_states, self.v41_runtime.input_ids)
        output, context = super().forward(hidden_states, *args, **kwargs)
        if self.v41_is_last:
            output = v41_aggregate(output, self.v41_runtime.pre, self.config.num_residual_streams)
        elif self.v41_output_plan is not None:
            output = v41_pack_carry(output, self.v41_runtime, self.v41_output_plan)
        return output, context


class V41TransformerBlock(TransformerBlock):
    """Use Core's layer schedule/offloading, but not V4's learned output
    contraction."""

    def __init__(self, config: TransformerConfig, spec: TransformerBlockSubmodules, **kwargs: Any) -> None:
        if not spec.layer_specs:
            raise ValueError("DeepSeek V4.1 currently requires nonempty, evenly partitioned pipeline stages.")
        runtime = spec.layer_specs[0].params.get("runtime")
        if not isinstance(runtime, V41Runtime) or any(
            layer.params.get("runtime") is not runtime for layer in spec.layer_specs
        ):
            raise ValueError("DeepSeek V4.1 layer specs must share their model instance's runtime.")
        self.v41_runtime = runtime
        self.v41_initial_plan = V41IdsOnlyPlan(config) if _uses_carry(config) else None
        super().__init__(config=config, spec=spec, **kwargs)

    def _build_hyper_connection_output_head(self) -> None:
        # The last layer already contracts using the preceding sublayer's pre.
        # V4.1 has no hc_head_fn/base/scale weights to allocate or load.
        return None

    def preprocess_for_layer_schedule(self, hidden_states: Any) -> torch.Tensor:
        hidden_states = super().preprocess_for_layer_schedule(hidden_states)
        if self.pre_process and self.v41_initial_plan is not None:
            hidden_states = v41_pack_carry(hidden_states, self.v41_runtime, self.v41_initial_plan)
        return hidden_states

    def postprocess_for_layer_schedule(
        self,
        hidden_states: torch.Tensor,
        *,
        extract_layer_indices: set[int] | None = None,
        return_mhc_multistream: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if self.final_layernorm is not None:
            if hidden_states.shape[-1] != self.config.hidden_size:
                raise ValueError("DeepSeek V4.1 final layer must contract mHC streams before final normalization.")
            hidden_states = apply_module(self.final_layernorm)(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        if not self.pre_process and len(self.layers) == 0 and self.final_layernorm is None:
            hidden_states = hidden_states.clone()
        return (hidden_states, None) if return_mhc_multistream else hidden_states


def get_v41_spec(
    config: TransformerConfig, runtime: V41Runtime, vp_stage: int | None = None, pp_rank: int | None = None
) -> TransformerBlockSubmodules:
    from .attention import DeepSeekV41Attention

    if config.experimental_attention_variant is not None:
        raise ValueError("DeepSeek V4.1 needs its own spec, not a native experimental attention variant.")
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config, vp_stage=vp_stage, pp_rank=pp_rank
    )
    for layer_spec in block_spec.layer_specs:
        layer_spec.module = V41TransformerLayer
        layer_spec.params = {**layer_spec.params, "runtime": runtime}
        layer_spec.submodules.self_attention = ModuleSpec(
            module=DeepSeekV41Attention,
            params={"runtime": runtime},
            metainfo={"fuse_input_layernorm": False},
        )
        layer_spec.submodules.self_attention_hyper_connection = ModuleSpec(
            module=V41HyperConnection, params={"runtime": runtime}
        )
        layer_spec.submodules.mlp_hyper_connection = ModuleSpec(module=V41HyperConnection, params={"runtime": runtime})
    return block_spec


def _freeze_routers(model: GPTModel) -> None:
    """Freeze routing weights while preserving gradients through routing
    inputs."""
    for layer in model.decoder.layers:
        router = getattr(layer.mlp, "router", None)
        if router is not None:
            for name in ("weight", "expert_bias"):
                value = getattr(router, name, None)
                if isinstance(value, torch.Tensor):
                    value.requires_grad_(False)


class DeepSeekV41GPTModel(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not callable(getattr(GPTModel, "_get_transformer_block_class", None)) or not callable(
            getattr(TransformerBlock, "_build_hyper_connection_output_head", None)
        ):
            raise RuntimeError(
                "DeepSeek V4.1 requires the updated Relax Megatron Core patch; rebuild the training image."
            )
        if "transformer_layer_spec" in kwargs:
            raise ValueError(
                "DeepSeek V4.1 constructs its own execution spec; external transformer_layer_spec is invalid."
            )
        if pg_collection is None:
            raise ValueError("DeepSeek V4.1 requires an explicit model process-group collection.")
        runtime = V41Runtime()
        self.v41_runtime = runtime
        pp_group = getattr(pg_collection, "pp", None)
        pp_rank = pp_group.rank() if pp_group is not None else 0
        block_spec = get_v41_spec(config, runtime, vp_stage=vp_stage, pp_rank=pp_rank)
        super().__init__(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            **kwargs,
        )
        _freeze_routers(self)

    def _get_transformer_block_class(self) -> type[TransformerBlock]:
        return V41TransformerBlock

    def _release_transient_state(self, *, for_offload: bool = True) -> None:
        """Drop microbatch activation references, retaining frozen host-backed
        Engram tables."""
        self.v41_runtime.reset()
        if for_offload:
            from .ops.position import precompute_freqs_cis

            # The shared cache owns immutable CUDA frequency tables too; they
            # are cheaply rebuilt after wake and must not pin selective-offload memory.
            precompute_freqs_cis.cache_clear()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context: Any = None,
        packed_seq_params: Any = None,
        extra_block_kwargs: dict[str, Any] | None = None,
        runtime_gather_output: bool | None = None,
        *,
        inference_params: Any = None,
        loss_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_processor: Any = None,
        output_processor_context: Any = None,
    ) -> torch.Tensor:
        if packed_seq_params is not None:
            raise NotImplementedError("DeepSeek V4.1 attention does not support packed sequences.")
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "DeepSeek V4.1 Megatron execution currently supports training, not KV-cache inference."
            )
        if attention_mask is not None:
            raise NotImplementedError(
                "DeepSeek V4.1 uses built-in causal attention; explicit attention masks are unsupported."
            )
        self.v41_runtime.reset()
        self.v41_runtime.input_ids = input_ids
        try:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
                output_processor=output_processor,
                output_processor_context=output_processor_context,
            )
        finally:
            # Full recompute restores state from each layer's carry during
            # backward. Sleep also clears anything left by that recomputation.
            self._release_transient_state(for_offload=False)

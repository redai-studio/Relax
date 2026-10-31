# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Any

import torch
from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.moe.moe_utils import router_gating_linear
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec

from .attention import KeyeCoreAttention, KeyeSelfAttention
from .linear import KeyeRowParallelLinear


class _StableTopK:
    """Preserve replay while fixing top-k order across grad/no-grad
    forwards."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate

    @property
    def layer_number(self) -> int | None:
        return getattr(self.delegate, "layer_number", None)

    @layer_number.setter
    def layer_number(self, value: int) -> None:
        if self.delegate is not None:
            self.delegate.layer_number = value

    def get_replay_topk(
        self, scores: torch.Tensor, topk: int, num_groups: Any, group_topk: Any, default_compute_topk: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def select(scores: torch.Tensor, topk: int, num_groups: Any = None, group_topk: Any = None) -> Any:
            if num_groups is not None:
                return default_compute_topk(scores, topk, num_groups, group_topk)
            return torch.topk(scores, k=topk, dim=1, sorted=True)

        if self.delegate is None:
            return select(scores, topk, num_groups, group_topk)
        return self.delegate.get_replay_topk(scores, topk, num_groups, group_topk, select)


class KeyeTopKRouter(TopKRouter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.router_replay = _StableTopK(self.router_replay)

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        # FP32 GEMM changes its reduction order with the token count. Compute
        # the dot in FP64, then keep the ordinary FP32 top-k/softmax contract.
        return router_gating_linear(input, self.weight, self.bias, torch.float64).float()


def keye_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    spec = transformer_engine_layer_spec(config)
    spec.submodules.input_layernorm = TENorm
    spec.submodules.self_attention.module = KeyeSelfAttention
    spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    spec.submodules.self_attention.submodules.core_attention = KeyeCoreAttention
    spec.submodules.self_attention.submodules.linear_proj = KeyeRowParallelLinear
    mlp = spec.submodules.mlp
    submodules = mlp.keywords["submodules"] if hasattr(mlp, "keywords") else mlp.submodules
    submodules.router = KeyeTopKRouter
    return spec


@dataclass
class KeyeModelProvider(GPTModelProvider):
    transformer_layer_spec = staticmethod(keye_layer_spec)
    keye_indexer_loss_coeff: float = 0.01
    keye_attention_chunk_size: int = 1024
    hf_model_path: str | None = None
    hf_config: Any = None
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def configure_runtime(self, args: Any) -> None:
        if getattr(args, "dynamic_context_parallel", False):
            raise ValueError("Keye currently supports static context parallelism only")
        if self.context_parallel_size > 1 and args.cp_partition_mode != "zigzag":
            raise ValueError("Keye context parallelism requires zigzag partitioning")
        # TE fused RoPE rounds once; the Keye reference rounds BF16 products separately.
        self.apply_rope_fusion = False
        if args.multimodal_keys:
            self.position_embedding_type = "mrope"
            args.multimodal_input_key_map = {"image_grid_thw": "grid_thws"}
            args.uses_unsplit_forward = True

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> Any:
        language = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        from transformer_engine.pytorch import RMSNorm as TERMSNorm

        from .norm import RMSNorm

        for module in language.modules():
            if isinstance(module, TERMSNorm):
                module.__class__ = RMSNorm
        if self.position_embedding_type != "mrope":
            return language
        from .model import KeyeVLModel

        model = KeyeVLModel(self, language)
        model.freeze(self.freeze_language_model, self.freeze_vision_model, self.freeze_vision_projection)
        return model

    def finalize(self) -> None:
        self.transformer_layer_spec = keye_layer_spec
        self.moe_router_fusion = False
        self.bias_activation_fusion = False
        super().finalize()
        self.experimental_attention_variant_loss_scale_func = DSAIndexerLossAutoScaler.set_loss_scale

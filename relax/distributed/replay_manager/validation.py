# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lightweight indexer replay capabilities; no model/backend imports at
validation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndexerReplayCapabilities:
    cp_partition_mode: str
    allgather_cp: bool
    rollout_dp_attention: bool
    pipeline_parallel_size: int | None = None
    disable_overlap_schedule: bool = False
    disable_radix_cache: bool = False
    topk_path: tuple[str, ...] = ("dsa_indexer_topk",)
    capture_ratio: int | None = 4
    hf_topk_path: tuple[str, ...] = ("index_topk",)
    qsa: bool = False

    def capture_layout(self, config: Any) -> tuple[int, tuple[int, ...]]:
        """Validate the actual training model, beyond the launch arguments."""
        if self.qsa:
            ratio = getattr(config, "qsa_indexer_compress_ratio", None)
            budget = getattr(config, "qsa_indexer_budget", None)
            pattern = getattr(config, "linear_attention_freq", None)
            if ratio != 4 or not budget or budget % ratio or getattr(config, "qsa_indexer_loss_coeff", 0):
                raise ValueError("QSA replay requires an R4 token budget and zero indexer loss")
            if not isinstance(pattern, (list, tuple)) or len(pattern) < config.num_layers:
                raise ValueError("QSA replay requires the explicit GDN/QSA layer map")
            layers = tuple(i for i, linear in enumerate(pattern[: config.num_layers]) if not linear)
            if not layers:
                raise ValueError("QSA capture layer map is empty")
            return budget // ratio, layers
        topk = config
        for field in self.topk_path:
            topk = topk.get(field) if isinstance(topk, dict) else getattr(topk, field, None)
        if topk is None or int(topk) <= 0:
            raise ValueError("Indexer capture requires a positive top-k")
        if self.capture_ratio is None:
            if getattr(config, "keye_indexer_loss_coeff", 0):
                raise ValueError("Keye replay requires zero indexer loss")
            return int(topk), tuple(range(config.num_layers))
        ratios = getattr(config, "csa_compress_ratios", None)
        if ratios is None or topk is None or getattr(config, "dsa_indexer_loss_coeff", 0):
            raise RuntimeError("Indexer replay requires a configured sparse-layer map and zero indexer loss")
        if len(ratios) < config.num_layers:
            raise ValueError("Indexer capture layer map is shorter than the model")
        layers = tuple(i for i, ratio in enumerate(ratios[: config.num_layers]) if ratio == self.capture_ratio)
        if not layers:
            raise ValueError("Indexer capture layer map is empty")
        return int(topk), layers


_CAPABILITIES = {
    "deepseek_v4": IndexerReplayCapabilities(
        cp_partition_mode="contiguous", allgather_cp=True, rollout_dp_attention=True
    ),
    "KeyeVL2": IndexerReplayCapabilities(
        cp_partition_mode="zigzag",
        allgather_cp=False,
        rollout_dp_attention=False,
        pipeline_parallel_size=1,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        topk_path=("hf_config", "sa_config", "topk"),
        capture_ratio=None,
        hf_topk_path=("sa_config", "topk"),
    ),
    "qwen4_exp": IndexerReplayCapabilities(
        cp_partition_mode="zigzag", allgather_cp=False, rollout_dp_attention=False, qsa=True
    ),
    "qwen4_exp_text": IndexerReplayCapabilities(
        cp_partition_mode="zigzag", allgather_cp=False, rollout_dp_attention=False, qsa=True
    ),
}


def indexer_capabilities(model_type: str | None) -> IndexerReplayCapabilities:
    try:
        return _CAPABILITIES[model_type]
    except KeyError:
        raise ValueError(f"Indexer replay is unsupported for model_type={model_type!r}") from None


def indexer_topk(hf_config: Any) -> int:
    """Read the capture payload width using the same model capability table."""
    config = getattr(hf_config, "text_config", hf_config)
    capabilities = indexer_capabilities(getattr(config, "model_type", None))
    if capabilities.qsa:
        budget = getattr(config, "indexer_budget", 0)
        ratio = getattr(config, "indexer_compress_ratio", None)
        if ratio != 4 or budget <= 0 or budget % ratio:
            raise ValueError("QSA replay requires a positive R4 token budget")
        return budget // ratio
    value = config
    for field in capabilities.hf_topk_path:
        value = value.get(field) if isinstance(value, dict) else getattr(value, field, None)
    if value is None or int(value) <= 0:
        raise ValueError("Indexer replay requires a positive capture top-k")
    return int(value)


def indexer_capture_layout(config: Any) -> tuple[int, tuple[int, ...]]:
    hf_config = getattr(config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None and getattr(config, "qsa_indexer_n_heads", None):
        model_type = "qwen4_exp_text"
    # Older DSV4 Megatron configs do not retain the HF config. Require their
    # explicit sparse-layout fields, never treat an unknown HF model as DSV4.
    if model_type is None and hasattr(config, "csa_compress_ratios") and hasattr(config, "dsa_indexer_topk"):
        model_type = "deepseek_v4"
    return indexer_capabilities(model_type).capture_layout(config)


def validate_indexer_replay_args(args: Any, *, is_offline: bool) -> None:
    from relax.utils.misc import get_hf_config

    model_type = getattr(get_hf_config(args.hf_checkpoint), "model_type", None)
    capabilities = indexer_capabilities(model_type)
    unsupported = []
    if is_offline:
        unsupported.append(f"offline training ({args.loss_type})")
    if args.train_backend != "megatron":
        unsupported.append(f"train_backend={args.train_backend}")
    if getattr(args, "multimodal_keys", None) is not None:
        unsupported.append("multimodal model")
    if not args.colocate:
        unsupported.append("non-colocated deployment")
    if args.partial_rollout:
        unsupported.append("partial rollout")
    if args.fully_async or args.hybrid:
        unsupported.append("fully-async/hybrid")
    if args.use_slime_router:
        unsupported.append("slime router")
    if args.use_agentic_rollout:
        unsupported.append("agentic rollout")
    if args.rollout_function_path != "relax.engine.rollout.sglang_rollout.generate_rollout":
        unsupported.append(f"custom rollout_function_path={args.rollout_function_path}")
    if args.custom_generate_function_path is not None:
        unsupported.append(f"custom generate_function_path={args.custom_generate_function_path}")
    if args.qkv_format != "thd":
        unsupported.append(f"qkv_format={args.qkv_format}")
    if capabilities.qsa and args.context_parallel_size != 1:
        unsupported.append("QSA replay requires context_parallel_size=1")
    if not capabilities.qsa and args.context_parallel_size <= 1:
        unsupported.append(f"context_parallel_size={args.context_parallel_size}")
    if not capabilities.qsa and (args.tensor_model_parallel_size != 1 or args.sequence_parallel):
        unsupported.append("TP>1/sequence parallel")
    if capabilities.qsa:
        from relax.utils.env import Envs

        from .config import references_enabled

        if not references_enabled():
            unsupported.append("QSA replay requires reference delivery (RELAX_REPLAY_REFERENCES=1)")
        if Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND != "mixed" or Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION:
            unsupported.append("QSA replay requires mixed attention with fused attention disabled")
        if not Envs.RELAX_QWEN38_QSA_TP_SELECTOR or Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
            unsupported.append("QSA replay requires TP selector with fused selector disabled")
        if getattr(args, "qsa_indexer_loss_coeff", 0) not in (None, 0, 0.0):
            unsupported.append("trainable QSA indexer")
    if args.dynamic_context_parallel:
        unsupported.append("dynamic CP")
    if args.enable_mtp_training or args.sglang_speculative_algorithm:
        unsupported.append("MTP/speculative decoding")
    if getattr(args, "sglang_enable_hierarchical_cache", False):
        unsupported.append("SGLang hierarchical cache")
    if args.dsa_indexer_loss_coeff not in (None, 0, 0.0):
        unsupported.append(f"dsa_indexer_loss_coeff={args.dsa_indexer_loss_coeff}")
    if args.recompute_granularity == "full" and args.recompute_method != "uniform":
        unsupported.append(f"full recompute_method={args.recompute_method}")
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        unsupported.append("combined-1f1b")
    if args.cp_partition_mode != capabilities.cp_partition_mode or args.allgather_cp != capabilities.allgather_cp:
        unsupported.append(
            f"requires cp_partition_mode={capabilities.cp_partition_mode}, allgather_cp={capabilities.allgather_cp}"
        )
    if args.sglang_enable_dp_attention != capabilities.rollout_dp_attention:
        unsupported.append(f"requires sglang_enable_dp_attention={capabilities.rollout_dp_attention}")
    if (
        capabilities.pipeline_parallel_size is not None
        and args.pipeline_model_parallel_size != capabilities.pipeline_parallel_size
    ):
        unsupported.append(f"requires pipeline_model_parallel_size={capabilities.pipeline_parallel_size}")
    for field in ("disable_overlap_schedule", "disable_radix_cache"):
        if getattr(capabilities, field) and not getattr(args, f"sglang_{field}", False):
            unsupported.append(f"requires sglang_{field}=True")
    if unsupported:
        raise ValueError(
            f"--use-rollout-indexer-replay ({model_type}): unsupported settings: {', '.join(unsupported)}"
        )


def validate_mm_embedding_replay_args(args: Any, *, is_offline: bool) -> None:
    """Validate the frozen Qwen3.8 CP1 reference consumer before launching
    workers."""
    from relax.utils.misc import get_hf_config

    config = get_hf_config(args.hf_checkpoint)
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", config)
    unsupported = []
    if getattr(config, "model_type", None) != "qwen4_exp" or vision is None:
        unsupported.append("requires the multimodal Qwen3.8 checkpoint")
    if vision is not None and (
        getattr(vision, "out_hidden_size", None) != 2560
        or getattr(vision, "spatial_merge_size", None) != 2
        or getattr(vision, "deepstack_visual_indexes", None)
        or getattr(text, "hidden_size", None) != 2560
    ):
        unsupported.append("requires 2560-wide R4 visual embeddings without DeepStack")
    if is_offline or getattr(args, "train_backend", None) != "megatron" or not getattr(args, "colocate", False):
        unsupported.append("requires online colocated Megatron training")
    if getattr(args, "context_parallel_size", 1) != 1 or getattr(args, "dynamic_context_parallel", False):
        unsupported.append("requires CP1 without dynamic CP")
    if not getattr(args, "freeze_vision_model", False) or not getattr(args, "freeze_vision_projection", False):
        unsupported.append("requires frozen vision model and projection/merger")
    if not getattr(args, "bf16", True):
        unsupported.append("requires BF16 visual embeddings")
    if getattr(args, "multimodal_keys", None) is None:
        unsupported.append("requires multimodal input mapping")
    from relax.algorithms import algorithm_needs_critic

    if algorithm_needs_critic(args):
        unsupported.append("critic-based algorithms do not yet consume MM replay")
    for field in (
        "fully_async",
        "hybrid",
        "partial_rollout",
        "use_agentic_rollout",
        "enable_mtp_training",
        "sglang_speculative_algorithm",
        "sglang_enable_hierarchical_cache",
        "use_slime_router",
        "use_opd",
    ):
        if getattr(args, field, False):
            unsupported.append(field)
    if (
        getattr(args, "rollout_function_path", "relax.engine.rollout.sglang_rollout.generate_rollout")
        != ("relax.engine.rollout.sglang_rollout.generate_rollout")
        or getattr(args, "custom_generate_function_path", None) is not None
    ):
        unsupported.append("custom rollout/generation function")
    if getattr(args, "qkv_format", "thd") != "thd":
        unsupported.append("requires packed THD")
    if unsupported:
        raise ValueError("--use-rollout-mm-embedding-replay: " + ", ".join(unsupported))

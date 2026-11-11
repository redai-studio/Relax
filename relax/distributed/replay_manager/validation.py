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

    def capture_layout(self, config: Any) -> tuple[int, tuple[int, ...]]:
        """Validate the actual training model, beyond the launch arguments."""
        topk = config
        for field in self.topk_path:
            topk = topk.get(field) if isinstance(topk, dict) else getattr(topk, field, None)
        if topk is None or int(topk) <= 0:
            raise ValueError("Indexer capture requires a positive top-k")
        if self.capture_ratio is None:
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
    value = config
    for field in capabilities.hf_topk_path:
        value = value.get(field) if isinstance(value, dict) else getattr(value, field, None)
    if value is None or int(value) <= 0:
        raise ValueError("Indexer replay requires a positive capture top-k")
    return int(value)


def indexer_capture_layout(config: Any) -> tuple[int, tuple[int, ...]]:
    hf_config = getattr(config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
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
    if args.context_parallel_size <= 1:
        unsupported.append(f"context_parallel_size={args.context_parallel_size}")
    if args.tensor_model_parallel_size != 1 or args.sequence_parallel:
        unsupported.append("TP>1/sequence parallel")
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

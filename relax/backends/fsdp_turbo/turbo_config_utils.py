# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Parse ``--turbo-config`` and validate the fsdp_turbo backend.

Model, optimizer and parallel-plan settings come from this config; Relax
CLI optimizer flags do not override them.

The raw YAML path or inline JSON stays serializable in ``args``; the heavy
``fsdp_turbo`` dependency is imported lazily in the train actor.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig


logger = get_logger(__name__)

_YAML_SUFFIXES = (".yaml", ".yml")


def load_turbo_config(source: str | None) -> "FSDPTurboConfig":
    """Load a YAML path or inline JSON object into ``FSDPTurboConfig``."""
    if not source:
        raise ValueError(
            "--train-backend fsdp_turbo requires --turbo-config pointing to a FSDPTurbo YAML file, "
            "e.g. configs/turbo_config/qwen3_8_flash_next.yaml."
        )

    from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass, load_config_from_yaml

    is_path = source.endswith(_YAML_SUFFIXES) or os.path.exists(source)
    if is_path:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"--turbo-config path not found: {source}")
        import yaml

        with open(source, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        config = load_config_from_yaml(source)
        logger.info(f"Loaded FSDPTurbo config from YAML: {source}")
    else:
        try:
            payload: Any = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--turbo-config is neither a YAML path nor valid inline JSON: {source[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise ValueError("--turbo-config inline JSON must be an object at the root.")
        config = _dict_to_dataclass(FSDPTurboConfig, payload)
        logger.info("Loaded FSDPTurbo config from inline JSON.")

    model_payload = payload.get("model") or {}
    optimizer_payload = payload.get("optimizer") or {}
    if "warmup_steps" in optimizer_payload:
        warmup_steps = optimizer_payload["warmup_steps"]
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, (int, float)) or warmup_steps < 0:
            raise ValueError("optimizer.warmup_steps must be a nonnegative number")
        config.optimizer.warmup_steps = warmup_steps
    from .performance_config import PerformanceConfig

    config.relax_performance = PerformanceConfig.from_dict(payload.get("relax_performance"))
    if "ple_layer_ids" in model_payload:
        # FSDPTurbo's ModelConfig drops unknown keys; preserve this Relax-side
        # qwen4_exp override so the actor can apply it before model construction.
        config.model.ple_layer_ids = model_payload["ple_layer_ids"]
    for _flag in ("mtp_detach_embedding", "mtp_detach_backbone", "mtp_detach_lm_head"):
        # The inline-JSON path drops unknown ModelConfig keys, unlike the YAML
        # loader; keep the MTP detach switches settable on both paths.
        if _flag in model_payload:
            setattr(config.model, _flag, bool(model_payload[_flag]))
    return config


def get_turbo_dp_size(args) -> int:
    """Return the input-data group size (DP*FSDP/CP) without importing
    ``fsdp_turbo``.

    Megatron tp/pp/cp args do not describe this backend. The controller must
    use the actor's DP size for TQ samplers, or unfetched logical slots can
    deadlock the fetch loop.
    """
    source = getattr(args, "turbo_config", None)
    if not source:
        return 1
    is_path = source.endswith(_YAML_SUFFIXES) or os.path.exists(source)
    if is_path:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"--turbo-config path not found: {source}")
        import yaml

        with open(source, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
    else:
        payload = json.loads(source)
    distributed = payload.get("distributed") or {}
    if int(distributed.get("kda_p2p_parallel_size") or 1) > 1:
        raise ValueError("fsdp_turbo does not support KDA P2P; keep distributed.kda_p2p_parallel_size=1.")
    replicas = int(distributed.get("data_parallel_size") or 1)
    shards = int(distributed.get("fully_shard_parallel_size") or 1)
    context = int(distributed.get("ulysses_parallel_size") or 1)
    tensor = int(distributed.get("tensor_parallel_size") or 1)
    resource = getattr(args, "resource", None) or {}
    world_size = int(resource["actor"][1]) if "actor" in resource else replicas * shards * tensor
    if (
        min(replicas, shards, context, tensor) < 1
        or world_size <= 0
        or world_size % (shards * tensor)
        or world_size % (context * tensor)
    ):
        raise ValueError("Turbo actor world size must be divisible by FSDP*TP and CP*TP")
    return world_size // (context * tensor)


def get_turbo_sglang_model_override(source: str, model_path: str, existing: str | None = None) -> str:
    """Keep rollout architecture aligned with the training plan in Turbo
    YAML."""
    from transformers import PretrainedConfig

    config = load_turbo_config(source)
    # Read the raw layer types: Transformers normalizes Qwen4's full_attention
    # to indexed_attention, while SGLang's decoder registry uses the former.
    hf_config, _ = PretrainedConfig.get_config_dict(model_path)
    config_key = next((key for key in ("text_config", "language_config") if hf_config.get(key) is not None), None)
    text_config = hf_config[config_key] if config_key else hf_config
    num_layers = config.model.num_hidden_layers
    if num_layers is None:
        num_layers = text_config["num_hidden_layers"]
    if num_layers <= 0 or num_layers > text_config["num_hidden_layers"]:
        raise ValueError(
            f"model.num_hidden_layers={num_layers} must be between 1 and "
            f"the checkpoint layer count ({text_config['num_hidden_layers']})."
        )

    overrides = json.loads(existing or "{}")
    text_overrides = overrides.setdefault(config_key, {}) if config_key else overrides
    text_overrides["num_hidden_layers"] = num_layers
    layer_types = text_config.get("layer_types")
    if layer_types is not None:
        text_overrides["layer_types"] = list(layer_types[:num_layers])

    if hf_config.get("model_type") == "qwen4_exp":
        ple_layer_ids = getattr(config.model, "ple_layer_ids", None)
        text_overrides["ple_layer_ids"] = list(
            ple_layer_ids if ple_layer_ids is not None else text_config["ple_layer_ids"]
        )
        conv_states = text_config.get("number_of_conv_states")
        if ple_layer_ids is not None and not ple_layer_ids:
            conv_states = 1
        if conv_states is not None:
            text_overrides["number_of_conv_states"] = conv_states
        text_overrides["mtp_num_layers"] = int(getattr(config.model, "mtp_num_layers", 0) or 0)

    return json.dumps(overrides)


def validate_turbo_config(config: "FSDPTurboConfig", allow_cp: bool = False, native_cp: bool = False) -> None:
    """Reject unsupported FSDPTurbo plans.

    Ulysses CP is SFT-only: Relax feeds whole batches and the model forward
    must shard internally, either natively (``native_cp``) or via
    ``module_patches``. Qwen4 SFT chunked projection also supports Ulysses,
    with TP=1 and MTP disabled. GRPO, KCP and KV-allgather CP remain
    unsupported.
    """
    distributed = config.distributed
    if int(getattr(distributed, "kda_p2p_parallel_size", 1) or 1) > 1:
        raise ValueError("fsdp_turbo does not support KDA P2P; keep distributed.kda_p2p_parallel_size=1.")
    if getattr(config.memory, "chunk_loss", False):
        if not allow_cp or not native_cp:
            raise ValueError("Relax chunk loss currently requires Qwen4 Exp SFT.")
        if any(
            int(getattr(distributed, field, 1) or 1) != 1
            for field in (
                "tensor_parallel_size",
                "kcp_parallel_size",
                "kv_allgather_parallel_size",
            )
        ):
            raise ValueError("Relax chunk loss requires TP=1 and supports only Ulysses CP.")
        if int(getattr(config.model, "mtp_num_layers", 0) or 0):
            raise ValueError("Relax chunk loss requires MTP disabled.")
    if not allow_cp:
        for field in ("data_parallel_size", "tensor_parallel_size"):
            if int(getattr(distributed, field, 1) or 1) != 1:
                raise ValueError(f"fsdp_turbo RL currently requires distributed.{field}=1.")
    if getattr(distributed, "kcp_parallel_size", 1) > 1:
        raise ValueError("fsdp_turbo backend does not support KCP in Relax; keep distributed.kcp_parallel_size=1.")
    # TODO: Support combined CP after aligning data slots, targets/masks, loss
    # reductions and evaluation deduplication with the complete CP group.
    if getattr(distributed, "kv_allgather_parallel_size", 1) > 1:
        raise ValueError("fsdp_turbo backend does not yet support KV-allgather CP; keep it at 1.")
    ulysses_size = int(getattr(distributed, "ulysses_parallel_size", 1) or 1)
    if ulysses_size > 1:
        if not allow_cp:
            raise ValueError(
                "Ulysses context parallel is supported for SFT only (CP-aware loss aggregation in "
                "backends/fsdp_turbo/model_utils.py); the GRPO logprob/advantage path is not "
                "CP-aware. Set distributed.ulysses_parallel_size=1 for RL runs."
            )
        if not config.module_patches and not native_cp:
            raise ValueError(
                "Ulysses context parallel requires module_patches that shard the fused sequence "
                "across the ulysses group inside the model forward, or a model built from "
                "fsdp_turbo's consolidated modeling (qwen4_exp modeling_qwen4_exp) whose forward "
                "shards natively — Relax feeds whole samples and never splits batches externally."
            )


def validate_fsdp_turbo_args(args) -> None:
    """Validate Relax args without importing ``fsdp_turbo`` during parsing.

    Only colocated, critic-free/KL-free GRPO and SFT are supported. Actor
    offload is a no-op because sharded weights stay resident; static memory
    must be budgeted with ``--sglang-mem-fraction-static``.
    """
    from relax.algorithms import algorithm_needs_critic

    is_sft = getattr(args, "loss_type", None) == "sft"

    problems: list[str] = []
    save_interval = args.save_interval
    if isinstance(save_interval, (int, float)) and save_interval <= 0:
        problems.append("--save-interval must be positive")
    max_keep = args.max_actor_ckpt_to_keep
    if max_keep is not None and max_keep < 0:
        problems.append("--max-actor-ckpt-to-keep must be non-negative")
    if is_sft and args.task_type != "causal_lm":
        problems.append("fsdp_turbo SFT only supports --task-type causal_lm")
    if getattr(args, "loss_type", None) not in ("sft", "policy_loss"):
        problems.append("only --loss-type sft and policy_loss are supported")
    if not getattr(args, "colocate", False):
        problems.append("--colocate is required (fsdp_turbo is colocate-only)")
    if getattr(args, "fully_async", False):
        problems.append("--fully-async is not supported")
    if getattr(args, "hybrid", False):
        problems.append("--hybrid is not supported")
    if getattr(args, "sft_predict_interval", None) is not None:
        problems.append("--sft-predict-interval is not supported (SFT on fsdp_turbo runs no rollout service)")
    if getattr(args, "kl_coef", 0.0) != 0:
        problems.append("--kl-coef must be 0 (reference model is not supported)")
    if getattr(args, "use_kl_loss", False):
        problems.append("--use-kl-loss is not supported")
    if not is_sft and algorithm_needs_critic(args):
        problems.append("critic-based advantage estimators are not supported; use a GRPO-family estimator")
    if not is_sft:
        if args.rollout_external:
            problems.append("fsdp_turbo RL requires local policy engines; --rollout-external is not supported")
        if int(getattr(args, "sglang_pp_size", 1) or 1) != 1:
            problems.append("fsdp_turbo weight synchronization requires --sglang-pp-size 1")
        if int(getattr(args, "sglang_dp_size", 1) or 1) != 1:
            problems.append("fsdp_turbo weight synchronization requires --sglang-dp-size 1")
        if int(getattr(args, "sglang_attn_dp_size", 1) or 1) != 1:
            problems.append("fsdp_turbo weight synchronization requires --sglang-attn-dp-size 1")
        rows = int(getattr(args, "rollout_batch_size", 0) or 0) * int(getattr(args, "n_samples_per_prompt", 1) or 1)
        if rows != int(args.global_batch_size or rows):
            problems.append(
                "fsdp_turbo requires rollout_batch_size * n_samples_per_prompt == global_batch_size "
                "(one optimizer batch per rollout)"
            )
        if getattr(args, "advantage_estimator", "grpo") != "grpo":
            problems.append("only --advantage-estimator grpo is implemented for fsdp_turbo RL")
        for name in ("normalize_advantages", "use_tis", "use_opd", "use_opsm", "custom_pg_loss_reducer_function_path"):
            if getattr(args, name, False):
                problems.append(f"--{name.replace('_', '-')} is not implemented for fsdp_turbo RL")
        if getattr(args, "entropy_coef", 0.0) != 0:
            problems.append("--entropy-coef must be 0 (entropy regularization is not implemented)")
        if float(getattr(args, "rollout_temperature", 1.0)) <= 0:
            problems.append("--rollout-temperature must be positive for RL log-probabilities")
    if not getattr(args, "turbo_config", None):
        problems.append("--turbo-config is required (path to a FSDPTurbo YAML, see configs/turbo_config/)")

    if problems:
        raise ValueError("Invalid fsdp_turbo configuration:\n  - " + "\n  - ".join(problems))

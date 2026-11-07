# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Parse ``--turbo-config`` and validate the fsdp_turbo backend.

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
            "e.g. configs/turbo_config/qwen3_5_moe_35b.yaml."
        )

    from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass, load_config_from_yaml

    is_path = source.endswith(_YAML_SUFFIXES) or os.path.exists(source)
    if is_path:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"--turbo-config path not found: {source}")
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
    return config


def get_turbo_dp_size(args) -> int:
    """Return the plan's DP size without importing ``fsdp_turbo``.

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
    return max(int(distributed.get("data_parallel_size") or 1), 1)


def validate_turbo_config(config: "FSDPTurboConfig", allow_cp: bool = False, native_cp: bool = False) -> None:
    """Reject unsupported FSDPTurbo plans.

    Ulysses CP is SFT-only: Relax feeds whole batches and the model forward
    must shard internally, either natively (``native_cp``) or via
    ``module_patches``. GRPO, KCP and KV-allgather CP remain unsupported.
    """
    distributed = config.distributed
    if getattr(distributed, "kcp_parallel_size", 1) > 1:
        raise ValueError("fsdp_turbo backend does not support KCP in Relax; keep distributed.kcp_parallel_size=1.")
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
    if not getattr(args, "turbo_config", None):
        problems.append("--turbo-config is required (path to a FSDPTurbo YAML, see configs/turbo_config/)")

    if problems:
        raise ValueError("Invalid fsdp_turbo configuration:\n  - " + "\n  - ".join(problems))

# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


STATIC_UPDATE_ENDPOINTS = frozenset(
    {
        "update_weights_from_tensor",
        "update_weights_from_distributed",
        "init_weights_update_group",
        "destroy_weights_update_group",
        "init_weights_send_group_for_remote_instance",
        "send_weights_to_remote_instance",
        "load_lora_adapter_from_tensors",
        "update_lora_from_distributed",
        "unload_lora_adapter",
        "post_process_weights",
        "register_dcs",
        "run_scale_weight_sync_precheck",
    }
)
_POLICY_SERVER_FIELDS = frozenset(
    {
        "model_path",
        "tokenizer_path",
        "served_model_name",
        "load_format",
        "enable_lora",
        "max_lora_rank",
        "max_loras_per_batch",
        "max_loaded_loras",
        "enable_weights_cpu_backup",
        "enable_draft_weights_cpu_backup",
        "quantization",
        "quantization_param_path",
        "enable_return_routed_experts",
    }
)


@dataclass(frozen=True)
class InferenceEngineSpec:
    role: str
    weight_source: str = "STATIC"
    strict_drain: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"genrm", "teacher"}:
            raise ValueError("Explicit static Engine spec requires GenRM or Teacher role")
        if self.weight_source != "STATIC":
            raise ValueError("GenRM and Teacher only support STATIC engine weights")
        if not isinstance(self.strict_drain, bool):
            raise ValueError("strict_drain must be boolean")

    def require_operation(self, operation: str) -> None:
        if operation in STATIC_UPDATE_ENDPOINTS:
            raise RuntimeError(f"STATIC {self.role} engine rejects dynamic operation {operation!r}")


def build_static_engine_config(
    args: Any, role: str, overrides: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    InferenceEngineSpec(role)
    if role == "genrm":
        effective = dict(getattr(args, "genrm_engine_config", None) or {})
        checkpoint = effective.get("model_path", args.genrm_model_path)
    else:
        effective = dict(overrides or {})
        checkpoint = effective.get("model_path", args.teacher_hf_checkpoint)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{role} requires its own checkpoint")
    effective.setdefault("load_format", "auto")
    if effective["load_format"] == "dummy":
        raise ValueError(f"STATIC {role} must load real checkpoint weights")
    if effective.get("enable_weights_cpu_backup", True) is not True:
        raise ValueError(f"STATIC {role} requires enable_weights_cpu_backup for offload recovery")
    effective["enable_weights_cpu_backup"] = True
    effective["model_path"] = checkpoint
    if effective.get("enable_lora") or any(key.startswith("lora_") and value for key, value in effective.items()):
        raise ValueError(f"STATIC {role} cannot enable dynamic LoRA adapters")
    effective["enable_lora"] = False
    cloned = copy.copy(args)
    for key in list(vars(cloned)):
        if not key.startswith("sglang_"):
            continue
        name = key[len("sglang_") :]
        if name in _POLICY_SERVER_FIELDS or name.startswith(("lora_", "speculative_")):
            delattr(cloned, key)
    cloned.hf_checkpoint = checkpoint
    cloned.sglang_hf_checkpoint = checkpoint
    cloned.model_source = None
    cloned.rollout_external = False
    cloned.enable_mtp_training = False
    cloned.lora_rank = 0
    cloned.lora_adapter_mode = False
    cloned.use_rollout_routing_replay = False
    cloned.optimize_routing_replay = False
    cloned.sglang_router_ip = None
    cloned.sglang_router_port = None
    if role == "teacher":
        effective.setdefault("pp_size", 1)
        effective.setdefault("dp_size", 1)
        effective.setdefault("ep_size", 1)
        effective.setdefault("moe_dense_tp_size", None)
    for key, value in effective.items():
        setattr(cloned, f"sglang_{key}", value)
    if role == "genrm":
        cloned.genrm_model_path = checkpoint
        cloned.genrm_engine_config = dict(effective)
    return cloned, effective


def validate_static_server_args(spec: InferenceEngineSpec, server_args: dict[str, Any]) -> None:
    if server_args.get("load_format") == "dummy":
        raise ValueError(f"STATIC {spec.role} cannot start with dummy weights")
    if server_args.get("enable_weights_cpu_backup") is not True:
        raise ValueError(f"Backend lacks CPU weight backup required by STATIC {spec.role}")
    if server_args.get("enable_lora"):
        raise ValueError(f"STATIC {spec.role} cannot start with dynamic LoRA adapters")


def drain_static_engine(
    *,
    pause: Callable[[float], Any],
    abort: Callable[[float], Any],
    flush: Callable[[float], bool],
    release: Callable[[float], Any],
    strict: bool,
    timeout: float,
    release_timeout: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    deadline = clock() + timeout

    def remaining() -> float:
        value = deadline - clock()
        if value <= 0:
            raise TimeoutError("Timeout while draining static inference before release")
        return value

    try:
        pause(remaining())
    except Exception:
        if strict:
            raise
    connection_errors = 0
    while True:
        remaining()
        try:
            abort(remaining())
        except Exception:
            if strict:
                raise
        try:
            if flush(remaining()):
                break
            connection_errors = 0
        except ConnectionError:
            connection_errors += 1
            if connection_errors >= 3:
                raise
        sleep(min(1.0, remaining()))
    return release(min(release_timeout, remaining()))

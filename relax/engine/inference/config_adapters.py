# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapt GenRM and Teacher arguments to unified inference model
configurations."""

import copy
from typing import Any

from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.engine.inference.config import EngineGroupSpec, InferenceModelSpec
from relax.engine.inference.phase_plans import PHASE_GENRM, PHASE_TEACHER, placement_phase
from relax.engine.inference.types import WeightSource
from relax.utils.env import Envs


# Each instance probes ports from its own window, so instances starting side
# by side never race for the same free port (probe-then-bind).
_GENRM_PORT_BASE = 16000
_GENRM_PORT_WINDOW_SIZE = 1000

# Teachers probe ports from their own windows, apart from rollout and GenRM.
_TEACHER_PORT_BASE = 26000
_TEACHER_PORT_WINDOW_SIZE = 500


def genrm_engine_env(args: Any) -> dict[str, str]:
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        # See rollout.py: recent SGLang reads SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
        # (default True) and the deprecation shim value-copies SGL_DISABLE_* into it,
        # so the old DISABLE vars re-enable the check. Set ENABLE=false directly.
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        # NOTE: disable custom all-reduce-v2, same as rollout.py — avoids
        # custom_all_reduce.cuh:37: CUDA error: invalid argument during CUDA graph capture.
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
    }
    if getattr(args, "fp16", False):
        env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    return env_vars


def genrm_role_models(args: Any, pg: Any) -> list[tuple[InferenceModelSpec, Any, dict[str, Any]]]:
    """Describe every GenRM instance for ``InferenceManager.create_role``.

    Under sync colocate the instances sit behind the rollout region of the
    shared placement group, one after another; a GenRM that shares the rollout
    bundles starts at bundle 0 and is only legal when it defers.
    """
    region_offset = (
        0 if args.fully_async or getattr(args, "_genrm_colocate_with_rollout", False) else args.rollout_num_gpus
    )
    models = []
    bundle_offset = 0
    for index, (key, spec) in enumerate(args._genrm_instances_resolved.items()):
        engine_args = copy.copy(args)
        engine_args.genrm_model_path = spec["model_path"]
        engine_args.genrm_num_gpus_per_engine = spec["num_gpus_per_engine"]
        engine_args.genrm_engine_config = spec["engine_config"]
        sampling_config = spec["sampling_config"] or {}
        config = InferenceModelSpec(
            key,
            spec["model_path"],
            engine_groups=[
                EngineGroupSpec(
                    "regular", spec["num_gpus"], spec["num_gpus_per_engine"], dict(spec["engine_config"] or {})
                )
            ],
            weight_source=WeightSource.STATIC,
            # The judge's request defaults live on its model, so the GenRM
            # service reads them from the manager instead of global arguments.
            sampling_defaults={
                "temperature": sampling_config.get("temperature", 0.2),
                "top_p": sampling_config.get("top_p", 1.0),
                "top_k": sampling_config.get("top_k", -1),
                "max_new_tokens": sampling_config.get("max_response_len", 1024),
            },
            chat_template_kwargs=dict(sampling_config.get("chat_template_kwargs") or {}),
            env_vars=genrm_engine_env(args),
            fault_tolerance_enabled=True,
        ).resolved(engine_args)
        placement = {
            "pg": pg,
            "bundle_offset": region_offset + bundle_offset,
            "phase": placement_phase(args, PHASE_GENRM),
            "base_port": _GENRM_PORT_BASE + index * _GENRM_PORT_WINDOW_SIZE,
            "ray_num_gpus": getattr(args, "genrm_ray_num_gpus", 0.2),
        }
        models.append((config, engine_args, placement))
        bundle_offset += spec["num_gpus"]
    return models


def _build_teacher_engine_env(args: Any) -> dict[str, str]:
    env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
        # OPD patches default off; enabled only when the corresponding env flag is
        # passed through from the driver. RELAX_OPD_PREEXPANDED_PATCH affects the
        # teacher engine only; RELAX_OPD_PER_POS_TOKEN_IDS affects teacher + student.
        "RELAX_OPD_PREEXPANDED_PATCH": str(int(Envs.RELAX_OPD_PREEXPANDED_PATCH)),
        "RELAX_OPD_PER_POS_TOKEN_IDS": str(int(Envs.RELAX_OPD_PER_POS_TOKEN_IDS)),
        "RELAX_OPD_TOKEN_IDS_LOGPROB_K": Envs.RELAX_OPD_TOKEN_IDS_LOGPROB_K,
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
    }
    if getattr(args, "fp16", False):
        env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    return env_vars


def teacher_role_model(
    args: Any,
    *,
    model_id: str,
    num_gpus: int,
    gpus_per_replica: int,
    pg: Any = None,
    bundle_offset: int = 0,
    index: int = 0,
) -> tuple[InferenceModelSpec, Any, dict[str, Any]]:
    """Describe one teacher for ``InferenceManager.create_role``.

    With ``pg`` the teacher sits in the shared actor placement group at
    ``bundle_offset`` within the teacher region; without it the manager creates
    a placement group for this teacher alone.
    """
    from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides, teacher_region_offset

    if gpus_per_replica > args.num_gpus_per_node and gpus_per_replica % args.num_gpus_per_node:
        raise ValueError("Multi-node teacher replicas must occupy complete nodes")
    overrides = build_teacher_overrides(args, colocate_sync=pg is not None)
    engine_args = build_teacher_engine_args(args, overrides)
    engine_args.use_slime_router = False
    config = InferenceModelSpec(
        model_id,
        overrides["model_path"],
        engine_groups=[EngineGroupSpec("regular", num_gpus, gpus_per_replica, dict(overrides))],
        weight_source=WeightSource.STATIC,
        env_vars=_build_teacher_engine_env(args),
        fault_tolerance_enabled=True,
    ).resolved(engine_args)
    placement = {
        "pg": pg,
        "bundle_offset": (teacher_region_offset(args) + bundle_offset) if pg is not None else 0,
        "phase": placement_phase(args, PHASE_TEACHER),
        "base_port": _TEACHER_PORT_BASE + index * _TEACHER_PORT_WINDOW_SIZE,
    }
    return config, engine_args, placement

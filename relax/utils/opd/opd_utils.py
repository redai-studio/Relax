# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import json
import os
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.utils.logging_utils import get_logger
from relax.utils.opd import opd_opsd_worker


if TYPE_CHECKING:
    from relax.utils.types import RolloutBatch, Sample


logger = get_logger(__name__)

OPD_TOKEN_SELECTIONS = ("student_sampled", "student_topk", "teacher_topk", "union")
OPD_KL_TYPES = ("reverse_kl", "forward_kl", "low_var_kl", "jsd")

OPD_ROLLOUT_LOG_SKIP_FIELDS = frozenset(
    {
        "opd_topk_token_ids",
        "opd_topk_student_log_probs",
        "opd_topk_teacher_log_probs",
        "opd_topk_ksz",
    }
)

OPD_CP_FLOAT_FIELDS = ("teacher_log_probs",)


def iter_opd_cp_float_fields() -> tuple[str, ...]:
    """Return OPD 1D per-token float field names that must be CP-sliced."""
    return OPD_CP_FLOAT_FIELDS


TEACHER_SGLANG_PREFIX = "teacher_sglang_"
_SGLANG_PASSTHROUGH_SKIP_ARGS = (
    "model_path",
    "config",
    "trust_remote_code",
    "random_seed",
    "enable_memory_saver",
    "tp_size",
    "port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "nccl_port",
    "skip_server_warmup",
    "enable_return_routed_experts",
)


def is_managed_opd_teacher_enabled(args: Any) -> bool:
    return (
        getattr(args, "use_opd", False)
        and getattr(args, "opd_type", None) == "sglang"
        and (
            getattr(args, "teacher_hf_checkpoint", None) is not None
            or getattr(args, "opd_teacher_routes", None) is not None
        )
        and getattr(args, "resource", None) is not None
        and "teacher" in args.resource
    )


def is_managed_opd_teacher_colocate(args: Any) -> bool:
    return (
        is_managed_opd_teacher_enabled(args)
        and getattr(args, "colocate", False)
        and not getattr(args, "hybrid", False)
        and "actor" in args.resource
        and "rollout" in args.resource
    )


def _mirror_teacher_sglang_server_args(parser: Any) -> None:
    import argparse

    from sglang.srt.server_args import ServerArgs

    old_add_argument = parser.add_argument

    def new_add_argument_wrapper(*name_or_flags: Any, **kwargs: Any) -> None:
        canonical_name = kwargs.get("dest")
        if not canonical_name:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    canonical_name = flag_name_candidate[2:].replace("-", "_")
                    break

        if canonical_name and canonical_name in _SGLANG_PASSTHROUGH_SKIP_ARGS:
            return

        final_name_or_flags = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                final_name_or_flags.append(f"--teacher-sglang-{item_flag.lstrip('-')}")
            else:
                final_name_or_flags.append(item_flag)

        final_kwargs = kwargs.copy()
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            if not original_dest.startswith(TEACHER_SGLANG_PREFIX):
                final_kwargs["dest"] = f"{TEACHER_SGLANG_PREFIX}{original_dest}"
        final_kwargs["default"] = argparse.SUPPRESS

        old_add_argument(*final_name_or_flags, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    try:
        ServerArgs.add_cli_args(parser)
    finally:
        parser.add_argument = old_add_argument


def teacher_sglang_parse_args() -> Namespace:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    _mirror_teacher_sglang_server_args(parser)
    args, _ = parser.parse_known_args()
    return args


def build_teacher_overrides(args: Any, colocate_sync: bool = False) -> dict[str, object]:
    overrides = {
        key[len(TEACHER_SGLANG_PREFIX) :]: value
        for key, value in vars(args).items()
        if key.startswith(TEACHER_SGLANG_PREFIX)
    }
    overrides["model_path"] = args.teacher_hf_checkpoint
    overrides.setdefault("load_format", "auto")
    overrides.setdefault("enable_memory_saver", colocate_sync)
    if overrides.get("enable_memory_saver"):
        # Release/resume without a CPU backup leaves weight pages uninitialized (uniform output).
        overrides.setdefault("enable_weights_cpu_backup", True)
    return overrides


def build_teacher_engine_args(args: Any, overrides: dict[str, object]) -> Any:
    teacher_args = copy.copy(args)
    for key, value in overrides.items():
        setattr(teacher_args, f"sglang_{key}", value)
    return teacher_args


def create_managed_opd_teacher_manager(
    args: Any,
    *,
    num_replicas: int,
    gpus_per_replica: int,
    pg: Any = None,
    shared_pg: bool = False,
    runtime_env: dict | None = None,
) -> tuple[Any, list[str]]:
    import ray

    from relax.distributed.ray.teacher_manager import TeacherManager

    teacher_manager = TeacherManager.options(
        **with_control_plane_affinity(
            args,
            {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env},
        )
    ).remote(
        args,
        num_replicas,
        gpus_per_replica,
        pg=pg,
        shared_pg=shared_pg,
    )

    urls = ray.get(teacher_manager.get_urls.remote())
    logger.info(f"[OPD teacher] TeacherManager initialized successfully: urls={urls}")

    if shared_pg and getattr(args, "offload_rollout", False):
        logger.info("[OPD teacher] Offloading teacher engines before actor init")
        ray.get(teacher_manager.offload.remote())

    return teacher_manager, urls


def maybe_start_managed_opd_teacher(args: Any, *, runtime_env: dict | None = None) -> tuple[Any, Any]:
    if not is_managed_opd_teacher_enabled(args) or getattr(args, "debug_train_only", False):
        return None, None

    # ── Multi-teacher (MOPD) path: --opd-teacher-routes ────────────────────
    routes_json = getattr(args, "opd_teacher_routes", None)
    if routes_json is not None:
        return _start_managed_multi_teacher(args, routes_json, runtime_env=runtime_env)

    # ── Single-teacher path: --teacher-hf-checkpoint ───────────────────────
    shared_pg = None
    shared_pg_enabled = is_managed_opd_teacher_colocate(args)
    if shared_pg_enabled:
        from relax.core.service import create_placement_group

        actor_gpus = args.resource["actor"][1]
        logger.info(
            f"[OPD teacher] pre-building shared PG: actor={actor_gpus}, "
            f"rollout={args.resource['rollout'][1]}, teacher={args.resource['teacher'][1]}"
        )
        shared_pg = create_placement_group(
            num_gpus=actor_gpus,
            node_group_affinity=getattr(args, "enable_affinity", True),
        )

    # args.resource["teacher"] = [num_cpus, num_gpus]. The teacher replica layout
    # is derived from the GPU total and --teacher-num-gpus-per-engine (TP per
    # engine). When the per-engine knob is unset, fall back to a single replica
    # that uses all teacher GPUs (backward-compatible with the previous default).
    _, teacher_total_gpus = args.resource["teacher"]
    gpus_per_replica = getattr(args, "teacher_num_gpus_per_engine", None) or teacher_total_gpus
    assert teacher_total_gpus % gpus_per_replica == 0, (
        f"teacher GPUs in --resource ({teacher_total_gpus}) must be divisible by "
        f"--teacher-num-gpus-per-engine ({gpus_per_replica})."
    )
    num_replicas = teacher_total_gpus // gpus_per_replica
    logger.info(
        f"[OPD teacher] managed teacher enabled: num_replicas={num_replicas}, "
        f"gpus_per_replica={gpus_per_replica}, teacher_total_gpus={teacher_total_gpus}, "
        f"shared_pg={shared_pg_enabled}"
    )
    teacher_manager, urls = create_managed_opd_teacher_manager(
        args,
        num_replicas=num_replicas,
        gpus_per_replica=gpus_per_replica,
        pg=shared_pg,
        shared_pg=shared_pg_enabled,
        runtime_env=runtime_env,
    )
    args.opd_teacher_url = urls[0]
    args.opd_teacher_urls = list(urls)
    logger.info(
        f"[OPD teacher] injected opd_teacher_url={args.opd_teacher_url} "
        f"opd_teacher_urls={args.opd_teacher_urls} ({len(urls)} replica(s))"
    )
    return shared_pg, teacher_manager


def _start_managed_multi_teacher(
    args: Any,
    routes_json: str,
    *,
    runtime_env: dict | None = None,
) -> tuple[Any, list]:
    """Launch multiple managed teachers for MOPD and inject routes into args.

    Each teacher (one per ``data_source``) gets an equal share of the 'teacher'
    resource GPUs; that share is further split into replicas of TP size
    ``--teacher-num-gpus-per-engine``. Every teacher's replica URLs are
    collected into a list and written to
    ``args.opd_teacher_routes_map[data_source]`` so the per-sample router
    (``_pick_teacher_url``) can route by data_source and then round-robin
    across that teacher's replicas.
    """
    import copy

    import ray

    routes_map: dict[str, str] = json.loads(routes_json)
    if not routes_map:
        raise ValueError("--opd-teacher-routes must be a non-empty JSON object.")

    num_teachers = len(routes_map)
    _, total_teacher_gpus = args.resource["teacher"]
    if total_teacher_gpus % num_teachers != 0:
        raise ValueError(
            f"Total teacher GPUs ({total_teacher_gpus}) must be evenly divisible by "
            f"number of teachers ({num_teachers})."
        )
    gpus_per_teacher = total_teacher_gpus // num_teachers

    # Per-teacher replica layout: TP size = --teacher-num-gpus-per-engine (default
    # = all of that teacher's GPUs, i.e. a single replica).
    gpus_per_replica = getattr(args, "teacher_num_gpus_per_engine", None) or gpus_per_teacher
    if gpus_per_teacher % gpus_per_replica != 0:
        raise ValueError(
            f"Per-teacher GPUs ({gpus_per_teacher}) must be divisible by "
            f"--teacher-num-gpus-per-engine ({gpus_per_replica})."
        )
    replicas_per_teacher = gpus_per_teacher // gpus_per_replica

    # MOPD is colocate-only: all teachers SHARE the actor/rollout placement group
    # (same as the single-teacher colocate path). During training the actor uses
    # all GPUs; during rollout student-rollout occupies bundles [0, rollout_gpus)
    # and teacher_k occupies [rollout_gpus + k*gpus_per_teacher, +gpus_per_teacher).
    # This requires rollout_gpus + total_teacher_gpus == actor_gpus.
    from relax.core.service import create_placement_group

    if not is_managed_opd_teacher_colocate(args):
        raise ValueError(
            "MOPD (--opd-teacher-routes) requires colocate mode: pass --colocate and "
            "include both 'actor' and 'rollout' in --resource. Dedicated teacher GPUs "
            "are no longer supported."
        )

    actor_gpus = args.resource["actor"][1]
    rollout_gpus = int(args.rollout_num_gpus)
    if rollout_gpus + total_teacher_gpus != actor_gpus:
        raise ValueError(
            f"MOPD colocate requires rollout_gpus + teacher_gpus == actor_gpus, but got "
            f"rollout={rollout_gpus} + teacher={total_teacher_gpus} != actor={actor_gpus}. "
            f"Teachers live inside the actor GPU pool (rollout at the front, teachers after "
            f"it), so shrink --rollout-num-gpus to leave room, e.g. actor=16 -> "
            f"--rollout-num-gpus 8 with resource['teacher'][1]=8."
        )

    shared_pg = create_placement_group(
        num_gpus=actor_gpus,
        node_group_affinity=getattr(args, "enable_affinity", True),
    )
    logger.info(
        f"[MOPD teacher] colocate mode: shared actor PG={actor_gpus} GPU, "
        f"rollout={rollout_gpus}, teachers start at bundle {rollout_gpus} "
        f"({gpus_per_teacher} GPU/teacher)"
    )

    logger.info(
        f"[MOPD teacher] launching {num_teachers} teachers x {replicas_per_teacher} replica(s), "
        f"{gpus_per_replica} GPU(s)/replica, {gpus_per_teacher} GPU(s)/teacher, total={total_teacher_gpus}"
    )

    from relax.distributed.ray.teacher_manager import TeacherManager

    teacher_managers = []
    url_routes: dict[str, list[str]] = {}

    for teacher_idx, (data_source, checkpoint_path) in enumerate(routes_map.items()):
        # Build per-teacher args with overridden model_path.
        teacher_args = copy.copy(args)
        teacher_args.teacher_hf_checkpoint = checkpoint_path

        teacher_manager = TeacherManager.options(
            **with_control_plane_affinity(
                teacher_args,
                {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env},
            )
        ).remote(
            teacher_args,
            replicas_per_teacher,
            gpus_per_replica,
            # Share the actor PG; this teacher takes its own bundle slice after
            # the rollout region.
            pg=shared_pg,
            shared_pg=True,
            bundle_offset=(teacher_idx * gpus_per_teacher),
        )
        urls = list(ray.get(teacher_manager.get_urls.remote()))
        # Append /generate to match the route format expected by _pick_teacher_url.
        replica_urls = [(u if u.endswith("/generate") else u.rstrip("/") + "/generate") for u in urls]

        url_routes[data_source] = replica_urls
        teacher_managers.append(teacher_manager)
        logger.info(
            f"[MOPD teacher] '{data_source}' → {checkpoint_path} "
            f"({replicas_per_teacher} replica(s), {gpus_per_replica} GPU(s) each) → {replica_urls}"
        )

    # Offload all teachers so the actor can load for the first training step. They
    # are onloaded/offloaded in lock-step with the actor thereafter.
    if getattr(args, "offload_rollout", False):
        ray.get([tm.offload.remote() for tm in teacher_managers])

    # Inject the routes map so per-sample routing works via _pick_teacher_url.
    args.opd_teacher_routes_map = url_routes
    opd_teacher_key = getattr(args, "opd_teacher_key", None) or "data_source"
    logger.info(f"[MOPD teacher] all teachers ready. key='{opd_teacher_key}', routes={list(url_routes.keys())}")

    # Return the shared PG so the controller reuses it for actor/rollout. Second
    # element is the list of managers for shutdown / offload-onload lock-step.
    return shared_pg, teacher_managers


async def set_managed_opd_teacher_on_actor_service(actor_service: Any, teacher_manager: Any, args: Any) -> None:
    if actor_service is None or teacher_manager is None or not is_managed_opd_teacher_colocate(args):
        return
    # Colocate teachers (single manager, or a list for MOPD multi-teacher) share the
    # actor placement group and are offloaded/onloaded in lock-step with training —
    # pass them through so the actor coordinates it.
    await actor_service.handle.set_teacher_manager.remote(teacher_manager)


def set_managed_opd_teacher_on_train_group(train_group: Any, teacher_manager: Any) -> None:
    import ray

    ray.get([actor.set_teacher_manager.remote(teacher_manager) for actor in train_group._actor_handlers])


def shutdown_managed_opd_teacher(teacher_manager: Any) -> None:
    if teacher_manager is None:
        return

    import ray

    # Support both single manager and list of managers (MOPD multi-teacher).
    managers = teacher_manager if isinstance(teacher_manager, list) else [teacher_manager]
    for mgr in managers:
        try:
            ray.get(mgr.shutdown.remote(), timeout=30)
            logger.info("OPD teacher shut down.")
        except Exception as e:
            logger.warning(f"Failed to shut down OPD teacher: {e}")


def has_managed_opd_teacher_manager(owner: Any) -> bool:
    # Truthy for a single manager or a non-empty list of managers (colocate MOPD).
    return bool(getattr(owner, "teacher_manager", None))


def append_managed_opd_teacher_offload_handle(handles: list[Any], owner: Any) -> None:
    teacher_manager = getattr(owner, "teacher_manager", None)
    if teacher_manager is None:
        return
    # Colocate multi-teacher stores a list; single-teacher stores one manager.
    for mgr in teacher_manager if isinstance(teacher_manager, list) else [teacher_manager]:
        handles.append(mgr.offload.remote())


def append_managed_opd_teacher_onload_handle(handles: list[Any], owner: Any) -> None:
    teacher_manager = getattr(owner, "teacher_manager", None)
    if teacher_manager is None:
        return
    for mgr in teacher_manager if isinstance(teacher_manager, list) else [teacher_manager]:
        handles.append(mgr.onload.remote())


def validate_managed_opd_teacher_colocate_args(args: Any) -> None:
    # Applies to both single-teacher (--teacher-hf-checkpoint) and multi-teacher
    # MOPD (--opd-teacher-routes): teachers share the actor placement group, so the
    # bundle-split constraint below (rollout + teacher == actor) holds for both.
    if args.offload_train is None:
        args.offload_train = True
    if args.offload_rollout is None:
        args.offload_rollout = True
    if args.rollout_num_gpus is None:
        if "rollout" not in args.resource:
            raise ValueError(
                "Managed OPD teacher colocate requires --rollout-num-gpus or a 'rollout' entry in --resource."
            )
        args.rollout_num_gpus = args.resource["rollout"][1]

    actor_total_gpus = args.resource.get("actor", [1, args.actor_num_gpus_per_node * args.actor_num_nodes])[1]
    if args.use_critic:
        actor_total_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    teacher_gpus = args.resource["teacher"][1]
    if args.rollout_num_gpus + teacher_gpus != actor_total_gpus:
        raise ValueError(
            "Managed OPD teacher colocate requires split bundles where "
            "--rollout-num-gpus + resource['teacher'][1] equals actor total GPUs. "
            f"Got rollout={args.rollout_num_gpus}, teacher={teacher_gpus}, actor total={actor_total_gpus}."
        )


def maybe_enable_sglang_weights_cpu_backup(args: Any) -> None:
    # Rollout sleep/wake without a CPU backup leaves weight pages uninitialized.
    if not getattr(args, "use_opd", False):
        return
    if not getattr(args, "offload_rollout", False):
        return
    if getattr(args, "sglang_enable_weights_cpu_backup", False):
        return
    logger.info("OPD with offloaded rollout: auto-enabling --sglang-enable-weights-cpu-backup.")
    args.sglang_enable_weights_cpu_backup = True


def add_opd_arguments(parser: Any) -> Any:
    parser.add_argument(
        "--use-opd",
        action="store_true",
        default=False,
        help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
    )
    parser.add_argument(
        "--opd-type",
        type=str,
        choices=["sglang", "megatron"],
        default=None,
        help=(
            "Type of on-policy distillation. "
            "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
        ),
    )
    parser.add_argument(
        "--opd-kl-coef",
        type=float,
        default=1.0,
        help=("On-policy distillation KL penalty coefficient. Default is 1.0."),
    )
    parser.add_argument(
        "--opd-loss-coef",
        type=float,
        default=0.0,
        help=("On-policy distillation KL coefficient, Default 0.0."),
    )

    parser.add_argument(
        "--opd-only-reward",
        "--opd-disable-rl-reward",
        action="store_true",
        default=False,
        dest="opd_only_reward",
        help=("Disable the base RL outcome reward."),
    )
    parser.add_argument(
        "--opd-teacher-load",
        type=str,
        default=None,
        help=(
            "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
            "The teacher model should have the same architecture as policy/ref model."
        ),
    )
    parser.add_argument(
        "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
    )
    parser.add_argument(
        "--opd-teacher-url",
        type=str,
        default=None,
        help=(
            "URL of the SGLang OPD teacher `/generate` endpoint "
            "(e.g. http://teacher-host:30001/generate). Kept separate from "
            "--rm-url so the OPD teacher and reward-model endpoint can be "
            "deployed and scaled independently."
        ),
    )
    parser.add_argument(
        "--opd-teacher-key",
        type=str,
        default=None,
        help=(
            "Sample metadata field used as the routing key for --opd-teacher-routes. "
            "Falls back to 'data_source' when OPD is enabled; kept None by default so "
            "compute_mopd_metrics stays a no-op (no per-source logging) in non-OPD runs."
        ),
    )
    parser.add_argument(
        "--teacher-hf-checkpoint",
        type=str,
        default=None,
        help=(
            "HF checkpoint path for a Relax-managed SGLang OPD teacher. "
            "When set with a 'teacher' entry in --resource, Relax launches the teacher "
            "and injects --opd-teacher-url at runtime. Teacher engine tensor-parallel "
            "size = --teacher-num-gpus-per-engine; number of teacher replicas = "
            "(teacher GPUs in --resource) / --teacher-num-gpus-per-engine."
        ),
    )
    parser.add_argument(
        "--teacher-num-gpus-per-engine",
        type=int,
        default=None,
        help=(
            "Number of GPUs per OPD teacher SGLang engine (== teacher TP size). "
            "When unset, defaults to the total teacher GPU count from --resource "
            "(single replica using all teacher GPUs). Set this lower to run multiple "
            "replicas, e.g. teacher GPUs=8 + --teacher-num-gpus-per-engine=4 -> "
            "2 replicas, each TP=4."
        ),
    )
    parser.add_argument(
        "--opd-teacher-routes",
        type=str,
        default=None,
        help=(
            "JSON map from data_source value to HF checkpoint path for multi-teacher "
            "on-policy distillation (MOPD). Relax launches one managed teacher per entry "
            "and routes each sample to the teacher whose key matches "
            "sample.metadata[--opd-teacher-key]. The 'teacher' entry in --resource specifies "
            "the TOTAL GPU budget, which is split equally among all teachers. "
            'Example: \'{"openai/gsm8k":"/path/Qwen3-8B",'
            '"hiyouga/geometry3k":"/path/Qwen3-VL-8B-Instruct"}\'. '
            "Mutually exclusive with --teacher-hf-checkpoint."
        ),
    )
    parser.add_argument(
        "--opd-teacher-timeout-s",
        type=float,
        default=30.0,
        help=(
            "Timeout (seconds) for OPD teacher HTTP requests when --opd-type=sglang. "
            "Increase this for long responses or high-latency cross-host teacher services."
        ),
    )
    parser.add_argument(
        "--opd-log-prob-top-k",
        type=int,
        default=0,
        help=("Top-k token ids to request/collect for OPD overlap metrics. Set to 0 to disable top-k collection."),
    )
    parser.add_argument(
        "--opd-kl-type",
        type=str,
        default="reverse_kl",
        choices=list(OPD_KL_TYPES),
        help=("Per-token OPD KL estimator."),
    )
    parser.add_argument(
        "--opd-jsd-alpha",
        type=float,
        default=0.5,
        help="Mixture coefficient for --opd-kl-type=jsd. 0.0 reduces to reverse_kl, 1.0 to forward_kl.",
    )
    parser.add_argument(
        "--opd-norm-mode",
        type=str,
        default="tail",
        choices=["tail", "norm", "trunc"],
        help=(
            "How to handle the tail probability in JSD / top-K KL computation: "
            "'tail' = append log(1 - sum p_i) as extra bin (default); "
            "'norm' = normalize top-K log-probs to sum-to-1 via logsumexp before KL; "
            "'trunc' = truncate (no tail), treat top-K as the full distribution."
        ),
    )
    parser.add_argument(
        "--opd-log-prob-min-clamp",
        type=float,
        default=None,
        help=(
            "When set, clamp both student/teacher log-probs from below by this value "
            "before computing the per-token OPD KL. This bounds the log-ratio to "
            "[clamp, -clamp] and avoids exp overflow in low_var_kl on outliers. "
            "Set to None (default) to disable."
        ),
    )

    parser.add_argument(
        "--opd-token-selection",
        type=str,
        default="student_sampled",
        choices=list(OPD_TOKEN_SELECTIONS),
        help=("Which token set the OPD KL signal is computed on."),
    )
    parser.add_argument(
        "--opd-teacher-prompt-key",
        type=str,
        default=None,
        help=("Dataset field name that holds the teacher-side prompt for On-Policy Self-Distillation (OPSD). "),
    )
    parser.add_argument(
        "--opd-teacher-image-key",
        type=str,
        default=None,
        help=("Dataset field name that holds the teacher-side images for *multimodal* OPSD ."),
    )
    parser.add_argument(
        "--opd-teacher-video-key",
        type=str,
        default=None,
        help="Same as --opd-teacher-image-key but for video modality.",
    )
    parser.add_argument(
        "--opd-teacher-audio-key",
        type=str,
        default=None,
        help="Same as --opd-teacher-image-key but for audio modality.",
    )

    parser.add_argument(
        "--opd-per-token-clip",
        type=float,
        default=None,
        help=(
            "Hard upper-bound on per-token OPD KL value, applied BEFORE advantage "
            "injection (as_adv) / loss aggregation (as_loss). Prevents KL spikes "
            "from destabilizing training (e.g. wrong answers with huge KL causing "
            "advantage explosion, or correct answers with huge KL flipping sign). "
            "Set to None (default) to disable."
        ),
    )
    parser.add_argument(
        "--opd-is-clip",
        type=float,
        default=None,
        help=(
            "Hard upper-bound on the importance-sampling ratio "
            "exp(log_p_new - log_p_old) applied to the per-token OPD KL "
            "before aggregation ."
            "Set to None (default) to disable."
        ),
    )
    return parser


def validate_opd_args(args: Namespace, *, is_sft: bool, log: Any = logger) -> None:
    if is_sft:
        return

    if not getattr(args, "use_opd", False):
        return

    # OPD is enabled here. Backfill the routing key so teacher routing AND the
    # per-source metrics (compute_mopd_metrics) get a consistent value even when
    # the user didn't pass --opd-teacher-key. The arg default stays None so that
    # compute_mopd_metrics remains a no-op in non-OPD runs.
    if getattr(args, "opd_teacher_key", None) is None:
        args.opd_teacher_key = "data_source"

    if args.opd_type is None:
        raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")
    if args.opd_teacher_timeout_s <= 0:
        raise ValueError("--opd-teacher-timeout-s must be > 0.")
    if args.opd_log_prob_top_k < 0:
        raise ValueError("--opd-log-prob-top-k must be >= 0.")
    token_selection = args.opd_token_selection
    if token_selection != "student_sampled":
        if args.opd_log_prob_top_k <= 0:
            raise ValueError(f"--opd-token-selection={token_selection} requires --opd-log-prob-top-k > 0 ")

    kl_type = args.opd_kl_type
    if token_selection == "student_sampled" and kl_type not in ("reverse_kl", "low_var_kl"):
        raise ValueError(
            f"--opd-kl-type={kl_type} is not valid for student_sampled tokens. "
            "Student sampled token KL estimator should be a reverse-kl approximation "
            "(choose reverse_kl or low_var_kl)."
        )

    opd_kl_coef = float(getattr(args, "opd_kl_coef", 0.0) or 0.0)
    opd_loss_coef = float(getattr(args, "opd_loss_coef", 0.0) or 0.0)

    is_adv_mode = opd_kl_coef != 0.0 and opd_loss_coef == 0.0
    is_loss_mode = opd_kl_coef == 0.0 and opd_loss_coef != 0.0
    if not is_adv_mode and not is_loss_mode:
        raise ValueError(
            "Exactly one of --opd-kl-coef / --opd-loss-coef must be non-zero. "
            f"Got opd_kl_coef={opd_kl_coef}, opd_loss_coef={opd_loss_coef}. "
            "Use --opd-kl-coef=X --opd-loss-coef=0.0 for advantage mode, or "
            "--opd-kl-coef=0.0 --opd-loss-coef=X for loss mode."
        )

    if getattr(args, "opd_teacher_prompt_key", None) is not None:
        if args.opd_type != "sglang":
            raise ValueError(
                "--opd-teacher-prompt-key currently only supports --opd-type=sglang "
                f"(got --opd-type={args.opd_type}). The megatron teacher path does not "
                "yet rebuild a teacher-side data_iterator from teacher_tokens."
            )

    if getattr(args, "opd_teacher_image_key", None) is not None:
        if args.opd_type != "sglang":
            raise ValueError("--opd-teacher-image-key currently only supports --opd-type=sglang.")
        if not getattr(args, "multimodal_keys", None):
            raise ValueError(
                "--opd-teacher-image-key requires --multimodal-keys to be set on the "
                "student side (the dataloader needs a processor and a parallel student field "
                "to extract teacher-side multimodal inputs)."
            )

    per_token_clip = getattr(args, "opd_per_token_clip", None)
    if per_token_clip is not None:
        if per_token_clip <= 0:
            raise ValueError(f"--opd-per-token-clip must be > 0, got {per_token_clip}.")

    is_clip = getattr(args, "opd_is_clip", None)
    if is_clip is not None:
        if is_clip <= 0:
            raise ValueError(f"--opd-is-clip must be > 0, got {is_clip}.")
        if opd_loss_coef == 0.0:
            log.info(
                "--opd-is-clip is set but --opd-loss-coef == 0; the clip "
                "only takes effect on the differentiable loss path."
            )

    if args.opd_type == "megatron":
        if args.opd_teacher_load is None:
            raise ValueError(
                "--opd-teacher-load is required when --opd-type=megatron. "
                "Please provide the path to the teacher model checkpoint."
            )
        if not os.path.exists(args.opd_teacher_load):
            raise FileNotFoundError(f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path.")
        if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
            log.info(
                f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    elif args.opd_type == "sglang":
        if args.opd_teacher_load is not None:
            raise ValueError(
                "--opd-teacher-load should not be set when --opd-type=sglang. "
                "In sglang mode, teacher log-probs are obtained from external server during rollout."
            )

        # --opd-teacher-routes: JSON map {data_source: HF checkpoint path} for
        # Relax-managed multi-teacher MOPD. The runtime URL map
        # (args.opd_teacher_routes_map) is populated at launch by
        # _start_managed_multi_teacher; here we only validate the spec.
        args.opd_teacher_routes_map = None
        opd_teacher_routes = getattr(args, "opd_teacher_routes", None)
        if opd_teacher_routes is not None:
            try:
                routes_map = json.loads(opd_teacher_routes)
            except json.JSONDecodeError as e:
                raise ValueError(f"--opd-teacher-routes must be valid JSON: {e}") from e
            if not isinstance(routes_map, dict) or not routes_map:
                raise ValueError("--opd-teacher-routes must be a non-empty JSON object.")
            log.info(
                "MOPD managed multi-teacher enabled: %d teachers, key='%s', sources=%s",
                len(routes_map),
                getattr(args, "opd_teacher_key", None) or "data_source",
                list(routes_map.keys()),
            )
            if getattr(args, "teacher_hf_checkpoint", None) is not None:
                raise ValueError(
                    "--opd-teacher-routes is mutually exclusive with --teacher-hf-checkpoint. "
                    "Use routes for multi-teacher, or a single checkpoint for single-teacher."
                )
            if getattr(args, "resource", None) is None or "teacher" not in args.resource:
                raise ValueError(
                    "--opd-teacher-routes requires a 'teacher' entry in --resource "
                    "specifying the total GPU budget to split among teachers."
                )

        has_managed_teacher = (
            getattr(args, "teacher_hf_checkpoint", None) is not None
            and getattr(args, "resource", None) is not None
            and "teacher" in args.resource
        )
        has_managed_multi_teacher = (
            opd_teacher_routes is not None
            and getattr(args, "resource", None) is not None
            and "teacher" in args.resource
        )
        if args.opd_teacher_url is None and not has_managed_teacher and not has_managed_multi_teacher:
            raise ValueError(
                "A teacher source is required when --opd-type=sglang. Set --opd-teacher-url for a "
                "single external teacher, or use the Relax-managed path: --teacher-hf-checkpoint "
                "(single) / --opd-teacher-routes (multi) with a 'teacher' entry in --resource."
            )


# ============================================================================
# Multimodal image encoding helpers (raw base64 PNG for opd_preexpanded_raw)
# ============================================================================


def _to_jsonable(x):
    """Convert a tensor / ndarray to nested Python lists for JSON transport."""
    if x is None:
        return None
    tolist = getattr(x, "tolist", None)
    return tolist() if callable(tolist) else x


async def _encode_images_b64(raw_images: list, cache_key: str, mm_dict: dict | None) -> list[str]:
    """Parallel-encode raw images to base64, with group-level dedup cache.

    ``cache_key`` is the attribute name on the shared ``mm_dict`` to cache the
    result (so 8 samples in a group encode each image at most once per step).
    """
    import asyncio

    from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

    if mm_dict is None:
        return list(await asyncio.gather(*(async_encode_image_for_rollout_engine(img) for img in raw_images)))
    cached = mm_dict.get(cache_key)
    if cached is None:
        cached = list(await asyncio.gather(*(async_encode_image_for_rollout_engine(img) for img in raw_images)))
        mm_dict[cache_key] = cached
    return cached


async def build_teacher_preexpanded_image_data(sample: "Sample") -> list | None:
    """Build image_data for the teacher forward request (OPSD path).

    Ships raw base64 images + teacher image_grid_thw via
    format=opd_preexpanded_raw. Returns None when there are no images (text-
    only path).
    """
    image_b64_list = getattr(sample, "teacher_image_b64_list", None)
    image_grid_thw = getattr(sample, "teacher_image_grid_thw", None)
    if not image_b64_list or image_grid_thw is None:
        return None
    return [
        {
            "format": opd_opsd_worker.PREEXPANDED_RAW_FORMAT,
            "images_b64": list(image_b64_list),
            "image_grid_thw": _to_jsonable(image_grid_thw),
        }
    ]


async def build_student_preexpanded_image_data(sample: "Sample") -> list | None:
    """Build image_data for the student extra forward request (student-at-
    teacher-topk).

    The student SGLang has the same opd_preexpanded_raw patch. Ships raw base64
    images (from sample.multimodal_inputs, cached) + student image_grid_thw.
    Returns None for text-only samples.
    """
    student_mm_train_inputs = getattr(sample, "multimodal_train_inputs", None) or {}
    image_grid_thw = student_mm_train_inputs.get("image_grid_thw")
    student_mm_in = sample.multimodal_inputs or {}
    raw_images = student_mm_in.get("images") or []
    if image_grid_thw is None or not raw_images:
        return None
    cached = await _encode_images_b64(raw_images, "_student_image_b64_cache", student_mm_in)
    return [
        {
            "format": opd_opsd_worker.PREEXPANDED_RAW_FORMAT,
            "images_b64": cached,
            "image_grid_thw": _to_jsonable(image_grid_thw),
        }
    ]


def _get_opd_transfer_schema(args: Namespace) -> list[str]:
    from relax.engine.rollout.on_policy_distillation import OpdManager

    return OpdManager(args).schema_opd_transfer_data()


def consume_opd_train_data(data_fields: list[str], args: Namespace) -> None:
    if not (getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"):
        return
    data_fields.extend(_get_opd_transfer_schema(args))


def consume_opd_advantage_data(data_fields: list[str], args: Namespace) -> None:
    if not getattr(args, "use_opd", False):
        return
    data_fields.extend(_get_opd_transfer_schema(args))


def build_opd_teacher_sample_fields(
    data: dict,
    tokenizer: Any,
    processor: Any,
    *,
    prompt_key: str,
    system_prompt: str | None,
    as_conversation: bool,
    multimodal_keys: dict | None,
    teacher_prompt_key: str | None,
    teacher_multimodal_keys: dict | None,
    custom_prompt_func: Callable[[Any, dict], Any] | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
    tools: Any,
    use_audio_in_video: bool | None,
    multimodal_config: Any,
    build_messages_fn: Callable[..., Any],
) -> tuple[str | list[dict[str, Any]] | None, Any]:
    teacher_prompt: str | list[dict[str, Any]] | None = None
    if teacher_prompt_key is not None and teacher_prompt_key in data:
        teacher_prompt_messages = build_messages_fn(
            data,
            teacher_prompt_key,
            system_prompt,
            as_conversation,
            teacher_multimodal_keys or multimodal_keys,
            custom_prompt_func,
        )
        if apply_chat_template:
            teacher_prompt = tokenizer.apply_chat_template(
                teacher_prompt_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                **(apply_chat_template_kwargs or {}),
            )
        else:
            teacher_prompt = teacher_prompt_messages

    teacher_multimodal_inputs = None
    if teacher_multimodal_keys is not None and processor:
        from relax.utils.data.processing_utils import process_vision_info

        teacher_messages_for_mm = build_messages_fn(
            data,
            teacher_prompt_key or prompt_key,
            system_prompt,
            True,
            teacher_multimodal_keys,
            custom_prompt_func,
        )
        teacher_multimodal_inputs = process_vision_info(
            teacher_messages_for_mm,
            processor,
            use_audio_in_video=use_audio_in_video,
            config=multimodal_config,
        )

    return teacher_prompt, teacher_multimodal_inputs


def validate_opd_topk_gather(args: Namespace, gather_topk_token_ids: list[torch.Tensor] | None) -> None:
    if gather_topk_token_ids is not None and getattr(args, "allgather_cp", False):
        raise NotImplementedError(
            "gather_topk_token_ids is not yet compatible with allgather_cp=True; "
            "the OPD top-K path needs an _allgather_cp_redistribute extension for "
            "the 2D 'topk_log_probs' key."
        )


def resolve_opd_gather_topk_token_ids(args: Namespace, batch: RolloutBatch) -> list[torch.Tensor] | None:
    token_selection = args.opd_token_selection
    if token_selection in ("student_topk", "teacher_topk", "union"):
        return batch.get("opd_topk_token_ids", None)
    return None


def compute_opd_topk_log_probs(
    logits_chunk: torch.Tensor,
    gather_topk_token_ids: list[torch.Tensor],
    sample_idx: int,
) -> torch.Tensor:
    from megatron.core import mpu

    if sample_idx >= len(gather_topk_token_ids):
        return logits_chunk.new_zeros((logits_chunk.size(0), 0))

    ids = gather_topk_token_ids[sample_idx]
    if ids is None or ids.numel() == 0:
        return logits_chunk.new_zeros((logits_chunk.size(0), 0))

    return compute_log_probs_on_topk_token_ids(logits_chunk, ids, mpu.get_tensor_model_parallel_group())


def compute_log_probs_on_topk_token_ids(
    logits: torch.Tensor,
    topk_token_ids: torch.Tensor,
    process_group: Any,
) -> torch.Tensor:
    """Vocab-parallel gather of log-probs at the given top-K token ids.

    Args:
        logits: ``[S, V/TP]`` vocab-parallel logits (this rank's vocab slice).
        topk_token_ids: ``[S, K]`` long, **global** vocab indices. Values
            ``< 0`` are treated as sentinels and yield ``-inf`` in the output.
        process_group: tensor-parallel group. ``None`` falls back to non-
            parallel semantics (single-rank gather).

    Returns:
        ``[S, K]`` float log-probabilities, identical on all TP ranks (after
        the cross-rank reductions). Gradient flows back to ``logits`` through
        both the gather and the logsumexp paths.
    """
    if logits.size(0) == 0:
        return logits.new_zeros((0, topk_token_ids.size(-1)))

    # === Step 1: vocab-parallel log-sum-exp over the vocab axis ===
    logits_max = logits.detach().max(dim=-1, keepdim=True).values  # [S, 1]
    if process_group is not None:
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
    shifted = logits - logits_max  # [S, V/TP]
    exp_sum_local = shifted.exp().sum(dim=-1, keepdim=True)  # [S, 1]
    if process_group is not None:
        from megatron.core.tensor_parallel.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        exp_sum = reduce_from_tensor_model_parallel_region(exp_sum_local)
    else:
        exp_sum = exp_sum_local
    logsumexp = logits_max + exp_sum.log()  # [S, 1]

    # === Step 2: vocab-parallel gather of logits at topk_token_ids ===
    from megatron.core import mpu

    if process_group is not None:
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
    else:
        tp_world_size = 1
        tp_rank = 0
    vocab_size_per_rank = logits.size(-1)
    vocab_start = tp_rank * vocab_size_per_rank
    vocab_end = vocab_start + vocab_size_per_rank

    ids = topk_token_ids.to(device=logits.device, dtype=torch.long)  # [S, K]
    invalid = ids < 0
    safe_ids = ids.clamp(min=0)
    in_range = (safe_ids >= vocab_start) & (safe_ids < vocab_end)
    local_ids = (safe_ids - vocab_start).clamp(min=0, max=vocab_size_per_rank - 1)

    topk_logits_local = torch.gather(logits, dim=-1, index=local_ids)  # [S, K]
    topk_logits_local = topk_logits_local.masked_fill(~in_range, 0.0)

    if process_group is not None and tp_world_size > 1:
        from megatron.core.tensor_parallel.mappings import (
            reduce_from_tensor_model_parallel_region,
        )

        topk_logits = reduce_from_tensor_model_parallel_region(topk_logits_local)
    else:
        topk_logits = topk_logits_local

    topk_lp = topk_logits - logsumexp  # [S, K]
    topk_lp = topk_lp.masked_fill(invalid, float("-inf"))
    return topk_lp


def compute_opd_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    kl_type: str = "reverse_kl",
    jsd_alpha: float = 0.5,
    norm_mode: str = "tail",
    log_prob_min_clamp: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token OPD KL signal.

    For 1D (sampled-token) path only ``reverse_kl`` and ``low_var_kl`` are valid.
    For 2D (top-K) path ``reverse_kl`` / ``forward_kl`` / ``jsd`` go through
    :func:`compute_opd_kl_topk`; ``low_var_kl`` is applied element-wise then
    summed over K.

    ``log_prob_min_clamp``: when not None, clamp both student/teacher log-probs
    from below before computing the log-ratio. This bounds ``log_ratio`` to
    ``[log_prob_min_clamp, -log_prob_min_clamp]`` and avoids ``exp`` overflow in
    ``low_var_kl`` on outliers.

    ``mask``: optional bool tensor ``[R, K]`` (``True`` = valid). Used by the
    union path where rows are padded to ``max_K'``. When provided, padding
    positions are set to ``-inf`` after clamp (so ``logsumexp`` ignores them)
    and their contributions are zeroed before ``.sum(dim=-1)``. ``None`` (topk
    path with fixed K) skips all masking — behavior unchanged.
    """
    s = student_log_probs.float()
    t = teacher_log_probs.float()
    if log_prob_min_clamp is not None:
        s = s.clamp_min(log_prob_min_clamp)
        t = t.clamp_min(log_prob_min_clamp)
    if mask is not None:
        # Re-apply -inf on padding (clamp may have lifted -inf to min_clamp).
        s = s.masked_fill(~mask, float("-inf"))
        t = t.masked_fill(~mask, float("-inf"))
    log_ratio = s - t

    if kl_type == "reverse_kl":
        return log_ratio
    if kl_type == "low_var_kl":
        per_kl = torch.exp(-log_ratio) - 1.0 + log_ratio
        if mask is not None:
            per_kl = per_kl.masked_fill(~mask, 0.0)
        return per_kl.sum(dim=-1) if per_kl.dim() > 1 else per_kl
    if kl_type == "jsd":
        if not (0.0 <= jsd_alpha <= 1.0):
            raise ValueError(f"jsd_alpha must be in [0, 1], got {jsd_alpha}")

        def _add_tail(lp: torch.Tensor) -> torch.Tensor:
            log_s = torch.logsumexp(lp, dim=-1, keepdim=True).clamp(max=-1e-7)
            tail = torch.log(-torch.expm1(log_s))
            return torch.cat([lp, tail], dim=-1)

        def _norm(lp: torch.Tensor) -> torch.Tensor:
            return lp - torch.logsumexp(lp, dim=-1, keepdim=True)

        if norm_mode == "tail":
            s_t = _add_tail(s)
            t_t = _add_tail(t)
        elif norm_mode == "norm":
            s_t = _norm(s)
            t_t = _norm(t)
        else:  # trunc
            s_t = s
            t_t = t

        K_orig = mask.size(-1) if mask is not None else s_t.size(-1)
        if jsd_alpha == 0.0:
            result = s_t.exp() * (s_t - t_t)
            if mask is not None:
                result = result[..., :K_orig].masked_fill(~mask, 0.0)
            return result.sum(dim=-1)
        if jsd_alpha == 1.0:
            result = t_t.exp() * (t_t - s_t)
            if mask is not None:
                result = result[..., :K_orig].masked_fill(~mask, 0.0)
            return result.sum(dim=-1)

        log_1ma = torch.log(torch.tensor(1.0 - jsd_alpha, device=s_t.device, dtype=s_t.dtype))
        log_a = torch.log(torch.tensor(jsd_alpha, device=s_t.device, dtype=s_t.dtype))
        m_t = torch.logsumexp(torch.stack([s_t + log_1ma, t_t + log_a]), dim=0)

        kl_student = s_t.exp() * (s_t - m_t)
        kl_teacher = t_t.exp() * (t_t - m_t)
        if mask is not None:
            kl_student = kl_student[..., :K_orig].masked_fill(~mask, 0.0)
            kl_teacher = kl_teacher[..., :K_orig].masked_fill(~mask, 0.0)
        return (1.0 - jsd_alpha) * kl_student.sum(dim=-1) + jsd_alpha * kl_teacher.sum(dim=-1)

    raise ValueError(f"Unknown opd_kl_type: {kl_type}. Choose one of {OPD_KL_TYPES}.")


def compute_opd_kl_topk(
    student_topk_lp: torch.Tensor,
    teacher_topk_lp: torch.Tensor,
    kl_type: str = "reverse_kl",
    jsd_alpha: float = 0.5,
    norm_mode: str = "tail",
    log_prob_min_clamp: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token KL from [R, K] top-K log-probs.

    ``mask``: optional bool ``[R, K]`` (union path only). ``None`` for fixed-K
    topk path — behavior unchanged.
    """
    if kl_type in ("reverse_kl", "forward_kl", "jsd"):
        if kl_type == "reverse_kl":
            alpha = 0.0
        elif kl_type == "forward_kl":
            alpha = 1.0
        else:
            alpha = float(jsd_alpha)
        return compute_opd_kl(
            student_topk_lp,
            teacher_topk_lp,
            kl_type="jsd",
            jsd_alpha=alpha,
            norm_mode=norm_mode,
            log_prob_min_clamp=log_prob_min_clamp,
            mask=mask,
        )
    # Element-wise fallback for low_var_kl on [R, K].
    per_kl = compute_opd_kl(
        student_topk_lp,
        teacher_topk_lp,
        kl_type=kl_type,
        log_prob_min_clamp=log_prob_min_clamp,
        mask=mask,
    )
    return per_kl.sum(dim=-1)


def _opd_compute_per_token_signal(
    *,
    student_lp_1d: torch.Tensor | None,
    teacher_lp_1d: torch.Tensor | None,
    student_lp_2d: torch.Tensor | None = None,
    teacher_lp_2d: torch.Tensor | None = None,
    token_selection: str,
    kl_type: str,
    jsd_alpha: float,
    norm_mode: str = "tail",
    log_prob_min_clamp: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if token_selection in ("student_topk", "teacher_topk", "union"):
        if (
            student_lp_2d is not None
            and teacher_lp_2d is not None
            and student_lp_2d.numel() > 0
            and teacher_lp_2d.numel() > 0
            and student_lp_2d.shape == teacher_lp_2d.shape
        ):
            return compute_opd_kl_topk(
                student_lp_2d,
                teacher_lp_2d,
                kl_type=kl_type,
                jsd_alpha=jsd_alpha,
                norm_mode=norm_mode,
                log_prob_min_clamp=log_prob_min_clamp,
                mask=mask,
            ).to(dtype=student_lp_2d.dtype)

    if student_lp_1d is None or teacher_lp_1d is None:
        raise ValueError(f"OPD per-token signal: token_selection={token_selection} but the required arms are missing.")

    return compute_opd_kl(
        student_lp_1d,
        teacher_lp_1d,
        kl_type=kl_type,
        jsd_alpha=jsd_alpha,
        norm_mode=norm_mode,
        log_prob_min_clamp=log_prob_min_clamp,
    ).to(dtype=student_lp_1d.dtype)


def apply_opd_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
) -> None:
    if args.opd_kl_coef == 0.0:
        return

    kl_type = args.opd_kl_type
    jsd_alpha = float(args.opd_jsd_alpha)
    norm_mode = getattr(args, "opd_norm_mode", "tail")
    log_prob_min_clamp = getattr(args, "opd_log_prob_min_clamp", None)
    token_selection = args.opd_token_selection
    is_topk = token_selection in ("student_topk", "teacher_topk", "union")

    if is_topk:
        student_topk_lp_list = rollout_data.get("opd_topk_student_log_probs")
        teacher_topk_lp_list = rollout_data.get("opd_topk_teacher_log_probs")
        if student_topk_lp_list is None or teacher_topk_lp_list is None:
            return
        device = advantages[0].device if advantages else torch.device("cpu")
        # union ：per-row valid length  → bool mask [R, max_K']
        k_lengths_list = rollout_data.get("opd_topk_ksz") if token_selection == "union" else None

        for i, adv in enumerate(advantages):
            s_lp_2d = None
            t_lp_2d = None
            if i < len(student_topk_lp_list) and isinstance(student_topk_lp_list[i], torch.Tensor):
                s_lp_2d = student_topk_lp_list[i].to(device=device)
            if i < len(teacher_topk_lp_list) and isinstance(teacher_topk_lp_list[i], torch.Tensor):
                t_lp_2d = teacher_topk_lp_list[i].to(device=device)

            mask = None
            if k_lengths_list is not None and i < len(k_lengths_list):
                kl = k_lengths_list[i]
                if kl is not None and t_lp_2d is not None:
                    kl = kl.to(device=device)
                    max_kp = t_lp_2d.size(-1)
                    mask = torch.arange(max_kp, device=device).unsqueeze(0) < kl.unsqueeze(1)

            kl_term = _opd_compute_per_token_signal(
                student_lp_1d=None,
                teacher_lp_1d=None,
                student_lp_2d=s_lp_2d,
                teacher_lp_2d=t_lp_2d,
                token_selection=token_selection,
                kl_type=kl_type,
                jsd_alpha=jsd_alpha,
                norm_mode=norm_mode,
                log_prob_min_clamp=log_prob_min_clamp,
                mask=mask,
            )

            per_token_clip = getattr(args, "opd_per_token_clip", None)
            if per_token_clip is not None:
                kl_term = torch.clamp(kl_term, max=float(per_token_clip))
            advantages[i] = adv - args.opd_kl_coef * kl_term.detach()
        return

    # sampled_token：1D teacher_log_probs + rollout_log_probs
    student_log_probs = rollout_data.get("rollout_log_probs")
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if student_log_probs is None or teacher_log_probs is None:
        return
    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    for i, adv in enumerate(advantages):
        kl_term = _opd_compute_per_token_signal(
            student_lp_1d=student_log_probs[i],
            teacher_lp_1d=teacher_log_probs[i],
            token_selection=token_selection,
            kl_type=kl_type,
            jsd_alpha=jsd_alpha,
            norm_mode=norm_mode,
            log_prob_min_clamp=log_prob_min_clamp,
        )
        advantages[i] = adv - args.opd_kl_coef * kl_term.detach()


def reduce_opd_loss(batch: RolloutBatch, values: torch.Tensor) -> torch.Tensor:
    chunks = torch.split(values, batch["response_lengths"], dim=0)
    masked_chunks = []
    for chunk, loss_mask in zip(chunks, batch["loss_masks"], strict=False):
        mask = loss_mask.to(device=chunk.device, dtype=chunk.dtype)
        masked = chunk * mask
        masked_chunks.append(masked)

    numerator = torch.cat(masked_chunks, dim=0).sum()
    denominator = sum(mask.to(device=values.device, dtype=values.dtype).sum() for mask in batch["loss_masks"])
    return numerator / torch.clamp_min(denominator, 1)


def compute_policy_opd_loss(
    *,
    args: Namespace,
    batch: RolloutBatch,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    log_probs_and_entropy: dict[str, list[torch.Tensor]],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    opd_loss_coef = float(getattr(args, "opd_loss_coef", 0.0))
    if opd_loss_coef == 0.0:
        return None, {}

    opd_kl_type = args.opd_kl_type
    opd_jsd_alpha = float(args.opd_jsd_alpha)
    opd_norm_mode = getattr(args, "opd_norm_mode", "tail")
    opd_log_prob_min_clamp = getattr(args, "opd_log_prob_min_clamp", None)
    token_selection = args.opd_token_selection
    is_topk = token_selection in ("student_topk", "teacher_topk", "union")

    student_topk_lp_list = log_probs_and_entropy.get("topk_log_probs")
    teacher_topk_lp_list = batch.get("opd_topk_teacher_log_probs")

    if is_topk:
        if not (student_topk_lp_list and teacher_topk_lp_list):
            return None, {}
        # union: each sample has a different K' (union of student/teacher top-K),
        # so we cannot torch.cat the per-sample [R_i, K'_i] along dim=0.
        # Compute per-sample KL (returns 1D [R_i]) then cat the 1D results.
        k_lengths_list = batch.get("opd_topk_ksz") if token_selection == "union" else None
        device = log_probs.device
        per_token_kl_chunks: list[torch.Tensor] = []
        for i, s_lp_2d in enumerate(student_topk_lp_list):
            t_lp_2d = teacher_topk_lp_list[i].to(device=device).detach()
            s_lp_2d = s_lp_2d.to(device=device)
            mask = None
            if k_lengths_list is not None and i < len(k_lengths_list):
                kl = k_lengths_list[i]
                if kl is not None and t_lp_2d is not None:
                    kl = kl.to(device=device)
                    max_kp = t_lp_2d.size(-1)
                    mask = torch.arange(max_kp, device=device).unsqueeze(0) < kl.unsqueeze(1)
            per_token_kl_chunks.append(
                compute_opd_kl_topk(
                    s_lp_2d,
                    t_lp_2d,
                    kl_type=opd_kl_type,
                    jsd_alpha=opd_jsd_alpha,
                    norm_mode=opd_norm_mode,
                    log_prob_min_clamp=opd_log_prob_min_clamp,
                    mask=mask,
                )
            )
        opd_per_token_kl = torch.cat(per_token_kl_chunks, dim=0).to(dtype=log_probs.dtype)
    else:
        if "teacher_log_probs" not in batch or batch["teacher_log_probs"] is None:
            return None, {}
        teacher_log_probs_loss = (
            torch.cat(batch["teacher_log_probs"], dim=0).to(device=log_probs.device, dtype=log_probs.dtype).detach()
        )
        opd_per_token_kl = compute_opd_kl(
            log_probs,
            teacher_log_probs_loss,
            kl_type=opd_kl_type,
            jsd_alpha=opd_jsd_alpha,
            norm_mode=opd_norm_mode,
            log_prob_min_clamp=opd_log_prob_min_clamp,
        ).to(dtype=log_probs.dtype)

    reported_loss: dict[str, torch.Tensor] = {}
    per_token_clip = getattr(args, "opd_per_token_clip", None)
    if per_token_clip is not None:
        tau = float(per_token_clip)
        before_clip = opd_per_token_kl
        opd_per_token_kl = torch.clamp(opd_per_token_kl, max=tau)
        with torch.no_grad():
            reported_loss["opd_per_token_clip_frac"] = (before_clip > tau).float().mean().clone().detach()

    is_clip = getattr(args, "opd_is_clip", None)
    if is_clip is not None:
        clip = float(is_clip)
        ratio = torch.exp(log_probs.detach() - old_log_probs.detach())
        ratio_clipped = torch.clamp(ratio, max=clip)
        opd_per_token_kl = opd_per_token_kl * ratio_clipped.to(dtype=opd_per_token_kl.dtype)
        with torch.no_grad():
            reported_loss["opd_is_clip_frac"] = (ratio > clip).float().mean().clone().detach()

    opd_loss = reduce_opd_loss(batch, opd_per_token_kl)
    return opd_loss_coef * opd_loss, reported_loss


# ---------------------------------------------------------------------------
# MOPD per-source metrics
# ---------------------------------------------------------------------------


def compute_mopd_metrics(args: Namespace, all_samples: list) -> dict:
    """Per-data-source accuracy and OPD distillation metrics for MOPD.

    Groups samples by ``metadata[opd_teacher_key]`` and reports for each source:
      - accuracy     : fraction of samples with reward == 1
      - mean_reward  : average reward value
      - teacher_logp : sequence-mean teacher log-prob
      - student_logp : sequence-mean student log-prob
      - logp_gap     : student_logp - teacher_logp (→ 0 as student converges)
      - rkl_approx   : E_student[log p_s - log p_t] per token, averaged across samples

    Returns ``{}`` when OPD is disabled (``opd_teacher_key`` absent).
    """
    from relax.utils.misc import group_by

    opd_teacher_key = getattr(args, "opd_teacher_key", None)
    if not opd_teacher_key:
        return {}

    by_source = group_by(all_samples, lambda s: (s.metadata or {}).get(opd_teacher_key, "unknown"))
    log_dict: dict = {}

    for source, samples in by_source.items():
        n = len(samples)
        if n == 0:
            continue

        label = source.replace("/", "_").replace("-", "_")
        prefix = f"by_source/{label}/"

        raw_rewards = [s.get_reward_value(args) for s in samples]
        log_dict[prefix + "accuracy"] = sum(1 for r in raw_rewards if r == 1) / n
        log_dict[prefix + "mean_reward"] = sum(raw_rewards) / n

        teacher_lp_seq: list[float] = []
        student_lp_seq: list[float] = []
        gap_seq: list[float] = []
        rkl_seq: list[float] = []

        for s in samples:
            t_lp = s.teacher_log_probs
            s_lp = s.rollout_log_probs
            t_mean = sum(t_lp) / len(t_lp) if t_lp and len(t_lp) > 0 else None
            s_mean = sum(s_lp) / len(s_lp) if s_lp and len(s_lp) > 0 else None
            if t_mean is not None:
                teacher_lp_seq.append(t_mean)
            if s_mean is not None:
                student_lp_seq.append(s_mean)
            # Pair gap/rkl per-sample: only when THIS sample has both teacher and
            # student log probs, so the values stay aligned to one sample (the
            # marginal teacher_lp_seq/student_lp_seq lists can differ in membership).
            if t_mean is not None and s_mean is not None:
                gap_seq.append(s_mean - t_mean)
            if t_lp and s_lp:
                n_tok = min(len(t_lp), len(s_lp))
                if n_tok > 0:
                    rkl_seq.append(sum(s_lp[i] - t_lp[i] for i in range(n_tok)) / n_tok)

        if teacher_lp_seq:
            log_dict[prefix + "teacher_logp"] = sum(teacher_lp_seq) / len(teacher_lp_seq)
        if student_lp_seq:
            log_dict[prefix + "student_logp"] = sum(student_lp_seq) / len(student_lp_seq)
        if gap_seq:
            log_dict[prefix + "logp_gap"] = sum(gap_seq) / len(gap_seq)
        if rkl_seq:
            log_dict[prefix + "rkl_approx"] = sum(rkl_seq) / len(rkl_seq)

    return log_dict

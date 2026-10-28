# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resolve the complete inference layout before allocating any engines."""

import json
from dataclasses import dataclass
from typing import Any


def validate_engine_config(width: int, overrides: dict, *, default_pp: int = 1) -> None:
    reserved = {"nnodes", "node_rank", "base_gpu_id", "gpu_id_step", "dist_init_addr", "host", "port", "nccl_port"}
    if reserved.intersection(overrides):
        raise ValueError("Engine config overrides planner-owned placement fields")
    pp = overrides.get("pp_size", default_pp)
    if not isinstance(pp, int) or isinstance(pp, bool) or pp <= 0 or width % pp:
        raise ValueError("Engine GPU width must be divisible by pipeline parallel size")
    tp = overrides.get("tp_size", width // pp)
    if not isinstance(tp, int) or isinstance(tp, bool) or tp <= 0 or tp * pp != width:
        raise ValueError("Engine TP x PP must match the planned GPU width")


@dataclass(frozen=True)
class ModelPlacement:
    role: str
    model: str
    pool: str
    offset: int
    num_gpus: int
    gpus_per_engine: int
    pg_owner: str
    phase: str


class PlacementPlanner:
    @staticmethod
    def resolve(args: Any) -> tuple[ModelPlacement, ...]:
        resource = getattr(args, "resource", {}) or {}
        if "rollout" not in resource:
            return ()
        colocate = bool(getattr(args, "colocate", False) and not getattr(args, "hybrid", False))
        teacher_defer = bool(getattr(args, "opd_teacher_defer", False))
        genrm_defer = bool(getattr(args, "defer_reward_to_post_process", False))
        if genrm_defer and not getattr(args, "_genrm_instances_resolved", None):
            raise ValueError("Deferred GenRM requires Relax-managed reward engines")
        if (teacher_defer or genrm_defer) and (not colocate or getattr(args, "fully_async", False)):
            raise ValueError("Deferred inference requires synchronous colocate")
        node_width = getattr(args, "num_gpus_per_node", None) or getattr(args, "actor_num_gpus_per_node", 1)
        rollout = getattr(args, "rollout_num_gpus", None) or resource["rollout"][1]
        if not colocate and rollout > resource["rollout"][1]:
            raise ValueError("Rollout GPU request exceeds its independent resource budget")
        specs = []
        rollout_width = getattr(args, "rollout_num_gpus_per_engine", 1)
        if getattr(args, "sglang_config", None):
            import yaml

            with open(args.sglang_config) as stream:
                config = yaml.safe_load(stream)
            names = [model["name"] for model in config["sglang"]]
            if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
                raise ValueError("Rollout model names must be nonempty and unique")
            for model in config["sglang"]:
                kinds = {group["worker_type"] for group in model["engine_groups"]}
                if not kinds <= {"regular", "prefill", "decode", "placeholder"}:
                    raise ValueError("Unknown inference worker type")
                if kinds & {"prefill", "decode"}:
                    if not {"prefill", "decode"} <= kinds or getattr(args, "use_slime_router", False):
                        raise ValueError("PD requires both prefill/decode groups and a PD-capable router")
                for index, group in enumerate(model["engine_groups"]):
                    width = group.get("num_gpus_per_engine") or model.get("num_gpus_per_engine") or rollout_width
                    if group["worker_type"] == "placeholder":
                        width = 1
                    else:
                        validate_engine_config(
                            width, group.get("overrides", {}), default_pp=getattr(args, "sglang_pp_size", 1)
                        )
                    specs.append(("rollout", f"{model['name']}/{index}", group["num_gpus"], width, False))
            if sum(spec[2] for spec in specs) != rollout:
                raise ValueError("Rollout engine group budgets must equal the Rollout GPU allocation")
        else:
            validate_engine_config(rollout_width, {}, default_pp=getattr(args, "sglang_pp_size", 1))
            prefill = getattr(args, "prefill_num_servers", None)
            if prefill is not None and (
                prefill <= 0 or prefill * rollout_width >= rollout or getattr(args, "use_slime_router", False)
            ):
                raise ValueError("PD requires positive prefill and decode GPU budgets and a PD-capable router")
            specs.append(("rollout", "default", rollout, rollout_width, False))
        for key, spec in (getattr(args, "_genrm_instances_resolved", {}) or {}).items():
            validate_engine_config(spec["num_gpus_per_engine"], spec.get("engine_config", {}))
            specs.append(("genrm", key, spec["num_gpus"], spec["num_gpus_per_engine"], genrm_defer))
        managed_teacher = (
            getattr(args, "use_opd", False)
            and getattr(args, "opd_type", None) == "sglang"
            and (getattr(args, "teacher_hf_checkpoint", None) or getattr(args, "opd_teacher_routes", None))
        )
        if teacher_defer and not managed_teacher:
            raise ValueError("Deferred Teacher requires Relax-managed teacher engines")
        if managed_teacher:
            if "teacher" not in resource:
                raise ValueError("Managed Teacher requires a teacher GPU resource budget")
            routes = (
                json.loads(args.opd_teacher_routes) if getattr(args, "opd_teacher_routes", None) else {"default": None}
            )
            total = resource["teacher"][1]
            if not routes or total % len(routes):
                raise ValueError("Teacher GPUs must divide evenly across teacher models")
            width = getattr(args, "teacher_num_gpus_per_engine", None)
            if width is None:
                width = total // len(routes)
            validate_engine_config(
                width,
                {
                    key[len("teacher_sglang_") :]: value
                    for key, value in vars(args).items()
                    if key.startswith("teacher_sglang_")
                },
            )
            specs.extend(("teacher", name, total // len(routes), width, teacher_defer) for name in routes)
        if genrm_defer and colocate and getattr(args, "dynamic_sampling_filter_path", None):
            raise ValueError("Deferred GenRM cannot filter samples using rewards before its scoring phase")
        allocations = []
        offsets: dict[str, int] = {}
        shared_cursor = 0
        for role, name, count, width, deferred in specs:
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (count, width, node_width)
            ):
                raise ValueError(f"{role}/{name}: GPU budgets and engine widths must be positive integers")
            if count % width or (width > node_width and width % node_width):
                raise ValueError(f"{role}/{name}: incomplete replica or invalid multi-node GPU width")
            phase = role if deferred else "inference"
            pool = "actor" if colocate else role
            if colocate and not deferred:
                offset = shared_cursor
                shared_cursor += count
            else:
                offset = offsets.get(role, 0)
                offsets[role] = offset + count
            if offset % min(width, node_width):
                raise ValueError(f"{role}/{name}: engine slice is not aligned to a local GPU span")
            local_width = min(width, node_width)
            if any(
                start % node_width + local_width > node_width for start in range(offset, offset + count, local_width)
            ):
                raise ValueError(f"{role}/{name}: local engine slice crosses a node boundary")
            if colocate and offset + count > resource["actor"][1]:
                raise ValueError("Inference split exceeds Actor GPU capacity; same-phase co-residency is unsupported")
            owner = "controller" if colocate else ("manager" if role == "teacher" else "service")
            allocations.append(ModelPlacement(role, name, pool, offset, count, width, owner, phase))
        for i, left in enumerate(allocations):
            for right in allocations[i + 1 :]:
                if left.pool == right.pool and left.phase == right.phase:
                    if max(left.offset, right.offset) < min(
                        left.offset + left.num_gpus, right.offset + right.num_gpus
                    ):
                        raise ValueError("Same-phase GPU co-residency is unsupported")
        if colocate and not (getattr(args, "debug_train_only", False) or getattr(args, "debug_rollout_only", False)):
            if not getattr(args, "offload_rollout", False) or not getattr(args, "offload_train", False):
                raise ValueError("Shared Actor/inference GPUs require train and rollout offload")
        if colocate and any(item.phase != "inference" for item in allocations):
            if getattr(args, "use_agentic_rollout", False) or getattr(args, "partial_rollout", False):
                raise ValueError("Deferred inference currently requires complete SGLang rollout batches")
            path = getattr(args, "rollout_function_path", "relax.engine.rollout.sglang_rollout.generate_rollout")
            if path != "relax.engine.rollout.sglang_rollout.generate_rollout":
                raise ValueError("Deferred inference requires the framework SGLang rollout pipeline")
        return tuple(allocations)

    @staticmethod
    def apply(args: Any) -> tuple[ModelPlacement, ...]:
        plan = PlacementPlanner.resolve(args)
        args._inference_placement = plan
        # Offloading CUDA memory does not release Ray actor resource claims.
        # Reserve enough scheduling capacity for every phase's resident actor,
        # including a colocated PPO critic, before launching any of them.
        node_width = getattr(args, "num_gpus_per_node", None) or getattr(args, "actor_num_gpus_per_node", 1)
        claims: dict[int, int] = {}
        for item in plan:
            if item.pool == "actor":
                for start in range(item.offset, item.offset + item.num_gpus, min(node_width, item.gpus_per_engine)):
                    claims[start] = claims.get(start, 0) + 1
        resource = getattr(args, "resource", {}) or {}
        shared_critic = getattr(args, "advantage_estimator", None) == "ppo" and resource.get("critic") == resource.get(
            "actor"
        )
        available = 0.2 if shared_critic else 0.6
        fraction = min(0.2, int(available / max(claims.values(), default=1) * 10000) / 10000)
        args._inference_ray_gpu_fraction = fraction
        custom_genrm = getattr(args, "genrm_ray_num_gpus", fraction)
        if any(item.role == "genrm" and item.pool == "actor" for item in plan) and custom_genrm > fraction:
            raise ValueError("GenRM Ray GPU reservation exceeds the capacity left by shared phase actors")
        teacher = [p for p in plan if p.role == "teacher"]
        genrm = [p for p in plan if p.role == "genrm"]
        if teacher:
            args._teacher_bundle_start = min(p.offset for p in teacher)
        if genrm:
            args._genrm_bundle_start = min(p.offset for p in genrm)
            args._genrm_colocate_with_rollout = any(p.pool == "actor" and p.phase == "genrm" for p in genrm)
        return plan


def validate_bundle_span(bundle_indices: list, gpu_ids: list, node_ids: dict, start: int, count: int) -> None:
    """Validate actual local topology, not merely adjacent logical bundles."""
    if start < 0 or start + count > len(bundle_indices) or len(gpu_ids) != len(bundle_indices):
        raise ValueError("Inference GPU span is outside its placement group")
    indices = bundle_indices[start : start + count]
    nodes = [node_ids.get(i, node_ids.get(str(i))) for i in indices]
    if any(not node for node in nodes) or len(set(nodes)) != 1:
        raise ValueError("A local inference worker's GPU span crosses node boundaries")
    devices = [int(i) for i in gpu_ids[start : start + count]]
    if devices != list(range(devices[0], devices[0] + count)):
        raise ValueError("Inference worker requires contiguous physical GPU IDs on its node")


def validate_pg_span(pg_tuple: tuple, start: int, count: int) -> None:
    import ray

    pg, bundles, devices = pg_tuple
    table = ray.util.placement_group_table(pg)
    validate_bundle_span(bundles, devices, table["bundles_to_node_id"], start, count)

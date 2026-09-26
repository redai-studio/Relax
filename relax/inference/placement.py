# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class InferencePlacement:
    role: str
    model_id: str
    num_gpus: int
    gpus_per_engine: int
    bundle_start: int
    pool: str
    owner: str
    phase: str
    deferred: bool = False
    num_gpus_per_node: int = 8
    tp_size: int = 1
    pp_size: int = 1
    group_id: str = "default"
    worker_type: str = "regular"

    def __post_init__(self) -> None:
        for key in ("num_gpus", "gpus_per_engine", "num_gpus_per_node", "tp_size", "pp_size"):
            _positive(getattr(self, key), key)
        if self.num_gpus % self.gpus_per_engine:
            raise ValueError("Placement GPU budget must divide by GPUs per engine")
        if isinstance(self.bundle_start, bool) or not isinstance(self.bundle_start, int) or self.bundle_start < 0:
            raise ValueError("bundle_start must be a nonnegative integer")

    @property
    def replicas(self) -> int:
        return 0 if self.worker_type == "placeholder" else self.num_gpus // self.gpus_per_engine

    @property
    def nodes_per_engine(self) -> int:
        return max(1, self.gpus_per_engine // self.num_gpus_per_node)

    @property
    def num_worker_slots(self) -> int:
        return self.replicas * self.nodes_per_engine

    @property
    def bundle_stop(self) -> int:
        return self.bundle_start + self.num_gpus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferencePlacementPlan:
    mode: str
    placements: tuple[InferencePlacement, ...]
    total_required_gpus: int
    pool_sizes: Mapping[str, int]
    training_roles: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "pool_sizes", MappingProxyType(dict(self.pool_sizes)))
        object.__setattr__(self, "training_roles", MappingProxyType(dict(self.training_roles)))

    @property
    def total_required(self) -> int:
        return self.total_required_gpus

    def for_role(self, role: str) -> tuple[InferencePlacement, ...]:
        return tuple(placement for placement in self.placements if placement.role == role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "placements": [placement.to_dict() for placement in self.placements],
            "total_required_gpus": self.total_required_gpus,
            "pool_sizes": dict(self.pool_sizes),
            "training_roles": dict(self.training_roles),
        }


def model_placement(args: Any, role: str, model_id: str = "__default__") -> InferencePlacement | None:
    plan = getattr(args, "_inference_placement_plan", None)
    if plan is None:
        return None
    placements = (
        plan.for_role(role)
        if isinstance(plan, InferencePlacementPlan)
        else tuple(InferencePlacement(**item) for item in plan["placements"] if item["role"] == role)
    )
    matches = tuple(placement for placement in placements if placement.model_id == model_id)
    if not matches and model_id == "__default__" and len(placements) == 1:
        return placements[0]
    if len(matches) > 1:
        raise ValueError(f"{role}/{model_id} has multiple engine groups; use plan.for_role()")
    return matches[0] if matches else None


def _resource(args: Any) -> dict[str, tuple[int, int]]:
    result = {}
    for key, entry in (getattr(args, "resource", None) or {}).items():
        role = str(key)
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"resource[{role!r}] must be [replicas, GPU count]")
        replicas = _positive(entry[0], f"resource[{role}].replicas")
        count = entry[1]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"resource[{role}].gpus must be a nonnegative integer")
        if replicas != 1:
            raise ValueError(f"Only one service replica is supported for role {role}")
        result[role] = (replicas, count)
    return result


def _managed_teacher(args: Any, resource: dict) -> bool:
    return bool(
        getattr(args, "use_opd", False)
        and getattr(args, "opd_type", None) == "sglang"
        and (
            getattr(args, "teacher_hf_checkpoint", None) is not None
            or getattr(args, "opd_teacher_routes", None) is not None
        )
        and "teacher" in resource
        and not getattr(args, "debug_train_only", False)
    )


def _rollout_groups(args: Any, total: int) -> list[tuple[str, str, str, int, int, dict]]:
    config = getattr(args, "_inference_rollout_config", None)
    if config is None:
        config = getattr(args, "sglang_config", None)
    default_engine = getattr(args, "rollout_num_gpus_per_engine", None) or total
    if config is None:
        prefill = getattr(args, "prefill_num_servers", 0) or 0
        if prefill:
            prefill_gpus = _positive(prefill, "prefill_num_servers") * default_engine
            return [
                ("default", "0", "prefill", prefill_gpus, default_engine, {}),
                ("default", "1", "decode", total - prefill_gpus, default_engine, {}),
            ]
        return [("default", "default", "regular", total, default_engine, {})]
    if not isinstance(config, Mapping) or not isinstance(config.get("sglang"), list):
        raise ValueError("Preflight requires parsed sglang config in _inference_rollout_config before allocation")
    groups, names = [], set()
    for model in config["sglang"]:
        name = model.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Rollout model names must be nonempty and unique")
        names.add(name)
        entries = model.get("engine_groups") or []
        if not entries:
            raise ValueError(f"Rollout model {name} has no engine groups")
        workers = {entry.get("worker_type", "regular") for entry in entries}
        if bool("prefill" in workers) != bool("decode" in workers):
            raise ValueError(f"Rollout model {name} requires both prefill and decode groups")
        if "regular" in workers and workers.intersection({"prefill", "decode"}):
            raise ValueError("A model cannot mix regular and PD groups")
        for index, entry in enumerate(entries):
            groups.append(
                (
                    name,
                    str(index),
                    entry.get("worker_type", "regular"),
                    entry.get("num_gpus"),
                    entry.get("num_gpus_per_engine") or model.get("num_gpus_per_engine") or default_engine,
                    dict(entry.get("overrides") or {}),
                )
            )
    if sum(group[3] for group in groups) != total:
        raise ValueError("Rollout engine group GPU total must match rollout budget")
    return groups


def _genrm_models(args: Any) -> dict[str, dict]:
    normalized = getattr(args, "_genrm_instances_resolved", None)
    if normalized is not None:
        return dict(normalized)
    if getattr(args, "genrm_instances", None) is not None:
        return {
            key: {
                **spec,
                "num_gpus_per_engine": spec.get("num_gpus_per_engine")
                or getattr(args, "genrm_num_gpus_per_engine", None),
                "engine_config": spec.get("engine_config", getattr(args, "genrm_engine_config", None)) or {},
            }
            for key, spec in args.genrm_instances.items()
        }
    if getattr(args, "genrm_model_path", None) is None:
        return {}
    return {
        "__default__": {
            "model_path": args.genrm_model_path,
            "num_gpus": args.genrm_num_gpus,
            "num_gpus_per_engine": args.genrm_num_gpus_per_engine,
            "engine_config": getattr(args, "genrm_engine_config", None) or {},
        }
    }


def _validate_persistent_ray_reservations(
    args: Any, placements: Sequence[InferencePlacement], training: Mapping[str, int], *, colocate: bool
) -> None:
    reservations: dict[tuple[str, int], list[tuple[str, float]]] = {}

    def reserve(pool: str, index: int, name: str, fraction: float) -> None:
        reservations.setdefault((pool, index), []).append((name, fraction))

    actor_gpus = training.get("actor", 0)
    for role, count in training.items():
        if role not in {"actor", "critic", "actor_fwd", "reference"}:
            continue
        shared = colocate and (role == "actor" or (role == "critic" and count == actor_gpus))
        for index in range(count):
            reserve("actor" if shared else role, index, role, 0.4)
    for placement in placements:
        if placement.worker_type == "placeholder":
            continue
        fraction = 0.2
        if placement.role == "genrm":
            default = 0.1 if getattr(args, "_genrm_colocate_with_rollout", False) else 0.2
            fraction = getattr(args, "genrm_ray_num_gpus", default)
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
                raise ValueError("genrm_ray_num_gpus must be a finite nonnegative number")
            if fraction < 0:
                raise ValueError("genrm_ray_num_gpus must be nonnegative")
        local_gpus = min(placement.gpus_per_engine, placement.num_gpus_per_node)
        for index in range(placement.bundle_start, placement.bundle_stop, local_gpus):
            reserve(placement.pool, index, f"{placement.role}/{placement.model_id}", fraction)
    for (pool, index), claims in reservations.items():
        total = sum(fraction for _name, fraction in claims)
        if total > 1.0 + 1e-9:
            details = ", ".join(f"{name}={fraction:g}" for name, fraction in claims)
            raise ValueError(
                f"Persistent Ray GPU reservations exceed bundle {pool}[{index}]: {total:g} > 1 ({details}). "
                "Offloading model memory does not release Ray actor GPU reservations"
            )


def plan_inference_placement(args: Any) -> InferencePlacementPlan:
    resource = _resource(args)
    per_node = _positive(getattr(args, "num_gpus_per_node", 8), "num_gpus_per_node")
    hybrid = bool(getattr(args, "hybrid", False))
    colocate = bool(getattr(args, "colocate", False)) and not hybrid
    if colocate and getattr(args, "fully_async", False):
        raise ValueError("fully_async+colocate requires the existing hybrid mode")
    if getattr(args, "rollout_external", False) and colocate:
        raise ValueError("External Rollout cannot participate in shared GPU placement")
    teacher_enabled = _managed_teacher(args, resource)
    genrm_models = _genrm_models(args) if not getattr(args, "debug_train_only", False) else {}
    if getattr(args, "rollout_external", False) and genrm_models:
        raise ValueError("Legacy rollout_external cannot express ownership of managed static models")
    training = {
        role: count for role, (_replicas, count) in resource.items() if role not in {"rollout", "genrm", "teacher"}
    }
    if getattr(args, "debug_rollout_only", False):
        training = {}
        colocate = False
    actor_gpus = training.get("actor", 0)
    if colocate and actor_gpus <= 0:
        raise ValueError("Shared inference placement requires a positive Actor GPU pool")
    deferred_roles = set(getattr(args, "inference_defer_roles", ()) or ())
    if getattr(args, "defer_reward_to_post_process", False):
        deferred_roles.add("genrm")
    if deferred_roles - {"genrm", "teacher"}:
        raise ValueError("Only GenRM and Teacher can be deferred")
    if "teacher" in deferred_roles and not teacher_enabled:
        raise ValueError("Deferred Teacher requires a managed Teacher model")
    if deferred_roles and not colocate:
        raise ValueError("Deferred inference requires shared synchronous placement")
    if "genrm" in deferred_roles and not genrm_models:
        raise ValueError("Deferred GenRM requested without a GenRM model")
    if getattr(args, "_genrm_colocate_with_rollout", False) and "genrm" not in deferred_roles:
        raise ValueError("Same-phase Rollout/GenRM residency on shared GPUs is unsupported; enable explicit defer")
    if (
        colocate
        and not getattr(args, "debug_train_only", False)
        and (getattr(args, "offload_train", None) is False or getattr(args, "offload_rollout", None) is False)
    ):
        raise ValueError("Shared GPU inference requires training and inference offload")
    raw = []
    if "rollout" in resource and not getattr(args, "debug_train_only", False):
        total = _positive(resource["rollout"][1], "rollout GPU budget")
        configured = getattr(args, "rollout_num_gpus", total)
        if configured != total:
            raise ValueError("rollout_num_gpus must equal resource rollout budget")
        for model_id, group_id, worker_type, count, engine_gpus, overrides in _rollout_groups(args, total):
            raw.append(("rollout", model_id, group_id, worker_type, count, engine_gpus, overrides))
    if genrm_models:
        if "genrm" not in resource or sum(spec["num_gpus"] for spec in genrm_models.values()) != resource["genrm"][1]:
            raise ValueError("GenRM resource budget must equal all named model budgets")
        for name, spec in genrm_models.items():
            if not isinstance(name, str) or not name:
                raise ValueError("GenRM model IDs must be nonempty strings")
            raw.append(
                (
                    "genrm",
                    name,
                    "default",
                    "regular",
                    spec["num_gpus"],
                    spec["num_gpus_per_engine"],
                    dict(spec.get("engine_config") or {}),
                )
            )
    if teacher_enabled:
        total = _positive(resource["teacher"][1], "Teacher GPU budget")
        routes = getattr(args, "opd_teacher_routes", None)
        routes = json.loads(routes) if isinstance(routes, str) else routes
        routes = routes if routes is not None else {"__default__": args.teacher_hf_checkpoint}
        if not isinstance(routes, Mapping) or not routes or any(not isinstance(key, str) or not key for key in routes):
            raise ValueError("Teacher routes must be a nonempty model mapping")
        if total % len(routes):
            raise ValueError("Teacher GPU budget must divide evenly across models")
        each = total // len(routes)
        per_engine = getattr(args, "teacher_num_gpus_per_engine", None) or each
        overrides = {
            key[len("teacher_sglang_") :]: value
            for key, value in vars(args).items()
            if key.startswith("teacher_sglang_")
        }
        for name in routes:
            raw.append(("teacher", name, "default", "regular", each, per_engine, overrides))
    placements, cursors = [], {}
    shared_cursor = 0
    for role, model_id, group_id, worker_type, count, engine_gpus, overrides in raw:
        count = _positive(count, f"{role}/{model_id} GPU budget")
        engine_gpus = _positive(engine_gpus, f"{role}/{model_id} GPUs per engine")
        if count % engine_gpus:
            raise ValueError(f"{role}/{model_id} GPU budget must divide by GPUs per engine")
        if engine_gpus > per_node and engine_gpus % per_node:
            raise ValueError(f"{role}/{model_id} multi-node engine must occupy whole configured nodes")
        if worker_type not in {"regular", "prefill", "decode", "placeholder"}:
            raise ValueError("Unsupported inference worker type")
        default_pp = getattr(args, "sglang_pp_size", 1) if role == "rollout" else 1
        pp = _positive(overrides.get("pp_size", default_pp), f"{role}/{model_id} PP")
        default_tp = engine_gpus if role == "genrm" else engine_gpus // pp
        tp = _positive(overrides.get("tp_size", default_tp), f"{role}/{model_id} TP")
        if tp * pp != engine_gpus:
            raise ValueError(f"{role}/{model_id} TP x PP must equal GPUs per engine")
        dp = _positive(
            overrides.get("dp_size", getattr(args, "sglang_dp_size", 1) if role == "rollout" else 1),
            f"{role}/{model_id} DP",
        )
        if overrides.get("enable_dp_attention", getattr(args, "sglang_enable_dp_attention", False)) and tp % dp:
            raise ValueError(f"{role}/{model_id} DP attention requires TP divisible by DP")
        deferred = role in deferred_roles
        pool = "actor" if colocate else role
        if colocate and not deferred:
            start = shared_cursor
            shared_cursor += count
        else:
            cursor_key = f"defer:{role}" if deferred else pool
            start = cursors.get(cursor_key, 0)
            cursors[cursor_key] = start + count
        if colocate and start + count > actor_gpus:
            raise ValueError(f"Combined inference layout exceeds Actor pool: {start + count} > {actor_gpus}")
        placements.append(
            InferencePlacement(
                role,
                model_id,
                count,
                engine_gpus,
                start,
                pool,
                "controller" if colocate else role,
                role + "_score" if deferred else "inference",
                deferred,
                per_node,
                tp,
                pp,
                group_id,
                worker_type,
            )
        )
    pools = {}
    if colocate:
        pools["actor"] = actor_gpus
    for role, count in training.items():
        if colocate and (role == "actor" or (role == "critic" and count == actor_gpus)):
            continue
        pools[role] = count
    for placement in placements:
        pools[placement.pool] = max(pools.get(placement.pool, 0), placement.bundle_stop)
    mode = "defer" if deferred_roles else "split" if colocate else "decoupled"
    _validate_persistent_ray_reservations(args, placements, training, colocate=colocate)
    return InferencePlacementPlan(mode, tuple(placements), sum(pools.values()), pools, training)


def validate_bound_placement(
    placement: InferencePlacement | Mapping[str, Any],
    topology: Sequence[Mapping[str, Any]],
    *,
    bundle_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    placement = placement if isinstance(placement, InferencePlacement) else InferencePlacement(**placement)
    by_index = {}
    for item in topology:
        index = item["bundle_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Physical bundle index must be a nonnegative integer")
        if index in by_index:
            raise ValueError("Duplicate physical bundle index")
        by_index[index] = dict(item)
    ordered_indices = list(bundle_indices) if bundle_indices is not None else sorted(by_index)
    if len(set(ordered_indices)) != len(ordered_indices) or any(index not in by_index for index in ordered_indices):
        raise ValueError("Bundle order contains duplicates or unknown indices")
    if placement.bundle_start < 0 or placement.bundle_stop > len(ordered_indices):
        raise ValueError("Planned inference slice exceeds bound placement group")
    selected = tuple(by_index[index] for index in ordered_indices[placement.bundle_start : placement.bundle_stop])
    identities = []
    for item in selected:
        node = item.get("node_id") or item.get("node_ip")
        if not isinstance(node, str) or not node:
            raise ValueError("Every GPU binding requires node identity")
        gpu = item.get("gpu_id")
        if isinstance(gpu, bool) or not str(gpu).isdigit():
            raise ValueError("GPU binding requires a nonnegative device index")
        identities.append((node, int(gpu)))
    if len(set(identities)) != len(identities):
        raise ValueError("Two inference bundles resolve to the same physical GPU")
    local = min(placement.gpus_per_engine, placement.num_gpus_per_node)
    for replica_start in range(0, placement.num_gpus, placement.gpus_per_engine):
        nodes = []
        for offset in range(0, placement.gpus_per_engine, local):
            chunk = identities[replica_start + offset : replica_start + offset + local]
            if len({node for node, _gpu in chunk}) != 1:
                raise ValueError("An inference worker must bind contiguous GPUs on one node")
            gpu_ids = [gpu for _node, gpu in chunk]
            if gpu_ids != list(range(gpu_ids[0], gpu_ids[0] + local)):
                raise ValueError("An inference worker requires ascending contiguous local GPU indices")
            nodes.append(chunk[0][0])
        if len(set(nodes)) != placement.nodes_per_engine:
            raise ValueError("A multi-node logical replica must span distinct nodes")
    return selected

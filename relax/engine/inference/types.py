# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable discovery observations; snapshots do not grant request
admission."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(str, Enum):
    ROLLOUT = "rollout"
    GENRM = "genrm"
    TEACHER = "teacher"


class WorkloadType(str, Enum):
    # Policy generation whose samples feed training.
    ROLLOUT = "rollout"
    # Reward scoring of finished samples (GenRM).
    REWARD = "reward"
    # Teacher log-probabilities for on-policy distillation.
    DISTILLATION = "distillation"


WORKLOAD_BY_ROLE = {
    Role.ROLLOUT: WorkloadType.ROLLOUT,
    Role.GENRM: WorkloadType.REWARD,
    Role.TEACHER: WorkloadType.DISTILLATION,
}


class DeploymentMode(str, Enum):
    # The role's engines run in a placement group of their own.
    DECOUPLED = "decoupled"
    # A slice of the shared actor placement group that no other role uses.
    SPLIT = "split"
    # Bundles shared with generation, used in a phase of its own.
    DEFER = "defer"


class WeightSource(str, Enum):
    # Loaded once from a checkpoint; never registers for dynamic updates.
    STATIC = "static"
    # Policy weights pushed by the trainer through DCS or its colocated path.
    DCS = "dcs"


class RouteMode(str, Enum):
    # Requests go through the model's SGLang Router.
    SGLANG_ROUTER = "sglang_router"
    # Requests go straight to a READY, direct-eligible replica.
    DIRECT = "direct"


class LifecycleState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    SLEEPING = "sleeping"
    ONLOADING = "onloading"
    DEAD = "dead"


@dataclass(frozen=True)
class ReplicaSnapshot:
    # Stable logical replica identity; it survives replacement of the process.
    engine_id: str
    state: LifecycleState
    base_url: str | None = None
    weight_version: str | None = None
    # Whether a client may send requests straight to this replica. Only a READY
    # replica of a direct-routed model is eligible; the Manager enforces it.
    direct_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the wire representation used by v2 discovery."""
        return {
            "engine_id": self.engine_id,
            "base_url": self.base_url,
            "state": self.state.value,
            "weight_version": self.weight_version,
            "direct_eligible": self.direct_eligible,
        }


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    replicas: tuple[ReplicaSnapshot, ...] = ()
    router_url: str | None = None
    state: LifecycleState | None = None
    admission: bool = False
    required_weight_version: str | None = None
    # PD prefill/decode workers as ``(worker_type, worker)``. Diagnostic only:
    # requests use the Router, so these are never replicas or direct targets.
    pd_workers: tuple[tuple[str, ReplicaSnapshot], ...] = ()

    def to_dict(self, status_filter: str | None = None) -> dict[str, Any]:
        if status_filter not in (None, "active", "dead"):
            raise ValueError("status_filter must be one of: active, dead")

        def keep(replica: ReplicaSnapshot) -> bool:
            return status_filter is None or status_filter == (
                "dead" if replica.state == LifecycleState.DEAD else "active"
            )

        replicas = [replica for replica in self.replicas if keep(replica)]
        return {
            "state": self.state.value if self.state is not None else None,
            "admission": self.admission,
            "router_url": self.router_url,
            "required_weight_version": self.required_weight_version,
            "engines": [replica.to_dict() for replica in replicas],
            "pd_workers": [
                {"worker_type": worker_type, **worker.to_dict()}
                for worker_type, worker in self.pd_workers
                if keep(worker)
            ],
        }


@dataclass(frozen=True)
class RoutingSpec:
    default_model: str | None = None
    route_key_to_model: tuple[tuple[str, str], ...] = ()
    config_version: int = 0

    def __post_init__(self) -> None:
        keys = [key for key, _ in self.route_key_to_model]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate route keys are not allowed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_model": self.default_model,
            "route_key_to_model": dict(self.route_key_to_model),
            "config_version": self.config_version,
        }


@dataclass(frozen=True)
class RoleSnapshot:
    role: Role
    # Stable identifier for one Manager/control-plane lifetime. A rebuild or
    # restart creates a new epoch; snapshots from different epochs are never
    # comparable, even when their numeric revisions happen to match.
    manager_epoch: str
    topology_revision: int = 0
    # Current phase reported by this role's own Manager. This is descriptive
    # only; it does not grant resource ownership or request admission.
    phase: str | None = None
    models: tuple[ModelSnapshot, ...] = ()
    routing: RoutingSpec = RoutingSpec()

    def __post_init__(self) -> None:
        if not self.manager_epoch:
            raise ValueError("Discovery requires a non-empty manager epoch")
        if self.topology_revision < 0:
            raise ValueError("Discovery topology revision must be non-negative")
        names = [model.model_id for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate model IDs are not allowed")

    def to_dict(self, status_filter: str | None = None) -> dict[str, Any]:
        """Serialize a complete, JSON-compatible v2 discovery snapshot."""
        if status_filter not in (None, "active", "dead"):
            raise ValueError("status_filter must be one of: active, dead")
        return {
            "schema_version": 2,
            "role": self.role.value,
            "manager_epoch": self.manager_epoch,
            "topology_revision": self.topology_revision,
            "phase": self.phase,
            "routing": self.routing.to_dict(),
            "models": {model.model_id: model.to_dict(status_filter) for model in self.models},
        }

    def to_legacy_dict(self, status_filter: str | None = None) -> dict[str, Any]:
        """Project discovery to the legacy rollout ``/engines`` shape.

        This exposes endpoint liveness only; ``active`` does not imply READY.
        """
        if status_filter not in (None, "active", "dead"):
            raise ValueError("status_filter must be one of: active, dead")
        models: dict[str, dict[str, Any]] = {}
        total_engines = 0
        for model in self.models:
            engines = []
            for rank, replica in enumerate(model.replicas):
                alive = replica.state != LifecycleState.DEAD
                if status_filter is not None and status_filter != ("active" if alive else "dead"):
                    continue
                engine = {"rank": rank, "status": "active" if alive else "dead"}
                if replica.base_url is not None:
                    engine["url"] = replica.base_url
                engines.append(engine)
            groups = []
            if engines:
                groups.append(
                    {
                        "group_index": 0,
                        "num_gpus_per_engine": None,
                        "num_new_engines": 0,
                        "engines": engines,
                    }
                )
            models[model.model_id] = {
                "router_ip": None,
                "router_port": None,
                "engine_groups": groups,
                "total_engines": len(engines),
            }
            total_engines += len(engines)
        return {"models": models, "total_engines": total_engines}


@dataclass(frozen=True)
class RouteTarget:
    """A routing candidate; a Gateway must still acquire a Manager permit."""

    model_id: str
    base_url: str

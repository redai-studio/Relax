# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared deployment configuration, independent of Ray and process creation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from relax.engine.inference.types import DeploymentMode, Role, RouteMode, RoutingSpec, WeightSource, WorkloadType


@dataclass(frozen=True)
class ReplicaSpec:
    replica_id: str
    node_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.replica_id or not self.node_ranks:
            raise ValueError("Replica identity and node ranks are required")
        if min(self.node_ranks) < 0 or len(set(self.node_ranks)) != len(self.node_ranks):
            raise ValueError("Node ranks must be unique and non-negative")


def replicas_from_slots(
    prefix: str, num_slots: int, nodes_per_engine: int, *, first_slot: int = 0
) -> tuple[ReplicaSpec, ...]:
    """Name the replicas of one engine group.

    ``first_slot`` offsets the identity, not the node ranks: a group that
    starts at engine slot N names its replicas from N while its ranks stay
    local to the group. That keeps an identity stable for the life of the
    replica even when another group is added or removed around it.
    """
    if num_slots < 0 or nodes_per_engine < 1 or num_slots % nodes_per_engine:
        raise ValueError("Engine slots must contain complete logical replicas")
    return tuple(
        ReplicaSpec(f"{prefix}/replica-{first_slot + head}", tuple(range(head, head + nodes_per_engine)))
        for head in range(0, num_slots, nodes_per_engine)
    )


@dataclass(frozen=True)
class EngineGroupTopology:
    """Runtime topology of an engine group; deployment parameters live in
    EngineGroupSpec."""

    group_id: str
    replicas: tuple[ReplicaSpec, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("Engine group identity is required")
        ranks = [rank for replica in self.replicas for rank in replica.node_ranks]
        if len(set(ranks)) != len(ranks):
            raise ValueError("Replicas cannot share node actor ranks within an engine group")


def validate_routes(routing: RoutingSpec, model_names: set[str]) -> None:
    targets = {model for _, model in routing.route_key_to_model}
    if routing.default_model is not None:
        targets.add(routing.default_model)
    if not targets <= model_names:
        raise ValueError(f"Routing references unregistered models: {sorted(targets - model_names)}")


@dataclass
class EngineGroupSpec:
    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    topology: EngineGroupTopology | None = None

    def __post_init__(self) -> None:
        valid_types = {"regular", "prefill", "decode", "placeholder"}
        assert self.worker_type in valid_types, (
            f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        )
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"


@dataclass
class InferenceModelSpec:
    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    engine_groups: list[EngineGroupSpec] = field(default_factory=list)
    weight_source: WeightSource = WeightSource.DCS
    route_mode: RouteMode = RouteMode.SGLANG_ROUTER
    # Request defaults a role applies when it builds an engine payload itself
    # (GenRM); a caller's own sampling parameters still win.
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Environment of every engine process of this model, so a static model never
    # inherits rollout-only settings from the global process environment.
    env_vars: dict[str, str] = field(default_factory=dict)
    # Whether the pool accepts scale-out/scale-in, and whether dead engines are
    # detected and rebuilt.
    elastic_enabled: bool = False
    fault_tolerance_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model identity is required")
        self.weight_source = WeightSource(self.weight_source)
        self.route_mode = RouteMode(self.route_mode)
        topologies = [group.topology for group in self.engine_groups if group.topology is not None]
        groups = [group.group_id for group in topologies]
        replicas = [replica.replica_id for group in topologies for replica in group.replicas]
        if len(set(groups)) != len(groups) or len(set(replicas)) != len(replicas):
            raise ValueError("Group and replica identities must be unique within a model")

    @property
    def model_id(self) -> str:
        return self.name

    @property
    def needs_weight_update(self) -> bool:
        return self.weight_source != WeightSource.STATIC

    @property
    def needs_dcs(self) -> bool:
        return self.weight_source == WeightSource.DCS

    @property
    def needs_router(self) -> bool:
        return self.route_mode == RouteMode.SGLANG_ROUTER

    def resolve(self, args: Any) -> None:
        """Resolve launch defaults in place (legacy API)."""
        default_gpus = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        self.model_path = self.model_path or args.sglang_hf_checkpoint or args.hf_checkpoint
        if not self.model_path:
            raise ValueError("Model checkpoint path is required")
        for group in self.engine_groups:
            if group.num_gpus_per_engine is None:
                group.num_gpus_per_engine = default_gpus
            group.overrides.setdefault("model_path", self.model_path)

    def resolved(self, args: Any) -> InferenceModelSpec:
        """Resolve once, then attach topology without a second configuration
        representation."""
        model = deepcopy(self)
        model.resolve(args)
        if args.num_gpus_per_node < 1:
            raise ValueError("GPUs per node must be positive")
        first_slot = 0
        for index, group in enumerate(model.engine_groups):
            group_id = f"{model.name}/group-{index}"
            gpus = group.num_gpus_per_engine
            placeholder = group.worker_type == "placeholder"
            if gpus < 1 or (not placeholder and group.num_gpus % gpus):
                raise ValueError("Engine group GPUs must contain complete replicas")
            if not placeholder and gpus > args.num_gpus_per_node and gpus % args.num_gpus_per_node:
                raise ValueError("Multi-node engines must occupy complete nodes")
            nodes = max(1, gpus // args.num_gpus_per_node)
            slots = group.num_gpus // gpus * nodes
            # Replicas are named after the model and their engine slot, which
            # is what discovery publishes, so the identity a group is created
            # with is the identity it keeps. A placeholder group names nothing
            # but still consumes its slot range, because the runtime advances
            # the engine offset over it too.
            replicas = replicas_from_slots(model.name, 0 if placeholder else slots, nodes, first_slot=first_slot)
            group.topology = EngineGroupTopology(group_id, replicas)
            first_slot += slots
        return model

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(group.worker_type in ("prefill", "decode") for group in self.engine_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(group.num_gpus for group in self.engine_groups)


@dataclass(frozen=True)
class DeploymentSpec:
    """How a role's engines share GPUs with training and other roles."""

    mode: DeploymentMode
    # The phase in which the role holds its GPUs.
    phase: str


@dataclass(frozen=True)
class InferenceRoleSpec:
    """One inference role of a task: its workload, deployment and models."""

    role: Role
    workload: WorkloadType
    deployment: DeploymentSpec
    routing: RoutingSpec
    models: tuple[InferenceModelSpec, ...]

    def __post_init__(self) -> None:
        names = [model.name for model in self.models]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Role {self.role.value} needs uniquely named models")
        validate_routes(self.routing, set(names))


@dataclass
class SglangConfig:
    models: list[InferenceModelSpec]

    @staticmethod
    def from_yaml(path: str) -> SglangConfig:
        import yaml

        with open(path) as stream:
            data = yaml.safe_load(stream)
        assert "sglang" in data, "sglang config must have a 'sglang' key"
        return SglangConfig(
            models=[
                InferenceModelSpec(
                    name=model["name"],
                    model_path=model.get("model_path"),
                    num_gpus_per_engine=model.get("num_gpus_per_engine"),
                    engine_groups=[EngineGroupSpec(**g) for g in model.get("engine_groups", [])],
                )
                for model in data["sglang"]
            ]
        )

    @staticmethod
    def from_prefill_num_servers(args: Any) -> SglangConfig:
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = args.rollout_num_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {args.rollout_num_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            [
                InferenceModelSpec(
                    "default",
                    engine_groups=[
                        EngineGroupSpec("prefill", prefill_gpus),
                        EngineGroupSpec("decode", decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(model.has_pd_disaggregation for model in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(model.total_num_gpus for model in self.models)

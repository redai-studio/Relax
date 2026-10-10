# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Topology snapshots of an inference role (rollout, GenRM, OPD teacher).

A snapshot answers "which models does this role serve, where, and can they
take a request right now". It is what a role's ``/engines?schema_version=2``
returns and what gateways and direct clients route on.

Only *logical* replicas appear in ``engines``: a TP/PP engine spanning several
nodes is one replica addressed through its head node, and PD prefill/decode
workers are listed separately as diagnostics because requests reach them only
through the router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 2


class EngineState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    SLEEPING = "sleeping"
    ONLOADING = "onloading"
    DEAD = "dead"


# A model with replicas in mixed states reports the first of these that any
# replica is in: one ready replica is enough to serve.
_MODEL_STATE_PRECEDENCE = (
    EngineState.READY,
    EngineState.ONLOADING,
    EngineState.STARTING,
    EngineState.DRAINING,
    EngineState.SLEEPING,
)


def format_base_url(host: str, port: int) -> str:
    host = host.strip("[]")
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


@dataclass(frozen=True)
class EngineSnapshot:
    engine_id: str
    base_url: str | None
    state: EngineState
    # True only when a client may send a request straight to ``base_url``.
    direct_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "base_url": self.base_url,
            "state": self.state.value,
            "direct_eligible": self.direct_eligible,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineSnapshot:
        return cls(
            engine_id=data["engine_id"],
            base_url=data.get("base_url"),
            state=EngineState(data["state"]),
            direct_eligible=bool(data.get("direct_eligible", False)),
        )


@dataclass(frozen=True)
class ModelSnapshot:
    name: str
    state: EngineState
    router_url: str | None = None
    engines: tuple[EngineSnapshot, ...] = ()
    # PD prefill/decode workers: reachable only through the router.
    diagnostic_workers: tuple[EngineSnapshot, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "router_url": self.router_url,
            "engines": [engine.to_dict() for engine in self.engines],
            "diagnostic_workers": [worker.to_dict() for worker in self.diagnostic_workers],
        }

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> ModelSnapshot:
        return cls(
            name=name,
            state=EngineState(data["state"]),
            router_url=data.get("router_url"),
            engines=tuple(EngineSnapshot.from_dict(item) for item in data.get("engines", ())),
            diagnostic_workers=tuple(EngineSnapshot.from_dict(item) for item in data.get("diagnostic_workers", ())),
        )


@dataclass(frozen=True)
class RoleSnapshot:
    role: str
    topology_revision: int
    models: tuple[ModelSnapshot, ...] = ()
    phase: str | None = None
    # Routing spec shared by the gateway and direct clients.
    default_model: str | None = None
    route_keys: Mapping[str, str] = field(default_factory=dict)

    def model(self, name: str) -> ModelSnapshot | None:
        for model in self.models:
            if model.name == name:
                return model
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "topology_revision": self.topology_revision,
            "phase": self.phase,
            "routing": {"default_model": self.default_model, "route_keys": dict(self.route_keys)},
            "models": {model.name: model.to_dict() for model in self.models},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleSnapshot:
        routing = data.get("routing") or {}
        return cls(
            role=data["role"],
            topology_revision=int(data["topology_revision"]),
            models=tuple(ModelSnapshot.from_dict(name, item) for name, item in data.get("models", {}).items()),
            phase=data.get("phase"),
            default_model=routing.get("default_model"),
            route_keys=dict(routing.get("route_keys") or {}),
        )


def aggregate_state(states: Iterable[EngineState]) -> EngineState:
    present = set(states)
    for state in _MODEL_STATE_PRECEDENCE:
        if state in present:
            return state
    return EngineState.DEAD


def build_model_snapshot(
    name: str,
    engines: Iterable[tuple[int, str | None, EngineState]],
    *,
    router_url: str | None = None,
    diagnostic_workers: Iterable[tuple[str, str | None, EngineState]] = (),
) -> ModelSnapshot:
    """Build a model snapshot from ``(replica index, base_url, state)`` rows.

    A replica is directly eligible only when it is ready and the model is not
    served through a router.
    """
    replicas = tuple(
        EngineSnapshot(
            engine_id=f"{name}/{index}",
            base_url=base_url,
            state=state,
            direct_eligible=state is EngineState.READY and router_url is None and base_url is not None,
        )
        for index, base_url, state in engines
    )
    worker_rows = tuple(diagnostic_workers)
    workers = tuple(
        EngineSnapshot(engine_id=f"{name}/{label}", base_url=base_url, state=state)
        for label, base_url, state in worker_rows
    )
    state = aggregate_state(replica.state for replica in replicas)
    if any(label.startswith(("prefill-", "decode-")) for label, _, _ in worker_rows):
        # A PD router needs both stages, even if regular replicas also exist.
        # Within each stage a healthy spare can replace a dead worker.
        stages = [
            aggregate_state(state for label, _, state in worker_rows if label.startswith(f"{stage}-"))
            for stage in ("prefill", "decode")
        ]
        if EngineState.DEAD in stages:
            state = EngineState.DEAD
        else:
            blocked = [stage for stage in stages if stage is not EngineState.READY]
            state = aggregate_state(blocked) if blocked else EngineState.READY
    return ModelSnapshot(
        name=name,
        state=state,
        router_url=router_url,
        engines=replicas,
        diagnostic_workers=workers,
    )


def model_snapshot_from_payload(name: str, payload: Mapping[str, Any]) -> ModelSnapshot:
    """Name a manager's ``get_inference_snapshot()`` payload.

    A manager does not know which model name its owner serves it under, so it
    reports replicas by index and the owner supplies the name.
    """
    return build_model_snapshot(
        name,
        [(row["index"], row.get("base_url"), EngineState(row["state"])) for row in payload.get("engines", ())],
        router_url=payload.get("router_url"),
        diagnostic_workers=[
            (row["label"], row.get("base_url"), EngineState(row["state"]))
            for row in payload.get("diagnostic_workers", ())
        ],
    )


def role_snapshot_from_payloads(
    role: str,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    phase: str | None = None,
    default_model: str | None = None,
    route_keys: Mapping[str, str] | None = None,
) -> RoleSnapshot:
    """Combine per-model manager payloads into a role snapshot.

    Each manager's revision is monotonic, so their sum is too, and it advances
    exactly when any model's topology changes.
    """
    return RoleSnapshot(
        role=role,
        topology_revision=sum(int(payload["topology_revision"]) for payload in payloads.values()),
        models=tuple(model_snapshot_from_payload(name, payload) for name, payload in payloads.items()),
        phase=phase,
        default_model=default_model,
        route_keys=dict(route_keys or {}),
    )


class TopologyRevision:
    """Monotonic counter that advances when the reachable topology changes.

    The signature covers router addresses, the replica set, replica addresses
    and each replica's incarnation (how many times it has been built), so a
    rebuild advances the revision even when the endpoint is preserved. Replica
    *state* is deliberately left out: sleeping and waking do not change where
    requests can go, only whether they can go there now.
    """

    def __init__(self) -> None:
        self._signature: Any = None
        self._revision = 0

    @property
    def value(self) -> int:
        return self._revision

    def observe(self, models: Iterable[ModelSnapshot], incarnations: Mapping[str, int] | None = None) -> int:
        incarnations = incarnations or {}
        signature = tuple(
            (
                model.name,
                model.router_url,
                tuple(
                    (engine.engine_id, engine.base_url, incarnations.get(engine.engine_id, 0))
                    for engine in model.engines
                ),
                tuple((worker.engine_id, worker.base_url) for worker in model.diagnostic_workers),
            )
            for model in models
        )
        if signature != self._signature:
            self._signature = signature
            self._revision += 1
        return self._revision

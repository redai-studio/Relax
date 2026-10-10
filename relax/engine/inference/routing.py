# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Routing rules shared by inference gateways and direct clients.

Two decisions, both pure functions of a topology snapshot:

1. Which model: explicit ``model`` -> ``route_key`` mapping -> default model.
2. Which replica: the model's router if it has one, otherwise one of its
   ready, directly eligible replicas.

Replica choice keeps its cursor and affinity memory in caller-owned
containers, so each caller decides the scope of its load-balancing state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, MutableMapping, Sequence, TypeVar

from relax.engine.inference.discovery import EngineSnapshot, EngineState, ModelSnapshot, RoleSnapshot


T = TypeVar("T")

# Bound on the affinity memory. It is a cache-locality heuristic, so clearing
# it wholesale only costs prefix reuse for groups still in flight.
DEFAULT_AFFINITY_MEMORY_CAP = 1 << 16


class RoutingError(Exception):
    status_code = 500


class UnknownModelError(RoutingError):
    """The request does not identify a model this role serves."""

    status_code = 400


class ModelUnavailableError(RoutingError):
    """The model exists but has nothing that can take a request right now."""

    status_code = 503


@dataclass
class RoutingState:
    """Load-balancing state of one gateway or client."""

    cursors: MutableMapping[str, int] = field(default_factory=dict)
    affinity_memory: MutableMapping[tuple[str, Any], int] = field(default_factory=dict)
    affinity_memory_cap: int = DEFAULT_AFFINITY_MEMORY_CAP


@dataclass(frozen=True)
class RouteTarget:
    model: str
    base_url: str
    # ``None`` when the request goes through the model's router.
    engine_id: str | None = None

    @property
    def via_router(self) -> bool:
        return self.engine_id is None


def select_model(snapshot: RoleSnapshot, model: str | None = None, route_key: str | None = None) -> ModelSnapshot:
    available = [item.name for item in snapshot.models]
    if model is not None:
        selected = snapshot.model(model)
        if selected is None:
            raise UnknownModelError(f"Unknown model {model!r} for role {snapshot.role!r}; available: {available}.")
        return selected

    if route_key is not None:
        selected = snapshot.model(snapshot.route_keys.get(route_key, route_key))
        if selected is None:
            raise UnknownModelError(
                f"No model registered for route_key {route_key!r} on role {snapshot.role!r}; available: {available}."
            )
        return selected

    if snapshot.default_model is not None:
        selected = snapshot.model(snapshot.default_model)
        if selected is not None:
            return selected
    if len(snapshot.models) == 1:
        return snapshot.models[0]
    raise UnknownModelError(
        f"Role {snapshot.role!r} serves several models; pass model or route_key. Available: {available}."
    )


def candidate_replicas(model: ModelSnapshot) -> tuple[EngineSnapshot, ...]:
    return tuple(engine for engine in model.engines if engine.state is EngineState.READY and engine.direct_eligible)


def select_replica(
    replicas: Sequence[T],
    *,
    cursor_key: str,
    cursors: MutableMapping[str, int],
    affinity_key: Hashable | None = None,
    affinity_memory: MutableMapping[tuple[str, Any], int] | None = None,
    affinity_memory_cap: int = DEFAULT_AFFINITY_MEMORY_CAP,
) -> T:
    """Pick one of ``replicas``.

    Without ``affinity_key`` the cursor advances on every call (round robin).
    With it, the cursor advances once per *new* key and the choice is
    remembered, so requests sharing a key (e.g. the samples of one GRPO group,
    which share a long prompt prefix) stay on one replica while distinct keys
    still spread evenly. Indexing by the key itself would collapse onto a
    subset of replicas whenever keys are not dense per ``cursor_key``.
    """
    if len(replicas) == 1:
        return replicas[0]

    if affinity_key is None or affinity_memory is None:
        index = cursors.get(cursor_key, 0)
        cursors[cursor_key] = index + 1
        return replicas[index % len(replicas)]

    memory_key = (cursor_key, affinity_key)
    remembered = affinity_memory.get(memory_key)
    if remembered is None:
        if len(affinity_memory) > affinity_memory_cap:
            affinity_memory.clear()
        remembered = cursors.get(cursor_key, 0)
        cursors[cursor_key] = remembered + 1
        affinity_memory[memory_key] = remembered
    return replicas[remembered % len(replicas)]


def select_target(
    snapshot: RoleSnapshot,
    state: RoutingState,
    *,
    model: str | None = None,
    route_key: str | None = None,
    affinity_key: Hashable | None = None,
) -> RouteTarget:
    selected = select_model(snapshot, model, route_key)
    if selected.router_url is not None:
        if selected.state is not EngineState.READY:
            raise ModelUnavailableError(f"Model {selected.name!r} is {selected.state.value}.")
        return RouteTarget(model=selected.name, base_url=selected.router_url)

    replicas = candidate_replicas(selected)
    if not replicas:
        raise ModelUnavailableError(
            f"Model {selected.name!r} is {selected.state.value}; no replica can take requests."
        )
    replica = select_replica(
        replicas,
        cursor_key=selected.name,
        cursors=state.cursors,
        affinity_key=affinity_key,
        affinity_memory=state.affinity_memory,
        affinity_memory_cap=state.affinity_memory_cap,
    )
    # A candidate replica is directly eligible, which implies it has an address.
    return RouteTarget(model=selected.name, base_url=str(replica.base_url), engine_id=replica.engine_id)

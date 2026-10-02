# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Parsing and a small HTTP client for the common inference discovery
contract."""

from typing import Any, Mapping
from uuid import uuid4

import httpx

from relax.engine.inference.routing import resolve_model, select_target
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RouteTarget,
    RoutingSpec,
)


def new_manager_epoch() -> str:
    """Create an opaque identifier for one Manager/control-plane lifetime."""
    return uuid4().hex


def _replica_from_dict(replica: Mapping[str, Any]) -> ReplicaSnapshot:
    return ReplicaSnapshot(
        engine_id=replica["engine_id"],
        base_url=replica.get("base_url"),
        state=LifecycleState(replica["state"]) if replica.get("state") else LifecycleState.STARTING,
        weight_version=replica.get("weight_version"),
        direct_eligible=bool(replica.get("direct_eligible", False)),
    )


def role_snapshot_from_dict(payload: Mapping[str, Any]) -> RoleSnapshot:
    """Parse a v2 JSON discovery response without trusting missing fields."""
    models = []
    for model_id, model_payload in dict(payload.get("models", {})).items():
        replicas = tuple(_replica_from_dict(replica) for replica in model_payload.get("engines", ()))
        pd_workers = tuple(
            (worker["worker_type"], _replica_from_dict(worker)) for worker in model_payload.get("pd_workers", ())
        )
        models.append(
            ModelSnapshot(
                model_id=model_id,
                replicas=replicas,
                router_url=model_payload.get("router_url"),
                state=LifecycleState(model_payload["state"]) if model_payload.get("state") else None,
                admission=bool(model_payload.get("admission", False)),
                required_weight_version=model_payload.get("required_weight_version"),
                pd_workers=pd_workers,
            )
        )
    routing_payload = payload.get("routing", {})
    return RoleSnapshot(
        role=Role(payload["role"]),
        manager_epoch=payload["manager_epoch"],
        topology_revision=int(payload.get("topology_revision", 0)),
        phase=payload.get("phase"),
        models=tuple(models),
        routing=RoutingSpec(
            default_model=routing_payload.get("default_model"),
            route_key_to_model=tuple(routing_payload.get("route_key_to_model", {}).items()),
            config_version=int(routing_payload.get("config_version", 0)),
        ),
    )


class InferenceDiscoveryClient:
    """Fetch discovery snapshots; request admission remains server-side."""

    def __init__(self, base_url: str, *, timeout: float = 10.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def get_snapshot(self, role: str, *, schema_version: int = 2, status_filter: str | None = None) -> RoleSnapshot:
        params: dict[str, Any] = {"schema_version": schema_version}
        if status_filter is not None:
            params["status_filter"] = status_filter
        response = self._client.get(f"{self.base_url}/{role}/engines", params=params)
        response.raise_for_status()
        return role_snapshot_from_dict(response.json())

    def resolve_model(
        self, snapshot: RoleSnapshot, *, model: str | None = None, route_key: str | None = None
    ) -> ModelSnapshot:
        return resolve_model(snapshot, model=model, route_key=route_key)

    def select_target(self, model: ModelSnapshot, *, cursor: int = 0) -> RouteTarget:
        return select_target(model, cursor=cursor)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "InferenceDiscoveryClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

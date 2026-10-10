# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from unittest.mock import Mock

import httpx

from relax.engine.inference.discovery import InferenceDiscoveryClient, role_snapshot_from_dict
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def _snapshot(*, router_url: str | None = None, topology_revision: int = 0, direct: bool = False) -> RoleSnapshot:
    replica = ReplicaSnapshot("teacher/replica-0", LifecycleState.READY, "http://teacher-0", direct_eligible=direct)
    return RoleSnapshot(
        role=Role.TEACHER,
        manager_epoch="epoch-a",
        topology_revision=topology_revision,
        models=(ModelSnapshot("teacher", (replica,), router_url, LifecycleState.READY, True),),
        routing=RoutingSpec(default_model="teacher"),
    )


def test_discovery_legacy_json_supports_status_filter() -> None:
    snapshot = RoleSnapshot(
        role=Role.ROLLOUT,
        manager_epoch="epoch-a",
        models=(
            ModelSnapshot(
                "student",
                (
                    ReplicaSnapshot("student/replica-0", LifecycleState.READY, "http://w0"),
                    ReplicaSnapshot("student/replica-1", LifecycleState.DEAD, "http://w1"),
                ),
            ),
        ),
    )
    active = snapshot.to_legacy_dict(status_filter="active")
    assert active["total_engines"] == 1
    assert "worker_type" not in active["models"]["student"]["engine_groups"][0]


def test_discovery_v2_round_trip_keeps_replica_direct_eligibility() -> None:
    snapshot = _snapshot(topology_revision=7, direct=True)
    payload = snapshot.to_dict()
    assert payload["models"]["teacher"]["engines"][0]["direct_eligible"] is True
    assert "direct_eligible" not in payload["models"]["teacher"]
    restored = role_snapshot_from_dict(payload)
    assert restored == snapshot
    assert restored.topology_revision == 7


def test_discovery_v2_round_trip_keeps_pd_workers_out_of_engines() -> None:
    snapshot = _snapshot(router_url="http://router")
    worker = ReplicaSnapshot("teacher/replica-1", LifecycleState.READY, "http://prefill")
    model = snapshot.models[0]
    snapshot = RoleSnapshot(
        role=snapshot.role,
        manager_epoch=snapshot.manager_epoch,
        models=(
            ModelSnapshot(
                model.model_id, model.replicas, model.router_url, model.state, True, pd_workers=(("prefill", worker),)
            ),
        ),
        routing=snapshot.routing,
    )
    payload = snapshot.to_dict()
    assert [engine["engine_id"] for engine in payload["models"]["teacher"]["engines"]] == ["teacher/replica-0"]
    assert payload["models"]["teacher"]["pd_workers"][0]["worker_type"] == "prefill"
    assert role_snapshot_from_dict(payload) == snapshot


def test_discovery_client_fetches_v2_snapshot() -> None:
    response = httpx.Response(
        200,
        json=_snapshot(router_url="http://router").to_dict(),
        request=httpx.Request("GET", "http://service/teacher/engines"),
    )
    transport = Mock()
    transport.get.return_value = response
    with InferenceDiscoveryClient("http://service", client=transport) as client:
        snapshot = client.get_snapshot("teacher")
        target = client.select_target(client.resolve_model(snapshot, model="teacher"))
    transport.get.assert_called_once_with("http://service/teacher/engines", params={"schema_version": 2})
    assert target.base_url == "http://router"


def test_discovery_client_selects_direct_replica_without_router() -> None:
    response = httpx.Response(
        200,
        json=_snapshot(direct=True).to_dict(),
        request=httpx.Request("GET", "http://service/teacher/engines"),
    )
    transport = Mock()
    transport.get.return_value = response
    with InferenceDiscoveryClient("http://service", client=transport) as client:
        snapshot = client.get_snapshot("teacher")
        target = client.select_target(client.resolve_model(snapshot))
    assert target.base_url == "http://teacher-0"


def test_discovery_defaults_topology_revision_for_payload_without_it() -> None:
    payload = _snapshot().to_dict()
    payload.pop("topology_revision")

    assert role_snapshot_from_dict(payload).topology_revision == 0

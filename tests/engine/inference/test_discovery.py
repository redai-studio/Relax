# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Topology snapshot structure and the revision counter."""

from relax.engine.inference.discovery import (
    SCHEMA_VERSION,
    EngineState,
    RoleSnapshot,
    TopologyRevision,
    aggregate_state,
    build_model_snapshot,
    format_base_url,
)


READY, SLEEPING, DEAD = EngineState.READY, EngineState.SLEEPING, EngineState.DEAD


def _teacher(engines, **kwargs):
    return build_model_snapshot("math", engines, **kwargs)


def test_discovery_snapshot_round_trips_through_dict():
    snapshot = RoleSnapshot(
        role="teacher",
        topology_revision=7,
        phase="score",
        default_model="math",
        route_keys={"algebra": "math"},
        models=(
            _teacher([(0, "http://node-1:15000", READY), (1, "http://node-2:15000", SLEEPING)]),
            build_model_snapshot(
                "policy",
                [(0, "http://node-3:15000", READY)],
                router_url="http://head:3000",
                diagnostic_workers=[("prefill-0", "http://node-4:15000", READY)],
            ),
        ),
    )

    data = snapshot.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["role"] == "teacher" and data["topology_revision"] == 7 and data["phase"] == "score"
    assert data["routing"] == {"default_model": "math", "route_keys": {"algebra": "math"}}
    assert data["models"]["math"]["engines"][0] == {
        "engine_id": "math/0",
        "base_url": "http://node-1:15000",
        "state": "ready",
        "direct_eligible": True,
    }
    assert RoleSnapshot.from_dict(data) == snapshot


def test_discovery_direct_eligible_requires_ready_and_no_router():
    direct = _teacher([(0, "http://a:1", READY), (1, "http://b:1", SLEEPING), (2, None, READY)])
    assert [engine.direct_eligible for engine in direct.engines] == [True, False, False]

    routed = _teacher([(0, "http://a:1", READY)], router_url="http://head:3000")
    assert [engine.direct_eligible for engine in routed.engines] == [False]
    assert routed.router_url == "http://head:3000"


def test_discovery_pd_workers_are_diagnostics_not_replicas():
    model = build_model_snapshot(
        "policy",
        [],
        router_url="http://head:3000",
        diagnostic_workers=[("prefill-0", "http://a:1", READY), ("decode-0", "http://b:1", READY)],
    )

    assert model.engines == ()
    assert [worker.engine_id for worker in model.diagnostic_workers] == ["policy/prefill-0", "policy/decode-0"]
    assert not any(worker.direct_eligible for worker in model.diagnostic_workers)


def test_discovery_model_state_follows_its_most_available_replica():
    assert aggregate_state([SLEEPING, READY, DEAD]) is READY
    assert aggregate_state([SLEEPING, EngineState.ONLOADING]) is EngineState.ONLOADING
    assert aggregate_state([SLEEPING, EngineState.DRAINING]) is EngineState.DRAINING
    assert aggregate_state([SLEEPING, DEAD]) is SLEEPING
    assert aggregate_state([DEAD]) is DEAD
    assert aggregate_state([]) is DEAD


def test_discovery_revision_increases_when_endpoint_changes():
    revision = TopologyRevision()
    first = revision.observe([_teacher([(0, "http://a:1", READY)])])

    assert revision.observe([_teacher([(0, "http://a:2", READY)])]) == first + 1
    # A replica added or removed, or a router address change, also counts.
    assert revision.observe([_teacher([(0, "http://a:2", READY), (1, "http://b:2", READY)])]) == first + 2
    assert revision.observe([_teacher([(0, "http://a:2", READY)], router_url="http://head:3000")]) == first + 3


def test_discovery_revision_increases_when_replica_is_rebuilt_in_place():
    revision = TopologyRevision()
    model = _teacher([(0, "http://a:1", READY)])
    first = revision.observe([model], {"math/0": 1})

    assert revision.observe([model], {"math/0": 2}) == first + 1


def test_discovery_revision_unchanged_on_state_change():
    revision = TopologyRevision()
    first = revision.observe([_teacher([(0, "http://a:1", READY)])])

    assert revision.observe([_teacher([(0, "http://a:1", SLEEPING)])]) == first
    assert revision.observe([_teacher([(0, "http://a:1", READY)])]) == first
    assert revision.value == first


def test_discovery_base_url_brackets_ipv6():
    assert format_base_url("10.0.0.1", 15000) == "http://10.0.0.1:15000"
    assert format_base_url("fe80::1", 15000) == "http://[fe80::1]:15000"
    assert format_base_url("[fe80::1]", 15000) == "http://[fe80::1]:15000"

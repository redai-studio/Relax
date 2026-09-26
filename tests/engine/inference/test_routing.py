# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace

import pytest

from relax.engine.inference.routing import RoutingError, resolve_model, select_target
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def _replica(
    name: str = "head", *, direct: bool = False, state: LifecycleState = LifecycleState.READY
) -> ReplicaSnapshot:
    return ReplicaSnapshot(
        engine_id=name,
        base_url=f"http://{name}.test",
        state=state,
        weight_version="v2",
        direct_eligible=direct,
    )


def _model(**kwargs) -> ModelSnapshot:
    return replace(
        ModelSnapshot(
            "student",
            (_replica(),),
            router_url="http://router.test",
            state=LifecycleState.READY,
            admission=True,
        ),
        **kwargs,
    )


def _snapshot() -> RoleSnapshot:
    return RoleSnapshot(
        role=Role.ROLLOUT,
        manager_epoch="epoch-a",
        phase="rollout",
        models=(_model(), ModelSnapshot("teacher")),
        routing=RoutingSpec(default_model="student", route_key_to_model=(("score", "teacher"),)),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, "student"), ({"route_key": "score"}, "teacher"), ({"model": "student", "route_key": "score"}, "student")],
)
def test_routing_selection_precedence(kwargs: dict, expected: str) -> None:
    assert resolve_model(_snapshot(), **kwargs).model_id == expected


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"model": "missing"}, "unknown_model"),
        ({"model": ""}, "unknown_model"),
    ],
)
def test_routing_invalid_explicit_model_never_falls_back(kwargs: dict, code: str) -> None:
    with pytest.raises(RoutingError) as error:
        resolve_model(_snapshot(), **kwargs)
    assert error.value.code == code
    assert error.value.status_code == 400


def test_routing_unmapped_route_key_falls_back_to_default_model() -> None:
    assert resolve_model(_snapshot(), route_key="missing").model_id == "student"


def test_routing_unmapped_route_key_without_default_is_rejected() -> None:
    snapshot = replace(_snapshot(), routing=RoutingSpec(route_key_to_model=(("score", "teacher"),)))
    with pytest.raises(RoutingError) as error:
        resolve_model(snapshot, route_key="missing")
    assert error.value.code == "unknown_route"
    assert error.value.status_code == 400


def test_routing_requires_configured_default_even_with_one_model() -> None:
    with pytest.raises(RoutingError, match="No model"):
        resolve_model(replace(_snapshot(), models=(_model(),), routing=RoutingSpec()))


@pytest.mark.parametrize("state", [None, LifecycleState.STARTING, LifecycleState.SLEEPING])
def test_routing_unknown_or_unready_model_is_unavailable(state: LifecycleState | None) -> None:
    with pytest.raises(RoutingError) as error:
        select_target(_model(state=state))
    assert error.value.status_code == 503


def test_routing_draining_model_closes_admission_even_with_ready_replica() -> None:
    with pytest.raises(RoutingError):
        select_target(_model(admission=False))


def test_routing_router_model_uses_router_even_when_replicas_are_ready() -> None:
    model = _model(replicas=(_replica("a"), _replica("b")))
    target = select_target(model, cursor=1)
    assert target.base_url == "http://router.test"


def test_routing_router_model_without_router_never_falls_back_to_replica() -> None:
    with pytest.raises(RoutingError) as error:
        select_target(_model(router_url=None))
    assert error.value.status_code == 503


def test_routing_direct_model_rotates_across_eligible_replicas() -> None:
    model = _model(router_url=None, replicas=(_replica("a", direct=True), _replica("b", direct=True)))
    assert [select_target(model, cursor=cursor).base_url for cursor in range(3)] == [
        "http://a.test",
        "http://b.test",
        "http://a.test",
    ]


def test_routing_direct_model_skips_ineligible_and_unready_replicas() -> None:
    model = _model(
        router_url=None,
        replicas=(
            _replica("sleeping", direct=True, state=LifecycleState.SLEEPING),
            _replica("ineligible"),
            _replica("ready", direct=True),
        ),
    )
    assert select_target(model, cursor=0).base_url == "http://ready.test"

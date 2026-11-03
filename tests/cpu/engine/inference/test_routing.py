# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Model and replica selection shared by gateways and direct clients."""

import pytest

from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RoutingState,
    UnknownModelError,
    candidate_replicas,
    select_model,
    select_replica,
    select_target,
)


READY, SLEEPING = EngineState.READY, EngineState.SLEEPING


def _role(*models, default_model=None, route_keys=None):
    return RoleSnapshot(
        role="teacher", topology_revision=1, models=models, default_model=default_model, route_keys=route_keys or {}
    )


def _model(name, states=(READY, READY), **kwargs):
    return build_model_snapshot(name, [(i, f"http://{name}-{i}:1", state) for i, state in enumerate(states)], **kwargs)


# ----------------------------------------------------------------------
# Which model.
# ----------------------------------------------------------------------


def test_routing_explicit_model_wins_over_route_key():
    role = _role(_model("math"), _model("code"), route_keys={"algebra": "math"})

    assert select_model(role, model="code", route_key="algebra").name == "code"


def test_routing_route_key_uses_mapping_then_falls_back_to_model_name():
    role = _role(_model("math"), _model("code"), route_keys={"algebra": "math"})

    assert select_model(role, route_key="algebra").name == "math"
    assert select_model(role, route_key="code").name == "code"


def test_routing_without_model_or_key_uses_default_or_the_sole_model():
    assert select_model(_role(_model("math"), _model("code"), default_model="code")).name == "code"
    assert select_model(_role(_model("math"))).name == "math"


def test_routing_unknown_model_lists_what_is_available():
    role = _role(_model("math"), _model("code"))

    for kwargs in ({"model": "physics"}, {"route_key": "physics"}, {}):
        with pytest.raises(UnknownModelError, match=r"\['math', 'code'\]") as excinfo:
            select_model(role, **kwargs)
        assert excinfo.value.status_code == 400


# ----------------------------------------------------------------------
# Which replica.
# ----------------------------------------------------------------------


def test_routing_model_with_router_is_reached_through_the_router():
    role = _role(_model("policy", router_url="http://head:3000"))

    target = select_target(role, RoutingState())

    assert (target.base_url, target.engine_id, target.via_router) == ("http://head:3000", None, True)


def test_routing_same_affinity_key_stays_on_one_replica():
    role, state = _role(_model("math")), RoutingState()

    urls = {select_target(role, state, affinity_key=7).base_url for _ in range(8)}

    assert len(urls) == 1


def test_routing_distinct_affinity_keys_spread_across_replicas():
    role, state = _role(_model("math")), RoutingState()

    # Sparse keys (only even ones) must not collapse onto one replica.
    urls = [select_target(role, state, affinity_key=key).base_url for key in (0, 2, 4, 6)]

    assert urls == ["http://math-0:1", "http://math-1:1", "http://math-0:1", "http://math-1:1"]


def test_routing_without_affinity_key_round_robins():
    role, state = _role(_model("math")), RoutingState()

    urls = [select_target(role, state).base_url for _ in range(4)]

    assert urls == ["http://math-0:1", "http://math-1:1", "http://math-0:1", "http://math-1:1"]


def test_routing_cursor_is_kept_per_model():
    role, state = _role(_model("math"), _model("code")), RoutingState()

    picks = [select_target(role, state, model=name).base_url for name in ("math", "code", "math", "code")]

    assert picks == ["http://math-0:1", "http://code-0:1", "http://math-1:1", "http://code-1:1"]


def test_routing_skips_replicas_that_cannot_take_requests():
    role, state = _role(_model("math", states=(SLEEPING, READY, EngineState.DEAD))), RoutingState()

    assert [engine.engine_id for engine in candidate_replicas(role.models[0])] == ["math/1"]
    assert {select_target(role, state).engine_id for _ in range(3)} == {"math/1"}


def test_routing_no_eligible_replica_is_unavailable():
    role = _role(_model("math", states=(SLEEPING, SLEEPING)))

    with pytest.raises(ModelUnavailableError, match="sleeping") as excinfo:
        select_target(role, RoutingState())
    assert excinfo.value.status_code == 503


def test_routing_sleeping_routed_model_is_unavailable():
    role = _role(_model("policy", states=(SLEEPING,), router_url="http://head:3000"))

    with pytest.raises(ModelUnavailableError):
        select_target(role, RoutingState())


def test_routing_affinity_memory_is_bounded():
    cursors, memory = {}, {}
    replicas = ["r0", "r1", "r2"]

    def pick(key):
        return select_replica(
            replicas,
            cursor_key="teacher",
            cursors=cursors,
            affinity_key=key,
            affinity_memory=memory,
            affinity_memory_cap=2,
        )

    assert [pick(key) for key in (0, 1, 2)] == ["r0", "r1", "r2"]
    # The fourth distinct key finds the memory over its cap and clears it...
    assert pick(3) == "r0"
    assert set(memory) == {("teacher", 3)}
    # ...so an earlier key is assigned afresh from the cursor rather than recalled.
    assert pick(0) == "r1"


def test_routing_single_replica_never_touches_the_cursor():
    cursors: dict = {}

    assert (
        select_replica(["only"], cursor_key="teacher", cursors=cursors, affinity_key=1, affinity_memory={}) == "only"
    )
    assert cursors == {}

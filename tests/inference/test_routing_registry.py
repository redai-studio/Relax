# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import json
import pickle
from dataclasses import FrozenInstanceError, replace

import pytest
import ray.cloudpickle
from ray.exceptions import RayError

from relax.inference.registry import InferenceRegistry
from relax.inference.routing import InferenceRoutingError, RouteResolver
from relax.inference.specs import ModelSnapshot, RegistrySnapshot, ReplicaSnapshot, RoutingSpec, validate_snapshot


@pytest.mark.parametrize("codec", [pickle, ray.cloudpickle, RayError])
@pytest.mark.parametrize("status_code", [400, 503])
def test_routing_error_serialization_preserves_status_and_message(codec, status_code):
    error = InferenceRoutingError(status_code, "Inference model 'default' is FAILED")
    if codec is RayError:
        restored = RayError.from_bytes(RayError.to_bytes(error))
    else:
        restored = codec.loads(codec.dumps(error))

    assert type(restored) is InferenceRoutingError
    assert restored.status_code == status_code
    assert restored.args == error.args
    assert str(restored) == str(error)


def _replica(engine_id: str, *, state: str = "READY", eligible: bool = True) -> ReplicaSnapshot:
    return ReplicaSnapshot(engine_id, f"http://{engine_id}.example:15000", state, eligible)


def _model(model_id: str = "policy", *, state: str = "READY") -> ModelSnapshot:
    return ModelSnapshot(model_id, state, "DIRECT", engines=(_replica(f"{model_id}-0"), _replica(f"{model_id}-1")))


def _registry() -> InferenceRegistry:
    registry = InferenceRegistry("rollout", registry_epoch="manager-1")
    registry.publish(
        [_model(), _model("judge")],
        RoutingSpec(default_model="policy", route_key_map={"math": "judge"}, aliases={"student": "policy"}),
    )
    return registry


def test_registry_snapshots_are_deeply_isolated_and_json_roundtrip():
    registry = _registry()
    snapshot = registry.snapshot()
    immutable = validate_snapshot(json.loads(json.dumps(snapshot)))

    assert immutable.to_dict() == snapshot
    with pytest.raises(FrozenInstanceError):
        immutable.role = "teacher"
    with pytest.raises(TypeError):
        immutable.models["policy"] = _model()
    with pytest.raises(TypeError):
        immutable.routing.route_key_map["math"] = "policy"

    snapshot["models"]["policy"]["engines"][0]["state"] = "SLEEPING"
    snapshot["routing"]["route_key_map"]["math"] = "policy"
    assert registry.snapshot() == immutable.to_dict()


def test_specs_copy_mutable_inputs_before_publication():
    replicas = [_replica("a")]
    routes = {"math": "policy"}
    model = ModelSnapshot("policy", "READY", "DIRECT", engines=replicas)
    routing = RoutingSpec(route_key_map=routes)

    replicas.clear()
    routes.clear()

    assert model.engines == (_replica("a"),)
    assert routing.route_key_map == {"math": "policy"}


def test_registry_revision_changes_for_endpoint_state_generation_routing_and_phase():
    registry = InferenceRegistry("rollout")
    assert registry.snapshot()["topology_revision"] == 0
    model = _model()
    routing = RoutingSpec(default_model="policy")
    assert registry.publish([model], routing)["topology_revision"] == 1
    assert registry.publish([model], routing)["topology_revision"] == 1
    model = replace(model, state="DRAINING")
    assert registry.publish([model], routing)["topology_revision"] == 2
    model = replace(model, engines=(replace(model.engines[0], base_url="http://replacement.example"),))
    assert registry.publish([model], routing)["topology_revision"] == 3
    model = replace(model, engines=(replace(model.engines[0], generation=1),))
    assert registry.publish([model], routing)["topology_revision"] == 4
    routing = replace(routing, policy_revision=1)
    assert registry.publish([model], routing)["topology_revision"] == 5
    snapshot = registry.publish([model], routing, phase_epoch=1, phase="training")
    assert snapshot["topology_revision"] == 6
    assert snapshot["phase_epoch"] == 1
    assert snapshot["phase"] == "training"


def test_registry_equivalent_candidate_order_does_not_change_revision():
    registry = InferenceRegistry("rollout")
    model = _model()
    original = registry.publish([model], RoutingSpec(default_model="policy"))

    reversed_engines = replace(model, engines=tuple(reversed(model.engines)))

    assert registry.publish([reversed_engines]) == original


def test_registry_failed_publication_keeps_last_valid_snapshot():
    registry = _registry()
    original = registry.snapshot()

    with pytest.raises(ValueError, match="unknown model"):
        registry.publish([_model()], RoutingSpec(default_model="missing"))

    assert registry.snapshot() == original


def test_registry_restart_changes_epoch_even_if_revisions_match():
    first, second = InferenceRegistry("teacher"), InferenceRegistry("teacher")
    first_snapshot = first.publish([_model("teacher")])
    second_snapshot = second.publish([_model("teacher")])

    assert first_snapshot["topology_revision"] == second_snapshot["topology_revision"]
    assert first_snapshot["registry_epoch"] != second_snapshot["registry_epoch"]


def test_registry_invalidation_fences_old_readiness_and_records_transient_changes():
    registry = InferenceRegistry("teacher")
    model, routing = _model(), RoutingSpec(default_model="policy")
    before = registry.publish([model], routing)

    invalidated = registry.invalidate()

    assert invalidated["topology_revision"] > before["topology_revision"]
    assert all(not engine["direct_eligible"] for engine in invalidated["models"]["policy"]["engines"])
    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(invalidated)
    assert error.value.status_code == 503
    after = registry.publish([model], routing)
    assert after["topology_revision"] > invalidated["topology_revision"]
    assert RouteResolver().resolve(after).engine_id == "policy-0"


@pytest.mark.parametrize(
    "capability", ["weight_update", "update_weights", "dcs", "DCS", "register_dcs", "seed_weight_sync"]
)
def test_static_models_reject_dynamic_capabilities(capability):
    with pytest.raises(ValueError, match="STATIC"):
        replace(_model(), capabilities=(capability,))


@pytest.mark.parametrize("role", ["teacher", "genrm"])
@pytest.mark.parametrize("weight_source", ["ACTOR", "DCS"])
def test_static_roles_reject_actor_weight_sources(role, weight_source):
    with pytest.raises(ValueError, match="STATIC"):
        InferenceRegistry(role).publish([replace(_model(), weight_source=weight_source)])


def test_dynamic_rollout_allows_weight_source_and_version():
    model = replace(_model(), weight_source="DCS", weight_version="step-2", capabilities=("dcs", "weight_update"))

    snapshot = InferenceRegistry("rollout").publish([model])

    assert snapshot["models"]["policy"]["weight_version"] == "step-2"


@pytest.mark.parametrize(
    ("selectors", "expected_model"),
    [
        ({}, "policy"),
        ({"model": "judge"}, "judge"),
        ({"model": "student"}, "policy"),
        ({"route_key": "math"}, "judge"),
        ({"model": "student", "route_key": "math"}, "policy"),
        ({"model": "judge", "route_key": "unknown"}, "judge"),
    ],
)
def test_routing_selector_precedence(selectors, expected_model):
    target = RouteResolver().resolve(_registry().snapshot(), **selectors)

    assert target.model_id == expected_model
    assert target.registry_epoch == "manager-1"
    assert target.topology_revision == 1


@pytest.mark.parametrize(
    "selectors",
    [{"model": "missing"}, {"route_key": "missing"}, {"model": ""}, {"model": 3}],
)
def test_routing_unknown_selector_never_falls_back_to_default(selectors):
    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(_registry().snapshot(), **selectors)

    assert error.value.status_code == 400


def test_routing_does_not_invent_default_for_single_model():
    snapshot = InferenceRegistry("teacher").publish([_model()])

    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(snapshot)

    assert error.value.status_code == 400


@pytest.mark.parametrize("state", ["UNKNOWN", "STARTING", "DRAINING", "SLEEPING", "ONLOADING", "FAILED", "DEAD"])
def test_routing_nonready_model_refuses_even_with_ready_replica(state):
    snapshot = InferenceRegistry("teacher").publish([_model(state=state)], RoutingSpec(default_model="policy"))

    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(snapshot)

    assert error.value.status_code == 503


def test_direct_routing_filters_sleeping_internal_and_missing_endpoints():
    candidates = (
        _replica("sleeping", state="SLEEPING"),
        _replica("internal", eligible=False),
        replace(_replica("unpublished"), base_url=None),
        replace(_replica("healthy"), generation=4),
    )
    snapshot = InferenceRegistry("teacher").publish(
        [replace(_model(), engines=candidates)], RoutingSpec(default_model="policy")
    )
    resolver = RouteResolver()

    for _ in range(4):
        target = resolver.resolve(snapshot)
        assert target.engine_id == "healthy"
        assert target.generation == 4


def test_direct_routing_with_no_ready_candidates_returns_unavailable():
    snapshot = InferenceRegistry("teacher").publish(
        [replace(_model(), engines=(_replica("sleeping", state="SLEEPING"),))],
        RoutingSpec(default_model="policy"),
    )

    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(snapshot)

    assert error.value.status_code == 503


def test_router_mode_never_routes_directly_to_pd_workers():
    model = replace(_model(), route_mode="SGLANG_ROUTER", router_url="http://router.example/")
    snapshot = InferenceRegistry("rollout").publish([model], RoutingSpec(default_model="policy"))

    target = RouteResolver().resolve(snapshot, affinity_key="session-1")

    assert target.base_url == "http://router.example"
    assert target.engine_id is None
    assert target.generation == 0


def test_round_robin_progress_is_isolated_per_model_and_resets_on_new_manager_epoch():
    snapshot = _registry().snapshot()
    resolver = RouteResolver()

    assert resolver.resolve(snapshot, model="policy").engine_id == "policy-0"
    assert resolver.resolve(snapshot, model="judge").engine_id == "judge-0"
    assert resolver.resolve(snapshot, model="policy").engine_id == "policy-1"
    assert resolver.resolve(snapshot, model="judge").engine_id == "judge-1"
    resolver.resolve(snapshot, model="policy")
    snapshot["registry_epoch"] = "manager-2"
    assert resolver.resolve(snapshot, model="policy").engine_id == "policy-0"


def test_affinity_matches_independent_resolvers_and_survives_endpoint_replacement():
    snapshot = _registry().snapshot()
    direct, gateway = RouteResolver(), RouteResolver()
    direct.resolve(snapshot)
    target = direct.resolve(snapshot, affinity_key="group-42")

    assert gateway.resolve(snapshot, affinity_key="group-42") == target
    replacement = snapshot["models"][target.model_id]["engines"]
    replacement.reverse()
    for engine in replacement:
        engine["base_url"] = "http://replacement.example"
        engine["generation"] += 1
    snapshot["topology_revision"] += 1

    replaced_target = gateway.resolve(snapshot, affinity_key="group-42")
    assert replaced_target.engine_id == target.engine_id
    assert replaced_target.generation == target.generation + 1
    assert replaced_target.base_url == "http://replacement.example"


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/engine",
        "http://user:password@host.example",
        "http://host.example/#fragment",
        "http://host.example/?x=1",
        "http://host.example:bad",
        "http://host.example:0",
        "http://host.example/a b",
    ],
)
def test_untrusted_snapshot_rejects_non_endpoint_urls(url):
    snapshot = _registry().snapshot()
    snapshot["models"]["policy"]["engines"][0]["base_url"] = url

    with pytest.raises(ValueError):
        RegistrySnapshot.from_dict(snapshot)
    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(snapshot)

    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "field,value", [("topology_revision", True), ("schema_version", 2.0), ("models", []), ("role", "worker")]
)
def test_untrusted_snapshot_rejects_invalid_schema_fields(field, value):
    snapshot = _registry().snapshot()
    snapshot[field] = value

    with pytest.raises(ValueError):
        validate_snapshot(snapshot)


def test_untrusted_snapshot_rejects_internal_worker_fields_and_model_mismatch():
    snapshot = _registry().snapshot()
    snapshot["models"]["policy"]["engines"][0]["pid"] = 42
    with pytest.raises(ValueError, match="Unknown replica"):
        validate_snapshot(snapshot)

    snapshot = _registry().snapshot()
    snapshot["models"]["policy"]["model_id"] = "judge"
    with pytest.raises(ValueError, match="model_id"):
        validate_snapshot(snapshot)


def test_untrusted_snapshot_without_redundant_model_id_accepts_design_schema():
    snapshot = _registry().snapshot()
    for model in snapshot["models"].values():
        model.pop("model_id")

    validated = validate_snapshot(snapshot)

    assert validated.models["policy"].model_id == "policy"


def test_untrusted_snapshot_rejects_duplicate_replica_identity():
    snapshot = _registry().snapshot()
    engines = snapshot["models"]["policy"]["engines"]
    engines[1]["engine_id"] = engines[0]["engine_id"]

    with pytest.raises(ValueError, match="Duplicate engine_id"):
        validate_snapshot(snapshot)


def test_registry_does_not_replace_explicit_invalid_epoch_with_random_identity():
    with pytest.raises(ValueError, match="registry_epoch"):
        InferenceRegistry("teacher", registry_epoch="")


def test_registry_mapping_publication_validates_and_detaches_json_inputs():
    model = _model().to_dict()
    registry = InferenceRegistry("teacher")
    published = registry.publish({"policy": model}, {"default_model": "policy"})
    model["engines"].clear()

    assert len(registry.snapshot()["models"]["policy"]["engines"]) == 2
    assert registry.publish(published["models"], published["routing"]) == published


def test_affinity_policy_requires_key_and_never_silently_uses_round_robin():
    snapshot = InferenceRegistry("teacher").publish([_model()], RoutingSpec(default_model="policy", policy="affinity"))

    with pytest.raises(InferenceRoutingError) as error:
        RouteResolver().resolve(snapshot)

    assert error.value.status_code == 400
    assert RouteResolver().resolve(snapshot, affinity_key="group").model_id == "policy"


@pytest.mark.parametrize("state", ["UNKNOWN", "STARTING", "DRAINING", "SLEEPING", "FAILED"])
def test_replica_publication_clears_direct_eligibility_without_ready_state(state):
    replica = _replica("engine", state=state, eligible=True)

    assert replica.direct_eligible is False
    assert replica.to_dict()["direct_eligible"] is False


@pytest.mark.parametrize("route_mode", ["DIRECT", "SGLANG_ROUTER"])
def test_routing_preserves_backend_served_name_separately_from_model_alias(route_mode):
    model = replace(
        _model(),
        route_mode=route_mode,
        router_url="http://router.example" if route_mode == "SGLANG_ROUTER" else None,
        served_model_name="organization/model-checkpoint",
    )
    registry = InferenceRegistry("rollout")
    snapshot = registry.publish([model], RoutingSpec(aliases={"student": "policy"}))

    target = RouteResolver().resolve(snapshot, model="student")

    assert target.model_id == "policy"
    assert target.served_model_name == "organization/model-checkpoint"
    assert validate_snapshot(snapshot).models["policy"].served_model_name == target.served_model_name
    renamed = registry.publish([replace(model, served_model_name="other-checkpoint")])
    assert renamed["topology_revision"] > snapshot["topology_revision"]


@pytest.mark.parametrize("served_model_name", ["", 1, ["model"]])
def test_untrusted_snapshot_rejects_invalid_served_model_name(served_model_name):
    snapshot = _registry().snapshot()
    snapshot["models"]["policy"]["served_model_name"] = served_model_name

    with pytest.raises(ValueError, match="served_model_name"):
        validate_snapshot(snapshot)

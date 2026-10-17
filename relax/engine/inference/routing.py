# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure selection shared by the Gateway and direct discovery clients.

Gateway callers must perform selection under Manager admission, not dispatch
from a previously observed snapshot. No function here sends or retries
requests.
"""

from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    RoleSnapshot,
    RouteTarget,
)


class RoutingError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int, model_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.model_id = model_id


def resolve_model(snapshot: RoleSnapshot, *, model: str | None = None, route_key: str | None = None) -> ModelSnapshot:
    """Resolve explicit model, then route key mapping, then the default model.

    An unmapped route key falls through to the default model; only a request
    that none of the three resolves is rejected.
    """
    selected = model
    if selected is None and route_key is not None:
        selected = dict(snapshot.routing.route_key_to_model).get(route_key)
    if selected is None:
        selected = snapshot.routing.default_model
    if selected is None:
        if route_key is not None:
            raise RoutingError("unknown_route", f"No model configured for route key {route_key!r}", status_code=400)
        raise RoutingError("model_required", "No model or default model selected", status_code=400)
    for candidate in snapshot.models:
        if candidate.model_id == selected:
            return candidate
    raise RoutingError("unknown_model", f"Unknown model {selected!r}", status_code=400, model_id=selected)


def select_target(model: ModelSnapshot, *, cursor: int = 0) -> RouteTarget:
    """Select the model's Router, or else a READY direct-eligible replica.

    A Router-routed model publishes a Router URL and no direct-eligible
    replica; a direct-routed model the reverse. ``cursor`` rotates among the
    eligible replicas and is ignored for a Router.
    """
    if cursor < 0:
        raise RoutingError("invalid_cursor", "Round-robin cursor must be non-negative", status_code=400)
    if not model.admission or model.state != LifecycleState.READY:
        raise RoutingError("unavailable", "Model is not accepting requests", status_code=503, model_id=model.model_id)
    if model.router_url:
        return RouteTarget(model_id=model.model_id, base_url=model.router_url)
    eligible = [
        replica
        for replica in model.replicas
        if replica.direct_eligible and replica.state == LifecycleState.READY and replica.base_url
    ]
    if not eligible:
        raise RoutingError(
            "unavailable", "Model has no Router or direct replica", status_code=503, model_id=model.model_id
        )
    return RouteTarget(model_id=model.model_id, base_url=eligible[cursor % len(eligible)].base_url)

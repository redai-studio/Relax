# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The routing contract shared by HTTP gateways and direct clients."""

from typing import Any


class RoutingError(ValueError):
    status_code = 400


class ModelUnavailable(RoutingError):
    status_code = 503


def select_model(snapshot: dict, model: str | None = None, route_key: str | None = None) -> str:
    models = snapshot["models"]
    routes = snapshot.get("routes", {})
    if model is not None:
        selected = model if model in models else routes.get(model)
    elif route_key is not None:
        selected = routes.get(route_key)
    else:
        selected = snapshot.get("default_model")
    if selected not in models:
        raise RoutingError(f"Unknown or ambiguous model {selected!r}; available={list(models)}")
    return selected


def select_endpoint(snapshot: dict, model: str, ordinal: int = 0) -> str:
    info = snapshot["models"][model]
    if info["state"] != "ready":
        raise ModelUnavailable(f"Model {model!r} is {info['state']}; retry after its next activation")
    if info.get("router_url"):
        return info["router_url"].rstrip("/")
    engines = [e for e in info["engines"] if e["state"] == "ready" and e["direct_eligible"]]
    if not engines:
        raise ModelUnavailable(f"Model {model!r} has no ready ingress")
    return engines[ordinal % len(engines)]["base_url"].rstrip("/")


def engine_record(engine_id: str, url: str | None, state: str, *, direct: bool = True) -> dict[str, Any]:
    return {
        "engine_id": engine_id,
        "base_url": url,
        "state": state,
        "direct_eligible": bool(url and direct and state == "ready"),
    }

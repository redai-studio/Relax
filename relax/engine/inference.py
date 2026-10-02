# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-light discovery and routing shared by gateways and clients."""

import copy
import threading
import uuid
from dataclasses import dataclass
from typing import Any


class InferenceError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def base_url(host: str, port: int) -> str:
    host = host.strip("[]")
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


class SnapshotVersion:
    """Publish detached snapshots; engine incarnations participate in
    revision."""

    def __init__(self) -> None:
        self.epoch = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._previous: Any = None
        self._revision = 0

    def publish(self, models: dict[str, Any], incarnation: Any = None) -> dict[str, Any]:
        value = copy.deepcopy(models)
        with self._lock:
            signature = (value, incarnation)
            if signature != self._previous:
                self._revision += 1
                self._previous = copy.deepcopy(signature)
            return {"models": value, "epoch": self.epoch, "revision": self._revision}


def actor_identity(engine: Any) -> str:
    actor_id = getattr(engine, "_actor_id", None)
    return actor_id.hex() if actor_id is not None else str(id(engine))


def model_state(engines: list[dict[str, Any]], fallback: str) -> str:
    if any(engine["state"] == "ready" for engine in engines):
        return "ready"
    return fallback if fallback != "ready" else "unavailable"


@dataclass(frozen=True)
class InferenceTarget:
    model: str
    base_url: str
    backend_model: str | None = None


class InferenceRouter:
    """Resolve the model identically in proxy and direct modes."""

    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}

    def select_model(self, snapshot: dict[str, Any], model: str | None, route_key: str | None) -> str:
        if any(value is not None and not isinstance(value, str) for value in (model, route_key)):
            raise InferenceError(400, "model and route_key must be strings")
        models = snapshot["models"]
        routing = snapshot.get("routing", {})
        if model is not None:
            selected = model if model in models else routing.get("aliases", {}).get(model)
        elif route_key is not None:
            selected = routing.get("route_keys", {}).get(route_key)
        else:
            selected = routing.get("default_model")
        if selected not in models:
            raise InferenceError(400, f"Unknown or missing model/route_key; available models: {list(models)}")
        return selected

    def select(
        self, snapshot: dict[str, Any], model: str | None = None, route_key: str | None = None
    ) -> InferenceTarget:
        name = self.select_model(snapshot, model, route_key)
        info = snapshot["models"][name]
        if info["state"] != "ready":
            raise InferenceError(503, f"Model {name!r} is {info['state']}")
        router_url = info.get("router_url")
        if router_url:
            return InferenceTarget(name, router_url, info.get("backend_model"))
        if info.get("router_required"):
            raise InferenceError(503, f"Model {name!r} requires an unavailable router")
        candidates = [e for e in info["engines"] if e["state"] == "ready" and e["direct_eligible"] and e["base_url"]]
        if not candidates:
            raise InferenceError(503, f"Model {name!r} has no eligible replica")
        cursor = self._cursors.get(name, 0)
        self._cursors[name] = cursor + 1
        return InferenceTarget(name, candidates[cursor % len(candidates)]["base_url"], info.get("backend_model"))


def prepare_payload(payload: dict[str, Any], target: InferenceTarget, *, chat: bool) -> dict[str, Any]:
    result = dict(payload)
    result.pop("route_key", None)
    result.pop("model", None)
    if chat:
        result["model"] = target.backend_model or target.model
    return result

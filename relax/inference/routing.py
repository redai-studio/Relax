# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from relax.inference.specs import RegistrySnapshot


class InferenceRoutingError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __reduce__(self) -> tuple[type[InferenceRoutingError], tuple[int, str]]:
        return type(self), (self.status_code, self.args[0])


@dataclass(frozen=True)
class RouteTarget:
    model_id: str
    base_url: str
    engine_id: str | None
    generation: int
    registry_epoch: str
    topology_revision: int
    served_model_name: str | None = None


class RouteResolver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursors: dict[tuple[str, str, str], int] = {}
        self._epochs: dict[str, str] = {}

    def resolve(
        self,
        snapshot: dict[str, Any],
        model: str | None = None,
        route_key: str | None = None,
        affinity_key: str | None = None,
    ) -> RouteTarget:
        try:
            registry = RegistrySnapshot.from_dict(snapshot)
        except (ValueError, TypeError) as exc:
            raise InferenceRoutingError(503, f"Invalid inference discovery: {exc}") from exc
        for name, value in (("model", model), ("route_key", route_key), ("affinity_key", affinity_key)):
            if value is not None and (not isinstance(value, str) or not value):
                raise InferenceRoutingError(400, f"{name} must be a nonempty string")
        routing = registry.routing
        if model is not None:
            model_id = model if model in registry.models else routing.aliases.get(model)
            if model_id is None:
                raise InferenceRoutingError(400, f"Unknown inference model: {model!r}")
        elif route_key is not None:
            model_id = routing.route_key_map.get(route_key)
            if model_id is None:
                raise InferenceRoutingError(400, f"Unknown inference route_key: {route_key!r}")
        else:
            model_id = routing.default_model
            if model_id is None:
                raise InferenceRoutingError(400, "An explicit model or route_key is required")
        selected = registry.models[model_id]
        if selected.state != "READY":
            raise InferenceRoutingError(503, f"Inference model {model_id!r} is {selected.state}")
        if selected.route_mode == "SGLANG_ROUTER":
            if selected.router_url is None:
                raise InferenceRoutingError(503, f"Inference model {model_id!r} has no Router endpoint")
            return RouteTarget(
                model_id,
                selected.router_url,
                None,
                0,
                registry.registry_epoch,
                registry.topology_revision,
                selected.served_model_name,
            )
        candidates = [
            engine
            for engine in selected.engines
            if engine.state == "READY" and engine.direct_eligible and engine.base_url is not None
        ]
        if not candidates:
            raise InferenceRoutingError(503, f"Inference model {model_id!r} has no READY direct replica")
        if affinity_key is not None:

            def score(engine: Any) -> bytes:
                key = json.dumps([registry.role, model_id, affinity_key, engine.engine_id], separators=(",", ":"))
                return hashlib.sha256(key.encode("utf-8")).digest()

            engine = max(candidates, key=score)
        else:
            if routing.policy == "affinity":
                raise InferenceRoutingError(400, "The affinity policy requires affinity_key")
            with self._lock:
                if self._epochs.get(registry.role) != registry.registry_epoch:
                    self._cursors = {key: value for key, value in self._cursors.items() if key[0] != registry.role}
                    self._epochs[registry.role] = registry.registry_epoch
                cursor_key = (registry.role, registry.registry_epoch, model_id)
                cursor = self._cursors.get(cursor_key, 0)
                engine = candidates[cursor % len(candidates)]
                self._cursors[cursor_key] = cursor + 1
        return RouteTarget(
            model_id,
            engine.base_url,
            engine.engine_id,
            engine.generation,
            registry.registry_epoch,
            registry.topology_revision,
            selected.served_model_name,
        )

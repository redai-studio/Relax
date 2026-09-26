# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from relax.inference.specs import ModelSnapshot, RegistrySnapshot, RoutingSpec


class InferenceRegistry:
    def __init__(self, role: str, registry_epoch: str | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = RegistrySnapshot(role, str(uuid.uuid4()) if registry_epoch is None else registry_epoch, 0, {})

    def publish(
        self,
        models: Iterable[ModelSnapshot] | Mapping[str, ModelSnapshot | Mapping[str, Any]],
        routing: RoutingSpec | Mapping[str, Any] | None = None,
        *,
        phase_epoch: int = 0,
        phase: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(models, Mapping):
            normalized = {
                key: value if isinstance(value, ModelSnapshot) else ModelSnapshot.from_dict(value, model_id=key)
                for key, value in models.items()
            }
        else:
            normalized = {}
            for model in models:
                if not isinstance(model, ModelSnapshot):
                    raise ValueError("models must contain ModelSnapshot objects")
                if model.model_id in normalized:
                    raise ValueError("Duplicate model_id")
                normalized[model.model_id] = model
        if routing is not None and not isinstance(routing, RoutingSpec):
            routing = RoutingSpec.from_dict(routing)
        with self._lock:
            current = self._snapshot
            candidate = RegistrySnapshot(
                role=current.role,
                registry_epoch=current.registry_epoch,
                topology_revision=current.topology_revision,
                models=normalized,
                routing=routing if routing is not None else current.routing,
                phase_epoch=phase_epoch,
                phase=phase,
            )
            if candidate != current:
                self._snapshot = replace(candidate, topology_revision=current.topology_revision + 1)
            return self._snapshot.to_dict()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot.to_dict()

    def invalidate(self) -> dict[str, Any]:
        with self._lock:
            current = self._snapshot
            models = {
                model_id: replace(
                    model,
                    state="UNKNOWN",
                    engines=tuple(replace(engine, state="UNKNOWN", direct_eligible=False) for engine in model.engines),
                )
                for model_id, model in current.models.items()
            }
            self._snapshot = replace(current, models=models, topology_revision=current.topology_revision + 1)
            return self._snapshot.to_dict()

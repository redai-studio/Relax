# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from relax.inference.registry import InferenceRegistry
from relax.inference.specs import ModelSnapshot, RegistrySnapshot, RoutingSpec


def teacher_base_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("Teacher endpoint must be an HTTP(S) base URL or /generate URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/generate"):
        path = path[: -len("/generate")]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def public_legacy_discovery(diagnostics: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(diagnostics)
    result["total_engines"] = 0
    for model in result.get("models", {}).values():
        model["total_engines"] = 0
        for group in model.get("engine_groups", []):
            engines = []
            if group.get("worker_type", "regular") == "regular":
                engines = [
                    {key: engine[key] for key in ("rank", "status", "url") if key in engine}
                    for engine in group.get("engines", [])
                    if engine.get("url")
                ]
            group["engines"] = engines
            model["total_engines"] += len(engines)
        result["total_engines"] += model["total_engines"]
    return result


class RoleDiscovery:
    def __init__(
        self, role: str, managers: Mapping[str, Any] | Callable[[], Mapping[str, Any]], *, timeout: float = 10.0
    ) -> None:
        self.role = role
        self._managers = managers
        self.timeout = timeout
        self.registry = InferenceRegistry(role)
        self._lock = asyncio.Lock()
        self._source_revisions: dict[str, tuple[str, int, int] | None] = {}

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            managers = self._managers() if callable(self._managers) else self._managers
            items = list(managers.items())

            async def fetch(manager: Any) -> dict[str, Any]:
                return await manager.get_inference_snapshot.remote()

            results = await asyncio.gather(
                *(asyncio.wait_for(fetch(manager), timeout=self.timeout) for _, manager in items),
                return_exceptions=True,
            )
            models = []
            source_revisions = {}
            for (model_id, _manager), result in zip(items, results, strict=True):
                if isinstance(result, BaseException):
                    source_revisions[model_id] = None
                    models.append(ModelSnapshot(model_id, "FAILED", "DIRECT"))
                    continue
                parsed = RegistrySnapshot.from_dict(result)
                source_revisions[model_id] = (parsed.registry_epoch, parsed.topology_revision, parsed.phase_epoch)
                if parsed.role != self.role or len(parsed.models) != 1:
                    raise ValueError("Legacy static manager must publish one model of the expected role")
                model = next(iter(parsed.models.values()))
                model_data = model.to_dict()
                model_data["model_id"] = model_id
                for engine in model_data["engines"]:
                    engine["engine_id"] = f"{model_id}/{parsed.registry_epoch}/{engine['engine_id']}"
                models.append(ModelSnapshot.from_dict(model_data))
            served_names: dict[str, list[str]] = {}
            for model in models:
                if model.served_model_name:
                    served_names.setdefault(model.served_model_name, []).append(model.model_id)
            routing = RoutingSpec(
                default_model=items[0][0] if len(items) == 1 else None,
                route_key_map={key: key for key, _ in items},
                aliases={name: ids[0] for name, ids in served_names.items() if len(ids) == 1 and name not in managers},
            )
            if source_revisions != self._source_revisions:
                self.registry.invalidate()
                self._source_revisions = source_revisions
            return self.registry.publish(models, routing=routing)

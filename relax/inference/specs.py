# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


INFERENCE_STATES = frozenset(
    {"UNKNOWN", "STARTING", "READY", "DRAINING", "SLEEPING", "ONLOADING", "FAILED", "STOPPING", "DEAD"}
)
INFERENCE_ROLES = frozenset({"rollout", "genrm", "teacher"})
ROUTE_MODES = frozenset({"DIRECT", "SGLANG_ROUTER"})
WEIGHT_SOURCES = frozenset({"STATIC", "ACTOR", "DCS"})
_DYNAMIC_CAPABILITIES = frozenset({"weight_update", "update_weights", "dcs", "register_dcs", "seed_weight_sync"})


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _choice(value: Any, choices: frozenset[str], name: str) -> str:
    _text(value, name)
    if value not in choices:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def _url(value: Any, name: str) -> str:
    value = _text(value, name)
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain whitespace or control characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (port is None or port > 0)
        )
    except ValueError as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not valid:
        raise ValueError(f"{name} must be an HTTP(S) base URL without credentials, query, or fragment")
    return value.rstrip("/")


def _string_map(value: Any, name: str) -> Mapping[str, str]:
    value = _mapping(value, name)
    return MappingProxyType({_text(key, name): _text(item, name) for key, item in value.items()})


def _fields(data: Any, allowed: set[str], required: set[str], name: str) -> Mapping:
    data = _mapping(data, name)
    if unknown := set(data) - allowed:
        raise ValueError(f"Unknown {name} fields: {sorted(str(key) for key in unknown)}")
    if missing := required - set(data):
        raise ValueError(f"Missing {name} fields: {sorted(missing)}")
    return data


@dataclass(frozen=True)
class ReplicaSnapshot:
    engine_id: str
    base_url: str | None
    state: str
    direct_eligible: bool
    generation: int = 0

    def __post_init__(self) -> None:
        _text(self.engine_id, "engine_id")
        if self.base_url is not None:
            object.__setattr__(self, "base_url", _url(self.base_url, "base_url"))
        _choice(self.state, INFERENCE_STATES, "replica state")
        if not isinstance(self.direct_eligible, bool):
            raise ValueError("direct_eligible must be a boolean")
        if self.state != "READY" or self.base_url is None:
            object.__setattr__(self, "direct_eligible", False)
        _integer(self.generation, "generation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "base_url": self.base_url,
            "state": self.state,
            "direct_eligible": self.direct_eligible,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplicaSnapshot:
        data = _fields(
            data,
            {"engine_id", "base_url", "state", "direct_eligible", "generation"},
            {"engine_id", "base_url", "state", "direct_eligible"},
            "replica",
        )
        return cls(**data)


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    state: str
    route_mode: str
    router_url: str | None = None
    engines: tuple[ReplicaSnapshot, ...] = ()
    weight_source: str = "STATIC"
    weight_version: str | None = None
    capabilities: tuple[str, ...] = ()
    served_model_name: str | None = None

    def __post_init__(self) -> None:
        _text(self.model_id, "model_id")
        _choice(self.state, INFERENCE_STATES, "model state")
        _choice(self.route_mode, ROUTE_MODES, "route_mode")
        _choice(self.weight_source, WEIGHT_SOURCES, "weight_source")
        if self.router_url is not None:
            object.__setattr__(self, "router_url", _url(self.router_url, "router_url"))
        if not isinstance(self.engines, (tuple, list)) or any(
            not isinstance(engine, ReplicaSnapshot) for engine in self.engines
        ):
            raise ValueError("engines must contain ReplicaSnapshot objects")
        engines = tuple(sorted(self.engines, key=lambda engine: engine.engine_id))
        if len({engine.engine_id for engine in engines}) != len(engines):
            raise ValueError("Duplicate engine_id in model")
        object.__setattr__(self, "engines", engines)
        if not isinstance(self.capabilities, (tuple, list)):
            raise ValueError("capabilities must be a sequence")
        capabilities = tuple(sorted({_text(value, "capability") for value in self.capabilities}))
        object.__setattr__(self, "capabilities", capabilities)
        if self.weight_version is not None:
            _text(self.weight_version, "weight_version")
        if self.served_model_name is not None:
            _text(self.served_model_name, "served_model_name")
        if self.weight_source == "STATIC":
            if self.weight_version is not None:
                raise ValueError("STATIC models cannot publish an Actor weight version")
            if _DYNAMIC_CAPABILITIES.intersection(capability.lower() for capability in capabilities):
                raise ValueError("STATIC models cannot enable DCS, weight updates, or seed weight sync")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "state": self.state,
            "route_mode": self.route_mode,
            "router_url": self.router_url,
            "engines": [engine.to_dict() for engine in self.engines],
            "weight_source": self.weight_source,
            "weight_version": self.weight_version,
            "capabilities": list(self.capabilities),
            "served_model_name": self.served_model_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, model_id: str | None = None) -> ModelSnapshot:
        data = _fields(
            data,
            {
                "model_id",
                "state",
                "route_mode",
                "router_url",
                "engines",
                "weight_source",
                "weight_version",
                "capabilities",
                "served_model_name",
            },
            {"state", "route_mode", "engines", "weight_source"},
            "model",
        )
        if model_id is not None and data.get("model_id", model_id) != model_id:
            raise ValueError("Model key disagrees with model_id")
        values = dict(data)
        values["model_id"] = values.get("model_id", model_id)
        if not isinstance(values["engines"], (list, tuple)):
            raise ValueError("engines must be an array")
        values["engines"] = tuple(ReplicaSnapshot.from_dict(engine) for engine in values["engines"])
        return cls(**values)


@dataclass(frozen=True)
class RoutingSpec:
    default_model: str | None = None
    route_key_map: Mapping[str, str] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    policy: str = "round_robin"
    policy_revision: int = 0

    def __post_init__(self) -> None:
        if self.default_model is not None:
            _text(self.default_model, "default_model")
        object.__setattr__(self, "route_key_map", _string_map(self.route_key_map, "route_key_map"))
        object.__setattr__(self, "aliases", _string_map(self.aliases, "aliases"))
        _choice(self.policy, frozenset({"round_robin", "affinity"}), "routing policy")
        _integer(self.policy_revision, "policy_revision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_model": self.default_model,
            "route_key_map": dict(self.route_key_map),
            "aliases": dict(self.aliases),
            "policy": self.policy,
            "policy_revision": self.policy_revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutingSpec:
        data = _fields(
            data,
            {"default_model", "route_key_map", "aliases", "policy", "policy_revision"},
            set(),
            "routing",
        )
        return cls(**data)


@dataclass(frozen=True)
class RegistrySnapshot:
    role: str
    registry_epoch: str
    topology_revision: int
    models: Mapping[str, ModelSnapshot]
    routing: RoutingSpec = field(default_factory=RoutingSpec)
    phase_epoch: int = 0
    phase: str | None = None

    def __post_init__(self) -> None:
        _choice(self.role, INFERENCE_ROLES, "role")
        _text(self.registry_epoch, "registry_epoch")
        _integer(self.topology_revision, "topology_revision")
        _integer(self.phase_epoch, "phase_epoch")
        if self.phase is not None:
            _text(self.phase, "phase")
        if not isinstance(self.routing, RoutingSpec):
            raise ValueError("routing must be a RoutingSpec")
        models = dict(_mapping(self.models, "models"))
        for model_id, model in models.items():
            if not isinstance(model, ModelSnapshot) or model.model_id != model_id:
                raise ValueError("Each model must be a ModelSnapshot keyed by its model_id")
            if self.role in {"genrm", "teacher"} and model.weight_source != "STATIC":
                raise ValueError(f"{self.role} only supports STATIC weights")
        object.__setattr__(self, "models", MappingProxyType(models))
        targets = list(self.routing.aliases.values()) + list(self.routing.route_key_map.values())
        if self.routing.default_model is not None:
            targets.append(self.routing.default_model)
        if any(target not in models for target in targets):
            raise ValueError("Routing references an unknown model")
        if any(alias in models and alias != target for alias, target in self.routing.aliases.items()):
            raise ValueError("Model aliases must not shadow another model ID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "role": self.role,
            "registry_epoch": self.registry_epoch,
            "topology_revision": self.topology_revision,
            "phase_epoch": self.phase_epoch,
            "phase": self.phase,
            "routing": self.routing.to_dict(),
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegistrySnapshot:
        data = _fields(
            data,
            {
                "schema_version",
                "role",
                "registry_epoch",
                "topology_revision",
                "phase_epoch",
                "phase",
                "routing",
                "models",
            },
            {"schema_version", "role", "registry_epoch", "topology_revision", "routing", "models"},
            "registry",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 2:
            raise ValueError("Unsupported inference discovery schema_version")
        models = _mapping(data["models"], "models")
        values = {key: value for key, value in data.items() if key != "schema_version"}
        values["models"] = {
            model_id: ModelSnapshot.from_dict(model, model_id=model_id) for model_id, model in models.items()
        }
        values["routing"] = RoutingSpec.from_dict(data["routing"])
        return cls(**values)


def validate_snapshot(data: Mapping[str, Any]) -> RegistrySnapshot:
    return RegistrySnapshot.from_dict(data)

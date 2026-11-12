# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch


@dataclass(frozen=True)
class ConditionFieldSpec:
    name: str
    fields: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    input_dependencies: tuple[str, ...]
    preprocessing_dependencies: tuple[str, ...]
    max_tensors: int
    max_depth: int = 4
    deterministic: bool = True
    adapter_version: int = 1
    structure_version: int = 1


@dataclass(frozen=True)
class ConditionRequest:
    metadata: dict[str, Any]
    eligible: bool = True


def condition_key(spec: ConditionFieldSpec, metadata: dict[str, Any], generation: int) -> str:
    required = {"weights", "text", "precision", *spec.input_dependencies, *spec.preprocessing_dependencies}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Missing condition dependencies: {sorted(missing)}")
    payload = {
        "schema_version": 1,
        "adapter_version": spec.adapter_version,
        "structure_version": spec.structure_version,
        "field": spec.name,
        "encoder_generation": generation,
        "dependencies": metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        storage.data_ptr(),
        storage.nbytes(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
        tensor._version,
        tensor.requires_grad,
    )


class ConditionCache:
    def __init__(
        self, specs: tuple[ConditionFieldSpec, ...], device: torch.device, max_entries: int, max_bytes: int
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (max_entries, max_bytes)):
            raise ValueError("Condition cache limits must be positive integers")
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Condition cache requires unique field specifications")
        for spec in specs:
            if not spec.fields or len(set(spec.fields)) != len(spec.fields):
                raise ValueError("Condition cache fields must be nonempty and unique")
            if spec.max_depth <= 0 or spec.max_tensors <= 0:
                raise ValueError("Condition structure limits must be positive")
        self.specs = {spec.name: spec for spec in specs}
        self.device = torch.device(device)
        if self.device.type == "meta" or (self.device.type == "cuda" and self.device.index is None):
            raise ValueError("Condition cache requires a concrete device")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries: OrderedDict[tuple[str, str], tuple[Any, tuple[Any, ...], int]] = OrderedDict()
        self._closed = False
        self._entry_bytes_histogram = [0] * 64
        self._stats = dict.fromkeys(
            (
                "lookups",
                "local_hits",
                "evictions",
                "oversized_skips",
                "charged_bytes",
                "peak_charged_bytes",
                "peak_entries",
                "inserts",
                "inserted_bytes",
            ),
            0,
        )

    def _check(self, spec_name: str, key: str) -> ConditionFieldSpec:
        if self._closed:
            raise RuntimeError("Condition cache is closed")
        if spec_name not in self.specs:
            raise ValueError(f"Unknown condition field: {spec_name}")
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Condition cache keys must be SHA256 digests")
        return self.specs[spec_name]

    def _inspect(self, spec: ConditionFieldSpec, value: Any, *, detach: bool) -> tuple[Any, tuple[Any, ...], int]:
        if not isinstance(value, Mapping) or set(value) != set(spec.fields):
            raise ValueError(f"Invalid output fields for {spec.name}")
        tensors: list[tuple[Any, ...]] = []
        storages: dict[tuple[torch.device, int], int] = {}
        nodes = 0

        def visit(item: Any, depth: int) -> Any:
            nonlocal nodes
            nodes += 1
            if depth > spec.max_depth or nodes > 4 * spec.max_tensors + len(spec.fields) + 1:
                raise ValueError("Condition structure exceeds its limits")
            if type(item) is torch.Tensor:
                if item.layout != torch.strided or item.device != self.device or item.requires_grad:
                    raise ValueError("Condition tensors must be frozen dense tensors on the cache device")
                if item.is_inference():
                    raise ValueError("Condition tensors must have version counters")
                tensor = item.detach() if detach else item
                tensors.append(_tensor_metadata(tensor))
                if len(tensors) > spec.max_tensors:
                    raise ValueError("Condition tensor count exceeds its limit")
                storage = tensor.untyped_storage()
                if storage.nbytes():
                    storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
                return tensor
            if isinstance(item, Mapping):
                if any(type(key) is not str for key in item):
                    raise ValueError("Condition field names must be strings")
                return MappingProxyType({key: visit(child, depth + 1) for key, child in sorted(item.items())})
            if type(item) is tuple:
                return tuple(visit(child, depth + 1) for child in item)
            raise ValueError("Condition outputs must contain only tensors, tuples and field mappings")

        result = visit(value, 0)
        if not tensors:
            raise ValueError("Condition outputs must contain tensors")
        return result, tuple(tensors), sum(storages.values())

    def get(self, spec_name: str, key: str) -> Any | None:
        spec = self._check(spec_name, key)
        self._stats["lookups"] += 1
        identity = (spec_name, key)
        entry = self._entries.get(identity)
        if entry is None:
            return None
        value = self._validate(spec, identity)
        self._entries.move_to_end(identity)
        self._stats["local_hits"] += 1
        return value

    def _validate(self, spec: ConditionFieldSpec, identity: tuple[str, str]) -> Any:
        value, expected, charged = self._entries[identity]
        try:
            _, actual, size = self._inspect(spec, value, detach=False)
            if actual != expected or size != charged:
                raise ValueError("Cached condition tensor metadata changed")
        except ValueError:
            self._remove(identity)
            raise
        return value

    def _remove(self, identity: tuple[str, str]) -> None:
        _, _, size = self._entries.pop(identity)
        self._stats["charged_bytes"] -= size

    def put(self, spec_name: str, key: str, value: Any) -> bool:
        spec = self._check(spec_name, key)
        entry = self._inspect(spec, value, detach=True)
        identity = (spec_name, key)
        if identity in self._entries:
            self._validate(spec, identity)
            return False
        size = entry[2]
        if size > self.max_bytes:
            self._stats["oversized_skips"] += 1
            return False
        while len(self._entries) >= self.max_entries or self._stats["charged_bytes"] + size > self.max_bytes:
            self._remove(next(iter(self._entries)))
            self._stats["evictions"] += 1
        self._entries[identity] = entry
        self._stats["charged_bytes"] += size
        self._stats["inserts"] += 1
        self._stats["inserted_bytes"] += size
        self._entry_bytes_histogram[min(size.bit_length(), 63)] += 1
        self._stats["peak_entries"] = max(self._stats["peak_entries"], len(self._entries))
        self._stats["peak_charged_bytes"] = max(self._stats["peak_charged_bytes"], self._stats["charged_bytes"])
        return True

    def clear(self, spec_name: str | None = None) -> None:
        for identity in tuple(self._entries):
            if spec_name is None or identity[0] == spec_name:
                self._remove(identity)

    def close(self) -> None:
        self.clear()
        self._closed = True

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "entries": len(self._entries),
            "entry_bytes_histogram": list(self._entry_bytes_histogram),
        }

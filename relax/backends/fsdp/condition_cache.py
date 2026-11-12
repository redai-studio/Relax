# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import time
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from relax.engine.sft.condition_cache import ConditionCache, ConditionFieldSpec, condition_key


class ConditionCacheBinding:
    def __init__(self, model: torch.nn.Module, adapter: Any, device: torch.device, config: dict[str, Any]) -> None:
        self.model, self.adapter = model, adapter
        self.specs = adapter.condition_field_specs(model)
        if len(self.specs) != 1:
            raise ValueError("SFT caching requires one complete condition field per micro-batch")
        self.cache = ConditionCache(
            self.specs,
            device,
            config.get("condition_cache_max_entries", 256),
            config.get("condition_cache_max_bytes", 1073741824),
        )
        self.generations = dict.fromkeys((spec.name for spec in self.specs), 0)
        self.changed = dict.fromkeys(self.generations, False)
        self.version_tags = dict.fromkeys(self.generations, None)
        self._eligible: dict[str, bool] = {}
        self._snapshots: dict[str, Any] = {}
        self._managed: set[str] = set()
        self._hooks: list[Any] = []
        self._refresh = True
        self._closed = False
        self._active = False
        self.requests: dict[str, Any] = {}
        self.keys: dict[str, str] = {}
        self.values: dict[str, Any] = {}
        self._hits: dict[str, Any] = {}
        self.counters = dict.fromkeys(
            (
                "effective_reuses",
                "requests",
                "group_forced_online",
                "version_invalidations",
                "skipped_prefix_encodes",
                "skipped_vision_encodes",
                "version_check_ms",
                "lookup_ms",
                "control_ms",
                "restore_ms",
                "build_host_ms",
            ),
            0,
        )
        try:
            self._refresh_dependencies()
        except BaseException:
            self.close()
            raise

    def _snapshot(self, spec: ConditionFieldSpec) -> tuple[Any, bool]:
        modules: dict[str, torch.nn.Module] = {}
        tensors: dict[str, tuple[torch.Tensor, Any]] = {}
        eligible = spec.deterministic
        for path in spec.dependency_paths:
            dependency = self.model.get_submodule(path)
            for suffix, module in dependency.named_modules():
                modules[f"{path}.{suffix}"] = module
                eligible = eligible and not module.training
            for kind, values in (("parameter", dependency.named_parameters()), ("buffer", dependency.named_buffers())):
                for suffix, tensor in values:
                    name = f"{path}.{kind}.{suffix}"
                    eligible = eligible and not tensor.requires_grad
                    if name in self._managed:
                        continue
                    if isinstance(tensor, DTensor):
                        self._managed.add(name)
                        continue
                    metadata = (tensor.shape, tensor.dtype, tensor.device, tensor.requires_grad, tensor._version)
                    tensors[name] = (tensor, metadata)
        state = (modules, {name: module.training for name, module in modules.items()}, tensors)
        return state, eligible

    @staticmethod
    def _same_snapshot(left: Any, right: Any) -> bool:
        left_modules, left_training, left_tensors = left
        right_modules, right_training, right_tensors = right
        return (
            left_modules.keys() == right_modules.keys()
            and all(module is right_modules[name] for name, module in left_modules.items())
            and left_training == right_training
            and left_tensors.keys() == right_tensors.keys()
            and all(
                tensor is right_tensors[name][0] and meta == right_tensors[name][1]
                for name, (tensor, meta) in left_tensors.items()
            )
        )

    def _refresh_dependencies(self) -> None:
        specs = self.adapter.condition_field_specs(self.model)
        if specs != self.specs:
            raise ValueError("Condition field declarations changed during training")
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        dependencies = {self.model.get_submodule(path) for spec in self.specs for path in spec.dependency_paths}
        for module in dependencies:
            self._hooks.append(module.register_load_state_dict_post_hook(self._loaded))
        self.adapter.bind_condition_cache(self.model, self)
        self._managed.clear()
        for spec in self.specs:
            self._snapshots[spec.name], _ = self._snapshot(spec)
        self._refresh = False

    def _loaded(self, module: torch.nn.Module, incompatible_keys: Any) -> None:
        self.bump_encoder_version()

    def bump_encoder_version(self, spec_name: str | None = None, version_tag: Any = None) -> None:
        if self._closed or self._active:
            raise RuntimeError("Encoder updates require an open, idle condition cache")
        names = tuple(self.generations) if spec_name is None else (spec_name,)
        for name in names:
            if name not in self.generations:
                raise ValueError(f"Unknown condition field: {name}")
            self.cache.clear(name)
            self.generations[name] += 1
            self.changed[name] = True
            self.version_tags[name] = version_tag
            self.counters["version_invalidations"] += 1
        self._refresh = True

    def prepare(self, data: Any) -> list[tuple[int, int, int, int]]:
        if self._closed or self._active:
            raise RuntimeError("Condition requests require an open, idle cache")
        self.counters["requests"] += 1
        started = time.monotonic()
        if self._refresh:
            self._refresh_dependencies()
        eligibility = {}
        for spec in self.specs:
            snapshot, eligible = self._snapshot(spec)
            if not self._same_snapshot(self._snapshots[spec.name], snapshot):
                self.bump_encoder_version(spec.name)
            eligibility[spec.name] = eligible
            self._snapshots[spec.name] = snapshot
        if self._refresh:
            self._refresh_dependencies()
        self.counters["version_check_ms"] += (time.monotonic() - started) * 1000
        started = time.monotonic()
        self.requests = self.adapter.prepare_condition_request(self.model, data)
        if self.requests.keys() != self.generations.keys():
            raise ValueError("Condition requests do not match field declarations")
        states = []
        for spec in self.specs:
            name = spec.name
            eligible = eligibility[name] and self.requests[name].eligible
            if name in self._eligible and eligible != self._eligible[name]:
                self.bump_encoder_version(name)
            self._eligible[name] = eligible
            hit = None
            if eligible:
                key = condition_key(spec, self.requests[name].metadata, self.generations[name])
                self.keys[name] = key
                hit = self.cache.get(name, key)
            else:
                self.cache.clear(name)
            self._hits[name] = hit
            states.append((int(self.changed[name]), self.generations[name], int(eligible), int(hit is not None)))
        if self._refresh:
            self._refresh_dependencies()
        self.counters["lookup_ms"] += (time.monotonic() - started) * 1000
        return states

    def decide(self, states: list[list[tuple[int, int, int, int]]]) -> None:
        for index, spec in enumerate(self.specs):
            name = spec.name
            ranks = [state[index] for state in states]
            changed = any(state[0] for state in ranks)
            versions = {state[1] for state in ranks}
            if not changed and len(versions) != 1:
                raise ValueError("Condition encoder generations differ without an update")
            if changed:
                self.cache.clear(name)
                self.generations[name] = max(versions) + 1
                self.changed[name] = False
                if self._eligible[name]:
                    self.keys[name] = condition_key(spec, self.requests[name].metadata, self.generations[name])
            reuse = not changed and all(state[2] and state[3] for state in ranks)
            if reuse:
                self.values[name] = self._hits[name]
                self.counters["effective_reuses"] += 1
            elif self._hits[name] is not None:
                self.counters["group_forced_online"] += 1
        self._hits.clear()
        self.requests.clear()
        self._active = True

    def put(self, spec_name: str, value: Any) -> bool:
        if not self._active:
            raise RuntimeError("Condition insertion requires an active micro-batch")
        if spec_name not in self.keys or not self._eligible[spec_name]:
            return False
        return self.cache.put(spec_name, self.keys[spec_name], value)

    def end_micro(self) -> None:
        self.values.clear()
        self.keys.clear()
        self._hits.clear()
        self.requests.clear()
        self._active = False

    def stats(self) -> dict[str, Any]:
        stats = {**self.cache.stats(), **self.counters}
        stats["local_hit_rate"] = stats["local_hits"] / max(stats["lookups"], 1)
        stats["effective_reuse_rate"] = stats["effective_reuses"] / max(stats["requests"], 1)
        return stats

    def close(self) -> None:
        if self._closed:
            return
        self.end_micro()
        self.cache.close()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._snapshots.clear()
        self._managed.clear()
        self.version_tags.clear()
        try:
            self.adapter.bind_condition_cache(self.model, None)
        finally:
            self.model = self.adapter = None
            self._closed = True

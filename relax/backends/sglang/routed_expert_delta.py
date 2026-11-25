# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Discover and load receiver-local slices through SGLang's native expert
loader."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps

import torch


_HF_EXPERT = re.compile(r"(.+\.experts)\.(0|[1-9][0-9]*)\.(gate_proj|up_proj|down_proj)\.weight")
EXPERT_ROUTE_CAPACITY = 64 << 10
_GROUPED_EXPERT = re.compile(r"(.+\.experts)\.(gate_up_proj|down_proj)(?:\.weight)?")
_PROJECTIONS = {"gate_proj": ("w1", 0), "up_proj": ("w3", 0), "down_proj": ("w2", 1)}
_NATIVE_METHODS = (
    "weight_loader",
    "_weight_loader_impl",
    "_load_gguf_weight",
    "_weight_loader_physical",
    "_map_global_expert_id_to_local_expert_id",
    "_load_model_weight_or_group_weight_scale",
    "_load_w13",
    "_load_w2",
)


@dataclass(frozen=True)
class ExpertSlice:
    """Canonical HF coordinate window, independent of the installed gate/up
    order."""

    shape: tuple[int, int]
    axis: int
    start: int
    length: int

    @property
    def local_shape(self) -> tuple[int, int]:
        return (self.shape[0], self.length) if self.axis else (self.length, self.shape[1])


@dataclass(frozen=True)
class _Geometry:
    expert_start: int
    expert_stop: int
    intermediate_start: int
    intermediate_stop: int
    intermediate_size: int
    hidden_size: int
    dtype: torch.dtype

    def shard(self, projection: str) -> ExpertSlice:
        _, axis = _PROJECTIONS[projection]
        shape = (self.hidden_size, self.intermediate_size) if axis else (self.intermediate_size, self.hidden_size)
        return ExpertSlice(shape, axis, self.intermediate_start, self.intermediate_stop - self.intermediate_start)

    @property
    def entries(self) -> int:
        return (self.expert_stop - self.expert_start) * len(_PROJECTIONS)


@dataclass
class _Layer:
    name: str
    module: torch.nn.Module
    geometry: _Geometry
    prefixes: set[str] = field(default_factory=set)


def _native_sglang():
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE, UnquantizedFusedMoEMethod

    return FusedMoE, UnquantizedFusedMoEMethod, get_global_expert_location_metadata


def _identity_experts(module, expert_locations: Callable[[], object]) -> bool:
    locations = expert_locations()
    if locations is None:
        return True
    physical = getattr(locations, "physical_to_logical_map_cpu", None)
    inverse = getattr(locations, "logical_to_all_physical_map_cpu", None)
    layer_id = module.layer_id
    experts = module._num_global_routed
    if (
        not isinstance(physical, torch.Tensor)
        or not isinstance(inverse, torch.Tensor)
        or physical.device.type != "cpu"
        or inverse.device.type != "cpu"
        or physical.ndim != 2
        or inverse.ndim != 3
        or type(layer_id) is not int
        or type(experts) is not int
        or not 0 <= layer_id < min(inverse.shape[0], physical.shape[0])
        or physical.shape[1] != experts
        or inverse.shape[1] != experts
        or inverse.shape[2] < 1
    ):
        return False
    logical = torch.arange(experts, dtype=physical.dtype)
    return (
        torch.equal(physical[layer_id], logical)
        and torch.equal(inverse[layer_id, :, 0], logical)
        and bool((inverse[layer_id, :, 1:] == -1).all())
    )


def _geometry(module, native_layer, unquantized, expert_locations) -> _Geometry | str:
    if (
        type(module.quant_method) is not unquantized
        or module.quant_config is not None
        or not module.moe_runner_config.is_gated
        or module.use_padded_loading
        or module.use_presharded_weights
        or module.use_triton_kernels
        or module.use_flashinfer_trtllm_moe
        or module._has_fused_shared
        or any(getattr(type(module), name) is not getattr(native_layer, name) for name in _NATIVE_METHODS)
    ):
        return "unsupported_native_expert_loader"
    parameters = (module.get_parameter("w13_weight"), module.get_parameter("w2_weight"))
    gate_up, down = parameters
    if (
        gate_up.ndim != 3
        or down.ndim != 3
        or not gate_up.is_floating_point()
        or down.dtype != gate_up.dtype
        or down.device != gate_up.device
        or any(
            getattr(p, "is_transposed", False)
            or getattr(p, "is_gguf_weight", False)
            or getattr(p, "is_gguf_weight_type", False)
            or getattr(p, "_sglang_require_global_experts", False)
            or getattr(p, "weight_loader", None) != module.weight_loader
            for p in parameters
        )
    ):
        return "unsupported_expert_storage"
    ep, ep_rank = module.moe_ep_size, module.moe_ep_rank
    tp, tp_rank = module.moe_tp_size, module.moe_tp_rank
    local = module._num_local_routed
    width = module.intermediate_size_per_partition
    hidden = gate_up.shape[2]
    if (
        any(type(x) is not int for x in (ep, tp, ep_rank, tp_rank, local, width))
        or min(ep, tp, local, width, hidden) <= 0
        or not (0 <= ep_rank < ep and 0 <= tp_rank < tp)
        or module._num_global_routed != ep * local
        or tuple(gate_up.shape) != (local, 2 * width, hidden)
        or tuple(down.shape) != (local, hidden, width)
    ):
        return "unsupported_expert_geometry"
    if not _identity_experts(module, expert_locations):
        return "remapped_or_redundant_experts"
    return _Geometry(
        ep_rank * local,
        (ep_rank + 1) * local,
        tp_rank * width,
        (tp_rank + 1) * width,
        tp * width,
        hidden,
        gate_up.dtype,
    )


class RoutedExpertReceiver:
    """Qualify each rank's destination with original FULL loader
    invocations."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        try:
            self._native = _native_sglang()
        except (ImportError, AttributeError):
            self._native = None
        self._layers: list[_Layer] = []
        self._observed: dict[tuple[str, str], set[tuple[int, str]]] = {}
        self._qualified: dict[str, _Layer] = {}
        self._fallbacks: Counter[str] = Counter()
        self._input_prefixes: set[str] = set()
        self._grouped_prefixes: set[str] = set()
        self._load_identity = self.model.load_weights
        if self._native is None:
            self._fallbacks["unsupported_sglang_native_api"] += 1
        else:
            for name, module in model.named_modules():
                if not isinstance(module, self._native[0]):
                    continue
                try:
                    geometry = _geometry(module, *self._native)
                except (AttributeError, IndexError):
                    geometry = "unsupported_sglang_native_api"
                if isinstance(geometry, _Geometry):
                    self._layers.append(_Layer(name, module, geometry))
                else:
                    self._fallbacks[geometry] += 1

    def load_full(self, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
        by_tensor = {id(value): (match, value) for name, value in weights if (match := _HF_EXPERT.fullmatch(name))}
        self._input_prefixes.update(match[1] for match, _ in by_tensor.values())
        self._grouped_prefixes.update(match[1] for name, _ in weights if (match := _GROUPED_EXPERT.fullmatch(name)))
        if not by_tensor or not self._layers:
            self.model.load_weights(weights)
            return
        restored = []
        observed = []

        def wrapper(layer, original, parameter, expected_axis):
            @wraps(original)
            def verify(target, loaded_weight, weight_name, shard_id, expert_id):
                entry = by_tensor.get(id(loaded_weight))
                if target is parameter and entry is not None:
                    match, value = entry
                    prefix, index, projection = match.groups()
                    index = int(index)
                    expected_shard, axis = _PROJECTIONS[projection]
                    geometry = layer.geometry
                    if (
                        shard_id == expected_shard
                        and axis == expected_axis
                        and expert_id == index
                        and geometry.expert_start <= index < geometry.expert_stop
                        and value.dtype == geometry.dtype
                        and tuple(value.shape) == geometry.shard(projection).shape
                    ):
                        observed.append((layer, prefix, index, projection))
                return original(target, loaded_weight, weight_name, shard_id, expert_id)

            return verify

        try:
            for layer in self._layers:
                for name, axis in (("w13_weight", 0), ("w2_weight", 1)):
                    parameter = layer.module.get_parameter(name)
                    original = parameter.weight_loader
                    restored.append((parameter, original))
                    parameter.weight_loader = wrapper(layer, original, parameter, axis)
            self.model.load_weights(weights)
        finally:
            for parameter, original in restored:
                parameter.weight_loader = original
        for layer, prefix, index, projection in observed:
            layer.prefixes.add(prefix)
            self._observed.setdefault((layer.name, prefix), set()).add((index, projection))

    def finalize_full(self) -> dict:
        self._qualified.clear()
        collisions = set()
        for layer in self._layers:
            if len(layer.prefixes) != 1:
                continue
            [prefix] = layer.prefixes
            if (layer.name, prefix) not in self._observed or len(
                self._observed[layer.name, prefix]
            ) != layer.geometry.entries:
                continue
            if prefix in self._qualified:
                collisions.add(prefix)
            self._qualified[prefix] = layer
        for prefix in collisions:
            del self._qualified[prefix]
        self._observed.clear()
        self._fallbacks["unqualified_native_inputs"] += len(self._input_prefixes - self._qualified.keys())
        self._fallbacks["grouped_expert_inputs"] += len(self._grouped_prefixes)
        self._input_prefixes.clear()
        self._grouped_prefixes.clear()
        return dict(
            routes=[self._descriptor(prefix, layer.geometry) for prefix, layer in sorted(self._qualified.items())],
            fallbacks={reason: count for reason, count in sorted(self._fallbacks.items()) if count},
        )

    def route_ack(self) -> bytes:
        report = self.finalize_full()
        raw = json.dumps(report, separators=(",", ":")).encode()
        if len(raw) + 4 > EXPERT_ROUTE_CAPACITY:
            self._qualified.clear()
            raw = b'{"routes":[],"fallbacks":{"route_directory_limit":1}}'
        return len(raw).to_bytes(4, "little") + raw

    @staticmethod
    def _descriptor(prefix: str, g: _Geometry) -> dict:
        return dict(
            prefix=prefix,
            expert_start=g.expert_start,
            expert_stop=g.expert_stop,
            intermediate_start=g.intermediate_start,
            intermediate_stop=g.intermediate_stop,
            intermediate_size=g.intermediate_size,
            hidden_size=g.hidden_size,
            dtype=str(g.dtype).removeprefix("torch."),
        )

    def _resolve(self, name: str) -> tuple[_Layer, ExpertSlice] | None:
        match = _HF_EXPERT.fullmatch(name)
        if match is not None:
            prefix, expert, projection = match.groups()
            layer = self._qualified.get(prefix)
            if layer is not None and layer.geometry.expert_start <= int(expert) < layer.geometry.expert_stop:
                return layer, layer.geometry.shard(projection)
        return None

    def local_slice(self, name: str) -> ExpertSlice | None:
        resolved = self._resolve(name)
        return resolved[1] if resolved is not None else None

    def load_local(self, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
        """Install a target-only chunk.

        The caller owns the device-local NaN mask.
        """
        layers: dict[str, _Layer] = {}
        for name, value in weights:
            resolved = self._resolve(name)
            if resolved is None:
                raise ValueError(f"Unqualified receiver-local expert slot: {name}")
            layer, shard = resolved
            if tuple(value.shape) != shard.local_shape or value.dtype != layer.geometry.dtype:
                raise ValueError(f"Receiver-local expert slice differs from native layout: {name}")
            layers[layer.name] = layer
        if (
            self.model.load_weights != self._load_identity
            or self._native is None
            or any(
                self.model.get_submodule(name) is not layer.module
                or _geometry(layer.module, *self._native) != layer.geometry
                for name, layer in layers.items()
            )
        ):
            raise ValueError("Receiver expert layout changed after its native FULL seed")
        try:
            for layer in layers.values():
                layer.module.use_presharded_weights = True
            self.model.load_weights(weights)
        finally:
            for layer in layers.values():
                layer.module.use_presharded_weights = False

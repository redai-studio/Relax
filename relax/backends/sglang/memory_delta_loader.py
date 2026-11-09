# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe the ordinary Dense BF16 loader once, then overwrite sparse
positions.

No global copy_ replacement, NaN sentinel, floating addition or disk
checkpoint. Only contiguous, value-preserving loader copies are admitted by the
first version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CopyRoute:
    parameter: str
    source_start: int
    target_start: int
    elements: int
    address: int
    shape: tuple[int, ...]


def record_full_load(model: Any, named_tensors: list[tuple[str, Any]]) -> list[CopyRoute]:
    """Use the real model loader for QKV/gate-up placement; observe this
    thread."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    parameters = dict(model.named_parameters(remove_duplicate=False))
    destinations = [(name, p, p.data_ptr(), p.numel() * p.element_size()) for name, p in parameters.items()]
    sources, offset = [], 0
    for name, tensor in named_tensors:
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("The memory loader requires contiguous BF16 HF inputs")
        sources.append((name, tensor.data_ptr(), tensor.numel() * 2, offset))
        offset += tensor.numel()
    routes = []
    protected = [(address, length) for _, _, address, length in destinations]
    protected.extend((address, length) for _, address, length, _ in sources)

    class ObserveCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is not torch.ops.aten.copy_.default:
                for index, argument in enumerate(func._schema.arguments):
                    value = args[index] if index < len(args) else (kwargs or {}).get(argument.name)
                    if argument.alias_info is not None and argument.alias_info.is_write:
                        values = value if isinstance(value, (list, tuple)) else [value]
                        if any(
                            isinstance(t, torch.Tensor) and address <= t.data_ptr() < address + length
                            for t in values
                            for address, length in protected
                        ):
                            raise ValueError("The model loader modifies protected values outside a direct copy")
            if func is torch.ops.aten.copy_.default:
                target, source = args[:2]
                if any(address <= target.data_ptr() < address + length for _, address, length, _ in sources):
                    raise ValueError("The model loader modified an HF source tensor")
                # Restrict observation to actual parameter storage, including fused views.
                for name, parameter, address, length in destinations:
                    target_address = target.data_ptr()
                    if address <= target_address < address + length:
                        if (
                            not target.is_contiguous()
                            or not source.is_contiguous()
                            or target.dtype != torch.bfloat16
                            or source.dtype != target.dtype
                            or target.numel() != source.numel()
                        ):
                            raise ValueError("The model loader needs a non-contiguous or dtype-changing mapping")
                        source_address = source.data_ptr()
                        match = next(
                            (
                                (start, base)
                                for _, start, size, base in sources
                                if start <= source_address and source_address + source.numel() * 2 <= start + size
                            ),
                            None,
                        )
                        if match is None or target_address + target.numel() * 2 > address + length:
                            raise ValueError("The loader transformed source values or wrote beyond a parameter")
                        start, base = match
                        routes.append(
                            CopyRoute(
                                name,
                                base + (source_address - start) // 2,
                                (target_address - address) // 2,
                                source.numel(),
                                address,
                                tuple(parameter.shape),
                            )
                        )
                        break
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), ObserveCopies():
        model.load_weights(named_tensors)
    intervals = sorted((r.source_start, r.source_start + r.elements) for r in routes)
    covered = 0
    for start, end in intervals:
        if start > covered:
            raise ValueError("The ordinary model loader ignored an HF tensor or slice")
        covered = max(covered, end)
    if covered != offset:
        raise ValueError("The ordinary model loader did not consume every HF value")
    # A custom operation can bypass copy_ observation. Prove every recorded
    # mapping preserved the input bits before admitting sparse updates.
    with torch.no_grad():
        flat = torch.cat([tensor.reshape(-1) for _, tensor in named_tensors]).view(torch.int16)
        for route in routes:
            target = parameters[route.parameter].view(torch.int16).reshape(-1)
            if not torch.equal(
                target[route.target_start : route.target_start + route.elements],
                flat[route.source_start : route.source_start + route.elements],
            ):
                raise ValueError("The ordinary model loader changed source value bits")
    return routes


def apply_sparse(model: Any, routes: list[CopyRoute], metadata: dict, payload: bytes) -> None:
    """The caller validates the frame and pauses generation before this
    call."""
    import numpy as np
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=False))
    validate_routes(model, routes)
    count = metadata["changed"]
    if count == 0:
        return
    positions_cpu = np.frombuffer(payload, dtype="<i4", count=count)
    device = next(iter(parameters.values())).device
    # One encoded H2D transfer. CPU searchsorted avoids CUDA nonzero per parameter.
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
    positions = raw[: count * 4].view(torch.int32)
    values = raw[count * 4 :].view(torch.int16)
    with torch.no_grad():
        for route in routes:
            parameter = parameters[route.parameter]
            left, right = np.searchsorted(
                positions_cpu, [route.source_start, route.source_start + route.elements]
            ).tolist()
            if left != right:
                indices = positions[left:right].to(torch.int64) - route.source_start + route.target_start
                parameter.view(torch.int16).reshape(-1).index_copy_(0, indices, values[left:right])


def validate_routes(model: Any, routes: list[CopyRoute]) -> None:
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=False))
    for route in routes:
        parameter = parameters.get(route.parameter)
        if (
            parameter is None
            or parameter.data_ptr() != route.address
            or tuple(parameter.shape) != route.shape
            or parameter.dtype != torch.bfloat16
            or not parameter.is_contiguous()
        ):
            raise ValueError("Inference parameter storage changed; require a new FULL load")


class MemoryModelReceiver:
    """Fail closed after a partial apply; a complete FULL reset can recover.

    This object is owned by one single-GPU SGLang worker. The engine group must
    remain paused until every worker ACKs and its KV cache has been flushed.
    """

    def __init__(self, model: Any, *, host_budget_bytes: int = 8 << 30) -> None:
        self.model, self.host_budget_bytes = model, host_budget_bytes
        self.stream: str | None = None
        self.version = -1
        self.schema: str | None = None
        self.update: str | None = None
        self.poisoned = False
        self.layouts: list[tuple] = []
        self.routes: list[list[CopyRoute]] = []

    def apply(self, manifest: dict, payloads: list[bytes]) -> None:
        import torch

        from relax.distributed.weight_sync.memory_delta import validate_bucket

        # Derived persistent buffers would need their own source/update rules.
        # Rotary/cache buffers marked nonpersistent are outside model weights.
        if any(
            value is not None and name not in module._non_persistent_buffers_set
            for module in self.model.modules()
            for name, value in module._buffers.items()
        ):
            raise ValueError("Memory delta currently requires a model without persistent buffers")
        metadata = manifest["buckets"]
        if len(metadata) != len(payloads) or sum(map(len, payloads)) > self.host_budget_bytes:
            raise ValueError("Invalid memory update payload count or host budget")
        for item, payload in zip(metadata, payloads):
            validate_bucket(item, payload, max_bytes=self.host_budget_bytes)
        reset = manifest.get("reset") is True
        if type(manifest.get("version")) is not int or type(manifest.get("base")) is not int:
            raise ValueError("Invalid memory version")
        if manifest["version"] != manifest["base"] + 1:
            raise ValueError("Invalid memory version sequence")
        if reset and any(item["encoding"] != "full" for item in metadata):
            raise ValueError("A memory reset requires every bucket in FULL form")
        if self.poisoned and not reset:
            raise RuntimeError("A partial memory apply requires a FULL reset")
        if not self.poisoned and manifest["update"] == self.update:
            for routes in self.routes:
                validate_routes(self.model, routes)
            return  # Lost socket ACK: same immutable update is already applied.
        if not reset and (
            self.stream != manifest["stream"] or self.schema != manifest["schema"] or self.version != manifest["base"]
        ):
            raise ValueError("Memory stream, schema or base version mismatch")
        if reset and self.stream == manifest["stream"] and manifest["version"] <= self.version:
            raise ValueError("A memory reset cannot roll back an applied version")
        layouts = [tuple((name, tuple(shape)) for name, shape in item["names"]) for item in metadata]
        if not reset and layouts != self.layouts:
            raise ValueError("Memory bucket layout changed")
        if reset:
            self.routes = [[] for _ in metadata]
        else:
            for routes in self.routes:
                validate_routes(self.model, routes)
        # Every byte and the entire protocol sequence passed before mutation.
        # Any exception below may leave GPU weights mixed; never ACK it.
        self.poisoned = True
        device = next(self.model.parameters()).device
        for index, (item, payload) in enumerate(zip(metadata, payloads)):
            if item["encoding"] == "full":
                flat = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).to(device)
                named, offset = [], 0
                for name, shape in layouts[index]:
                    size = torch.Size(shape).numel()
                    named.append((name, flat[offset : offset + size].view(shape)))
                    offset += size
                self.routes[index] = record_full_load(self.model, named)
            else:
                apply_sparse(self.model, self.routes[index], item, payload)
        self._check_coverage()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.stream, self.schema = manifest["stream"], manifest["schema"]
        self.version, self.update = manifest["version"], manifest["update"]
        self.layouts, self.poisoned = layouts, False

    def _check_coverage(self) -> None:
        # Tied parameter aliases share a single storage interval. Every actual
        # parameter element must be supplied by the ordinary FULL loader.
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        intervals: dict[tuple, list[tuple[int, int]]] = {}
        for routes in self.routes:
            for route in routes:
                key = (parameters[route.parameter].device, route.address, route.shape)
                intervals.setdefault(key, []).append((route.target_start, route.target_start + route.elements))
        for parameter in parameters.values():
            key = (parameter.device, parameter.data_ptr(), tuple(parameter.shape))
            covered = 0
            for start, end in sorted(intervals.get(key, [])):
                if start < covered:
                    raise ValueError("Overlapping FULL routes require an explicit verified alias mapping")
                if start > covered:
                    raise ValueError("The FULL update did not cover every inference parameter")
                covered = max(covered, end)
            if covered != parameter.numel():
                raise ValueError("The FULL update did not cover every inference parameter")

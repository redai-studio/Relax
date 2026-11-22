# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Single-node read-only CUDA IPC views and checked rectangle copies.

The caller retains source storage and waits for every consumer's CUDA
completion before allowing training to modify it. Create one export per
consumer, not one export reused by multiple consumers (PyTorch IPC refcounts
are per consumer).
"""

import base64
import json
import math
import time
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from .plan import fingerprint, validate_coverage
from .rectangle_batch import rectangle_metadata


_FAILED_READS: list[tuple] = []


def apply_ipc_regions(
    parameters: Mapping[str, torch.Tensor],
    writes: list[dict],
    bindings: list[dict],
    *,
    batched: bool = True,
    owner_rotation: int = 0,
    metadata_builder=None,
    timings: dict | None = None,
    verify_native: bool = False,
) -> None:
    """Bind fresh sources, execute checked regions, fence and retire imports.

    Every binding owns one consumer export. A view value of None selects the
    complete exported tensor; other views are relative to that export's flat
    storage. Family adapters validate coverage and version before this call. No
    source or target reference survives a successful completion. A failed fence
    retains all imported allocations and launch metadata until teardown.
    """
    if verify_native and not batched:
        raise ValueError("Native bitwise validation requires batched IPC")
    expected = {(write["source_rank"], write["source"]) for write in writes}
    keys = [(binding["source_rank"], name) for binding in bindings for name in binding["views"]]
    if len(keys) != len(set(keys)) or set(keys) != expected or any(not item["views"] for item in bindings):
        raise ValueError("IPC bindings differ from the complete source plan")
    peers, copies, metadata, imports = {}, [], [], []
    native_checks = [] if verify_native else None
    native_mismatch = None
    source = target = storage = None
    started = time.monotonic() if timings is not None else None
    try:
        for binding in bindings:
            storage = import_tensor(binding["payload"], torch.cuda.current_device())
            imports.append(storage)
            views = binding["views"]
            if any(value is None for value in views.values()):
                if len(views) != 1 or next(iter(views.values())) is not None:
                    raise ValueError("A complete IPC tensor cannot mix with storage views")
                peers[binding["source_rank"], next(iter(views))] = storage
            else:
                peers.update(
                    ((binding["source_rank"], name), tensor)
                    for name, tensor in bind_storage_views(storage, views).items()
                )
            storage = None
        if timings is not None:
            timings["import"] = time.monotonic() - started
            started = time.monotonic()
        if batched:
            from .rectangle_batch import launch_rectangles

            groups = (
                metadata_builder(peers, parameters)
                if metadata_builder is not None
                else rectangle_metadata(peers, parameters, writes, owner_rotation=owner_rotation)
            )
            if groups:
                if verify_native:
                    launch_rectangles(groups, next(iter(peers.values())).device, metadata, native_checks=native_checks)
                    native_mismatch = torch.stack([check.any() for check in native_checks]).any()
                    metadata.append(native_mismatch)
                else:
                    launch_rectangles(groups, next(iter(peers.values())).device, metadata)
        else:
            # Validate every rectangle before enqueuing the first write.
            copies.extend(
                rectangle_views(peers[write["source_rank"], write["source"]], parameters[write["param"]], write)
                for write in writes
            )
            with torch.no_grad():
                for source, target in copies:
                    target.copy_(source)
        if timings is not None:
            timings["launch"] = time.monotonic() - started
            started = time.monotonic()
    finally:
        try:
            torch.cuda.synchronize()
        except BaseException:
            _FAILED_READS.append((parameters, peers, copies, metadata, imports))
            raise
        source = target = storage = None
        copies.clear()
        metadata.clear()
        peers.clear()
        imports.clear()
        if timings is not None:
            timings["fence_close"] = time.monotonic() - started
    # Cold only: the existing completion fence covers every copy and audit
    # tile. No model snapshot or model-sized GPU-to-CPU transfer is needed.
    if verify_native and (native_mismatch is None or native_mismatch.item()):
        raise RuntimeError("Expert IPC differs from the native-loader reference or has unverified tiles")


def group_storage_views(tensors: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    """Describe current storage aliases without copying or caching addresses.

    The returned flat views are live only for this transfer. Descriptions stay
    relative to their allocation; fresh allocations may have different offsets.
    """
    allocations, sources, views = {}, {}, {}
    for name, tensor in sorted(tensors.items()):
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or not tensor.numel():
            raise ValueError("Nonempty contiguous BF16 storage requires native reseeding")
        storage = tensor.untyped_storage()
        capacity, offset = storage.nbytes(), tensor.storage_offset()
        if capacity % 2 or offset < 0 or (offset + tensor.numel()) * 2 > capacity:
            raise ValueError("IPC source view exceeds its storage")
        identity = (tensor.device, storage.data_ptr())
        if identity not in allocations:
            key = f"storage-{len(sources)}"
            allocations[identity] = key
            count = capacity // 2
            sources[key] = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                storage, 0, (1, count), (count, 1)
            )
        views[name] = dict(source=allocations[identity], offset=offset, shape=list(tensor.shape))
    return sources, views


def bind_storage_views(storage: torch.Tensor, views: dict) -> dict[str, torch.Tensor]:
    """Restore checked contiguous views from a single imported allocation."""
    if storage.dtype != torch.bfloat16 or not storage.is_contiguous() or storage.ndim != 2 or storage.shape[0] != 1:
        raise ValueError("IPC binding requires one flat BF16 allocation")
    if not views:
        raise ValueError("IPC allocation has no consumers")
    result, flat = {}, storage.view(-1)
    for name, spec in views.items():
        shape, offset = spec["shape"], spec["offset"]
        if (
            not isinstance(name, str)
            or not name
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
            or type(offset) is not int
            or offset < 0
            or offset + math.prod(shape) > storage.numel()
        ):
            raise ValueError("IPC binding view exceeds its allocation")
        result[name] = flat.narrow(0, offset, math.prod(shape)).view(shape)
    return result


def export_tensor(tensor: torch.Tensor) -> dict:
    if tensor.dtype != torch.bfloat16 or tensor.ndim != 2 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("Direct IPC requires a contiguous two-dimensional BF16 CUDA source")
    rebuild, args = reduce_tensor(tensor.detach())
    if rebuild is not rebuild_cuda_tensor or len(args) != 15 or args[10]:
        raise RuntimeError("Unsupported PyTorch CUDA IPC reduction protocol")
    return dict(
        shape=list(args[1]),
        stride=list(args[2]),
        offset=args[3],
        storage_size=args[8],
        storage_offset=args[9],
        handle=base64.b64encode(args[7]).decode(),
        ref_handle=base64.b64encode(args[11]).decode(),
        ref_offset=args[12],
        event=base64.b64encode(args[13]).decode(),
        event_sync=args[14],
    )


def import_tensor(payload: dict, device: int) -> torch.Tensor:
    shape, stride = payload["shape"], payload["stride"]
    if len(shape) != 2 or min(shape) <= 0 or stride != [shape[1], 1] or payload["offset"] < 0:
        raise ValueError("Invalid direct IPC source layout")
    if (payload["offset"] + shape[0] * shape[1]) * 2 > payload["storage_size"]:
        raise ValueError("Direct IPC view exceeds exported storage")
    # Map the allocation into the consumer's context. Source and destination
    # Ray workers can have different CUDA_VISIBLE_DEVICES namespaces.
    return rebuild_cuda_tensor(
        torch.Tensor,
        tuple(shape),
        tuple(stride),
        payload["offset"],
        torch.storage.TypedStorage,
        torch.bfloat16,
        device,
        base64.b64decode(payload["handle"]),
        payload["storage_size"],
        payload["storage_offset"],
        False,
        base64.b64decode(payload["ref_handle"]),
        payload["ref_offset"],
        base64.b64decode(payload["event"]),
        payload["event_sync"],
    )


def rectangle_views(source: torch.Tensor, destination: torch.Tensor, write: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not source.is_contiguous() or not destination.is_contiguous() or source.device != destination.device:
        raise ValueError("IPC views must be contiguous and mapped into the receiving CUDA context")
    if source.dtype != destination.dtype:
        raise ValueError("IPC rectangle source and destination dtypes differ")
    dtype = source.dtype
    source = source.view(torch.uint8).reshape(-1)
    destination = destination.detach().view(torch.uint8).reshape(-1)
    rows, width = write["rows"], write["row_bytes"]
    if min(rows, width) <= 0:
        raise ValueError("Empty IPC rectangle")
    views = []
    for tensor, side in ((source, "src"), (destination, "dst")):
        offset, stride = write[f"{side}_offset"], write[f"{side}_stride"]
        if offset < 0 or stride < width or offset + (rows - 1) * stride + width > tensor.numel():
            raise ValueError("IPC rectangle exceeds parameter bounds")
        views.append(tensor.as_strided((rows, width), (stride, 1), tensor.storage_offset() + offset))
    # Keep element width in the copy kernel. Byte-wide strided kernels waste
    # bandwidth for BF16 weights even though the wire plan uses byte offsets.
    return views[0].view(dtype), views[1].view(dtype)


def copy_rectangle(source: torch.Tensor, destination: torch.Tensor, write: dict) -> None:
    source_view, destination_view = rectangle_views(source, destination, write)
    destination_view.copy_(source_view)


def validate_targets(
    writes: list[dict], parameters: Mapping[str, torch.Tensor], expected: dict, *, source_world_size: int
) -> None:
    """Require exact coverage of both native matrices in every MoE layer."""
    if set(parameters) != set(expected):
        raise ValueError("Expert IPC geometry does not describe the complete native parameter set")
    layouts, allocations = {}, {}
    for name, shape in expected.items():
        parameter = parameters[name]
        if parameter.dtype != torch.bfloat16 or not parameter.is_contiguous() or tuple(parameter.shape) != shape:
            raise ValueError(f"Native direct projection storage differs from negotiated geometry: {name}")
        size = parameter.numel() * 2
        storage = parameter.untyped_storage()
        offset = parameter.storage_offset() * 2
        if offset < 0 or offset + size > storage.nbytes():
            raise ValueError(f"Native expert view exceeds its storage: {name}")
        # Distinct storage wrappers (for example DLPack views) may still
        # overlap physically. A wrapper's base is not an allocation identity.
        pointer = parameter.data_ptr()
        allocations.setdefault(parameter.device, []).append((pointer, pointer + size))
        layouts[name] = {"nbytes": size}
    for ranges in allocations.values():
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("Native expert parameters physically overlap")
    covered = validate_coverage(writes, layouts, source_world_size=source_world_size)
    if covered != set(layouts):
        raise ValueError("Expert IPC plan does not cover every native parameter")


def _selected_writes(writes: list[dict], write_ids: frozenset[str] | None) -> list[dict]:
    """Select delivery only after the caller validates the complete union."""
    if write_ids is None:
        return writes
    if not isinstance(write_ids, frozenset) or not write_ids <= {write["id"] for write in writes}:
        raise ValueError("Expert IPC selection contains unknown write IDs")
    return [write for write in writes if write["id"] in write_ids]


def apply_ipc_plan(
    parameters: Mapping[str, torch.Tensor],
    writes: list[dict],
    bindings: dict,
    geometry: dict,
    *,
    target_shapes: dict,
    batched: bool = False,
    owner_rotation: int = 0,
    write_ids: frozenset[str] | None = None,
) -> None:
    """Validate complete coverage, then copy this receiver's selected IPC
    delivery and close all imports before returning.

    The caller owns session/version validation and producer lifetime. Bindings
    are fresh, private exports for this consumer. No CUDA graph or tensor view
    survives the request, so TMS may release/remap targets after the ACK.
    """
    validate_targets(writes, parameters, target_shapes, source_world_size=geometry["source_world_size"])
    writes = _selected_writes(writes, write_ids)
    apply_ipc_regions(
        parameters,
        writes,
        [dict(source_rank=rank, payload=payload, views={name: None}) for (rank, name), payload in bindings.items()],
        batched=batched,
        owner_rotation=owner_rotation,
    )


def _layout(tensors: Mapping, *, disjoint: bool = False) -> tuple[tuple, list[int]]:
    """Describe allocation and view aliases without retaining addresses."""
    allocations, views, schema, pointers, ranges = {}, {}, [], [], []
    for key in sorted(tensors):
        tensor = tensors[key]
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or not tensor.numel():
            raise ValueError("Compiled IPC requires nonempty contiguous BF16 storage")
        storage = tensor.untyped_storage()
        pointer, size = tensor.data_ptr(), tensor.numel() * 2
        offset, capacity = tensor.storage_offset() * 2, storage.nbytes()
        base = storage.data_ptr()
        if (
            not 0 < pointer <= pointer + size <= (1 << 63) - 1
            or pointer != base + offset
            or offset < 0
            or offset + size > capacity
        ):
            raise ValueError("Compiled IPC allocation exceeds address or storage bounds")
        allocation = allocations.setdefault((tensor.device, base), len(allocations))
        view = views.setdefault((tensor.device, pointer), len(views))
        schema.append(
            (
                key,
                str(tensor.dtype),
                str(tensor.device),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                offset,
                capacity,
                allocation,
                view,
            )
        )
        pointers.append(pointer)
        ranges.append((pointer, pointer + size))
    if len({str(tensor.device) for tensor in tensors.values()}) != 1:
        raise ValueError("Compiled IPC allocations must use one device")
    if disjoint:
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("Compiled IPC target parameters physically overlap")
    return tuple(schema), pointers


class CompiledIPC:
    """Own a fixed logical plan, retaining only CPU schemas and templates.

    Construction checks exact logical coverage and target storage. The first
    apply also checks every source rectangle before compiling. Later applies
    accept fresh addresses only when the complete allocation/view layout is
    unchanged. Any failure requires a new native seed and a new plan object.
    Session, version and native-reference validation belong to the caller.
    """

    def __init__(
        self,
        parameters: Mapping,
        writes: list[dict],
        geometry: dict,
        *,
        target_shapes: dict,
        owner_rotation: int = 0,
        write_ids: frozenset[str] | None = None,
    ) -> None:
        # JSON copy both establishes ownership and rejects tensors/handles in
        # the description. Nested geometry is used only during construction.
        description = json.loads(json.dumps(dict(geometry=geometry, writes=writes)))
        self.plan_hash = fingerprint(description)
        writes, geometry = description["writes"], description["geometry"]
        validate_targets(writes, parameters, target_shapes, source_world_size=geometry["source_world_size"])
        if set(parameters) != {write["param"] for write in writes}:
            raise ValueError("Compiled IPC parameters differ from the complete plan")
        if any(any(type(value) not in (str, int) for value in write.values()) for write in writes):
            raise ValueError("Compiled IPC rectangles must contain scalar descriptions only")
        writes = _selected_writes(writes, write_ids)
        self._writes = tuple(MappingProxyType(write) for write in writes)
        self._expected = frozenset((write["source_rank"], write["source"]) for write in writes)
        self._target_signature, _ = _layout(parameters, disjoint=True)
        self._source_signature = None
        self._groups = None
        self._owner_rotation = owner_rotation
        self.failed = False

    @property
    def template_statistics(self) -> tuple[int, int]:
        """Return descriptor count and CPU template bytes, without exposing
        cache state."""
        templates = (self._groups or {}).values()
        return sum(value.shape[0] for value in templates), sum(
            value.numel() * value.element_size() for value in templates
        )

    def _metadata(self, sources: Mapping, parameters: Mapping) -> dict:
        from .rectangle_batch import rectangle_metadata

        target_signature, target_pointers = _layout(parameters, disjoint=True)
        source_signature, source_pointers = _layout(sources) if sources else ((), [])
        if target_signature != self._target_signature or (
            self._source_signature is not None and source_signature != self._source_signature
        ):
            raise ValueError("Compiled IPC layout changed; native reseeding is required")
        if set(sources) != self._expected:
            raise ValueError("Compiled IPC bindings differ from the complete source plan")
        if self._groups is None:
            groups = rectangle_metadata(sources, parameters, self._writes, owner_rotation=self._owner_rotation)
            source_slots = {pointer: index for index, pointer in enumerate(source_pointers)}
            target_slots = {pointer: index for index, pointer in enumerate(target_pointers)}
            self._groups = {
                group: torch.tensor(
                    [[source_slots[row[0]], target_slots[row[1]], *row[2:]] for row in rows],
                    dtype=torch.int64,
                    device="cpu",
                )
                for group, rows in groups.items()
            }
            self._source_signature = source_signature
        source_addresses = torch.tensor(source_pointers, dtype=torch.int64, device="cpu")
        target_addresses = torch.tensor(target_pointers, dtype=torch.int64, device="cpu")
        output = {}
        for group, template in self._groups.items():
            descriptors = template.clone()
            descriptors[:, 0] = source_addresses.index_select(0, template[:, 0])
            descriptors[:, 1] = target_addresses.index_select(0, template[:, 1])
            output[group] = descriptors
        return output

    def apply(self, parameters: Mapping, bindings: dict, *, verify_native: bool = False) -> None:
        if self.failed:
            raise RuntimeError("Compiled IPC session failed; native reseeding is required")
        self.failed = True
        if set(bindings) != self._expected:
            raise ValueError("Compiled IPC bindings differ from the complete source plan")
        signature, _ = _layout(parameters, disjoint=True)
        if signature != self._target_signature:
            raise ValueError("Compiled IPC target layout changed; native reseeding is required")
        apply_ipc_regions(
            parameters,
            self._writes,
            [
                dict(source_rank=rank, payload=payload, views={name: None})
                for (rank, name), payload in bindings.items()
            ],
            metadata_builder=self._metadata,
            **({"verify_native": True} if verify_native else {}),
        )
        self.failed = False

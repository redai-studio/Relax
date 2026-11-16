# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""In-place U16 weight updates in a drained SGLang worker.

Prepare authenticates the offline packages, checks the resident base against
their tensor hashes, and pins one recovery base. The owner loads every other
configured checkpoint once and issues adopt to verify its runtime bytes and
auxiliary buffers. Subsequent switches read only pinned residuals.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


_RUNTIME = "sglang-dense-v1"
_GROUP_BYTES = 64 * 1024**2
_STATE_ATTRIBUTE = "_relax_adp_teacher_state"
_GEMMA_NORM_CLASS = "sglang.srt.layers.layernorm.GemmaRMSNorm"


@dataclass
class _State:
    current: str | None = None
    poisoned: bool = False
    prepared: bool = False
    manifest_digest: str | None = None
    bindings: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    plans: dict[tuple[str, str], Any] = field(default_factory=dict)
    teachers: frozenset[str] = frozenset()
    verified_teachers: set[str] = field(default_factory=set)
    config_json: str | None = None
    base_teacher: str | None = None
    recovery_params: dict[str, Any] = field(default_factory=dict)
    recovery_auxiliary: dict[str, Any] = field(default_factory=dict)
    teacher_hashes: dict[str, dict[str, str]] = field(default_factory=dict)
    auxiliary_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class _RuntimeView:
    params: dict[str, Any]
    auxiliary: dict[str, Any]
    metadata: dict[str, Any]
    bindings: dict[str, tuple[Any, ...]]
    config_json: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _layout(tensor: Any) -> dict[str, Any]:
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "stride": list(tensor.stride())}


def _binding(tensor: Any) -> tuple[Any, ...]:
    return (
        id(tensor),
        _class_name(tensor),
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
        tensor.storage_offset(),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        str(tensor.layout),
    )


def _check_parameter(name: str, tensor: Any) -> None:
    import torch

    _require(tensor.device.type == "cuda", f"parameter {name} must be CUDA resident")
    _require(tensor.layout == torch.strided and tensor.is_contiguous(), f"noncontiguous parameter: {name}")
    _require(tensor.dtype in (torch.bfloat16, torch.float16, torch.float32), f"unsupported parameter dtype: {name}")
    _require(tensor.numel() > 0, f"empty parameter: {name}")


def _check_gemma_buffer(name: str, tensor: Any, weight: Any) -> None:
    import torch

    _require(isinstance(tensor, torch.Tensor) and isinstance(weight, torch.Tensor), f"missing Gemma weight: {name}")
    _require(
        tensor.dtype == weight.dtype
        and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and tensor.shape == weight.shape
        and tensor.ndim == 1
        and tensor.numel() > 0
        and tensor.layout == weight.layout == torch.strided
        and tensor.is_contiguous()
        and weight.is_contiguous()
        and tensor.device == weight.device,
        f"unsupported Gemma buffer layout: {name}",
    )
    _require(
        tensor.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr(),
        f"Gemma buffer must have independent storage: {name}",
    )


def _is_tensor_free_metadata(value: Any, seen: set[int]) -> bool:
    """Prove a skipped branch contains only known metadata, including its
    keys."""
    scalar_types = (type(None), bool, int, float, complex, str, bytes)
    if type(value) in scalar_types:
        return True
    if isinstance(value, Enum):
        return type(value.value) in scalar_types
    if type(value) not in (dict, list, tuple) or id(value) in seen:
        return False
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return all(
            _is_tensor_free_metadata(key, seen) and _is_tensor_free_metadata(child, seen)
            for key, child in value.items()
        )
    return all(_is_tensor_free_metadata(child, seen) for child in value)


def _tensor_attributes(value: Any, path: str, seen: set[int]) -> Iterable[tuple[str, Any]]:
    """Inspect tensor attributes and ordinary containers without following
    modules."""
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (dict, list, tuple)):
        _require(id(value) not in seen, f"cyclic auxiliary container: {path}")
        seen = seen | {id(value)}
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            if not isinstance(key, (str, int)):
                # Qwen3-VL/Qwen3.5 has {Modality.IMAGE: True, ...} configuration
                # maps. Skip only proven metadata; the ordinary tensor walker
                # cannot prove that sets or unknown objects contain no tensors.
                _require(not isinstance(key, torch.Tensor), f"tensor auxiliary container key: {path}")
                _require(
                    _is_tensor_free_metadata(key, seen) and _is_tensor_free_metadata(child, seen),
                    f"unsupported auxiliary container key containing tensors or unknown metadata: {path}",
                )
                continue
            suffix = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
            yield from _tensor_attributes(child, f"{path}[{suffix}]", seen)


def _parameter_view(
    name: str, tensor: Any, storage_parameters: dict[tuple[str, int], list[tuple[str, Any]]]
) -> dict[str, Any] | None:
    """Describe a nonempty view covered by one contiguous parameter allocation.

    Storage identity alone is insufficient: parameters can occupy disjoint
    slices of a larger allocation. The complete strided access span must be
    inside one parameter's actual byte range. Unknown overlapping views fail
    closed instead of being silently treated as constant auxiliary state.
    """
    storage = tensor.untyped_storage()
    parameters = storage_parameters.get((str(tensor.device), storage.data_ptr()))
    if not parameters or tensor.numel() == 0:
        return None
    _require(not tensor.is_conj() and not tensor.is_neg(), f"lazy conjugate/negative parameter view: {name}")
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    _require(len(shape) == len(strides) and all(stride >= 0 for stride in strides), f"invalid view strides: {name}")
    element_bytes = tensor.element_size()
    start = tensor.storage_offset() * element_bytes
    span = (1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))) * element_bytes
    end = start + span
    _require(0 <= start < end <= storage.nbytes(), f"parameter view outside its storage: {name}")
    _require(tensor.data_ptr() == storage.data_ptr() + start, f"inconsistent parameter view pointer: {name}")
    for parameter_name, parameter in parameters:
        parameter_storage = parameter.untyped_storage()
        parameter_start = parameter.storage_offset() * parameter.element_size()
        parameter_end = parameter_start + parameter.numel() * parameter.element_size()
        if parameter_start <= start and end <= parameter_end:
            _require(parameter.is_contiguous(), f"parameter view owner is not contiguous: {name}")
            _require(parameter.dtype == tensor.dtype, f"parameter view dtype reinterpretation: {name}")
            _require(
                parameter_storage.nbytes() == storage.nbytes()
                and parameter.data_ptr() == storage.data_ptr() + parameter_start,
                f"inconsistent shared parameter storage: {name}",
            )
            return {
                "parameter": parameter_name,
                "offset_bytes": start - parameter_start,
                "span_bytes": span,
                **_layout(tensor),
            }
    raise ValueError(f"parameter view is not fully covered by one parameter: {name}")


def _runtime_view(model: Any, *, prepared: _State | None = None) -> _RuntimeView:
    import torch

    params = dict(model.named_parameters())
    mutable_buffers = {}
    for module_name, module in model.named_modules():
        if _class_name(module) != _GEMMA_NORM_CLASS:
            continue
        prefix = f"{module_name}." if module_name else ""
        name, parameter = f"buffer:{prefix}gemma_weight", f"{prefix}weight"
        tensor, weight = module._buffers.get("gemma_weight"), params.get(parameter)
        _require(name not in params and weight is getattr(module, "weight", None), f"Gemma owner differs: {name}")
        _check_gemma_buffer(name, tensor, weight)
        params[name] = tensor
        mutable_buffers[name] = {"parameter": parameter, "module_class": _GEMMA_NORM_CLASS}
    canonical = {id(tensor): name for name, tensor in params.items()}
    aliases = {
        name: canonical[id(tensor)]
        for name, tensor in model.named_parameters(remove_duplicate=False)
        if name != canonical[id(tensor)]
    }
    ranges: list[tuple[int, int, str]] = []
    storage_parameters: dict[tuple[str, int], list[tuple[str, Any]]] = {}
    for name, tensor in params.items():
        _check_parameter(name, tensor)
        ranges.append((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size(), name))
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        storage_parameters.setdefault(key, []).append((name, tensor))
    _require(len({str(tensor.device) for tensor in params.values()}) == 1, "parameters span multiple devices")
    ranges.sort()
    _require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "distinct parameters overlap in storage")

    auxiliary = {
        f"buffer:{name}": tensor
        for name, tensor in model.named_buffers(remove_duplicate=False)
        if id(tensor) not in canonical
    }
    for module_name, module in model.named_modules():
        for key in ("quant_config", "quantization_config"):
            _require(getattr(module, key, None) is None, f"quantized model is unsupported: {module_name}.{key}")
        method = getattr(module, "quant_method", None)
        _require(
            method is None or type(method).__name__.startswith("Unquantized"),
            f"quantized/unknown weight method is unsupported: {module_name}",
        )
        for attr, value in vars(module).items():
            if attr in ("_parameters", "_buffers", "_modules", _STATE_ATTRIBUTE):
                continue
            for path, tensor in _tensor_attributes(value, f"attr:{module_name}:{attr}", set()):
                if id(tensor) not in canonical:
                    auxiliary[path] = tensor
    parameter_views, view_tensors = {}, {}
    for name, tensor in list(auxiliary.items()):
        _require(tensor.layout == torch.strided, f"unsupported auxiliary tensor layout: {name}")
        _require(tensor.device.type in ("cpu", "cuda"), f"unsupported auxiliary tensor device: {name}")
        view = _parameter_view(name, tensor, storage_parameters)
        if view is not None:
            parameter_views[name], view_tensors[name] = view, tensor
            del auxiliary[name]

    config = getattr(model, "config", None)
    config_json = None
    if config is not None:
        config = config.to_dict() if hasattr(config, "to_dict") else config
        _require(isinstance(config, dict), "model config must provide to_dict() or be a mapping")
        config = {
            k: v for k, v in config.items() if k not in ("_name_or_path", "_commit_hash", "transformers_version")
        }
        _require(config.get("quantization_config") is None, "quantized model config is unsupported")
        config_json = json.dumps(config, sort_keys=True, default=str)
    if prepared is not None:
        _require(config_json == prepared.config_json, "runtime parameter/auxiliary layout or storage changed: config")
        config = prepared.metadata["config_sha256"]
    else:
        config = hashlib.sha256(config_json.encode()).hexdigest() if config_json is not None else None
    metadata = {
        "model_class": _class_name(model),
        "config_sha256": config,
        "layout": {name: _layout(tensor) for name, tensor in params.items()},
        "aliases": aliases,
        "auxlayout": {name: _layout(tensor) for name, tensor in auxiliary.items()},
    }
    # Preserve capture compatibility for models without parameter-backed views.
    # Models with such views require new captures with this explicit contract.
    if parameter_views:
        metadata["parameter_views"] = parameter_views
    if mutable_buffers:
        metadata["mutable_buffers"] = mutable_buffers
    bindings = {f"param:{name}": _binding(tensor) for name, tensor in params.items()}
    bindings.update({f"aux:{name}": _binding(tensor) for name, tensor in auxiliary.items()})
    bindings.update({f"view:{name}": _binding(tensor) for name, tensor in view_tensors.items()})
    return _RuntimeView(params, auxiliary, metadata, bindings, config_json)


def _hash_tensor(tensor: Any) -> str:
    """Slow preparation-only byte hash, with bounded GPU-to-CPU transfers."""
    import torch

    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), _GROUP_BYTES):
        chunk = raw[offset : offset + _GROUP_BYTES].to(device="cpu")
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def _tp_context() -> tuple[Any, Any, int, int]:
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    _require(group is not None and dist.is_initialized(), "SGLang TP CPU process group is not initialized")
    return dist, group, dist.get_rank(group=group), dist.get_world_size(group=group)


def _poison(state: _State) -> None:
    state.current = None
    state.poisoned = True


def _consensus(
    context: tuple[Any, Any, int, int],
    state: _State,
    route: tuple[Any, ...],
    error: str | None,
    digest: str | None = None,
) -> None:
    dist, group, _, size = context
    records: list[Any] = [None] * size
    local = {"route": route, "error": error, "digest": digest}
    try:
        dist.all_gather_object(records, local, group=group)
    except Exception:
        _poison(state)
        raise
    errors = [f"TP rank {rank}: {item['error']}" for rank, item in enumerate(records) if item["error"]]
    if any(item["route"] != route for item in records):
        errors.append("TP workers disagree on command or current teacher state")
    if not errors and any(item["digest"] != digest for item in records):
        errors.append("TP workers read different manifests")
    if errors:
        if any(item["route"][1] in ("switch", "restore", "adopt") for item in records):
            _poison(state)
        raise RuntimeError("ADP TP consensus failed: " + "; ".join(errors))


def _relative_file(root: Path, value: Any) -> Path:
    path = (root / value).resolve()
    _require(root in path.parents, "package path escapes RELAX_ADP_TEACHER_DIR")
    return path


def _package_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        _require(stream.read(8) == b"ADP1\0\0\0\0", "invalid ADP1 magic")
        size = struct.unpack("<Q", stream.read(8))[0]
        _require(0 < size < _GROUP_BYTES, "invalid ADP1 metadata length")
        header = stream.read(size)
    return json.loads(header)


def _synchronize(view: _RuntimeView) -> None:
    import torch

    devices = {tensor.device for tensor in (*view.params.values(), *view.auxiliary.values()) if tensor.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)


def _allocate_pinned(tensor: Any) -> Any:
    import torch

    return torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu", pin_memory=True)


def _prepare_recovery(view: _RuntimeView, candidate: _State, teacher: str) -> None:
    """Retain one anchor only, with one phase sync for all asynchronous D2H
    copies."""
    import torch

    snapshots: dict[int, Any] = {}
    banks = []
    try:
        with torch.inference_mode():
            for values in (view.params, view.auxiliary):
                bank = {}
                for name, tensor in values.items():
                    # Exact-object auxiliary aliases need only one retained allocation.
                    if id(tensor) not in snapshots:
                        snapshot = _allocate_pinned(tensor)
                        snapshots[id(tensor)] = snapshot
                        snapshot.copy_(tensor.detach(), non_blocking=True)
                    bank[name] = snapshots[id(tensor)]
                banks.append(bank)
    finally:
        # Drain copies even on allocation/copy failure before releasing CPU banks.
        _synchronize(view)
    candidate.recovery_params, candidate.recovery_auxiliary = banks
    candidate.base_teacher = teacher


def _restore(view: _RuntimeView, state: _State, teacher: str) -> None:
    import torch

    try:
        with torch.inference_mode():
            for live, bank in ((view.params, state.recovery_params), (view.auxiliary, state.recovery_auxiliary)):
                for name, tensor in live.items():
                    tensor.copy_(bank[name], non_blocking=True)
            if teacher != state.base_teacher:
                state.plans[state.base_teacher, teacher].apply(teacher)
    finally:
        # Includes the base==target path and failure after a partial asynchronous copy.
        _synchronize(view)


def _verify(view: _RuntimeView, state: _State, teacher: str) -> None:
    """Verify the checkpoint loaded from disk before adopting it as a delta
    source."""
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.params.items()} == state.teacher_hashes[teacher],
        "runtime parameter bytes do not match the tracked teacher",
    )
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.auxiliary.items()} == state.auxiliary_hashes,
        "runtime auxiliary bytes do not match preparation",
    )


def _prepare(view: _RuntimeView, root: Path, teacher: str, rank: int, size: int) -> _State:
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest["version"] == 1 and manifest["runtime"] == _RUNTIME, "unsupported manifest runtime/version")
    _require(manifest["tp_size"] == size, "manifest TP size mismatch")
    _require(manifest["initial_teacher"] == teacher, "prepare must name the manifest initial_teacher")
    _require(manifest.get("group_bytes", _GROUP_BYTES) == _GROUP_BYTES, "ADP requires 64 MiB bounded groups")
    teachers = manifest["teachers"]
    hashes: dict[str, dict[str, str]] = {name: {} for name in teachers}
    expected_edges = {(source, target) for source in teachers for target in teachers if source != target}
    edges = manifest["edges"]
    _require(manifest.get("packet_codec") == "u16-bitpack-v1", "expected a U16 Teacher package")
    from .adp.bitpack import PackedU16Plan

    plans, physical_plans = {}, {}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        reverse = edge.get("reverse", False)
        package_source, package_target = (target, source) if reverse else (source, target)
        packages = edge["packages"]
        _require(len(packages) == size, "package paths must cover every TP rank")
        relative = packages[rank]
        path = _relative_file(root, relative)
        if path in physical_plans:
            header, plan = physical_plans[path]
        else:
            header, plan = _package_header(path), None
        _require(
            header["format"] == "ADP1" and header["version"] == 1 and header["policy"] == "size",
            "requires ADP1 size policy",
        )
        provenance = header["provenance"]
        for key, value in {
            "runtime": _RUNTIME,
            "source": package_source,
            "target": package_target,
            "tp_rank": rank,
            "tp_size": size,
        }.items():
            _require(provenance.get(key) == value, f"package provenance mismatch: {key}")
        _require(
            set(header["tensors"]) == set(view.params), "package must cover all parameters, including SKIP tensors"
        )
        for name, tensor in view.params.items():
            entry = header["tensors"][name]
            _require(
                entry["shape"] == list(tensor.shape) and entry["dtype"] == str(tensor.dtype),
                f"package layout mismatch: {name}",
            )
            for model_id, digest in ((package_source, entry["a_sha256"]), (package_target, entry["b_sha256"])):
                _require(
                    hashes[model_id].setdefault(name, digest) == digest,
                    f"conflicting package teacher hash: {model_id}/{name}",
                )
        if plan is None:
            plan = PackedU16Plan(path, view.params, header, manifest["package_integrity"][relative]["sha256"])
            physical_plans[path] = header, plan
        plans[source, target] = plan
    _require(plans.keys() == expected_edges, "bank lacks a directed Teacher pair")
    candidate = _State(
        current=teacher,
        prepared=True,
        manifest_digest=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        bindings=view.bindings,
        metadata=view.metadata,
        plans=plans,
        teachers=frozenset(teachers),
        verified_teachers={teacher},
        config_json=view.config_json,
        teacher_hashes=hashes,
    )
    # One D2H copy produces both the recovery anchor and the bytes to attest.
    _prepare_recovery(view, candidate, teacher)
    with ThreadPoolExecutor(max_workers=4) as pool:
        initial = dict(
            zip(candidate.recovery_params, pool.map(_hash_tensor, candidate.recovery_params.values()), strict=True)
        )
        _require(initial == hashes[teacher], "resident initial Teacher weights differ from the package")
        candidate.auxiliary_hashes = dict(
            zip(
                candidate.recovery_auxiliary,
                pool.map(_hash_tensor, candidate.recovery_auxiliary.values()),
                strict=True,
            )
        )
    return candidate


def adp_teacher_loader(model: Any, named_tensors: Any) -> None:
    """SGLang custom_weight_loader callback; every TP rank must enter together.

    TP ranks agree before acknowledging an update. A partial switch or restore
    poisons every reachable worker: restart/reload the engine to recover. A
    later prepare cannot bless potentially half-switched weights.
    """
    context = _tp_context()
    _, _, rank, size = context
    state = getattr(model, _STATE_ATTRIBUTE, None)
    if state is None:
        state = _State()
        setattr(model, _STATE_ATTRIBUTE, state)
    operation = teacher = None
    error = None
    view = None
    try:
        ((marker, _),) = named_tensors
        _, operation, teacher = marker.split(":", 2)
        _require(not state.poisoned, "ADP worker state is unknown; engine restart/reload is required")
        view = _runtime_view(model, prepared=state if state.prepared else None)
        if state.prepared:
            _require(
                view.metadata == state.metadata and view.bindings == state.bindings,
                "runtime parameter/auxiliary layout or storage changed",
            )
        if operation == "prepare":
            _require(not state.prepared, "ADP is already prepared")
        elif operation == "switch":
            _require(state.prepared and state.current is not None, "ADP prepare is required before switch")
            _require(teacher in state.verified_teachers, "target checkpoint has not been verified")
            _require(
                teacher == state.current or (state.current, teacher) in state.plans,
                "directed teacher package is unavailable",
            )
        elif operation == "restore":
            _require(teacher in state.verified_teachers, "target checkpoint has not been verified")
        elif operation == "adopt":
            _require(state.prepared and teacher in state.teachers, "unprepared or unknown disk checkpoint")
        else:
            raise ValueError(f"Unknown Teacher operation: {operation}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    route = ("preflight", operation, teacher, state.current, state.prepared, state.poisoned, state.base_teacher)
    _consensus(context, state, route, error, state.manifest_digest)
    candidate = None
    error = None
    try:
        if operation in ("switch", "restore"):
            _synchronize(view)
        if operation == "prepare":
            candidate = _prepare(view, Path(os.environ["RELAX_ADP_TEACHER_DIR"]).resolve(), teacher, rank, size)
        elif operation == "restore":
            _restore(view, state, teacher)
        elif operation == "adopt":
            _verify(view, state, teacher)
        elif operation == "switch" and teacher != state.current:
            try:
                state.plans[state.current, teacher].apply(teacher)
            finally:
                _synchronize(view)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    route = ("commit", operation, teacher, state.current, state.prepared, state.poisoned, state.base_teacher)
    _consensus(context, state, route, error, candidate.manifest_digest if candidate else state.manifest_digest)
    if candidate is not None:
        setattr(model, _STATE_ATTRIBUTE, candidate)
    elif operation in ("switch", "restore", "adopt"):
        state.current = teacher
        if operation == "adopt":
            state.verified_teachers.add(teacher)

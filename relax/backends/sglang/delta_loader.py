# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lossless bucket loading through SGLang's ordinary model loader.

Only one complete conversion group is materialized at a time. The receiver owns
workspace admission and live destination storage; no HF model mirror or
Megatron layout knowledge lives here. Reviewed loader/layout capabilities are
required before the sender may omit values for a conversion group.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

from relax.distributed.weight_sync.codec import tensors, validate_directory
from relax.utils.sparse_tensor import FullTensorRequired, SparseTensor, storage_identity

from .delta_capability import full_loader_inputs, sparse_loader_identity


LOADER = "relax.backends.sglang.delta_loader.load_delta_weights"


def _model_storage(model: Any) -> frozenset[tuple[str, int]]:
    return frozenset(storage_identity(tensor) for tensor in (*model.parameters(), *model.buffers()))


def _model_signature(model: Any) -> tuple:
    result = []
    for name, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        if isinstance(tensor, SparseTensor):
            raise FullTensorRequired("The ordinary loader retained a sparse input as live model state")
        result.append((name, tuple(tensor.shape), tensor.dtype, tuple(tensor.stride()), storage_identity(tensor)))
    return tuple(result)


class _ImmutableInputs(TorchFunctionMode):
    """Block Python Tensor.data/Parameter paths before wrapper storage escapes.

    These setters bypass ATen dispatch. This thread-local mode leaves torch's
    global functions and ordinary FULL inputs unchanged.
    """

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        if getattr(func, "__name__", "") in ("__set__", "requires_grad_") and any(
            isinstance(value, SparseTensor) for value in tree_leaves((args, kwargs))
        ):
            raise FullTensorRequired("Storage/grad metadata replacement requires ordinary FULL input")
        return func(*args, **kwargs)


def workspace_bytes(metadata: dict) -> int:
    """The explicit decode allocation floor, excluding existing model weights.

    FULL clones isolate ordinary loaders from the immutable producer. Sparse
    inputs use dense value + validity workspaces and an integer masked-copy
    result. Groups may be buffered by the ordinary loader, so sum their
    retained input allocations rather than assume the generator is consumed
    eagerly. Arbitrary loader layout temporaries are not statically bounded
    here: FULL seed measures them and applies a posterior budget gate before
    any ACK.
    """
    required, largest_copy, largest_indices = 0, 0, 0
    for entry in metadata["tensors"]:
        elements = math.prod(entry["shape"])
        width = torch.empty((), dtype=getattr(torch, entry["dtype"])).element_size()
        required += elements * width
        if not entry["full"]:
            required += elements
            largest_copy = max(largest_copy, elements * width)
        if not entry["full"]:
            largest_indices = max(largest_indices, entry["count"] * 8)
    return required + largest_copy + largest_indices


class DeltaReceiver:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.version: str | None = None
        self.active: dict | None = None
        self.failed = False
        self.schema: dict[str, tuple] = {}
        self.full_only: set[str] = set()
        self.workspace_peaks: dict[tuple, int] = {}
        self.last_workspace_peak_bytes: int | None = None
        self.document: dict | None = None
        self.force_full_seed = False
        self.capability: tuple | None = None
        self.destinations = _model_storage(model)

    def begin(self, control: dict) -> None:
        if self.active is not None or self.failed:
            raise RuntimeError("Delta receiver has an unfinished or failed update")
        if type(control.get("seed")) is not bool or not isinstance(control.get("version"), str):
            raise ValueError("Delta begin requires a version and an explicit FULL seed decision")
        if not control["seed"] and control.get("base") != self.version:
            raise ValueError("Delta receiver does not have the acknowledged source baseline")
        destinations = _model_storage(self.model)
        capability = sparse_loader_identity(self.model)
        if not control["seed"] and self.version is not None and capability != self.capability:
            raise ValueError("Changed acknowledged loader geometry requires a new FULL seed")
        if not control["seed"] and destinations != self.destinations and capability is not None:
            raise ValueError("Reallocated sparse model storage requires a new FULL seed")
        self.capability = capability
        if capability is not None:
            self.full_only.update(full_loader_inputs(self.model))
        self.destinations = destinations
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            expected = control.get("uuid")
            actual = str(torch.cuda.get_device_properties(device).uuid)
            if (
                not isinstance(expected, str)
                or expected.removeprefix("GPU-").lower() != actual.removeprefix("GPU-").lower()
            ):
                raise ValueError("Delta proxy and model worker must own the same physical GPU")
        self.active = dict(control=control, next_frame=0, schema={})

    def _workspace_key(self, metadata: dict) -> tuple:
        directory = tuple((e["name"], tuple(e["shape"]), e["dtype"]) for e in metadata["tensors"])
        return directory, self.destinations

    def accept_document(self, control: dict) -> None:
        if self.active is None or control.get("kind") != "frame":
            raise ValueError("Frame metadata requires an active receiver update")
        identity, index, count, data = (control[key] for key in ("id", "index", "count", "data"))
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or type(index) is not int
            or type(count) is not int
            or not 0 <= index < count <= 65536
            or not isinstance(data, str)
            or len(data.encode()) > 65536
        ):
            raise ValueError("Invalid bounded delta metadata fragment")
        if self.document is None:
            self.document = dict(id=identity, count=count, next=0, parts=[], bytes=0)
        doc = self.document
        if doc["id"] != identity or doc["count"] != count or doc["next"] != index:
            raise ValueError("Delta metadata fragments must arrive once in order")
        doc["bytes"] += len(data.encode())
        if doc["bytes"] > 16 << 20:
            raise ValueError("Delta metadata document exceeds its bounded capacity")
        doc["parts"].append(data)
        doc["next"] += 1
        if doc["next"] == count:
            encoded = "".join(doc.pop("parts"))
            if hashlib.sha256(encoded.encode()).hexdigest() != identity:
                raise ValueError("Delta metadata document identity mismatch")
            doc["metadata"] = json.loads(encoded)

    def _admit(self, metadata: dict, raw: torch.Tensor) -> None:
        control = self.active["control"]
        if raw.device != next(self.model.parameters()).device:
            raise ValueError("Delta IPC payload must reside on the model worker's device")
        budget = control["receiver_gpu_budget_bytes"]
        validate_directory(metadata, max_bytes=budget)
        indices = max((e["count"] * 8 for e in metadata["tensors"] if not e["full"]), default=0)
        required = max(workspace_bytes(metadata), self.workspace_peaks.get(self._workspace_key(metadata), 0) + indices)
        # Proxy-owned resident staging and SG-owned workspace are distinct
        # allocations sharing the configured synchronization budget.
        staging = control["staging_budget_bytes"]
        if required + staging > budget:
            raise ValueError("Receiver workspace plus proxy staging exceeds its configured GPU budget")
        if raw.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(raw.device)
            reusable = torch.cuda.memory_reserved(raw.device) - torch.cuda.memory_allocated(raw.device)
            if required > free + reusable:
                raise ValueError("Receiver loader workspace exceeds physically available device memory")

    def _weights(self, metadata: dict, raw: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
        for entry, positions, values in tensors(metadata, raw):
            shape = tuple(entry["shape"])
            if positions is None:
                value = values.clone().view(shape)
                yield entry["name"], value
                continue
            if values.dtype not in (torch.bfloat16, torch.float32):
                raise FullTensorRequired("This storage dtype requires FULL loader input")
            flat = torch.empty(math.prod(shape), dtype=values.dtype, device=raw.device)
            valid = torch.zeros(flat.numel(), dtype=torch.bool, device=raw.device)
            indices = positions.to(torch.int64)
            words = torch.int16 if values.element_size() == 2 else torch.int32
            flat.view(words).index_copy_(0, indices, values.view(words))
            valid.index_fill_(0, indices, True)
            del indices
            yield entry["name"], SparseTensor(flat.view(shape), valid.view(shape), self.destinations)

    @torch.no_grad()
    def consume(self, control: dict, raw: torch.Tensor, status: torch.Tensor) -> None:
        if (
            self.active is None
            or control.get("version") != self.active["control"]["version"]
            or control["frame"] != self.active["next_frame"]
        ):
            raise ValueError("Delta frames must be consumed once in source order")
        if "metadata" in control:
            metadata = control["metadata"]
        elif self.document is not None and control.get("metadata_id") == self.document["id"]:
            metadata = self.document["metadata"]
        else:
            raise ValueError("Delta frame lacks its complete metadata document")
        self._admit(metadata, raw)
        entries = list(tensors(metadata, raw))
        if status.dtype != torch.uint8 or status.shape != (len(entries),) or status.device != raw.device:
            raise ValueError("Delta capability output must have one byte per input tensor on the same device")
        if storage_identity(status) == storage_identity(raw) or storage_identity(status) in self.destinations:
            raise ValueError("Capability output cannot alias immutable payloads or model storage")
        invalid_positions = []
        for entry, positions, _values in entries:
            if entry["name"] in self.active["schema"]:
                raise ValueError("Duplicate HF input across delta frames")
            if self.active["control"]["seed"] and not entry["full"]:
                raise ValueError("A FULL seed cannot contain omitted values")
            signature = (tuple(entry["shape"]), entry["dtype"])
            if not self.active["control"]["seed"] and self.schema.get(entry["name"]) != signature:
                raise ValueError("Delta HF name/shape/dtype differs from the acknowledged schema")
            if not entry["full"] and entry["name"] in self.full_only:
                raise FullTensorRequired("The receiver requires this complete conversion group as FULL")
            if positions is not None and positions.numel():
                invalid = (positions[0] < 0) | (positions[-1] >= math.prod(entry["shape"]))
                invalid |= torch.any(positions[1:] <= positions[:-1])
                # Explicit readiness validation, before the ordinary loader can
                # write any live storage. No numeric floating comparison.
                invalid_positions.append(invalid)
            self.active["schema"][entry["name"]] = signature
        if invalid_positions and bool(torch.stack(invalid_positions).any().cpu()):
            raise ValueError("Sparse coordinates must be sorted, unique and in bounds")

        current_capability = sparse_loader_identity(self.model)
        if current_capability != self.capability:
            raise FullTensorRequired("Receiver loader/layout capability changed before sparse installation")
        full_group = all(entry["full"] for entry, _, _ in entries)
        full_required = self.capability is None
        if not full_group and full_required:
            raise FullTensorRequired("Unknown receiver loader requires complete ordinary FULL inputs")
        allocated_before = None
        if raw.device.type == "cuda":
            torch.cuda.synchronize(raw.device)
            allocated_before = torch.cuda.memory_allocated(raw.device)
            torch.cuda.reset_peak_memory_stats(raw.device)
        # FULL executes the original ordinary loader exactly once. Capability
        # is established from reviewed code/layout, never by running speculative
        # writes or replaying unknown Python side effects on the live model.
        with _ImmutableInputs():
            self.model.load_weights(self._weights(metadata, raw))
        _model_signature(self.model)
        self.destinations = _model_storage(self.model)
        if full_required:
            self.full_only.update(entry["name"] for entry, _, _ in entries)
        required = [full_required or entry["name"] in self.full_only for entry, _, _ in entries]
        if all(required) or not any(required):
            status.fill_(int(all(required)))
        else:
            status.copy_(torch.tensor(required, dtype=torch.uint8, device=status.device))
        if raw.device.type == "cuda":
            torch.cuda.synchronize(raw.device)
            peak = max(0, torch.cuda.max_memory_allocated(raw.device) - allocated_before)
            self.last_workspace_peak_bytes = peak
            key = self._workspace_key(metadata)
            self.workspace_peaks[key] = max(self.workspace_peaks.get(key, 0), peak)
            if (
                peak + self.active["control"]["staging_budget_bytes"]
                > self.active["control"]["receiver_gpu_budget_bytes"]
            ):
                raise ValueError("Measured receiver workspace plus proxy staging exceeds its GPU budget")
        self.active["next_frame"] += 1
        self.document = None

    def finalize(self, control: dict) -> None:
        if (
            self.active is None
            or control.get("version") != self.active["control"]["version"]
            or control["frames"] != self.active["next_frame"]
        ):
            raise ValueError("Cannot finalize an incomplete delta update")
        schema = self.active["schema"]
        if not schema or (not self.active["control"]["seed"] and schema != self.schema):
            raise ValueError("Delta update did not cover the acknowledged HF directory")
        # Applicable model/session finalization runs in the existing engine
        # lifecycle before its final ACK; this boundary does not invent a
        # generic postprocess hook or repeat quantization on every model.
        self.force_full_seed = self.capability is None or sparse_loader_identity(self.model) != self.capability
        if self.force_full_seed:
            self.full_only.update(schema)
        self.destinations = _model_storage(self.model)
        devices = {parameter.device for parameter in self.model.parameters()}
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        self.schema = schema
        self.version = self.active["control"]["version"]
        self.active = None


def load_delta_weights(model: Any, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Registered SGLang custom loader; exceptions propagate to the caller."""
    weights = list(weights)
    if not 1 <= len(weights) <= 2:
        raise ValueError("Delta loader requires one control/payload and optional capability output")
    control = json.loads(weights[0][0])
    receiver = getattr(model, "_relax_delta_receiver", None)
    if receiver is None:
        receiver = DeltaReceiver(model)
        model._relax_delta_receiver = receiver
    try:
        if control["op"] == "document":
            receiver.accept_document(control)
        elif control["op"] == "begin":
            receiver.begin(control)
        elif control["op"] == "consume" and len(weights) == 2 and weights[1][0] == "__delta_capability__":
            receiver.consume(control, weights[0][1], weights[1][1])
        elif control["op"] == "finalize":
            receiver.finalize(control)
            if len(weights) == 2:
                status = weights[1][1]
                if weights[1][0] != "__delta_capability__" or status.dtype != torch.uint8 or status.numel() != 1:
                    raise ValueError("Final receiver capability output must be one byte")
                status.fill_(int(receiver.force_full_seed))
                if status.device.type == "cuda":
                    torch.cuda.synchronize(status.device)
        else:
            raise ValueError("Unknown or incomplete delta loader operation")
    except BaseException:
        receiver.failed = True
        raise

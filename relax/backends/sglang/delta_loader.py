# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lossless bucket loading through SGLang's ordinary model loader.

Only one complete conversion group is materialized at a time. The receiver owns
workspace admission and live destination storage; no HF model mirror or
Megatron layout knowledge lives here. Reviewed loader/layout capabilities are
required before the sender may omit values for a conversion group.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Iterable
from contextlib import nullcontext
from itertools import chain
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

from relax.distributed.weight_sync.codec import WIRE_ALIGNMENT, schema_digest, tensors, validate_directory
from relax.utils.logging_utils import get_logger
from relax.utils.sparse_tensor import FullTensorRequired, SparseTensor, storage_identity

from .delta_capability import declared_copy_layout, full_loader_inputs, requires_full_input, sparse_loader_identity
from .delta_model_loaders import stock_observed_experts_only, stock_rank_skips_input
from .delta_shard_layout import BroadcastTargets, ShardFrame, marked_ordinals, validate_transport_storage
from .delta_target_lane import ExpertTargets


logger = get_logger(__name__)

LOADER = "relax.backends.sglang.delta_loader.load_delta_weights"


MAX_HTTP_BYTES = 64 << 10
MAX_DOCUMENT_BYTES = 16 << 20


def _inline_directory(control: dict, *, encoded_limit: int = MAX_HTTP_BYTES) -> Any:
    inline = control["metadata_zlib"]
    if not isinstance(inline, dict) or set(inline) != {"bytes", "data"}:
        raise ValueError("Invalid inline delta directory envelope")
    size, data = inline["bytes"], inline["data"]
    if (
        type(size) is not int
        or not 0 < size <= MAX_DOCUMENT_BYTES
        or not isinstance(data, str)
        or len(data) > encoded_limit
    ):
        raise ValueError("Invalid inline delta directory size")
    return _decode_directory(base64.b64decode(data, validate=True), size, control.get("metadata_id"))


def _decode_directory(compressed: bytes, size: int, identity: str) -> Any:
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(compressed, size + 1)
        if len(decoded) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Invalid delta directory stream or length")
        if hashlib.sha256(decoded).hexdigest() != identity:
            raise ValueError("Delta directory identity mismatch")
        return json.loads(decoded.decode("ascii"))
    except zlib.error as exc:
        raise ValueError("Invalid delta directory compression") from exc


def _model_storage(model: Any) -> frozenset[tuple[str, int]]:
    destinations = set()
    for tensor in chain(model.parameters(), model.buffers()):
        if isinstance(tensor, SparseTensor):
            raise FullTensorRequired("The ordinary loader retained a sparse input as live model state")
        destinations.add(storage_identity(tensor))
    return frozenset(destinations)


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
        self._schema_order: tuple[str, ...] = ()
        self.full_only: set[str] = set()
        self.workspace_peaks: dict[tuple, int] = {}
        self.last_workspace_peak_bytes: int | None = None
        self.document: dict | None = None
        self.force_full_seed = False
        self.capability: tuple | None = None
        self.destinations = _model_storage(model)
        self._target_plan: ExpertTargets | None = None
        self._broadcast_plan: BroadcastTargets | None = None
        self._capture: BroadcastTargets | None = None
        self._declared: tuple | None = None
        self._observed_only = False
        self._seed_directory: tuple = ()
        self.target_shard_metrics: dict | None = None

    def begin(self, control: dict) -> None:
        if self.active is not None or self.failed:
            raise RuntimeError("Delta receiver has an unfinished or failed update")
        if type(control.get("seed")) is not bool or not isinstance(control.get("version"), str):
            raise ValueError("Delta begin requires a version and an explicit FULL seed decision")
        if not control["seed"] and control.get("base") != self.version:
            raise ValueError("Delta receiver does not have the acknowledged source baseline")
        requested = control.get("target_shard_request", False)
        if type(requested) is not bool or (
            requested
            and (type(control.get("target_shard_rank")) is not int or not 0 <= control["target_shard_rank"] < 4)
        ):
            raise ValueError("Invalid target shard request binding")
        if control["seed"]:
            self._broadcast_plan = None
            self._schema_order = ()
            self.full_only.clear()
        if control["seed"] or not requested:
            self._target_plan = None
            self._seed_directory = ()
        elif self._target_plan is not None and self._target_plan.rank != control["target_shard_rank"]:
            raise ValueError("Confirmed target shard rank changed before publication")
        destinations = _model_storage(self.model)
        capability = sparse_loader_identity(self.model)
        if not control["seed"] and self.version is not None and capability != self.capability:
            raise ValueError("Changed acknowledged loader geometry requires a new FULL seed")
        if not control["seed"] and destinations != self.destinations and capability is not None:
            raise ValueError("Reallocated sparse model storage requires a new FULL seed")
        self.capability = capability
        self._declared = declared_copy_layout(self.model)
        self._observed_only = self._declared is not None or stock_observed_experts_only(self.model)
        self._capture = BroadcastTargets(self.model) if control["seed"] and capability is not None else None
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
        self.target_shard_metrics = (
            dict(frames=0, entries=0, words=0, wire_bytes=0, tiles=0, scatter_launches=0) if requested else None
        )
        if not requested:
            self.target_shard_metrics = dict(
                mode="broadcast_hf",
                frames=0,
                entries=0,
                nonempty_entries=0,
                ordinary_entries=0,
                ordinary_full_entries=0,
                words=0,
                wire_bytes=0,
                tiles=0,
                scatter_launches=0,
                local_words=0,
            )

    def _workspace_key(self, metadata: dict, local: frozenset = frozenset()) -> tuple:
        directory = tuple((e["name"], tuple(e["shape"]), e["dtype"]) for e in metadata["tensors"])
        return (directory, self.destinations, local) if local else (directory, self.destinations)

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
            or len(data.encode()) > MAX_HTTP_BYTES
        ):
            raise ValueError("Invalid bounded delta metadata fragment")
        decoded_size = control.get("decoded_bytes")
        if "decoded_bytes" in control and (
            type(decoded_size) is not int or not 0 < decoded_size <= MAX_DOCUMENT_BYTES
        ):
            raise ValueError("Invalid compressed directory size")
        if self.document is None:
            self.document = dict(id=identity, count=count, next=0, parts=[], bytes=0, decoded_size=decoded_size)
        doc = self.document
        if (
            doc["id"] != identity
            or doc["count"] != count
            or doc["next"] != index
            or doc["decoded_size"] != decoded_size
        ):
            raise ValueError("Delta metadata fragments must arrive once in order")
        doc["bytes"] += len(data.encode())
        if doc["bytes"] > MAX_DOCUMENT_BYTES:
            raise ValueError("Delta metadata document exceeds its bounded capacity")
        doc["parts"].append(data)
        doc["next"] += 1
        if doc["next"] == count:
            encoded = "".join(doc.pop("parts"))
            if decoded_size is not None:
                doc["metadata"] = _inline_directory(
                    dict(metadata_zlib=dict(data=encoded, bytes=decoded_size), metadata_id=identity),
                    encoded_limit=MAX_DOCUMENT_BYTES,
                )
                return
            if hashlib.sha256(encoded.encode()).hexdigest() != identity:
                raise ValueError("Delta metadata document identity mismatch")
            doc["metadata"] = json.loads(encoded)

    def _admit(
        self, metadata: dict, raw: torch.Tensor, local: frozenset = frozenset(), ipc_directory_bytes: int = 0
    ) -> None:
        control = self.active["control"]
        if raw.device != next(self.model.parameters()).device:
            raise ValueError("Delta IPC payload must reside on the model worker's device")
        budget = control["receiver_gpu_budget_bytes"]
        validate_directory(metadata, max_bytes=budget)
        ordinary = [e for i, e in enumerate(metadata["tensors"]) if i not in local] if local else metadata["tensors"]
        entries = metadata["tensors"]
        ends = metadata.get("loader_ends", [len(entries)])
        groups = (
            [entries[i] for i in range(start, end) if i not in local]
            for start, end in zip([0, *ends[:-1]], ends, strict=True)
        )
        decode_peak = max(workspace_bytes({"tensors": group}) for group in groups)
        indices = max((e["count"] * 8 for e in ordinary if not e["full"]), default=0)
        tile_bytes = sum(
            48 * ((metadata["tensors"][i]["count"] + ShardFrame.BLOCK - 1) // ShardFrame.BLOCK) for i in local
        )
        required = max(
            decode_peak + ((tile_bytes + 511) // 512 * 512 + 1024 if local and raw.is_cuda else tile_bytes),
            self.workspace_peaks.get(self._workspace_key(metadata, local), 0) + indices,
        )
        # Proxy-owned resident staging and SG-owned workspace are distinct
        # allocations sharing the configured synchronization budget.
        staging = control["staging_budget_bytes"]
        if required + ipc_directory_bytes + staging > budget:
            raise ValueError("Receiver workspace plus proxy staging exceeds its configured GPU budget")
        if raw.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(raw.device)
            reusable = torch.cuda.memory_reserved(raw.device) - torch.cuda.memory_allocated(raw.device)
            if required > free + reusable:
                raise ValueError("Receiver loader workspace exceeds physically available device memory")

    def _decoded_weights(self, decoded: Iterable) -> Iterable[tuple[str, torch.Tensor]]:
        for entry, positions, values in decoded:
            shape = tuple(entry["shape"])
            if positions is None:
                value = values.clone().view(shape)
                if self._capture is not None:
                    yield from self._capture.observe(entry["name"], value)
                else:
                    yield entry["name"], value
                continue
            if values.dtype not in (torch.bfloat16, torch.float32):
                raise FullTensorRequired("This storage dtype requires FULL loader input")
            flat = torch.empty(math.prod(shape), dtype=values.dtype, device=values.device)
            valid = torch.zeros(flat.numel(), dtype=torch.bool, device=values.device)
            indices = positions.to(torch.int64)
            words = torch.int16 if values.element_size() == 2 else torch.int32
            flat.view(words).index_copy_(0, indices, values.view(words))
            valid.index_fill_(0, indices, True)
            del indices
            yield entry["name"], SparseTensor(flat.view(shape), valid.view(shape), self.destinations)

    def _consume_ordered(self, control: dict, raw: torch.Tensor, status: torch.Tensor) -> None:
        active = self.active["control"]
        if (
            self._target_plan is None
            or active["seed"]
            or not active.get("ordered_target_consume")
            or not active.get("target_shard_request")
            or self.document is not None
        ):
            raise ValueError("Ordered frames require a confirmed target shard seed")
        frames = _inline_directory(control) if "metadata_zlib" in control else control["metadata"]
        if not isinstance(frames, list) or not 2 <= len(frames) <= 65536:
            raise ValueError("Invalid ordered target frame count")
        if (
            raw.dtype != torch.uint8
            or raw.ndim != 1
            or not raw.is_contiguous()
            or raw.numel() > active["staging_budget_bytes"]
            or status.dtype != torch.uint8
            or status.ndim != 1
            or not status.is_contiguous()
            or status.device != raw.device
            or status.numel() != sum(len(frame) for frame in frames)
        ):
            raise ValueError("Invalid ordered target payload or capability storage")
        # Validate every frame before the first loader invocation or sparse write.
        self._target_plan.check(self, raw, status)
        if sparse_loader_identity(self.model) != self.capability:
            raise FullTensorRequired("Receiver loader geometry changed before ordered installation")
        if self.capability is None or raw.device != next(self.model.parameters()).device:
            raise ValueError("Ordered target scope differs from the acknowledged model")
        if "metadata_zlib" in control and "metadata" in control:
            raise ValueError("Conflicting ordered target directories")
        required = status.numel() + 1024 + sum(ShardFrame.ordered_workspace(frame) for frame in frames)
        budget = active["receiver_gpu_budget_bytes"] - active["staging_budget_bytes"]
        if required > budget:
            raise ValueError("Ordered target workspace exceeds its GPU budget")
        if raw.is_cuda:
            free, _ = torch.cuda.mem_get_info(raw.device)
            allocated = torch.cuda.memory_allocated(raw.device)
            if required > free + torch.cuda.memory_reserved(raw.device) - allocated:
                raise ValueError("Ordered target workspace exceeds available device memory")
            torch.cuda.reset_peak_memory_stats(raw.device)
        pending, invalid = [], []
        end = cursor = 0
        for frame in frames:
            if not isinstance(frame, list) or not 0 < len(frame) <= 65536:
                raise ValueError("Invalid ordered target directory")
            entries, size = [], 0
            for ordinal, count in frame:
                if (
                    type(ordinal) is not int
                    or not 0 <= ordinal < len(self._seed_directory)
                    or type(count) is not int
                    or count < 0
                ):
                    raise ValueError("Invalid ordered target seed ordinal or count")
                name, (shape, dtype) = self._seed_directory[ordinal]
                if dtype != "bfloat16" or name not in self._target_plan.targets:
                    raise ValueError("Ordered target inputs must belong to the admitted expert layout")
                offset = (size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
                value_offset = offset + count * 4
                size = value_offset + count * 2
                entries.append(
                    dict(
                        name=name,
                        shape=list(shape),
                        dtype=dtype,
                        full=False,
                        count=count,
                        offset=offset,
                        value_offset=value_offset,
                    )
                )
            start = (end + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT if size else 0
            if size:
                end = start + size
            if end > raw.numel():
                raise ValueError("Ordered directory exceeds its IPC arena span")
            marker = dict(format=1, tp_rank=self._target_plan.rank, tp_size=4, entries=list(range(len(entries))))
            directory = dict(format=1, bytes=size, tensors=entries, target_shard=marker)
            self.consume(
                dict(frame=self.active["next_frame"], version=active["version"], metadata=directory),
                raw[start : start + size],
                status[cursor : cursor + len(entries)],
                pending=(pending, invalid),
            )
            cursor += len(entries)
        if end != raw.numel():
            raise ValueError("Ordered target directory did not cover its IPC arena span")
        if invalid and bool(torch.stack(invalid).any().cpu()):
            raise ValueError("Ordered sparse coordinates must be sorted, unique and in bounds")
        with _ImmutableInputs():
            for shard in pending:
                self._load_group(shard.runs, shard, raw, status, recheck=True)
        status.zero_()
        if raw.is_cuda:
            torch.cuda.synchronize(raw.device)
            peak = max(0, torch.cuda.max_memory_allocated(raw.device) - allocated)
            self.last_workspace_peak_bytes = peak
            if peak + status.numel() > budget:
                raise ValueError("Measured ordered target workspace exceeds its GPU budget")

    @torch.no_grad()
    def consume(
        self, control: dict, raw: torch.Tensor, status: torch.Tensor, *, pending: tuple[list, list] | None = None
    ) -> None:
        if (
            self.active is None
            or control.get("version") != self.active["control"]["version"]
            or control["frame"] != self.active["next_frame"]
        ):
            raise ValueError("Delta frames must be consumed once in source order")
        if control.get("ordered_target") is True:
            self._consume_ordered(control, raw, status)
            return
        ipc_directory_bytes = 0
        if "metadata_ipc" in control:
            if (
                self.document is not None
                or "metadata" in control
                or "metadata_zlib" in control
                or self.active["control"].get("target_shard_request")
            ):
                raise ValueError("Conflicting IPC delta directories")
            descriptor = control["metadata_ipc"]
            if not isinstance(descriptor, list) or len(descriptor) != 3 or any(type(n) is not int for n in descriptor):
                raise ValueError("Invalid IPC delta directory descriptor")
            count, compressed, size = descriptor
            if (
                not 0 < count <= 65536
                or not 0 < compressed <= size <= MAX_DOCUMENT_BYTES
                or type(status) is not torch.Tensor
                or status.dtype != torch.uint8
                or status.device != raw.device
                or status.shape != (count + compressed,)
                or not status.is_contiguous()
            ):
                raise ValueError("Invalid IPC delta directory size or storage")
            ipc_directory_bytes = ((status.numel() + 511) // 512) * 512
            active = self.active["control"]
            if ipc_directory_bytes > active["receiver_gpu_budget_bytes"] - active["staging_budget_bytes"]:
                raise ValueError("IPC delta directory exceeds the receiver workspace budget")
            # Validate the entire exported status allocation before reading its
            # immutable tail. The original capability ACK only writes the prefix.
            validate_transport_storage(self.model, (raw, status))
            metadata = _decode_directory(status[count:].cpu().numpy().tobytes(), size, control.get("metadata_id"))
            status = status[:count]
        elif "metadata_zlib" in control:
            if self.document is not None or "metadata" in control:
                raise ValueError("Conflicting inline and staged delta directories")
            metadata = _inline_directory(control)
        elif "metadata" in control:
            metadata = control["metadata"]
        elif self.document is not None and control.get("metadata_id") == self.document["id"]:
            metadata = self.document["metadata"]
        else:
            raise ValueError("Delta frame lacks its complete metadata document")
        local = frozenset()
        lane_ordinals = None
        plan = None
        if isinstance(metadata, dict) and "target_shard" in metadata:
            if (
                not self.active["control"].get("target_shard_request")
                or self.active["control"]["seed"]
                or self._target_plan is None
            ):
                raise ValueError("Target shard lacks a confirmed ordinary FULL receiver layout")
            metadata = dict(metadata)
            marker = metadata.pop("target_shard")
            if "loader_ends" in metadata:
                raise ValueError("A target-lane frame cannot contain packed ordinary groups")
            plan = self._target_plan
            lane_ordinals = frozenset(marked_ordinals(marker, plan.rank, len(metadata["tensors"])))
            local = lane_ordinals
            # Observation-only loaders require a bound region for each sparse input.
            # Unmarked slots in a mixed lane keep their original HF coordinates.
            if self._observed_only:
                local = local.union(
                    i
                    for i, entry in enumerate(metadata["tensors"])
                    if not entry["full"] and entry["name"] in plan.bindings.targets
                )
        elif self._broadcast_plan is not None:
            validate_directory(metadata, max_bytes=raw.numel())
            plan = self._broadcast_plan
            local = frozenset(
                i for i, entry in enumerate(metadata["tensors"]) if not entry["full"] and entry["name"] in plan.targets
            )
            if local and (not plan.current(self, raw, status) or not status.is_contiguous()):
                if self._observed_only:
                    raise FullTensorRequired("Observed copy layout changed before installation")
                self._broadcast_plan, plan, local = None, None, frozenset()
        if pending is None:
            self._admit(metadata, raw, local, ipc_directory_bytes)
        entries = list(tensors(metadata, raw))
        if status.dtype != torch.uint8 or status.shape != (len(entries),) or status.device != raw.device:
            raise ValueError("Delta capability output must have one byte per input tensor on the same device")
        if pending is None and not ipc_directory_bytes and not (local and plan.broadcast):
            validate_transport_storage(self.model, (raw, status))
        if storage_identity(status) == storage_identity(raw) or storage_identity(status) in self.destinations:
            raise ValueError("Capability output cannot alias immutable payloads or model storage")
        invalid_positions = []
        shard = None
        if local:
            if not status.is_contiguous():
                raise ValueError("Target shard capability output must be contiguous")
            if pending is None and not plan.broadcast:
                plan.check(self, raw, status)
            shard = ShardFrame(plan, local, entries, raw, metadata.get("loader_ends"), lane_ordinals=lane_ordinals)
            invalid = shard.invalid_positions()
            if invalid is not None:
                invalid_positions.append(invalid)
        for i, (entry, positions, _values) in enumerate(entries):
            if entry["name"] in self.active["schema"]:
                raise ValueError("Duplicate HF input across delta frames")
            if self.active["control"]["seed"] and (
                entry["name"] not in self._declared[-1]
                if self._declared is not None
                else requires_full_input(self.model, entry["name"])
            ):
                self.full_only.add(entry["name"])
            if self.active["control"]["seed"] and not entry["full"]:
                raise ValueError("A FULL seed cannot contain omitted values")
            ordinal = len(self.active["schema"])
            if self._schema_order and (
                ordinal >= len(self._schema_order) or self._schema_order[ordinal] != entry["name"]
            ):
                raise ValueError("Delta HF input order differs from the acknowledged schema")
            signature = (tuple(entry["shape"]), entry["dtype"])
            if not self.active["control"]["seed"] and self.schema.get(entry["name"]) != signature:
                raise ValueError("Delta HF name/shape/dtype differs from the acknowledged schema")
            if not entry["full"] and entry["name"] in self.full_only:
                raise FullTensorRequired("The receiver requires this complete conversion group as FULL")
            if i not in local and positions is not None and positions.numel():
                invalid = (positions[0] < 0) | (positions[-1] >= math.prod(entry["shape"]))
                invalid |= torch.any(positions[1:] <= positions[:-1])
                # Explicit readiness validation, before the ordinary loader can
                # write any live storage. No numeric floating comparison.
                invalid_positions.append(invalid)
            self.active["schema"][entry["name"]] = signature
        if pending is not None:
            if shard is None or len(local) != len(entries):
                raise ValueError("Only pure target frames may defer their install fence")
            pending[0].append(shard)
            pending[1].extend(invalid_positions)
            self._record_shard(shard, lane_ordinals if lane_ordinals is not None else local, entries)
            self.active["next_frame"] += 1
            return
        if invalid_positions and bool(torch.stack(invalid_positions).any().cpu()):
            raise ValueError("Sparse coordinates must be sorted, unique and in bounds")

        if sparse_loader_identity(self.model) != self.capability:
            raise FullTensorRequired("Receiver loader/layout capability changed before sparse installation")
        full_group = all(entry["full"] for entry, _, _ in entries)
        full_required = self.capability is None
        if self._observed_only and any(
            not entry["full"] and i not in local for i, (entry, _, _) in enumerate(entries)
        ):
            raise FullTensorRequired("Sparse input has no acknowledged copy region")
        if not full_group and full_required:
            raise FullTensorRequired("Unknown receiver loader requires complete ordinary FULL inputs")
        allocated_before = None
        if raw.device.type == "cuda":
            torch.cuda.synchronize(raw.device)
            allocated_before = torch.cuda.memory_allocated(raw.device)
            if shard is not None and shard.table is not None:
                allocated_before -= shard.table.numel() * shard.table.element_size() + shard.invalid.element_size()
            torch.cuda.reset_peak_memory_stats(raw.device)
        # FULL executes the original ordinary loader exactly once. Capability
        # is established from reviewed code/layout, never by running speculative
        # writes or replaying unknown Python side effects on the live model.
        with _ImmutableInputs():
            # Keep model-owned setup/teardown at every original invocation.
            start = 0
            ordinary_ran = False
            for group, end in enumerate(metadata.get("loader_ends", [len(entries)])):
                if ordinary_ran:
                    if sparse_loader_identity(self.model) != self.capability:
                        raise FullTensorRequired("Ordinary loader capability changed between packed groups")
                    self.destinations = _model_storage(self.model)
                runs = shard.group_runs[group] if shard is not None else [(False, entries[start:end])]

                self._load_group(runs, shard, raw, status, recheck=ordinary_ran)
                ordinary_ran = any(not direct for direct, _ in runs)
                start = end
        self.destinations = _model_storage(self.model)
        if self._capture is not None:
            for entry, _, _ in entries:
                name = entry["name"]
                # A loader can consume the iterator into a list, then write
                # later in the same call. Only a known off-stage input can
                # turn an unobserved copy into an acknowledged local no-op.
                if name in self.full_only or (
                    self._capture.targets.get(name) is None
                    and (
                        self._declared is not None
                        or (self._observed_only and name not in self._capture.targets)
                        or not stock_rank_skips_input(self.model, name)
                    )
                ):
                    self._capture.targets.pop(name, None)
                    if self._observed_only:
                        self.full_only.add(name)
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
            key = self._workspace_key(metadata, local)
            self.workspace_peaks[key] = max(self.workspace_peaks.get(key, 0), peak)
            if (
                peak + ipc_directory_bytes + self.active["control"]["staging_budget_bytes"]
                > self.active["control"]["receiver_gpu_budget_bytes"]
            ):
                raise ValueError("Measured receiver workspace plus proxy staging exceeds its GPU budget")
        if shard is not None:
            self._record_shard(shard, lane_ordinals if lane_ordinals is not None else local, entries)
        if self.target_shard_metrics is not None and self.target_shard_metrics.get("mode") == "broadcast_hf":
            self.target_shard_metrics["nonempty_entries"] += sum(entries[i][0]["count"] > 0 for i in local)
            self.target_shard_metrics["ordinary_entries"] += len(entries) - len(local)
            self.target_shard_metrics["ordinary_full_entries"] += sum(entry["full"] for entry, _, _ in entries)
        self.active["next_frame"] += 1
        self.document = None

    def _load_group(
        self, runs: list, shard: ShardFrame | None, raw: torch.Tensor, status: torch.Tensor, *, recheck: bool
    ) -> None:
        pending = sum(direct for direct, _ in runs)

        def weights():
            nonlocal recheck, pending
            for direct, batch in runs:
                if direct:
                    plan = shard.plan
                    plan.check_run(self.model, batch)
                    if (recheck or self._observed_only) and (
                        not plan.current(self, raw, status) or sparse_loader_identity(self.model) != self.capability
                    ):
                        raise FullTensorRequired("Ordinary loader changed the admitted destination")
                    shard.scatter(batch)
                    pending -= 1
                    recheck = False
                else:
                    yield from self._decoded_weights(batch)
                    recheck = True

        with self._capture if self._capture is not None else nullcontext():
            self.model.load_weights(weights())
        if pending:
            raise FullTensorRequired("Ordinary loader did not consume every admitted scatter")

    def _record_shard(self, shard: ShardFrame, local: frozenset, entries: list) -> None:
        m = self.target_shard_metrics
        if shard.plan.broadcast:
            if m.get("mode") != "broadcast_hf":
                return  # A negotiated lane connection may still deliver an ordinary HF frame.
            m["local_words"] += shard.local_words
        m["frames"] += 1
        m["entries"] += len(local)
        m["wire_bytes"] += shard.raw.numel()
        m["words"] += sum(entries[i][0]["count"] for i in local)
        m["tiles"] += 0 if shard.table is None else shard.table.shape[0]
        m["scatter_launches"] += sum(
            first != end and (not shard.plan.broadcast or key[0] is not None)
            for direct, batch in shard.runs
            if direct
            for key, first, end, _ in batch
        )

    def finalize(self, control: dict, status: torch.Tensor | None = None) -> None:
        if (
            self.active is None
            or control.get("version") != self.active["control"]["version"]
            or self.document is not None
            or type(control.get("frames")) is not int
            or control["frames"] != self.active["next_frame"]
        ):
            raise ValueError("Cannot finalize an incomplete delta update")
        schema = self.active["schema"]
        final_mask = "full_required_count" in control
        if final_mask and (
            status is None
            or not self.active["control"]["seed"]
            or self.active["control"].get("target_shard_request")
            or type(control["full_required_count"]) is not int
            or control["full_required_count"] != len(schema)
            or control.get("full_required_schema") != schema_digest(schema)
        ):
            raise ValueError("Final capability mask differs from the ordered ordinary seed schema")
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
        if self.active["control"]["seed"]:
            if final_mask:
                self._schema_order = tuple(schema)
            if (
                self._capture is not None
                and self._capture.seal(self, final_mask=final_mask)
                and not self.force_full_seed
            ):
                self._broadcast_plan = self._capture
            elif self._observed_only or (self._capture and self._capture.conflicts - self.full_only):
                self.force_full_seed = True
                self.full_only.update(schema)
            self._capture = None
        if self.active["control"].get("target_shard_request"):
            if self.active["control"]["seed"]:
                self._target_plan = ExpertTargets.build(self)
                self._seed_directory = (
                    tuple(schema.items())
                    if self._target_plan and self.active["control"].get("ordered_target_consume")
                    else ()
                )
            if self.force_full_seed or (
                self._target_plan is not None and self._target_plan.rank != self.active["control"]["target_shard_rank"]
            ):
                self._target_plan = None
        if self.target_shard_metrics is not None and self.target_shard_metrics.get("mode") == "broadcast_hf":
            if isinstance(self.target_shard_metrics["local_words"], torch.Tensor):
                self.target_shard_metrics["local_words"] = int(self.target_shard_metrics["local_words"].cpu())
            runtime = sys.modules.get("sglang.srt.runtime_context")
            parallel = runtime.get_parallel() if runtime is not None and hasattr(runtime, "get_parallel") else None
            try:
                tp_rank = getattr(parallel, "tp_rank", None)
                pp_rank = getattr(parallel, "pp_rank", None)
            except (AssertionError, RuntimeError):
                # Standalone loaders can run before SGLang publishes its rank context.
                tp_rank = pp_rank = None
            logger.info(
                "Broadcast delta receiver version=%s seed=%s tp_rank=%s pp_rank=%s metrics=%s",
                self.active["control"]["version"],
                self.active["control"]["seed"],
                tp_rank,
                pp_rank,
                self.target_shard_metrics,
            )
        if status is not None:
            if final_mask:
                # A single copy under the original final publication fence.
                bits = [self.force_full_seed, *(name in self.full_only for name in schema)]
                status.copy_(torch.tensor(bits, dtype=torch.uint8, device="cpu"), non_blocking=True)
            elif self.active["control"].get("target_shard_request"):
                status[0].fill_(int(self.force_full_seed))
                status[1].fill_(
                    2 if self._seed_directory and self._target_plan else int(self._target_plan is not None)
                )
            else:
                status.fill_(int(self.force_full_seed))
            if status.device.type == "cuda":
                torch.cuda.synchronize(status.device)
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
    if receiver.failed:
        raise RuntimeError("Delta receiver has a failed update")
    try:
        if control["op"] == "document":
            receiver.accept_document(control)
        elif control["op"] == "begin":
            receiver.begin(control)
        elif control["op"] == "consume" and len(weights) == 2 and weights[1][0] == "__delta_capability__":
            receiver.consume(control, weights[0][1], weights[1][1])
        elif control["op"] == "finalize":
            requested = receiver.active is not None and receiver.active["control"].get("target_shard_request", False)
            count = control.get("full_required_count", 0)
            if ("full_required_count" in control or "full_required_schema" in control) and (
                requested or type(count) is not int or count <= 0 or "full_required_schema" not in control
            ):
                raise ValueError("Invalid final capability mask request")
            if (requested or count) and len(weights) != 2:
                raise ValueError("Finalization requires acknowledged capability output")
            if len(weights) == 2:
                status = weights[1][1]
                if weights[1][0] != "__delta_capability__" or status.dtype != torch.uint8:
                    raise ValueError("Invalid final receiver capability output")
                if (
                    type(status) is not torch.Tensor
                    or status.shape != (2 if requested else 1 + count,)
                    or not status.is_contiguous()
                    or status.device != next(model.parameters()).device
                ):
                    raise ValueError("Invalid final receiver status storage")
                validate_transport_storage(model, (weights[0][1], status))
            receiver.finalize(control, weights[1][1] if len(weights) == 2 else None)
        else:
            raise ValueError("Unknown or incomplete delta loader operation")
    except BaseException:
        receiver.failed = True
        raise

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install a bounded verl delta_sharded stream in the SGLang model worker."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import zlib
from collections.abc import Iterable
from typing import Any

import torch

from relax.distributed.weight_sync.codec import frame_flushes, schema_digest, validate_directory

from .routed_expert_delta import EXPERT_ROUTE_CAPACITY, RoutedExpertReceiver
from .verl_nan_mask import CHUNK_BYTES, apply_flush


LOADER = "relax.backends.sglang.delta_loader.load_delta_weights"
MAX_HTTP_BYTES = 64 << 10
MAX_DOCUMENT_BYTES = 16 << 20
VERIFY_ACK_BYTES = 8


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


def validate_transport_storage(model: Any, io: tuple[torch.Tensor, ...]) -> None:
    """A receiver may only acknowledge storage disjoint from model parameters
    and buffers."""

    def span(tensor):
        return (tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes)

    regions = [span(tensor) for tensor in io if tensor.numel()]
    if any(a < d and c < b for i, (a, b) in enumerate(regions) for c, d in regions[i + 1 :]):
        raise ValueError("IPC payload and acknowledgement overlap")
    for value in (*model.parameters(), *model.buffers()):
        if value.device == io[0].device and value.numel():
            store = value.untyped_storage()
            lo, hi = store.data_ptr(), store.data_ptr() + store.nbytes()
            if any(lo < end and start < hi for start, end in regions):
                raise ValueError("IPC storage aliases live model state")


def workspace_bytes(metadata: dict) -> int:
    """Bound NaN expansion, mask/cast scratch and the int64 index
    conversion."""
    sizes = [math.prod(e["shape"]) * getattr(torch, e["dtype"]).itemsize for e in metadata["tensors"]]
    largest = max(sizes, default=0)
    return (
        min(sum(sizes), CHUNK_BYTES)
        + 4 * largest
        + 8 * max((e["count"] for e in metadata["tensors"] if not e["full"]), default=0)
    )


class DeltaReceiver:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.version: str | None = None
        self.active: dict | None = None
        self.schema: dict = {}
        self.document: dict | None = None
        self.failed = False
        self.expert_receiver: RoutedExpertReceiver | None = None
        self.verifier = None

    def begin(self, control: dict) -> None:
        if self.active is not None or self.failed:
            raise RuntimeError("Receiver already has an active or failed publication")
        if type(control.get("seed")) is not bool or not isinstance(control.get("version"), str):
            raise ValueError("Missing explicit seed decision or version")
        if not control["seed"] and (self.version is None or control.get("base") != self.version):
            raise ValueError("DELTA does not match the acknowledged version")
        if (
            type(control.get("expert_direct", False)) is not bool
            or type(control.get("verify", False)) is not bool
            or (control["seed"] and control.get("verify"))
        ):
            raise ValueError("Invalid expert direct capability or verification request")
        device = next(self.model.parameters()).device
        if device.type == "cuda" and str(torch.cuda.get_device_properties(device).uuid).lower().removeprefix(
            "gpu-"
        ) != str(control.get("uuid")).lower().removeprefix("gpu-"):
            raise ValueError("SGLang and its CUDA IPC proxy must share a physical GPU")
        if control["seed"]:
            self.expert_receiver = RoutedExpertReceiver(self.model) if control.get("expert_direct") else None
        self.active = dict(control=control, next_frame=0, schema={}, verified_schema={})
        self.document = None

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

    def consume(self, control: dict, raw: torch.Tensor, status: torch.Tensor) -> None:
        active = self.active
        if (
            active is None
            or control.get("version") != active["control"]["version"]
            or control.get("frame") != active["next_frame"]
        ):
            raise ValueError("Frames must arrive once in publication order")
        if (
            type(raw) is not torch.Tensor
            or raw.dtype != torch.uint8
            or raw.ndim != 1
            or not raw.is_contiguous()
            or type(status) is not torch.Tensor
            or status.dtype != torch.uint8
            or status.ndim != 1
            or not status.is_contiguous()
            or status.device != raw.device
            or raw.device != next(self.model.parameters()).device
        ):
            raise ValueError("Invalid CUDA IPC payload or status")
        validate_transport_storage(self.model, (raw, status))
        budget = active["control"]["receiver_gpu_budget_bytes"] - active["control"]["staging_budget_bytes"]
        if "metadata_ipc" in control:
            count, compressed, size = control["metadata_ipc"]
            if (
                "metadata" in control
                or "metadata_zlib" in control
                or self.document is not None
                or any(type(n) is not int for n in (count, compressed, size))
                or not 0 < count <= 65536
                or not 0 < compressed <= size <= MAX_DOCUMENT_BYTES
                or status.numel() != count + compressed
                or status.numel() > budget
            ):
                raise ValueError("Invalid CUDA IPC directory")
            directory = _decode_directory(status[count:].cpu().numpy().tobytes(), size, control.get("metadata_id"))
            status = status[:count]
            budget -= compressed + count
        elif "metadata_zlib" in control:
            directory = _inline_directory(control)
        elif "metadata" in control:
            directory = control["metadata"]
        elif self.document is not None and control.get("metadata_id") == self.document["id"]:
            directory = self.document["metadata"]
        else:
            raise ValueError("Frame lacks its complete manifest")
        if not isinstance(directory, dict):
            raise ValueError("Delta frame lacks a manifest object")
        direct = directory.get("expert_direct", False)
        verify = directory.get("verify", False)
        seed = active["control"]["seed"]
        if (
            type(direct) is not bool
            or type(verify) is not bool
            or ((direct or verify) and seed)
            or (verify and not active["control"].get("verify"))
            or (direct and (self.expert_receiver is None or verify))
            or (self.verifier is not None and not verify)
        ):
            raise ValueError("Invalid expert-direct or verification frame phase")
        validate_directory(directory, max_bytes=active["control"]["staging_budget_bytes"])
        entries = directory["tensors"]
        if directory["bytes"] != raw.numel() or status.numel() != len(entries) or workspace_bytes(directory) > budget:
            raise ValueError("Frame exceeds its admitted workspace or has an invalid length")
        for entry in entries:
            name = entry["name"]
            shape = tuple(entry["shape"])
            dtype = entry["dtype"]
            if seed or verify:
                observed = active["schema" if seed else "verified_schema"]
                if not entry["full"] or name in observed:
                    raise ValueError("A native FULL pass must enumerate each complete weight once")
                observed[name] = (shape, dtype)
            if seed:
                continue
            if entry.get("expert_direct"):
                if not direct:
                    raise ValueError("Receiver-local expert slice requires explicit lane delivery")
                shard = self.expert_receiver.local_slice(name)
                if shard is None or entry["full"] or shape != shard.local_shape:
                    raise ValueError(f"Unqualified receiver-local expert slice: {name}")
                shape = shard.shape
            if self.schema.get(name) != (shape, dtype):
                raise ValueError(f"Delta slot is absent from the FULL seed or changed shape/dtype: {name}")
        if verify and self.verifier is None:
            from .installed_weight_verify import InstalledWeightVerifier

            self.verifier = InstalledWeightVerifier(self.model)
        for spec, positions, values in frame_flushes(directory, raw):
            load_weights = None
            if spec.get("expert_direct"):
                load_weights = self.expert_receiver.load_local
            elif seed and self.expert_receiver is not None:
                load_weights = self.expert_receiver.load_full
            apply_flush(self.model, spec, positions, values, load_weights=load_weights)
        status.zero_()
        if raw.is_cuda:
            torch.cuda.synchronize(raw.device)
        active["next_frame"] += 1
        self.document = None

    def finalize(self, control: dict, status: torch.Tensor) -> None:
        active = self.active
        if (
            active is None
            or self.document is not None
            or control.get("version") != active["control"]["version"]
            or control.get("frames") != active["next_frame"]
        ):
            raise ValueError("Incomplete version at SGLang final acknowledgement")
        schema = active["schema"] if active["control"]["seed"] else self.schema
        count = control.get("full_required_count", 0)
        route_capacity = control.get("expert_route_capacity", 0)
        verify_capacity = control.get("verify_ack_bytes", 0)
        verify = active["control"].get("verify", False)
        route_ack = 0
        if active["control"]["seed"] and self.expert_receiver is not None:
            route_ack = EXPERT_ROUTE_CAPACITY
        if (
            not schema
            or type(count) is not int
            or route_capacity != route_ack
            or verify_capacity != (VERIFY_ACK_BYTES if verify else 0)
            or status.numel() != 1 + count + route_capacity + verify_capacity
            or (count and (count != len(schema) or control.get("full_required_schema") != schema_digest(schema)))
        ):
            raise ValueError("Final receiver schema or acknowledgement size differs")
        payload = b""
        if verify:
            if self.verifier is None or active["verified_schema"] != schema:
                raise ValueError("Native verification did not cover the complete acknowledged schema")
            compared = self.verifier.finish(self.model)
            self.close_verifier()
            if type(compared) is not int or not 0 < compared < 1 << 64:
                raise ValueError("Native verification did not report compared weight bytes")
            payload = compared.to_bytes(VERIFY_ACK_BYTES, "little")
        if route_capacity:
            payload = self.expert_receiver.route_ack()
        status.zero_()
        if payload:
            status[1 + count : 1 + count + len(payload)].copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        if status.is_cuda:
            torch.cuda.synchronize(status.device)
        self.schema, self.version = schema, control["version"]
        self.active = None

    def close_verifier(self) -> None:
        if self.verifier is not None:
            try:
                self.verifier.close()
            finally:
                self.verifier = None


def load_delta_weights(model: Any, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    weights = list(weights)
    if not 1 <= len(weights) <= 2:
        raise ValueError("Expected one CUDA IPC command and optional status")
    control = json.loads(weights[0][0])
    receiver = getattr(model, "_relax_delta_receiver", None)
    if receiver is None:
        receiver = model._relax_delta_receiver = DeltaReceiver(model)
    if receiver.failed:
        raise RuntimeError("Receiver has a failed publication")
    try:
        op = control["op"]
        if op == "document":
            receiver.accept_document(control)
        elif op == "begin":
            receiver.begin(control)
        elif len(weights) == 2 and weights[1][0] == "__delta_capability__" and op in ("consume", "finalize"):
            raw, status = weights[0][1], weights[1][1]
            if op == "consume":
                receiver.consume(control, raw, status)
            else:
                if (
                    type(status) is not torch.Tensor
                    or status.dtype != torch.uint8
                    or status.ndim != 1
                    or not status.is_contiguous()
                    or status.device != next(model.parameters()).device
                ):
                    raise ValueError("Invalid final receiver acknowledgement")
                validate_transport_storage(model, (raw, status))
                receiver.finalize(control, status)
        else:
            raise ValueError("Unknown or incomplete receiver command")
    except BaseException as exc:
        receiver.failed = True
        try:
            receiver.close_verifier()
        except Exception as close_error:
            exc.add_note(f"Receiver verification cleanup failed: {close_error}")
        raise

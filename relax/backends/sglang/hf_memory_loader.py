# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Typed HF cache feeding the ordinary synchronous SGLang weight loader.

Wire buckets are storage units, not loader invocation boundaries. Each update
loads the complete ordered HF directory once. Coverage requires synchronous
per-item parameter writes; buffered multi-input writes, undeclared aliases,
padding and persistent model buffers need explicit contracts before admission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any


HF_MEMORY_LOADER = "relax.backends.sglang.hf_memory_loader.load_hf_memory_weights"


def _parameter_storage(model: Any) -> dict[str, tuple]:
    result = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        storage = parameter.untyped_storage()
        result[name] = (
            str(parameter.device),
            int(storage._cdata),
            int(storage.data_ptr()),
            parameter.dtype,
            tuple(parameter.shape),
            tuple(parameter.stride()),
            int(parameter.storage_offset()),
        )
    return result


class HFMemoryReceiver:
    """One worker's cache and strict stream state, with no automatic recovery.

    The GPU budget covers the complete HF cache, one encoded staging frame,
    sparse int64 indices and an explicit ordinary-loader workspace reserve. The
    reserve is a loader contract/estimate, not a hard limit on arbitrary Python
    loaders; physical GPU peak measurements remain an acceptance gate. Live
    model storage is outside this synchronization allocation budget. Any
    failure invalidates the stream; recovery requires a fresh FULL stream. The
    caller keeps generation paused across the entire fixed worker group.
    """

    def __init__(
        self,
        model: Any,
        *,
        host_budget_bytes: int = 8 << 30,
        receiver_gpu_budget_bytes: int = 12 << 30,
        loader_workspace_budget_bytes: int = 0,
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (host_budget_bytes, receiver_gpu_budget_bytes)):
            raise ValueError("HF receiver budgets must be positive integers")
        if type(loader_workspace_budget_bytes) is not int or loader_workspace_budget_bytes < 0:
            raise ValueError("HF loader workspace reserve must be a nonnegative integer")
        self.model = model
        self.host_budget_bytes = host_budget_bytes
        self.receiver_gpu_budget_bytes = receiver_gpu_budget_bytes
        self.loader_workspace_budget_bytes = loader_workspace_budget_bytes
        self.stream: str | None = None
        self.plan_fingerprint: str | None = None
        self.version = -1
        self.schema: str | None = None
        self.update: str | None = None
        self.poisoned = False
        self.cache_schema: tuple = ()
        self.cache_bytes = 0
        self.peak_bytes = 0
        self._cache: list[Any] = []
        self._blocked_streams: set[str] = set()
        self._storage = _parameter_storage(model)
        if not self._storage:
            raise ValueError("HF receiver requires live model parameters")
        devices = {item[0] for item in self._storage.values()}
        if len(devices) != 1 or not all(device == "cpu" or device.startswith("cuda") for device in devices):
            raise ValueError("HF receiver requires one CPU or CUDA parameter device")
        self.device = next(model.parameters()).device
        self._check_model()

    def _check_model(self) -> None:
        if _parameter_storage(self.model) != self._storage:
            raise ValueError("HF receiver model parameter storage/schema changed")
        if any(
            value is not None and name not in module._non_persistent_buffers_set
            for module in self.model.modules()
            for name, value in module._buffers.items()
        ):
            raise ValueError("HF receiver has no update contract for persistent model buffers")

    def poison(self, stream: str | None = None) -> None:
        """Invalidate both the active stream and a failed incoming stream."""
        self.poisoned = True
        for value in (self.stream, stream):
            if isinstance(value, str) and value:
                self._blocked_streams.add(value)

    def apply(self, manifest: dict, payloads: list[bytes]) -> None:
        """Validate immutable frames before any cache or model mutation."""
        from relax.distributed.weight_sync.memory_delta import validate_bucket

        try:
            metadata = manifest["buckets"]
            if len(metadata) != len(payloads) or sum(map(len, payloads)) > self.host_budget_bytes:
                raise ValueError("HF payload count or host budget mismatch")
            if any(type(payload) is not bytes for payload in payloads):
                raise ValueError("Direct HF apply requires immutable payload bytes")
            for item, payload in zip(metadata, payloads, strict=True):
                validate_bucket(item, payload, max_bytes=self.host_budget_bytes)
            self._apply_validated(manifest, payloads)
        except BaseException:
            self.poison(manifest.get("stream"))
            raise

    def _preflight(self, manifest: dict, payloads: list[bytes | bytearray]) -> tuple[tuple, int, int]:
        from relax.distributed.weight_sync.memory_delta import _validate_bucket_structure

        metadata = manifest.get("buckets")
        if not isinstance(metadata, list) or not metadata or len(metadata) != len(payloads):
            raise ValueError("HF update requires a complete bucket directory")
        if sum(map(len, payloads)) > self.host_budget_bytes:
            raise ValueError("HF update exceeds its host budget")
        stream, version, base = manifest.get("stream"), manifest.get("version"), manifest.get("base")
        if not isinstance(stream, str) or not stream or stream in self._blocked_streams:
            raise ValueError("HF stream is invalid or retired; require a new FULL stream")
        if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in ("schema", "update")):
            raise ValueError("HF update requires schema and update identities")
        if type(version) is not int or type(base) is not int or version < 0 or version != base + 1:
            raise ValueError("Invalid HF version/base sequence")
        if type(manifest.get("reset")) is not bool:
            raise ValueError("HF reset must be a boolean")
        fresh = stream != self.stream
        if fresh:
            if not manifest["reset"] or version != 0 or base != -1:
                raise ValueError("A new HF stream requires FULL reset version zero")
        elif self.poisoned or manifest["reset"] or base != self.version or manifest["schema"] != self.schema:
            raise ValueError("HF stream, schema or base version mismatch")

        seen: set[str] = set()
        layout = []
        cache_bytes = staging_bytes = 0
        for item, payload in zip(metadata, payloads, strict=True):
            _validate_bucket_structure(item, payload, max_bytes=self.host_budget_bytes)
            if fresh and item["encoding"] != "full":
                raise ValueError("A new HF stream requires every frame in FULL form")
            names = tuple((name, tuple(shape)) for name, shape in item["names"])
            for name, _shape in names:
                if name in seen:
                    raise ValueError("Duplicate HF name across wire buckets")
                seen.add(name)
            dtype = item.get("dtype", "bfloat16")
            layout.append((names, dtype))
            cache_bytes += item["elements"] * (2 if dtype == "bfloat16" else 4)
            scratch = item["changed"] * 8 if item["encoding"] == "sparse" else 0
            staging_bytes = max(staging_bytes, len(payload) + scratch)
        layout = tuple(layout)
        schema = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
        if schema != manifest["schema"]:
            raise ValueError("HF schema identity does not match the complete bucket directory")
        if not fresh and layout != self.cache_schema:
            raise ValueError("HF cache names, shapes, dtypes or bucket order changed")
        required = cache_bytes + staging_bytes + self.loader_workspace_budget_bytes
        if required > self.receiver_gpu_budget_bytes:
            raise ValueError("Complete HF cache plus staging, sparse scratch and loader workspace exceeds GPU budget")
        self._check_model()
        return layout, cache_bytes, required

    def _apply_validated(self, manifest: dict, payloads: list[bytes | bytearray]) -> None:
        """Consume transport-owned frames already integrity checked by TCP."""
        import torch

        from relax.backends.sglang.weight_write_coverage import record_parameter_write_coverage

        try:
            layout, cache_bytes, required = self._preflight(manifest, payloads)
            fresh = manifest["stream"] != self.stream
            self.poisoned = True
            if fresh:
                if self.stream is not None:
                    self._blocked_streams.add(self.stream)
                # Discard a failed/old cache before allocating its replacement;
                # the budget never assumes two complete HF mirrors fit.
                self._cache.clear()
                self.cache_bytes = 0
                self.cache_schema = layout
                for item in manifest["buckets"]:
                    self._cache.append(
                        torch.empty(
                            item["elements"], dtype=getattr(torch, item.get("dtype", "bfloat16")), device=self.device
                        )
                    )
                self.cache_bytes = cache_bytes
            self.peak_bytes = max(self.peak_bytes, required)
            with torch.no_grad():
                for cache, item, payload in zip(self._cache, manifest["buckets"], payloads, strict=True):
                    self._apply_frame(cache, item, payload)
                names = {name for bucket, _dtype in self.cache_schema for name, _shape in bucket}
                with record_parameter_write_coverage(self.model, required_hf_names=names) as coverage:
                    self.model.load_weights(coverage.tag_weights(self._named_weights()))
            self._check_model()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.stream, self.schema = manifest["stream"], manifest["schema"]
            self.version, self.update = manifest["version"], manifest["update"]
            self.poisoned = False
        except BaseException:
            self.poison(manifest.get("stream"))
            raise

    def _apply_frame(self, cache: Any, item: dict, payload: bytes | bytearray) -> None:
        import torch

        if not payload:
            return
        # TCP private_buffers=True already owns a writable host allocation.
        # Direct callers provide bytes, which need one writable host wrapper.
        host = payload if isinstance(payload, bytearray) else bytearray(payload)
        raw = torch.frombuffer(host, dtype=torch.uint8).to(self.device)
        words = torch.int16 if cache.element_size() == 2 else torch.int32
        if item["encoding"] == "full":
            cache.view(words).copy_(raw.view(words))
        else:
            count = item["changed"]
            indices = raw[: count * 4].view(torch.int32).to(torch.int64)
            values = raw[count * 4 :].view(words)
            cache.view(words).index_copy_(0, indices, values)

    def _named_weights(self) -> Iterator[tuple[str, Any]]:
        import math

        for cache, (names, _dtype) in zip(self._cache, self.cache_schema, strict=True):
            offset = 0
            for name, shape in names:
                size = math.prod(shape)
                yield name, cache[offset : offset + size].view(shape)
                offset += size


def load_hf_memory_weights(model: Any, weights: list[tuple[str, Any]]) -> None:
    """Custom loader control contains private TCP credentials, never logged."""
    from relax.distributed.weight_sync.memory_transport import receive_memory_update

    receiver = getattr(model, "_relax_hf_memory_receiver", None)
    failed_initializations = getattr(model, "_relax_hf_failed_streams", set())
    stream = None
    try:
        weights = list(weights)
        if len(weights) != 1 or weights[0][1].numel() != 0 or len(weights[0][0]) > 65536:
            raise ValueError("HF memory loader expects one empty control tensor")
        control = json.loads(weights[0][0])
        stream = control["stream"]
        if not isinstance(stream, str) or not stream:
            raise ValueError("HF memory control requires the expected stream")
        if stream in failed_initializations:
            raise ValueError("HF stream is retired after receiver initialization failure")
        plan = control["plan_fingerprint"]
        if not isinstance(plan, str) or len(plan) != 64 or any(c not in "0123456789abcdef" for c in plan):
            raise ValueError("HF control requires a valid synchronization plan")
        if receiver is None:
            receiver = HFMemoryReceiver(
                model,
                host_budget_bytes=control["host_budget_bytes"],
                receiver_gpu_budget_bytes=control["receiver_gpu_budget_bytes"],
                loader_workspace_budget_bytes=control["loader_workspace_budget_bytes"],
            )
            model._relax_hf_memory_receiver = receiver
        elif (
            receiver.host_budget_bytes != control["host_budget_bytes"]
            or receiver.receiver_gpu_budget_bytes != control["receiver_gpu_budget_bytes"]
            or receiver.loader_workspace_budget_bytes != control["loader_workspace_budget_bytes"]
        ):
            raise ValueError("HF receiver budgets changed")
        if receiver.device.type == "cuda" and receiver.loader_workspace_budget_bytes <= 0:
            raise ValueError("GPU ordinary loader requires an explicit positive workspace reserve")
        if stream in receiver._blocked_streams:
            raise ValueError("HF stream is retired; require a new FULL stream")

        def apply(manifest: dict, payloads: list[bytearray]) -> None:
            if manifest["stream"] != stream or manifest.get("group_update") != plan:
                raise ValueError("HF memory control and manifest stream/plan mismatch")
            if receiver.stream == stream and receiver.plan_fingerprint not in (None, plan):
                raise ValueError("HF synchronization plan changed within a stream")
            receiver._apply_validated(manifest, payloads)
            receiver.plan_fingerprint = plan

        receive_memory_update(
            control["descriptor"],
            control["member"],
            apply,
            timeout_s=control["timeout_s"],
            host_budget_bytes=receiver.host_budget_bytes,
            private_buffers=True,
        )
    except BaseException:
        if receiver is not None:
            receiver.poison(stream)
        elif isinstance(stream, str) and stream:
            # Admission may fail before a receiver exists. Changing its budget
            # or model cannot turn that failed stream into an accepted retry.
            failed_initializations.add(stream)
            model._relax_hf_failed_streams = failed_initializations
        raise

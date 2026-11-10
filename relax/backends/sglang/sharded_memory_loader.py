# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-owner expert shards and typed HF buffers for a paused MoE group."""

from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any

from relax.utils.logging_utils import get_logger


SHARDED_MEMORY_LOADER = "relax.backends.sglang.sharded_memory_loader.load_sharded_memory_weights"
_EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(w13_weight|w2_weight)")
_PREFIX = "relax.expert_slice.v1:"
logger = get_logger(__name__)


class ShardedMemoryReceiver:
    """A failed multi-stream install stays poisoned until a new complete FULL.

    Expert buffers are the existing native parameter slices. Only nonexpert HF
    tensors persist separately, under a per-worker GPU budget. The updater owns
    the fixed group, pause lock and cross-source commit gate.
    """

    def __init__(self, model: Any, *, gpu_budget_bytes: int) -> None:
        if type(gpu_budget_bytes) is not int or gpu_budget_bytes <= 0:
            raise ValueError("A positive receiver GPU budget is required")
        self.model, self.gpu_budget_bytes = model, gpu_budget_bytes
        self.version = -1
        self.root: str | None = None
        self.poisoned = False
        self.layouts: dict[str, list] = {}
        self.schemas: dict[str, str] = {}
        self.hf: dict[str, Any] = {}
        self.storage: dict[str, tuple] = {}
        self.hf_bytes = 0
        self._seen_hf: set[str] = set()

    def apply_group(self, control: dict, receive: Any) -> dict:
        import torch

        started = monotonic()
        receive_and_validate_s, gpu_apply_s = 0.0, 0.0
        root, version, group_update = control["stream"], control["version"], control["group_update"]
        reset = control.get("reset") is True
        tp_rank, tp_size, owners = control["tp_rank"], control["tp_size"], control["owners"]
        if (
            not isinstance(root, str)
            or not root
            or not isinstance(group_update, str)
            or not group_update
            or type(version) is not int
            or version < 0
            or type(tp_rank) is not int
            or tp_size != 2
            or not 0 <= tp_rank < tp_size
            or type(owners) is not int
            or owners <= 0
        ):
            raise ValueError("Invalid sharded group identity or topology")
        expected = [f"{root}:expert:{owner}:tp:{tp_rank}" for owner in range(owners)] + [f"{root}:nonexpert"]
        streams = control["streams"]
        if [item.get("stream") for item in streams] != expected:
            raise ValueError("Sharded request must contain the exact ordered owner stream set")
        if not reset and (self.poisoned or root != self.root or version != self.version + 1):
            raise ValueError("A failed or mismatched group needs a new FULL stream")
        if reset and root == self.root:
            raise ValueError("FULL reset requires a new group stream identity")
        if any(
            value is not None and name not in module._non_persistent_buffers_set
            for module in self.model.modules()
            for name, value in module._buffers.items()
        ):
            raise ValueError("Persistent inference buffers require explicit synchronization rules")
        self._validate_storage()
        self.poisoned = True
        if reset:
            self.hf.clear()
            self.hf_bytes = 0
            self.layouts.clear()
            self.schemas.clear()
        coverage: dict[str, list[tuple[int, int]]] = {}
        self._seen_hf = set()
        device = next(self.model.parameters()).device
        for index, item in enumerate(streams):
            stream = expected[index]
            kind = "nonexpert" if index == owners else "expert"

            def apply(manifest: dict, payloads: list[bytes]) -> None:
                nonlocal gpu_apply_s
                apply_started = monotonic()
                if (
                    manifest["stream"] != stream
                    or manifest.get("group_update") != group_update
                    or manifest["version"] != version
                    or manifest["base"] != version - 1
                    or manifest.get("reset") is not reset
                ):
                    raise ValueError("Mixed source update identities or versions")
                metadata = manifest["buckets"]
                layouts = [tuple((name, tuple(shape)) for name, shape in bucket["names"]) for bucket in metadata]
                if not reset and (
                    layouts != self.layouts.get(stream) or manifest["schema"] != self.schemas.get(stream)
                ):
                    raise ValueError("Shard schema or bucket layout changed")
                if reset and any(bucket["encoding"] != "full" for bucket in metadata):
                    raise ValueError("Every reset stream must be FULL")
                if len(metadata) != len(payloads):
                    raise ValueError("Shard payload count mismatch")
                for bucket, payload in zip(metadata, payloads, strict=True):
                    self._apply_bucket(bucket, payload, kind, coverage, device, owner=index, owners=owners)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                gpu_apply_s += monotonic() - apply_started
                self.layouts[stream], self.schemas[stream] = layouts, manifest["schema"]

            receive_started, prior_apply_s = monotonic(), gpu_apply_s
            receive(item["descriptor"], item["member"], apply)
            receive_and_validate_s += monotonic() - receive_started - (gpu_apply_s - prior_apply_s)
        self._check_expert_coverage(coverage)
        # The actual ordinary TP-aware loader handles dense/GDN/shared tensors.
        # It consumes the persistent, typed HF subset; experts never enter it.
        from relax.backends.sglang.weight_write_coverage import record_weight_write_coverage

        expert_names = {
            name for name, _ in self.model.named_parameters(remove_duplicate=False) if _EXPERT.fullmatch(name)
        }
        load_started = monotonic()
        with (
            torch.no_grad(),
            record_weight_write_coverage(
                self.model, required_hf_names=set(self.hf), excluded_expert_names=expert_names
            ) as recorder,
        ):
            self.model.load_weights(recorder.tag_weights(self.hf.items()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        hf_load_s = monotonic() - load_started
        self._remember_storage()
        self.version, self.root, self.poisoned = version, root, False
        self.last_metrics = dict(
            version=version,
            streams=len(streams),
            hf_bytes=self.hf_bytes,
            receive_and_validate_s=receive_and_validate_s,
            gpu_apply_and_fence_s=gpu_apply_s,
            hf_load_and_fence_s=hf_load_s,
            receiver_total_s=monotonic() - started,
        )
        if device.type == "cuda":
            self.last_metrics.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        logger.info("Sharded memory receiver stages: %s", json.dumps(self.last_metrics, sort_keys=True))
        return self.last_metrics

    def _apply_bucket(
        self, metadata: dict, payload: bytes, kind: str, coverage: dict, device: Any, *, owner: int, owners: int
    ) -> None:
        import numpy as np
        import torch

        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[metadata["dtype"]]
        if kind == "expert" and dtype != torch.bfloat16:
            raise ValueError("Native expert shards must be BF16")
        width = torch.empty((), dtype=dtype).element_size()
        # Include encoded H2D staging and the worst per-bucket sparse index
        # expansion, in addition to all persistent nonexpert buffers.
        scratch = len(payload) + metadata["changed"] * 8
        if self.hf_bytes + scratch > self.gpu_budget_bytes:
            raise ValueError("Receiver staging exceeds its GPU budget")
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device) if payload else None
        sparse = metadata["encoding"] == "sparse"
        changed = metadata["changed"]
        words = torch.int16 if width == 2 else torch.int32
        positions_cpu = np.frombuffer(payload, dtype="<i4", count=changed) if sparse else None
        positions = raw[: changed * 4].view(torch.int32) if sparse and changed else None
        values = raw[changed * 4 :].view(words) if sparse and changed else None
        offset = 0
        for name, shape in metadata["names"]:
            count = torch.Size(shape).numel()
            if kind == "expert":
                target = self._expert_target(name, shape, coverage, owner=owner, owners=owners)
            else:
                if name.startswith(_PREFIX) or ".experts." in name:
                    raise ValueError("Expert tensor in the nonexpert HF stream")
                if name in self._seen_hf:
                    raise ValueError("Duplicate nonexpert tensor in a group update")
                self._seen_hf.add(name)
                target = self.hf.get(name)
                if target is None:
                    size = count * width
                    if sparse or self.hf_bytes + size + scratch > self.gpu_budget_bytes:
                        raise ValueError("Nonexpert FULL buffers exceed their GPU budget")
                    target = torch.empty(tuple(shape), dtype=dtype, device=device)
                    self.hf[name] = target
                    self.hf_bytes += size
            if target.dtype != dtype or tuple(target.shape) != tuple(shape) or not target.is_contiguous():
                raise ValueError("Shard target shape, dtype or storage layout mismatch")
            with torch.no_grad():
                flat = target.view(words).reshape(-1)
                if sparse and changed:
                    left, right = np.searchsorted(
                        positions_cpu, np.asarray([offset, offset + count], dtype=positions_cpu.dtype)
                    ).tolist()
                    if left != right:
                        indices = positions[left:right].to(torch.int64)
                        indices.sub_(offset)
                        flat.index_copy_(0, indices, values[left:right])
                elif not sparse:
                    flat.copy_(raw[offset * width : (offset + count) * width].view(words))
            offset += count

    def _expert_target(self, descriptor: str, shape: list, coverage: dict, *, owner: int, owners: int) -> Any:
        import torch

        if not descriptor.startswith(_PREFIX):
            raise ValueError("A native expert shard requires a slice descriptor")
        name, start, full_shape = json.loads(descriptor[len(_PREFIX) :])
        if (
            not isinstance(name, str)
            or _EXPERT.fullmatch(name) is None
            or type(start) is not int
            or len(shape) != 3
            or len(full_shape) != 3
            or any(type(value) is not int or value <= 0 for value in full_shape)
            or full_shape[0] % owners
            or tuple(shape[1:]) != tuple(full_shape[1:])
            or not 0 <= start < start + shape[0] <= full_shape[0]
            or not owner * (full_shape[0] // owners)
            <= start
            < start + shape[0]
            <= (owner + 1) * (full_shape[0] // owners)
        ):
            raise ValueError("Invalid expert shard descriptor")
        parameter = self.model.get_parameter(name)
        if tuple(parameter.shape) != tuple(full_shape) or parameter.dtype != torch.bfloat16:
            raise ValueError("Expert descriptor does not match the native parameter")
        coverage.setdefault(name, []).append((start, start + shape[0]))
        return parameter.narrow(0, start, shape[0])

    def _check_expert_coverage(self, coverage: dict) -> None:
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            if not _EXPERT.fullmatch(name):
                continue
            end = 0
            for start, stop in sorted(coverage.get(name, [])):
                if start != end:
                    raise ValueError("Missing or overlapping native expert owner slices")
                end = stop
            if end != parameter.shape[0]:
                raise ValueError("Native expert owner slices do not cover every local expert")

    def _remember_storage(self) -> None:
        self.storage = {
            name: (p.data_ptr(), tuple(p.shape), p.dtype, p.device)
            for name, p in self.model.named_parameters(remove_duplicate=False)
        }

    def _validate_storage(self) -> None:
        if self.storage:
            current = {
                name: (p.data_ptr(), tuple(p.shape), p.dtype, p.device)
                for name, p in self.model.named_parameters(remove_duplicate=False)
            }
            if current != self.storage:
                raise ValueError("Inference parameter storage changed; restart the entire group")


def load_sharded_memory_weights(model: Any, weights: list[tuple[str, Any]]) -> None:
    """One engine supplies a separate control tensor for each TP worker."""
    from relax.distributed.weight_sync.memory_transport import receive_memory_update

    weights = list(weights)
    if len(weights) != 1 or weights[0][1].numel() or len(weights[0][0]) > 65536:
        raise ValueError("Sharded memory loader requires one bounded empty control tensor")
    control = json.loads(weights[0][0])
    receiver = getattr(model, "_relax_sharded_memory_receiver", None)
    if receiver is None:
        receiver = ShardedMemoryReceiver(model, gpu_budget_bytes=control["gpu_budget_bytes"])
        model._relax_sharded_memory_receiver = receiver
    elif receiver.gpu_budget_bytes != control["gpu_budget_bytes"]:
        raise ValueError("Receiver GPU budget changed")

    def receive(descriptor: dict, member: str, apply: Any) -> None:
        receive_memory_update(
            descriptor, member, apply, timeout_s=control["timeout_s"], host_budget_bytes=control["host_budget_bytes"]
        )

    receiver.apply_group(control, receive)

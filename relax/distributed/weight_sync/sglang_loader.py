# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Single-process admission gate for an embedded, eager SGLang engine.

The engine must be private to this adapter. External HTTP/Router entrypoints
need their own shared lifecycle gate and are deliberately not implied here.
"""

import asyncio
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .deadline import checkpoint as check_deadline
from .deadline import operation_remaining, remaining, time_budget
from .diagnostics import observe, save_diagnostic
from .model_config import configuration_signature, configured_experts, unpacked_expert_schema


logger = get_logger(__name__)

# Per-call telemetry, isolated between concurrent verifications. CUDA enqueue
# timings are host times; byte_compare_sync includes queued GPU work and sync.
_verification_timings = ContextVar("weight_verification_timings", default=None)


def _record_verification_phase(name: str, started: float) -> None:
    timings = _verification_timings.get()
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


# One verification owns two bounded staging slots. No target data or result is
# trusted across invocations; every byte is compared on the live CUDA device.
_verification_comparator = ContextVar("weight_verification_comparator", default=None)


class _AsyncByteComparator:
    def __init__(self, device, *, chunk_bytes=8 * 1024**2, slots=2, batch_size=128):
        import torch

        self.device, self.chunk_bytes, self.batch_size = device, chunk_bytes, batch_size
        self.slots, self.index, self.closed = [], 0, False
        self.copiers = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="weight-stage")
        self.metrics = dict(
            h2d_bytes=0,
            windows=0,
            h2d_device_seconds=0.0,
            compare_device_seconds=0.0,
            reuse_wait_seconds=0.0,
            cpu_copy_worker_seconds=0.0,
            host_staging_bytes=slots * chunk_bytes,
            device_staging_bytes=slots * (2 * chunk_bytes + batch_size),
        )
        for _ in range(slots):
            self.slots.append(
                dict(
                    host=torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
                    reference=torch.empty(chunk_bytes, dtype=torch.uint8, device=device),
                    different=torch.empty(chunk_bytes, dtype=torch.bool, device=device),
                    flags=torch.empty(batch_size, dtype=torch.bool, device=device),
                    host_flags=torch.empty(batch_size, dtype=torch.bool, pin_memory=True),
                    stream=torch.cuda.Stream(device=device),
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(3)],
                    pending=False,
                    transfer=False,
                    labels=[],
                    staged=None,
                )
            )

    def _wait(self, slot):
        if not slot["pending"]:
            return
        started = time.perf_counter()
        first, copied, done = slot["events"]
        if not done.query():
            done.synchronize()  # only before reusing a still-busy pinned slot
        self.metrics["reuse_wait_seconds"] += time.perf_counter() - started
        if slot["transfer"]:
            self.metrics["h2d_device_seconds"] += first.elapsed_time(copied) / 1000
        self.metrics["compare_device_seconds"] += copied.elapsed_time(done) / 1000
        slot["pending"] = False
        _record_verification_phase("staging_slot_wait", started)

    def _collect(self, slot, mismatches):
        self._wait(slot)
        if not slot["labels"]:
            return
        import torch

        started = time.perf_counter()
        count = len(slot["labels"])
        with torch.cuda.stream(slot["stream"]):
            slot["host_flags"][:count].copy_(slot["flags"][:count], non_blocking=True)
        slot["stream"].synchronize()
        mismatches.extend(
            label for label, failed in zip(slot["labels"], slot["host_flags"][:count].tolist(), strict=True) if failed
        )
        slot["labels"].clear()
        _record_verification_phase("byte_compare_sync", started)

    @staticmethod
    def _copy_host(destination, source):
        started = time.perf_counter()
        destination.copy_(source)
        return time.perf_counter() - started

    def _submit_staged(self, slot):
        staged = slot["staged"]
        if staged is not None:
            future, left, right, label = staged
            started = time.perf_counter()
            self.metrics["cpu_copy_worker_seconds"] += future.result()
            _record_verification_phase("reference_staging_wait", started)
            self._enqueue(slot, left, slot["host"][: left.numel()], label, transfer=True)
            slot["staged"] = None

    def compare(self, left, right, label, mismatches):
        slot = self.slots[self.index % len(self.slots)]
        self.index += 1
        self._submit_staged(slot)
        self._wait(slot)
        if len(slot["labels"]) == self.batch_size:
            self._collect(slot, mismatches)
        size = left.numel()
        if size > self.chunk_bytes:
            raise ValueError("Comparison exceeds its bounded staging slot")
        if right.device.type == "cpu":
            # Two tasks at most: overlap mmap faults/CPU copies with the other
            # slot's H2D and comparison without retaining whole-model buffers.
            future = self.copiers.submit(self._copy_host, slot["host"][:size], right.reshape(-1))
            slot["staged"] = (future, left, right, label)
            self.metrics["h2d_bytes"] += size
            for other in self.slots:
                if other is not slot and other["staged"] is not None and other["staged"][0].done():
                    self._submit_staged(other)
        else:
            self._enqueue(slot, left, right.reshape(-1), label, transfer=False)
        self.metrics["windows"] += 1

    def _enqueue(self, slot, left, source, label, *, transfer):
        import torch

        size = left.numel()
        started = time.perf_counter()
        stream = slot["stream"]
        # Covers noncontiguous physical views and packed experts created on
        # the caller's stream. Model mutation stays fenced by the worker RPC.
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            slot["events"][0].record()
            slot["reference"][:size].copy_(source, non_blocking=True)
            slot["events"][1].record()
            _record_verification_phase("reference_transfer_enqueue", started)
            started = time.perf_counter()
            torch.ne(left.reshape(-1), slot["reference"][:size], out=slot["different"][:size])
            torch.any(slot["different"][:size], out=slot["flags"][len(slot["labels"])])
            slot["events"][2].record()
        left.record_stream(stream)
        if source.device.type == "cuda":
            source.record_stream(stream)
        slot["pending"], slot["transfer"] = True, transfer
        slot["labels"].append(label)
        _record_verification_phase("byte_compare_enqueue", started)

    def finish(self, mismatches):
        for slot in self.slots:
            self._submit_staged(slot)
            self._collect(slot, mismatches)

    def close(self):
        # Also drains enqueued copies when layout validation/deadlines fail.
        # Host buffers cannot be released while DMA still reads their bytes.
        if not self.closed:
            self.copiers.shutdown(wait=True)
            for slot in self.slots:
                slot["stream"].synchronize()
            self.closed = True


def _compare_bytes(
    actual, wanted, label: str, mismatches: list[str], *, chunk_bytes: int = 8 * 1024**2, schema_only: bool = False
) -> None:
    """Compare row windows; fused expert matrices never need a second full
    copy."""
    import torch

    if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
        mismatches.append(
            f"Physical schema mismatch: {label}; actual={tuple(actual.shape)}/{actual.dtype}, "
            f"expected={tuple(wanted.shape)}/{wanted.dtype}"
        )
        return
    if schema_only:
        return
    if actual.ndim == 0:
        actual, wanted = actual.reshape(1), wanted.reshape(1)
    row_bytes = max(1, actual[0].numel() * actual.element_size()) if actual.shape[0] else 1
    rows = max(1, chunk_bytes // row_bytes)
    for start in range(0, actual.shape[0], rows):
        check_deadline("physical tensor verification")
        started = time.perf_counter()
        left = actual[start : start + rows].detach().contiguous().view(torch.uint8)
        _record_verification_phase("physical_contiguous_enqueue", started)
        started = time.perf_counter()
        right = wanted[start : start + rows].contiguous().view(torch.uint8)
        _record_verification_phase(
            "reference_contiguous_cpu" if wanted.device.type == "cpu" else "reference_contiguous_device_enqueue",
            started,
        )
        comparator = _verification_comparator.get()
        if comparator is not None and left.device == comparator.device and left.numel() <= comparator.chunk_bytes:
            comparator.compare(
                left,
                right,
                f"Physical weight mismatch: {label}, row window [{start}, {start + len(left)})",
                mismatches,
            )
            continue
        # Exact bit comparison (including signed zero and NaN payloads) on
        # the live device avoids copying and scanning the model on the CPU.
        # Only one bounded row window is staged; no full model GPU copy is
        # retained. torch.equal synchronizes before reporting a result.
        started = time.perf_counter()
        right = right.to(device=left.device)
        _record_verification_phase("reference_transfer", started)
        started = time.perf_counter()
        matches = torch.equal(left, right)
        _record_verification_phase("byte_compare_sync", started)
        if not matches:
            mismatches.append(f"Physical weight mismatch: {label}, row window [{start}, {start + len(left)})")
            return


def _compare_stacked(actual, parts, label: str, mismatches: list[str], *, schema_only: bool = False) -> None:
    offset = 0
    for wanted in parts:
        size = wanted.shape[0]
        _compare_bytes(actual[offset : offset + size], wanted, label, mismatches, schema_only=schema_only)
        offset += size
    if actual.shape[0] != offset:
        mismatches.append(f"Physical fused dimension mismatch: {label}; actual={actual.shape[0]}, expected={offset}")


def _compare_packed_expert(
    actual,
    wanted,
    label: str,
    mismatches: list[str],
    *,
    fused_gate: bool,
    tp_size: int,
    tp_rank: int,
    schema_only: bool = False,
    chunk_bytes: int = 8 * 1024**2,
) -> None:
    # HF stores [input, output]. Transfer that contiguous storage once for a
    # bounded expert, then transpose/split on its live device. Splitting first
    # makes CPU gate/up views strided and repeatedly copies the same pages.
    # Larger experts retain the bounded row-window path below.
    if (
        not schema_only
        and actual.device.type == "cuda"
        and wanted.device.type == "cpu"
        and wanted.numel() * wanted.element_size() <= chunk_bytes
        and wanted.transpose(-1, -2).is_contiguous()
    ):
        check_deadline("packed expert verification transfer")
        started = time.perf_counter()
        wanted = wanted.transpose(-1, -2).to(device=actual.device).transpose(-1, -2)
        _record_verification_phase("packed_expert_transfer", started)
    if fused_gate:
        parts = (part.chunk(tp_size, dim=0)[tp_rank] for part in wanted.chunk(2, dim=0))
        _compare_stacked(actual, parts, label, mismatches, schema_only=schema_only)
    else:
        _compare_bytes(
            actual,
            wanted.chunk(tp_size, dim=1)[tp_rank],
            label,
            mismatches,
            chunk_bytes=chunk_bytes,
            schema_only=schema_only,
        )


class CheckpointTensors(Mapping):
    """Keep the shard index, materializing one logical tensor when
    requested."""

    def __init__(self, directory: Path, *, metadata_only: bool = False) -> None:
        from safetensors import safe_open

        self.metadata_only = metadata_only
        self._readers = None
        self.paths = {}
        self.dtypes = set()
        self.schema = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in self.paths:
                        raise ValueError("Duplicate logical tensor")
                    self.paths[name] = path
                    view = source.get_slice(name)
                    dtype = view.get_dtype()
                    self.dtypes.add(dtype)
                    self.schema[name] = (tuple(view.get_shape()), dtype)

    def __getitem__(self, key: str) -> object:
        started = time.perf_counter()
        try:
            return self._get_tensor(key)
        finally:
            # mmap page faults during later copies belong to those copy times.
            _record_verification_phase("checkpoint_tensor_mapping", started)

    def _get_tensor(self, key: str) -> object:
        from safetensors import safe_open

        if self.metadata_only:
            import torch
            from safetensors.torch import _getdtype

            shape, dtype = self.schema[key]
            return torch.empty(shape, dtype=_getdtype(dtype), device="meta")
        if self._readers is not None:
            path = self.paths[key]
            if path not in self._readers:
                if len(self._readers) == 4:
                    _, reader = self._readers.popitem(last=False)
                    reader.__exit__(None, None, None)
                reader = safe_open(path, framework="pt", device="cpu")
                reader.__enter__()
                self._readers[path] = reader
            self._readers.move_to_end(path)
            return self._readers[path].get_tensor(key)
        with safe_open(self.paths[key], framework="pt", device="cpu") as source:
            return source.get_tensor(key)

    def __enter__(self) -> "CheckpointTensors":
        if self._readers is not None:
            raise RuntimeError("Checkpoint reader scope is already open")
        self._readers = OrderedDict()
        return self

    def __exit__(self, *exception: object) -> None:
        readers, self._readers = self._readers, None
        for reader in readers.values():
            reader.__exit__(*exception)

    def __iter__(self):
        return iter(self.paths)

    def __contains__(self, key):
        # Mapping's default implementation calls __getitem__, which opens a
        # shard and materializes a tensor just to check whether it exists.
        return key in self.paths

    def __len__(self) -> int:
        return len(self.paths)


class _CheckpointInspection:
    """Single-load metadata reuse, bound to unchanged local file identities.

    This never replaces checkpoint hashing or worker byte verification. Each
    session preflight constructs a new inspection, and load consumes it once.
    """

    def __init__(self, checkpoint: Path, tensors: CheckpointTensors):
        self.checkpoint = checkpoint.resolve()
        self.paths = [self.checkpoint, self.checkpoint / "config.json", *sorted(set(tensors.paths.values()))]
        self.names = set(tensors)
        self.identity = self._identity()

    def _identity(self):
        return tuple(
            (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            for path in self.paths
            for stat in [path.stat()]
        )

    def matches(self, checkpoint):
        return Path(checkpoint).resolve() == self.checkpoint and self._identity() == self.identity


def validate_tied_embeddings(config: dict, tensors: CheckpointTensors) -> tuple[str, ...]:
    text = config.get("text_config", config)
    if not config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)):
        return ()
    embedding = "model.language_model.embed_tokens.weight" if "text_config" in config else "model.embed_tokens.weight"
    aliases = (embedding, "lm_head.weight")
    if all(name in tensors for name in aliases):
        mismatches = []
        _compare_bytes(tensors[aliases[0]], tensors[aliases[1]], "tied embedding/lm_head", mismatches)
        if mismatches:
            raise ValueError("; ".join(mismatches))
        return aliases
    return ()


def verify_qwen3_dense(
    checkpoint: Path,
    dump: Path | None,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    layouts: dict | None = None,
    physical_tensors: Mapping | None = None,
    worker_rank: int | None = None,
    schema_only: bool = False,
    timings: dict | None = None,
    pipeline_metrics: dict | None = None,
) -> dict:
    """Compare all saved physical parameters to an independent HF layout map.

    Supports unquantized Qwen3 dense/MoE with static expert placement.
    Unsupported shapes or uncovered parameters fail rather than sampling.
    """
    # An unpacked MoE checkpoint has thousands of tensors in each shard.
    # Reopening it per tensor reparses the entire header quadratically. Keep
    # at most four mmap readers during this verification, never across targets.
    started = time.perf_counter()
    phases = {} if timings is not None else None
    comparator = None
    if not schema_only and physical_tensors:
        devices = {value.device for value in physical_tensors.values()}
        if len(devices) == 1 and next(iter(devices)).type == "cuda":
            comparator = _AsyncByteComparator(next(iter(devices)))
    token = _verification_timings.set(phases)
    comparison_token = _verification_comparator.set(comparator)
    try:
        with CheckpointTensors(checkpoint, metadata_only=schema_only) as expected:
            inventory_seconds = time.perf_counter() - started
            result = _verify_qwen3_dense(
                checkpoint,
                dump,
                expected=expected,
                tp_size=tp_size,
                pp_size=pp_size,
                layouts=layouts,
                physical_tensors=physical_tensors,
                worker_rank=worker_rank,
                schema_only=schema_only,
            )
        if timings is not None:
            layout_seconds = time.perf_counter() - started - inventory_seconds
            timings.update(
                checkpoint_inventory=inventory_seconds,
                layout_and_content=layout_seconds,
                layout_and_dispatch=layout_seconds - sum(phases.values()),
                **phases,
            )
        if pipeline_metrics is not None and comparator is not None:
            pipeline_metrics.update(comparator.metrics)
        return result
    finally:
        try:
            if comparator is not None:
                comparator.close()
        finally:
            _verification_comparator.reset(comparison_token)
            _verification_timings.reset(token)


def _verify_qwen3_dense(
    checkpoint: Path,
    dump: Path | None,
    *,
    expected: CheckpointTensors,
    tp_size: int,
    pp_size: int,
    layouts: dict | None,
    physical_tensors: Mapping | None,
    worker_rank: int | None,
    schema_only: bool,
    tied_aliases: tuple[str, ...] | None = None,
) -> dict:
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_bytes())
    if config.get("model_type") not in ("qwen3", "qwen3_moe", "qwen3_vl", "qwen3_vl_moe"):
        raise ValueError("A Qwen3 or Qwen3-VL Dense/MoE layout is required")
    is_vl = config["model_type"] in ("qwen3_vl", "qwen3_vl_moe")
    text_config = config.get("text_config", config)
    heads = text_config["num_attention_heads"]
    kv_heads = text_config["num_key_value_heads"]
    if heads % tp_size or (kv_heads % tp_size and tp_size % kv_heads):
        raise ValueError("Incompatible attention head partition")
    if layouts is None:
        if pp_size != 1:
            raise ValueError("Pipeline verification requires runtime layer ranges")
        layouts = {
            rank: {"moe_ep_rank": 0, "moe_ep_size": 1, "moe_tp_rank": rank, "moe_tp_size": tp_size}
            for rank in range(tp_size)
        }
    if set(layouts) != set(range(tp_size * pp_size)):
        raise ValueError("Incomplete worker layouts")
    end = 0
    for pp in range(pp_size):
        expert_members, ranges = set(), set()
        for rank in range(tp_size):
            layout = layouts[pp * tp_size + rank]
            if layout.get("tp_rank", rank) != rank or layout.get("pp_rank", pp) != pp:
                raise ValueError("Worker coordinates disagree with its receipt")
            ep, etp = layout["moe_ep_size"], layout["moe_tp_size"]
            er, tr = layout["moe_ep_rank"], layout["moe_tp_rank"]
            if ep * etp != tp_size or not 0 <= er < ep or not 0 <= tr < etp:
                raise ValueError(f"Invalid expert layout for rank {rank}")
            expert_members.add((ep, etp, er, tr))
            if pp_size > 1:
                ranges.add((layout["start_layer"], layout["end_layer"]))
        if len(expert_members) != tp_size or len({(e, t) for e, t, _, _ in expert_members}) != 1:
            raise ValueError("Expert worker layouts overlap or disagree")
        if pp_size > 1:
            if len(ranges) != 1:
                raise ValueError("Tensor-parallel workers disagree on their pipeline range")
            start, stop = ranges.pop()
            if start != end or stop <= start:
                raise ValueError("Pipeline layer ranges overlap or leave a gap")
            end = stop
    if pp_size > 1 and end != text_config["num_hidden_layers"]:
        raise ValueError("Pipeline layer ranges do not cover the model")
    if tied_aliases is None:
        tied_aliases = validate_tied_embeddings(config, CheckpointTensors(checkpoint) if schema_only else expected)
    physical = {} if physical_tensors is None else physical_tensors
    for path in sorted(dump.glob("*.safetensors")) if dump is not None else ():
        # SGLang ShardedStateLoader default filename: model-rank-R-part-P.
        match = re.fullmatch(r"model-rank-(\d+)-part-\d+\.safetensors", path.name)
        if not match:
            raise ValueError("Unknown shard filename")
        rank = int(match[1])
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if (rank, name) in physical:
                    raise ValueError("Duplicate physical parameter")
                physical[rank, name] = path
    required_ranks = set(range(tp_size * pp_size)) if worker_rank is None else {worker_rank}
    if not required_ranks <= set(layouts) or {rank for rank, _ in physical} != required_ranks:
        raise ValueError("Incomplete worker dump")
    if is_vl:
        # SGLang 0.5.17 replicates the vision blocks across PP stages; only
        # the first stage owns the positional embedding.
        vision_names = {
            name.removeprefix("model.").replace(".attn.qkv.", ".attn.qkv_proj.")
            for name in expected
            if name.startswith("model.visual.")
        }
        for worker in required_ranks:
            actual_names = {name for rank, name in physical if rank == worker and name.startswith("visual.")}
            stage_names = vision_names if worker // tp_size == 0 else vision_names - {"visual.pos_embed.weight"}
            if actual_names != stage_names:
                raise ValueError(
                    f"Incomplete vision parameter inventory on worker {worker}: "
                    f"missing={sorted(stage_names - actual_names)}, unexpected={sorted(actual_names - stage_names)}"
                )
    for pp in range(pp_size):
        names = {name for rank, name in physical if rank == pp * tp_size}
        for rank in range(pp * tp_size, (pp + 1) * tp_size) if worker_rank is None else ():
            if {name for r, name in physical if r == rank} != names:
                raise ValueError("Physical parameter coverage differs between tensor-parallel workers")
    covered, checked, mismatches = set(), 0, []
    for (worker, name), path in physical.items():
        pp, rank = divmod(worker, tp_size)
        if physical_tensors is None:
            with safe_open(path, framework="pt", device="cpu") as source:
                actual = source.get_tensor(name)
        else:
            actual = path
        logical_name = name
        parts = None
        wanted = None
        label = f"rank={worker} {name}"
        if pp_size > 1:
            layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            layout = layouts[worker]
            if layer and not layout["start_layer"] <= int(layer[1]) < layout["end_layer"]:
                raise ValueError(f"Parameter is outside its pipeline range: {label}")
            if name == "model.embed_tokens.weight" and pp != 0:
                raise ValueError(f"Input parameter belongs on the first pipeline stage: {label}")
            if name in ("lm_head.weight", "model.norm.weight") and pp != pp_size - 1:
                raise ValueError(f"Output parameter belongs on the last pipeline stage: {label}")
        if is_vl:
            if name.startswith("visual."):
                logical_name = "model." + name.replace(".attn.qkv_proj.", ".attn.qkv.")
            elif name.startswith("model."):
                logical_name = name.replace("model.", "model.language_model.", 1)
        if name.startswith("visual."):
            wanted = expected[logical_name]
            covered.add(logical_name)
            if ".attn.qkv_proj." in name:
                parts = [part.chunk(tp_size, dim=0)[rank] for part in wanted.chunk(3, dim=0)]
                wanted = None
            elif ".linear_fc1." in name:
                wanted = wanted.chunk(tp_size, dim=0)[rank]
            elif name.endswith((".attn.proj.weight", ".linear_fc2.weight")):
                wanted = wanted.chunk(tp_size, dim=1)[rank]
            elif name == "visual.pos_embed.weight":
                wanted = wanted.chunk(tp_size, dim=0)[rank]
        elif name.endswith(".experts.w13_weight") or name.endswith(".experts.w2_weight"):
            prefix = logical_name.rsplit(".", 1)[0]
            expert_count = configured_experts(text_config)
            if type(expert_count) is not int or expert_count < 1:
                raise ValueError("Missing expert count in model configuration")
            layout = layouts[worker]
            ep_size, ep_rank = layout["moe_ep_size"], layout["moe_ep_rank"]
            moe_tp_size, moe_tp_rank = layout["moe_tp_size"], layout["moe_tp_rank"]
            if expert_count % ep_size:
                raise ValueError("Expert count must divide evenly across workers")
            local_experts = expert_count // ep_size
            if actual.ndim != 3 or actual.shape[0] != local_experts:
                raise ValueError(f"Unsupported expert layout: {label}, {tuple(actual.shape)}")
            fused_gate = name.endswith("w13_weight")
            packed_key = f"{prefix}.{'gate_up_proj' if fused_gate else 'down_proj'}"
            packed = expected[packed_key] if packed_key in expected else None
            if packed is not None:
                # HF also stores all experts in one tensor. Keep that logical
                # identity while selecting the worker's experts and TP slices.
                if packed.ndim != 3 or packed.shape[0] != expert_count:
                    raise ValueError(f"Invalid packed expert shape: {packed_key}, {tuple(packed.shape)}")
                # Published Qwen3-VL packed tensors use [expert, input, output];
                # SGLang's fused weights use [expert, output, input].
                packed = packed.transpose(-1, -2)
                axis, divisor = (1, 2 * moe_tp_size) if fused_gate else (2, moe_tp_size)
                if packed.shape[axis] == 0 or packed.shape[axis] % divisor:
                    raise ValueError(f"Uneven packed expert partition: {packed_key}")
                covered.add(packed_key)
            for local_expert in range(local_experts):
                expert = ep_rank * local_experts + local_expert
                if packed is not None:
                    _compare_packed_expert(
                        actual[local_expert],
                        packed[expert],
                        f"{label} expert={expert}",
                        mismatches,
                        fused_gate=fused_gate,
                        tp_size=moe_tp_size,
                        tp_rank=moe_tp_rank,
                        schema_only=schema_only,
                    )
                    continue
                if fused_gate:
                    keys = [f"{prefix}.{expert}.{part}.weight" for part in ("gate_proj", "up_proj")]
                    logical_parts = (expected[key] for key in keys)
                    _compare_stacked(
                        actual[local_expert],
                        (part.chunk(moe_tp_size, dim=0)[moe_tp_rank] for part in logical_parts),
                        f"{label} expert={expert}",
                        mismatches,
                        schema_only=schema_only,
                    )
                else:
                    keys = [f"{prefix}.{expert}.down_proj.weight"]
                    logical_down = expected[keys[0]]
                    _compare_bytes(
                        actual[local_expert],
                        logical_down.chunk(moe_tp_size, dim=1)[moe_tp_rank],
                        f"{label} expert={expert}",
                        mismatches,
                        schema_only=schema_only,
                    )
                covered.update(keys)
        elif ".qkv_proj." in name:
            parts = []
            for part in ("q_proj", "k_proj", "v_proj"):
                key = logical_name.replace("qkv_proj", part)
                tensor = expected[key]
                if part != "q_proj" and kv_heads < tp_size:
                    parts.append(tensor.chunk(kv_heads, dim=0)[rank // (tp_size // kv_heads)])
                else:
                    if tensor.shape[0] % tp_size:
                        raise ValueError("Uneven QKV partition")
                    parts.append(tensor.chunk(tp_size, dim=0)[rank])
                covered.add(key)
        elif ".gate_up_proj." in name:
            keys = [logical_name.replace("gate_up_proj", p) for p in ("gate_proj", "up_proj")]
            parts = [expected[k].chunk(tp_size, dim=0)[rank] for k in keys]
            covered.update(keys)
        else:
            key = logical_name
            if (
                key == "lm_head.weight"
                and key not in expected
                and config.get("tie_word_embeddings", text_config.get("tie_word_embeddings", False))
            ):
                key = "model.language_model.embed_tokens.weight" if is_vl else "model.embed_tokens.weight"
            wanted = expected[key]
            covered.add(key)
            if any(p in name for p in (".o_proj.weight", ".down_proj.weight")):
                wanted = wanted.chunk(tp_size, dim=1)[rank]
            elif name in ("model.embed_tokens.weight", "lm_head.weight"):
                if wanted.shape[0] % tp_size:
                    raise ValueError("Padded vocabulary is not supported yet")
                wanted = wanted.chunk(tp_size, dim=0)[rank]
        if parts is not None:
            _compare_stacked(actual, parts, label, mismatches, schema_only=schema_only)
        elif wanted is not None:
            _compare_bytes(actual, wanted, label, mismatches, schema_only=schema_only)
        checked += actual.numel() * actual.element_size()
    comparator = _verification_comparator.get()
    if comparator is not None:
        comparator.finish(mismatches)
    if mismatches:
        raise ValueError("; ".join(mismatches))
    if covered.intersection(tied_aliases):
        covered.update(tied_aliases)
    if worker_rank is None and covered != set(expected):
        raise ValueError(f"Uncovered logical weights: {sorted(set(expected) - covered)}")
    result = {"physical_parameters": len(physical), "logical_parameters": len(covered), "bytes_checked": checked}
    if worker_rank is not None:
        result.update(covered_logical_names=sorted(covered), physical_names=sorted(name for _, name in physical))
    return result


class EmbeddedSGLangGroup:
    def __init__(
        self,
        engine: object | None,
        *,
        tp_size: int,
        audit_directory: str | Path,
        model_path: str | Path | None = None,
        pp_size: int | None = None,
    ) -> None:
        self.engine = engine
        model_path = model_path if model_path is not None else engine.server_args.model_path
        self.config_hash = configuration_signature(json.loads((Path(model_path) / "config.json").read_bytes()))
        self.startup_schema = CheckpointTensors(Path(model_path)).schema
        self.tp_size = tp_size
        self.pp_size = pp_size if pp_size is not None else getattr(getattr(engine, "server_args", None), "pp_size", 1)
        self.audit_directory = Path(audit_directory)
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        self._gate = threading.RLock()
        self._serving = False
        self.verification = None
        self._rpc_failed = False
        self._workers = self._rpc("identify")
        self.layouts = {item["rank"]: item["layout"] for item in self.worker_receipts}
        self.physical_schema = {item["rank"]: item["physical_schema"] for item in self.worker_receipts}

    def _rpc(self, phase: str, *, command_id: str | None = None, **parameters) -> dict:
        if self._rpc_failed:
            raise RuntimeError("Worker RPC timed out; restart the engine and restore a retained checkpoint")
        if self.pp_size > 1 and phase in ("load", "dump", "verify", "flush"):
            # PP forwards requests sequentially. Collect every stage's identity
            # check before a second request can mutate any stage's weights.
            self._rpc("check", **parameters)
        request_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="rpc-", dir=self.audit_directory) as directory:
            request = {
                "request_id": request_id,
                "receipts": directory,
                "phase": phase,
                "command_id": command_id or request_id,
                "workers": getattr(self, "_workers", {}),
                **parameters,
                "timeout": operation_remaining(120, "worker command"),
            }
            paths = [Path(directory) / f"rank-{rank}.json" for rank in range(self.tp_size * self.pp_size)]
            receipts = []
            try:
                self._send_rpc(json.dumps(request))
                # SGLang forwards control requests through PP stages after the
                # first stage replies. Wait for the final stage's durable receipt.
                deadline = time.monotonic() + operation_remaining(120, "pipeline receipts")
                while not all(path.is_file() for path in paths):
                    if time.monotonic() >= deadline:
                        self._rpc_failed = True
                        raise TimeoutError(
                            "Pipeline worker receipts are incomplete; restart the engine before recovery"
                        )
                    time.sleep(0.01)
                receipts = [json.loads(path.read_bytes()) for path in paths]
                if any(item["request_id"] != request_id or not item["success"] for item in receipts):
                    if all(
                        item["request_id"] == request_id
                        and not item["success"]
                        and item.get("error_code") == "WORKER_BUSY"
                        for item in receipts
                    ):
                        from .sglang_plugin import WorkerBusyError

                        raise WorkerBusyError(f"Workers have generation cleanup pending: {receipts}")
                    raise RuntimeError(f"Worker installation failed: {receipts}")
                if phase in ("identify", "begin", "check") and hasattr(self, "layouts"):
                    if {item["rank"]: item["layout"] for item in receipts} != self.layouts:
                        raise RuntimeError("Worker layout changed; create a new consumer session")
                self.worker_receipts = receipts
                return {str(item["rank"]): item["identity"] for item in receipts}
            finally:
                if phase in ("load", "verify"):
                    self._save_load_receipts(request, paths, receipts)

    def _save_load_receipts(self, request: dict, paths: list[Path], receipts: list[dict]) -> None:
        """Retain loader observations after temporary RPC receipts are
        removed."""
        from .store import canonical

        # Reuse the receipts already read for correctness checks. A second
        # diagnostic read can stall on the same filesystem after the RPC ends.
        workers = receipts
        errors = []
        if not workers:

            def collect():
                partial, failures = [], []
                for path in paths:
                    try:
                        partial.append(json.loads(path.read_bytes()))
                    except (OSError, ValueError) as error:
                        failures.append({"worker": path.name, "error": f"{type(error).__name__}: {error}"})
                return partial, failures

            observed = observe(collect, "partial worker diagnostics")
            if observed.error:
                errors.append({"error": observed.error})
            else:
                workers, errors = observed.value
        record = {
            "record_type": "loader_observation" if request["phase"] == "load" else "worker_verification",
            "request_id": request["request_id"],
            "version": request.get("version"),
            "checkpoint": request.get("checkpoint"),
            "workers": workers,
            "expected_workers": len(paths),
            "errors": errors,
        }
        try:
            save_diagnostic(
                self.audit_directory / f"{request['phase']}-receipts-{request['request_id']}.json", canonical(record)
            )
        except OSError:
            logger.warning("Loader diagnostic output failed", exc_info=True)

    def _send_rpc(self, payload: str) -> None:
        import zmq
        from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput, sock_recv, sock_send

        socket = self.engine.send_to_rpc
        timeout = operation_remaining(120, "worker RPC")
        deadline = time.monotonic() + timeout
        old_timeout = socket.getsockopt(zmq.SNDTIMEO)
        try:
            socket.setsockopt(zmq.SNDTIMEO, max(1, int(timeout * 1000)))
            sock_send(socket, RpcReqInput(method="relax_weight_version", parameters={"payload": payload}))
        finally:
            socket.setsockopt(zmq.SNDTIMEO, old_timeout)
        if not socket.poll(max(0, int((deadline - time.monotonic()) * 1000)), zmq.POLLIN):
            self._rpc_failed = True
            raise TimeoutError("Worker RPC exceeded its deadline; restart the engine before recovery")
        reply = sock_recv(socket)
        if not isinstance(reply, RpcReqOutput) or not reply.success:
            raise RuntimeError(f"Worker RPC failed: {reply}")

    def begin_install(self, installation_id: str, root: str, epoch: int) -> int:
        current = self._rpc("identify")
        epoch = max(epoch, max(item["epoch"] for item in current.values()) + 1)
        self._workers = current
        self._workers = self._rpc("begin", installation_id=installation_id, root=root, epoch=epoch)
        return epoch

    def identity(self) -> str:
        pids = self.engine.get_all_child_pids()
        if not pids or os.name != "posix":
            raise RuntimeError("Cannot establish live scheduler membership")
        identities = []
        for pid in pids:
            os.kill(pid, 0)
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                raise RuntimeError("Scheduler exited")
            identities.append((pid, fields[19]))
        workers = self._rpc("identify")
        return json.dumps([identities, {rank: item["instance"] for rank, item in workers.items()}])

    def close(self) -> None:
        self._serving = False
        if not self._gate.acquire(timeout=operation_remaining(120, "generation drain")):
            raise TimeoutError("Generation did not drain; restart the engine before checkpoint installation")
        self._gate.release()

    def pause(self) -> None:
        # Calls are serialized through generate; close waits for the current call.
        if self._serving:
            raise RuntimeError("Admission gate must be closed")

    def preflight(self, checkpoint: Path) -> None:
        config = json.loads((checkpoint / "config.json").read_bytes())
        if configuration_signature(config) != self.config_hash:
            raise ValueError("Disk update cannot change the running model configuration")
        tensors = CheckpointTensors(checkpoint, metadata_only=True)
        expected = unpacked_expert_schema(self.startup_schema, config)
        candidate_schema = unpacked_expert_schema(tensors.schema, config)
        text = config.get("text_config", config)
        if config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)):
            embedding = (
                "model.language_model.embed_tokens.weight" if "text_config" in config else "model.embed_tokens.weight"
            )
            if "lm_head.weight" in tensors and "lm_head.weight" not in expected:
                expected["lm_head.weight"] = expected[embedding]
        if set(candidate_schema) != set(expected):
            raise ValueError(
                f"Checkpoint inventory differs from startup model: missing={sorted(set(expected) - set(candidate_schema))}, extra={sorted(set(candidate_schema) - set(expected))}"
            )
        if any(candidate_schema[name][0] != schema[0] for name, schema in expected.items()):
            raise ValueError("Checkpoint tensor shapes differ from the startup model inventory")
        # Validate every logical tensor against the live parameter/buffer
        # layout. A mixed checkpoint must not bypass dtype checks or rely on
        # a loader's implicit casts: version installation preserves exact bits.
        import torch

        self._rpc("identify")
        physical = {
            (item["rank"], name): torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")), device="meta")
            for item in self.worker_receipts
            for name, (shape, dtype) in item["physical_schema"].items()
        }
        # Reuse this invocation's index for the layout pass. Tied aliases
        # still need their independent content check, once per preflight.
        aliases = ()
        if config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)):
            with CheckpointTensors(checkpoint) as content:
                aliases = validate_tied_embeddings(config, content)
        _verify_qwen3_dense(
            checkpoint,
            None,
            expected=tensors,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            layouts={item["rank"]: item["layout"] for item in self.worker_receipts},
            physical_tensors=physical,
            worker_rank=None,
            schema_only=True,
            tied_aliases=aliases,
        )
        self._pending_preflight = _CheckpointInspection(Path(checkpoint), tensors)

    def load(self, checkpoint: Path, version: str) -> None:
        inspection = getattr(self, "_pending_preflight", None)
        if inspection is None or not inspection.matches(checkpoint):
            self.preflight(checkpoint)
            inspection = self._pending_preflight
        # Consume before the RPC: a lost reply cannot leave reusable permission.
        self._pending_preflight = None
        self._loaded_inspection = None
        self._rpc("load", command_id="load", checkpoint=str(checkpoint), version=version)
        self.load_receipts = self.worker_receipts
        self._loaded_version = version
        self._loaded_inspection = inspection

    def verify(self, checkpoint: Path) -> None:
        self._rpc(
            "verify",
            checkpoint=str(checkpoint),
            version=getattr(self, "_loaded_version", None),
            layouts=self.layouts,
            expected_physical_schema=self.physical_schema,
            timeout=operation_remaining(120, "worker verification"),
        )
        results = {item["rank"]: item["result"]["verification"] for item in self.worker_receipts}
        covered = {name for item in results.values() for name in item["covered_logical_names"]}
        inspection = getattr(self, "_loaded_inspection", None)
        if inspection is not None:
            if not inspection.matches(checkpoint):
                raise ValueError("Checkpoint files changed during device installation/verification")
            expected = inspection.names
        else:
            expected = set(CheckpointTensors(checkpoint))
        if covered != expected:
            raise ValueError(f"Worker verification has incomplete logical coverage: {sorted(expected - covered)}")
        for pp in range(self.pp_size):
            names = results[pp * self.tp_size]["physical_names"]
            if any(results[pp * self.tp_size + tp]["physical_names"] != names for tp in range(self.tp_size)):
                raise ValueError("Physical parameter coverage differs between tensor-parallel workers")
        self.verification = {
            "physical_parameters": sum(item["physical_parameters"] for item in results.values()),
            "logical_parameters": len(covered),
            "bytes_checked": sum(item["bytes_checked"] for item in results.values()),
            "method": "worker_local",
        }

    def verify_dump(self, checkpoint: Path) -> dict:
        """Independent file-dump reference for integration tests and audits."""
        with tempfile.TemporaryDirectory(prefix="verify-", dir=self.audit_directory) as directory:
            self._rpc("dump", dump=directory)
            return verify_qwen3_dense(
                checkpoint, Path(directory), tp_size=self.tp_size, pp_size=self.pp_size, layouts=self.layouts
            )

    def flush(self) -> None:
        from .sglang_plugin import WorkerBusyError
        from .store import canonical

        # A generated response can reach the tokenizer before the scheduler
        # retires its last batch. Only a complete, identity-checked BUSY reply
        # is retryable; stale identities, partial results and real errors fail.
        observation = {"record_type": "flush_drain", "attempts": 0, "busy_attempts": 0, "passed": False}
        started = time.monotonic()
        observation["start_unix_ns"] = time.time_ns()
        try:
            with time_budget(operation_remaining(120, "cache flush drain")):
                while True:
                    check_deadline("cache flush drain")
                    observation["attempts"] += 1
                    try:
                        self._rpc("flush")
                        observation["passed"] = True
                        return
                    except WorkerBusyError:
                        observation["busy_attempts"] += 1
                        delay = min(0.01 * observation["busy_attempts"], 0.1)
                        time.sleep(min(delay, remaining(0.1, "cache flush drain")))
        finally:
            observation.update(seconds=time.monotonic() - started, end_unix_ns=time.time_ns())
            try:
                save_diagnostic(self.audit_directory / f"flush-drain-{uuid.uuid4().hex}.json", canonical(observation))
            except OSError:
                logger.warning("Cache drain diagnostic output failed", exc_info=True)

    def resume(self) -> None:
        self.identity()
        self.engine.tokenizer_manager._update_weight_version_if_provided(self._loaded_version)
        # Exercise lazy kernels before admitting external requests. The engine
        # is private and the public gate remains closed throughout this probe.
        timeout = operation_remaining(120, "generation probe")
        try:
            self.engine.loop.run_until_complete(
                asyncio.wait_for(
                    self.engine.async_generate(input_ids=[0], sampling_params={"temperature": 0, "max_new_tokens": 1}),
                    timeout,
                )
            )
        except TimeoutError:
            self._rpc_failed = True
            raise TimeoutError(
                "Generation probe timed out; restart the engine and restore the retained checkpoint"
            ) from None
        self.flush()

    def open(self) -> None:
        if self._rpc_failed:
            raise RuntimeError("Restart the timed-out engine and recover a retained checkpoint before activation")
        with self._gate:
            self._serving = True

    def generate(self, **kwargs: object) -> object:
        with self._gate:
            if not self._serving:
                raise RuntimeError("Weight installation is not active")
            return self.engine.generate(**kwargs)


class SGLangHTTPGroup(EmbeddedSGLangGroup):
    """Coordinate a node-local HTTP engine through the same worker protocol."""

    def __init__(
        self,
        url: str,
        model_path: str | Path,
        *,
        tp_size: int,
        audit_directory: str | Path,
        api_key: str | None = None,
        pp_size: int = 1,
    ):
        self.url = url.rstrip("/")
        self.owner = uuid.uuid4().hex
        self.installation_identity = None
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(
            None, model_path=model_path, tp_size=tp_size, pp_size=pp_size, audit_directory=audit_directory
        )

    def _control(self, action: str, **payload):
        import requests
        from urllib3.util import Timeout

        budget = operation_remaining(130, f"engine control {action}")
        from .metrics import SyncMetrics

        metrics = getattr(self, "control_metrics", None)
        if metrics is None:
            metrics = self.control_metrics = SyncMetrics("engine-control:" + self.owner)
        response = None
        try:
            response = requests.post(
                self.url + "/relax_weight_version",
                json={
                    "action": action,
                    "owner": self.owner,
                    "identity": self.installation_identity,
                    **payload,
                    "timeout": min(float(payload.get("timeout", budget)), budget),
                },
                headers=self.headers,
                timeout=Timeout(total=budget, connect=min(5, budget)),
            )
            request_body = getattr(getattr(response, "request", None), "body", None)
            reply_body = getattr(response, "content", None)
            if isinstance(request_body, str):
                request_body = request_body.encode()
            if isinstance(request_body, bytes) and isinstance(reply_body, bytes):
                metrics.transferred("C", len(request_body), link=f"http:{action}:request", scope="http_body")
                metrics.transferred("C", len(reply_body), link=f"http:{action}:response", scope="http_body")
            else:
                metrics.unmeasured("C", "HTTP client did not expose serialized request/response bodies")
        except requests.RequestException:
            metrics.unmeasured("C", "Failed HTTP operation may have sent a partial request")
            raise
        finally:
            try:
                metrics.save(self.audit_directory / f"control-metrics-{metrics.operation_id}.json")
            except OSError:
                logger.warning("Engine control diagnostic output failed", exc_info=True)
        if not response.ok:
            raise RuntimeError(f"Weight-version endpoint: {response.status_code}: {response.text}")
        return response.json()

    def _send_rpc(self, payload: str) -> None:
        self._control("rpc", payload=payload)

    def identity(self) -> str:
        workers = self._rpc("identify")
        return json.dumps({rank: item["instance"] for rank, item in workers.items()}, sort_keys=True)

    def close(self) -> None:
        if self.installation_identity is None:
            self.claim_install("session-" + self.owner, "bootstrap", 0)
        self._serving = False
        self._control("close")

    def claim_install(self, installation_id: str, root: str, epoch: int) -> int:
        workers = self._rpc("identify")
        gate = self._control("status")["identity"]
        epoch = max(epoch, max(item["epoch"] for item in workers.values()) + 1, (gate or {}).get("epoch", -1) + 1)
        self.installation_identity = {
            "installation_id": installation_id,
            "root": root,
            "epoch": epoch,
            "workers": {rank: item["instance"] for rank, item in workers.items()},
        }
        self._workers = workers
        self._control("claim")
        return epoch

    def begin_install(self, installation_id: str, root: str, epoch: int) -> int:
        self._workers = self._rpc("begin", installation_id=installation_id, root=root, epoch=epoch)
        return epoch

    def commit_install(self) -> None:
        self._control("commit")

    def resume(self) -> None:
        self.identity()
        self._control("probe", version=self._loaded_version)
        self.flush()

    def open(self) -> None:
        self._control("open")
        self._serving = True

    def generate(self, **kwargs):
        import requests

        response = requests.post(self.url + "/generate", json=kwargs, headers=self.headers, timeout=(5, 120))
        response.raise_for_status()
        return response.json()

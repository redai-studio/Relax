# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stream a torch_dist checkpoint to HF on CPU without constructing a model.

DCP global offsets recover TP/PP/EP shards; mappings consume only model
tensors. PLE tables and stacked vision layers use range-sized destination
buffers. Large uncompressed DCP chunks use bounded range reads; smaller chunks
use the DCP loader and reserve its deserialization buffers. The mapping module selects supported layouts from
the HF configuration and rejects unsupported architectures or tensors.

Input batches default to 2 GiB and use four concurrent file readers. The memory
budget can split batches further; an indivisible conversion can exceed the batch
target but must still fit the budget. --cpu-threads controls tensor operations,
independently of --read-workers. One batch is prefetched when the memory budget
allows it. --no-prefetch disables overlap. Missing frozen tensors stream through
a bounded copy buffer. Load and write work can overlap; load_wait measures the
main thread's wait. Write time is included in append/frozen/finalize, and
preflight is excluded from seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.utils.logging_utils import get_logger
from scripts.tools.convert_checkpoint_mapping import Conversion, TensorSlice, build_conversions, model_key
from scripts.tools.convert_checkpoint_reader import READ_BLOCK_BYTES, prepare_range_reads, read_tensor_range


logger = get_logger(__name__)


def _needs_owned_copy(tensor: Any) -> bool:
    return (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
    )


def _dtype_name(dtype: Any) -> str:
    return {
        "torch.float64": "F64",
        "torch.float32": "F32",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.int64": "I64",
        "torch.int32": "I32",
        "torch.int16": "I16",
        "torch.int8": "I8",
        "torch.uint8": "U8",
        "torch.bool": "BOOL",
    }[str(dtype)]


def resolve_checkpoint(path: Path) -> Path:
    if (path / ".metadata").is_file():
        return path.resolve()
    tracker = path / "latest_checkpointed_iteration.txt"
    tag = tracker.read_text().strip()
    if tag == "release":
        path = path / tag
    elif tag.isdecimal():
        path = path / f"iter_{int(tag):07d}"
    else:
        raise ValueError(f"Invalid checkpoint tracker: {tracker}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(path / ".metadata")
    return path.resolve()


def _read_hf_header(path: Path) -> tuple[int, dict[str, Any]]:
    from safetensors import safe_open

    # The official parser checks dtypes, offsets, holes and payload lengths
    # without materializing tensors. Raw copies must retain these checks.
    with safe_open(path, framework="pt", device="cpu"):
        pass
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size < 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        header = json.loads(handle.read(size))
    return size + 8, header


def read_hf_schema(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    index = path / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    filenames = sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
    schema: dict[str, dict[str, Any]] = {}
    actual_map: dict[str, str] = {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ValueError(f"Expected a flat HF shard filename: {filename}")
        _, header = _read_hf_header(path / filename)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in schema:
                raise ValueError(f"Duplicate HF tensor: {name}")
            schema[name] = {"shape": entry["shape"], "dtype": entry["dtype"]}
            actual_map[name] = filename
    if weight_map is not None and weight_map != actual_map:
        raise ValueError("HF index does not match safetensors headers")
    if not schema:
        raise ValueError("Empty HF model")
    return actual_map, schema


def check_chunks(key: str, metadata: Any) -> None:
    """Reject holes and overlaps before DCP can leave uninitialized values."""
    shape = tuple(metadata.size)
    rectangles = []
    volume = 0
    for chunk in metadata.chunks:
        offset, size = tuple(chunk.offsets), tuple(chunk.sizes)
        if (
            len(offset) != len(shape)
            or len(size) != len(shape)
            or any(a < 0 or b <= 0 or a + b > n for a, b, n in zip(offset, size, shape))
        ):
            raise ValueError(f"Invalid DCP chunk bounds: {key}")
        volume += math.prod(size)
        rectangles.append((offset, tuple(a + b for a, b in zip(offset, size))))
    if volume != math.prod(shape):
        raise ValueError(f"Incomplete DCP tensor: {key}")
    if shape:
        axis = max(range(len(shape)), key=lambda i: len({start[i] for start, _ in rectangles}))
        active = []
        for start, end in sorted(rectangles, key=lambda item: item[0][axis]):
            active = [(a, b) for a, b in active if b[axis] > start[axis]]
            if any(
                all(a < d and c < b for a, b, c, d in zip(start, end, other_start, other_end))
                for other_start, other_end in active
            ):
                raise ValueError(f"Overlapping DCP chunks: {key}")
            active.append((start, end))


def make_reader(checkpoint: Path, read_workers: int, range_records: dict[Any, Any] | None = None) -> Any:
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    class ParallelReader(FileSystemReader):
        def read_file(self, plan: Any, planner: Any) -> Any:
            regular = []
            for request in plan.items:
                record = (range_records or {}).get(request.storage_index)
                if record is None:
                    regular.append(request)
                else:
                    info = self.storage_data[request.storage_index]
                    with (checkpoint / info.relative_path).open("rb") as stream:
                        target = planner.resolve_tensor(request).detach()
                        read_tensor_range(stream, record, request, target)
                        planner.commit_tensor(request, target)
            return super().read_data(replace(plan, items=regular), planner)

        def read_data(self, plan: Any, planner: Any) -> Any:
            files: dict[str, list[Any]] = {}
            for request in plan.items:
                info = self.storage_data[request.storage_index]
                files.setdefault(info.relative_path, []).append(request)
            for requests in files.values():
                requests.sort(key=lambda request: self.storage_data[request.storage_index].offset)
            if read_workers == 1 or len(files) <= 1:
                return self.read_file(
                    replace(plan, items=[item for requests in files.values() for item in requests]), planner
                )
            # Each worker owns its file handle. Validated chunks write disjoint
            # destination ranges; SlicePlanner has no mutable commit state.
            with ThreadPoolExecutor(max_workers=read_workers) as pool:
                futures = [
                    pool.submit(self.read_file, replace(plan, items=requests), planner) for requests in files.values()
                ]
                for future in futures:
                    future.result().wait()
            result = torch.futures.Future()
            result.set_result(None)
            return result

    return ParallelReader(checkpoint)


def plan_batches(
    jobs: list[Conversion],
    input_bytes: list[int],
    extra_bytes: list[int],
    batch_bytes: int,
    memory_budget: int,
    read_buffer_bytes: int | Callable[[list[int]], int],
) -> list[list[int]]:
    batches: list[list[int]] = []
    batch: list[int] = []
    keys: set[str] = set()
    size = extra = 0
    for index, job in enumerate(jobs):
        incoming = {item.key for item in job.inputs}
        read_bytes = read_buffer_bytes(batch + [index]) if callable(read_buffer_bytes) else read_buffer_bytes
        if batch and (
            keys & incoming
            or size + input_bytes[index] > batch_bytes
            or size + input_bytes[index] + max(extra, extra_bytes[index]) + read_bytes > memory_budget
        ):
            batches.append(batch)
            batch, keys, size, extra = [], set(), 0, 0
        batch.append(index)
        keys.update(incoming)
        size += input_bytes[index]
        extra = max(extra, extra_bytes[index])
    if batch:
        batches.append(batch)
    return batches


def _reader_workspace(
    jobs: list[Conversion],
    tensors: dict[str, Any],
    storage_data: dict[Any, Any],
    read_workers: int,
    range_records: dict[Any, Any] | None = None,
) -> Callable[[list[int]], int]:
    import torch

    chunk_bytes = {
        (key, tuple(chunk.offsets)): math.prod(chunk.sizes)
        * torch.empty((), dtype=value.properties.dtype).element_size()
        for key, value in tensors.items()
        for chunk in value.chunks
    }
    chunks = dict.fromkeys(tensors, 0)
    archives = dict.fromkeys(tensors, 0)
    ranged: set[str] = set()
    files: dict[str, set[str]] = {key: set() for key in tensors}
    for index, info in storage_data.items():
        if index.fqn in tensors:
            if index in (range_records or {}):
                ranged.add(index.fqn)
            else:
                chunks[index.fqn] = max(chunks[index.fqn], chunk_bytes[index.fqn, tuple(index.offset)])
                archives[index.fqn] = max(archives[index.fqn], info.length)
            files[index.fqn].add(info.relative_path)
    job_chunks = [max(chunks[item.key] for item in job.inputs) for job in jobs]
    job_archives = [max(archives[item.key] for item in job.inputs) for job in jobs]
    job_files = [set().union(*(files[item.key] for item in job.inputs)) for job in jobs]
    job_ranges = [any(item.key in ranged for item in job.inputs) for job in jobs]

    def workspace(batch: list[int]) -> int:
        readers = min(read_workers, len(set().union(*(job_files[i] for i in batch))))
        # A reader may retain the previous chunk while deserializing the next.
        # Include every chunk of each key, even for partial tensor slices, and
        # reserve only for files this batch can actually read concurrently.
        return readers * max(
            2 * max(job_chunks[i] for i in batch) + max(job_archives[i] for i in batch),
            READ_BLOCK_BYTES if any(job_ranges[i] for i in batch) else 0,
        )

    return workspace


def load_slices(reader: Any, items: tuple[TensorSlice, ...], metadata: Any) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
    from torch.distributed.checkpoint.planner import LoadPlan
    from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list

    if len({item.key for item in items}) != len(items):
        raise ValueError("A load batch must contain only one slice per checkpoint key")

    class SlicePlanner(dcp.DefaultLoadPlanner):
        def resolve_tensor(self, read_item: Any) -> Any:
            # dest_index.offset describes the global range; our plain tensor
            # holds only that range. dest_offsets are already relative to it.
            return self.transform_tensor(read_item, self.state_dict[read_item.dest_index.fqn])

        def create_local_plan(self) -> LoadPlan:
            requests = []
            for item in items:
                chunk = ChunkStorageMetadata(torch.Size(item.offset), torch.Size(item.shape))
                requests.extend(
                    create_read_items_for_chunk_list(item.key, metadata.state_dict_metadata[item.key], [chunk])
                )
            return LoadPlan(requests)

    state = {
        item.key: torch.empty(item.shape, dtype=metadata.state_dict_metadata[item.key].properties.dtype, device="cpu")
        for item in items
    }
    # Explicit no_dist prevents an inherited environment from initializing Gloo.
    # Disabling flattening preserves the literal persistent Megatron keys.
    dcp.load(
        state,
        storage_reader=reader,
        planner=SlicePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )
    return state


class ShardWriter:
    def __init__(self, path: Path, shard_bytes: int) -> None:
        self.path = path
        self.shard_bytes = shard_bytes
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.files: list[str] = []
        self.seconds = 0.0
        self.copy_read_seconds = 0.0

    def add(self, name: str, tensor: Any) -> None:
        import torch

        if name in self.weight_map or name in self.pending:
            raise ValueError(f"Duplicate output tensor: {name}")
        if tensor.device.type != "cpu":
            raise ValueError(f"Expected CPU tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending_bytes + size > self.shard_bytes:
            self.flush()
        # Own views so a small projection cannot pin a huge input allocation.
        if _needs_owned_copy(tensor):
            tensor = tensor.clone(memory_format=torch.contiguous_format)
        self.pending[name] = tensor
        self.pending_bytes += size
        self.total_bytes += size
        if self.pending_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        from safetensors.torch import save_file

        if not self.pending:
            return
        filename = f"shard-{len(self.files) + 1:05d}.safetensors"
        started = time.monotonic()
        save_file(self.pending, self.path / filename, metadata={"format": "pt"})
        self.seconds += time.monotonic() - started
        self.files.append(filename)
        self.weight_map.update(dict.fromkeys(self.pending, filename))
        self.pending.clear()
        self.pending_bytes = 0

    def finish(self) -> None:
        self.flush()
        renames = {}
        for index, old in enumerate(self.files, 1):
            new = f"model-{index:05d}-of-{len(self.files):05d}.safetensors"
            (self.path / old).rename(self.path / new)
            renames[old] = new
        self.weight_map = {key: renames[value] for key, value in self.weight_map.items()}
        (self.path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map}, indent=2) + "\n"
        )

    def copy_frozen(
        self, model: Path, files: dict[str, list[str]], schema: dict[str, dict[str, Any]], buffer_bytes: int
    ) -> None:
        self.flush()
        buffer = bytearray(buffer_bytes)
        ranges: list[tuple[str, Path, int, int]] = []
        size = 0
        seen = set(self.weight_map)
        for filename, names in files.items():
            start, header = _read_hf_header(model / filename)
            for name in sorted(names, key=lambda key: header[key]["data_offsets"][0]):
                entry = header[name]
                if {key: entry[key] for key in ("shape", "dtype")} != schema[name] or name in seen:
                    raise ValueError(f"Frozen tensor schema changed or key duplicated: {name}")
                seen.add(name)
                begin, end = entry["data_offsets"]
                if ranges and size + end - begin > self.shard_bytes:
                    self._copy_ranges(ranges, schema, buffer)
                    ranges, size = [], 0
                ranges.append((name, model / filename, start + begin, end - begin))
                size += end - begin
                if size >= self.shard_bytes:
                    self._copy_ranges(ranges, schema, buffer)
                    ranges, size = [], 0
        if ranges:
            self._copy_ranges(ranges, schema, buffer)

    def _copy_ranges(
        self, ranges: list[tuple[str, Path, int, int]], schema: dict[str, dict[str, Any]], buffer: bytearray
    ) -> None:
        header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
        size = 0
        for name, _, _, length in ranges:
            header[name] = {**schema[name], "data_offsets": [size, size + length]}
            size += length
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        filename = f"shard-{len(self.files) + 1:05d}.safetensors"
        view = memoryview(buffer)
        with (self.path / filename).open("wb") as target, ExitStack() as sources:
            started = time.monotonic()
            target.write(struct.pack("<Q", len(raw)))
            target.write(raw)
            self.seconds += time.monotonic() - started
            previous = source = None
            for name, path, offset, remaining in ranges:
                if path != previous:
                    sources.close()
                    source = sources.enter_context(path.open("rb"))
                    previous = path
                source.seek(offset)
                while remaining:
                    started = time.monotonic()
                    count = source.readinto(view[: min(remaining, len(buffer))])
                    self.copy_read_seconds += time.monotonic() - started
                    if not count:
                        raise ValueError(f"Truncated frozen tensor payload: {name}")
                    started = time.monotonic()
                    target.write(view[:count])
                    self.seconds += time.monotonic() - started
                    remaining -= count
        self.files.append(filename)
        self.weight_map.update(dict.fromkeys((item[0] for item in ranges), filename))
        self.total_bytes += size


def _fallback_group(key: str) -> str | None:
    if key.startswith("model.visual."):
        return "vision"
    if re.search(r"\.ple\.ple_embedding\.", key):
        return "ple"
    if key.startswith("mtp."):
        return "mtp"
    return None


def _copy_assets(source: Path, output: Path) -> None:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
    }
    for item in source.iterdir():
        if item.is_file() and (item.name in names or item.suffix in (".model", ".tiktoken", ".jinja", ".py")):
            shutil.copyfile(item, output / item.name)
    if (source / "chat_templates").is_dir():
        shutil.copytree(source / "chat_templates", output / "chat_templates")


def export_checkpoint(
    checkpoint: Path,
    model: Path,
    output: Path,
    *,
    shard_bytes: int = 2 * 1024**3,
    memory_budget: int = 64 * 1024**3,
    batch_bytes: int = 2 * 1024**3,
    read_workers: int = 4,
    prefetch: bool = True,
    allow_base_fallback: tuple[str, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("CPU conversion must run without CUDA or a distributed process group")
    preflight_started = time.monotonic()
    if min(shard_bytes, memory_budget, batch_bytes, read_workers) <= 0:
        raise ValueError("Shard size, memory budget, batch size and read workers must be positive")
    checkpoint, model, output = resolve_checkpoint(checkpoint), model.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a fresh output directory")
    if output.is_relative_to(checkpoint) or output.is_relative_to(model):
        raise ValueError("Output must be outside the source checkpoint/model directories")
    config = json.loads((model / "config.json").read_text())
    if config.get("quantization_config") or config.get("text_config", {}).get("quantization_config"):
        raise ValueError("Quantized source models are not supported")
    fingerprint = hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
    reader = make_reader(checkpoint, read_workers)
    metadata = reader.read_metadata()
    # Metadata can be large; each partial load reuses this exact snapshot.
    reader.read_metadata = lambda: metadata
    tensors = {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata) and model_key(key) and "_extra_state" not in key
    }
    if not tensors:
        raise ValueError("Checkpoint contains no supported model tensors")
    for key, value in tensors.items():
        check_chunks(key, value)
    range_records = prepare_range_reads(checkpoint, metadata, tensors)
    if range_records:
        reader = make_reader(checkpoint, read_workers, range_records)
        reader.read_metadata = lambda: metadata
    weight_map, schema = read_hf_schema(model)
    jobs = build_conversions(tensors, config, schema)
    expected: dict[str, tuple[tuple[int, ...], Any]] = {}
    expected_schema = dict(schema)
    dtype_changes = {}
    input_bytes: list[int] = []
    extra_bytes: list[int] = []
    for job in jobs:
        for item in job.inputs:
            full_shape = tensors[item.key].size
            if (
                len(item.shape) != len(full_shape)
                or len(item.offset) != len(full_shape)
                or any(
                    start < 0 or size <= 0 or start + size > full
                    for start, size, full in zip(item.offset, item.shape, full_shape)
                )
            ):
                raise ValueError(f"Invalid checkpoint slice: {item}")
        state = {
            item.key: torch.empty(item.shape, dtype=tensors[item.key].properties.dtype, device="meta")
            for item in job.inputs
        }
        input_bytes.append(sum(t.numel() * t.element_size() for t in state.values()))
        converted = job.apply(state, config)
        # Meta tensors preserve aliases. Count each allocation once, including
        # when the converted tensors share storage with their DCP input.
        storages = {
            tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
            for tensor in (*state.values(), *converted.values())
        }
        owned_copy_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in converted.values() if _needs_owned_copy(tensor)
        )
        largest_output = max(tensor.numel() * tensor.element_size() for tensor in converted.values())
        extra_bytes.append(
            sum(storages.values())
            - input_bytes[-1]
            + owned_copy_bytes
            + shard_bytes
            + max(shard_bytes, largest_output)
        )
        for name, tensor in converted.items():
            if name in expected or name not in schema or list(tensor.shape) != schema[name]["shape"]:
                raise ValueError(f"Duplicate/unexpected HF key or shape mismatch: {name}: {tuple(tensor.shape)}")
            expected[name] = (tuple(tensor.shape), tensor.dtype)
            dtype = _dtype_name(tensor.dtype)
            if dtype != schema[name]["dtype"]:
                floating = {"F64", "F32", "F16", "BF16"}
                if dtype not in floating or schema[name]["dtype"] not in floating:
                    raise ValueError(f"Incompatible checkpoint dtype: {name}: {dtype} vs {schema[name]['dtype']}")
                dtype_changes[name] = {"base": schema[name]["dtype"], "export": dtype}
            expected_schema[name] = {"shape": list(tensor.shape), "dtype": dtype}
    missing = sorted(schema.keys() - expected.keys())
    forbidden = [key for key in missing if _fallback_group(key) not in allow_base_fallback]
    if forbidden:
        raise ValueError(
            f"Missing {len(forbidden)} HF tensors; base fallback is disabled for these keys: {forbidden[:5]}"
        )
    fallback_files: dict[str, list[str]] = {}
    for name in missing:
        fallback_files.setdefault(weight_map[name], []).append(name)
    copy_buffer_bytes = min(64 * 1024**2, max((math.prod(schema[key]["shape"]) * 8 for key in missing), default=0))
    read_workspace = _reader_workspace(jobs, tensors, metadata.storage_data, read_workers, range_records)
    batches = plan_batches(jobs, input_bytes, extra_bytes, batch_bytes, memory_budget, read_workspace)
    batch_read_buffers = [read_workspace(batch) for batch in batches]
    batch_inputs = [sum(input_bytes[i] for i in batch) for batch in batches]
    batch_buffers = [size + max(extra_bytes[i] for i in batch) for size, batch in zip(batch_inputs, batches)]
    overlap_buffers = [
        current + following + read_bytes
        for current, following, read_bytes in zip(batch_buffers, batch_inputs[1:], batch_read_buffers[1:])
    ]
    prefetch_after = [prefetch and size <= memory_budget for size in overlap_buffers]
    reservation = max(
        max(size + read_bytes for size, read_bytes in zip(batch_buffers, batch_read_buffers)),
        max((size for size, enabled in zip(overlap_buffers, prefetch_after) if enabled), default=0),
        copy_buffer_bytes,
    )
    report = {
        "checkpoint": str(checkpoint),
        "model": str(model),
        "output": str(output),
        "model_tensors": len(tensors),
        "conversion_groups": len(jobs),
        "hf_tensors": len(schema),
        "base_fallback_keys": missing,
        "max_input_bytes": max(input_bytes),
        "load_batches": len(batches),
        "prefetched_batches": sum(prefetch_after),
        "copy_buffer_bytes": copy_buffer_bytes,
        "batch_bytes": batch_bytes,
        "peak_batch_input_bytes": max(sum(input_bytes[i] for i in batch) for batch in batches),
        "read_workers": read_workers,
        "range_read_chunks": len(range_records),
        "read_buffer_bytes": max(batch_read_buffers),
        "shard_bytes": shard_bytes,
        "planned_buffer_bytes": reservation,
        "memory_budget_bytes": memory_budget,
        "metadata_sha256": fingerprint,
        "dtype_changes": dtype_changes,
    }
    if reservation > memory_budget:
        raise ValueError(
            f"Planned buffers need {reservation / 1024**3:.1f} GiB, above budget {memory_budget / 1024**3:.1f} GiB; increase --memory-budget-gb. This excludes Python and OS overhead."
        )
    logger.info("CPU export plan: %s", json.dumps(report, ensure_ascii=False))
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
    started = time.monotonic()
    timing = {"preflight": started - preflight_started, "load": 0.0, "load_wait": 0.0, "map": 0.0, "append": 0.0}

    def load_batch(batch: list[int]) -> tuple[dict[str, Any], float]:
        begin = time.monotonic()
        state = load_slices(reader, tuple(item for i in batch for item in jobs[i].inputs), metadata)
        return state, time.monotonic() - begin

    try:
        writer = ShardWriter(staging, shard_bytes)
        with ThreadPoolExecutor(max_workers=1) as loader:
            future = None
            for number, batch in enumerate(batches):
                phase_started = time.monotonic()
                state, elapsed = load_batch(batch) if future is None else future.result()
                future = None
                timing["load_wait"] += time.monotonic() - phase_started
                timing["load"] += elapsed
                if number < len(prefetch_after) and prefetch_after[number]:
                    future = loader.submit(load_batch, batches[number + 1])
                for i in batch:
                    job = jobs[i]
                    phase_started = time.monotonic()
                    selected = {item.key: state.pop(item.key) for item in job.inputs}
                    converted = job.apply(selected, config)
                    del selected
                    timing["map"] += time.monotonic() - phase_started
                    phase_started = time.monotonic()
                    for name in list(converted):
                        tensor = converted.pop(name)
                        if (tuple(tensor.shape), tensor.dtype) != expected[name]:
                            raise ValueError(f"Conversion changed after preflight: {name}")
                        writer.add(name, tensor)
                        del tensor
                    timing["append"] += time.monotonic() - phase_started
                    del converted
                del state
                logger.info(
                    "Converted %d/%d load batches (%d/%d tensor groups)",
                    number + 1,
                    len(batches),
                    batch[-1] + 1,
                    len(jobs),
                )
        phase_started = time.monotonic()
        writer.flush()
        timing["append"] += time.monotonic() - phase_started
        timing["trained_export"] = time.monotonic() - started
        logger.info("Trained export timing: %s", json.dumps(timing))
        phase_started = time.monotonic()
        if fallback_files:
            writer.copy_frozen(model, fallback_files, schema, copy_buffer_bytes)
        timing["append_frozen"] = time.monotonic() - phase_started
        timing["frozen_read"] = writer.copy_read_seconds
        phase_started = time.monotonic()
        _copy_assets(model, staging)
        writer.finish()
        _, actual = read_hf_schema(staging)
        if actual != expected_schema:
            raise ValueError("Final HF schema verification failed")
        if hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest() != fingerprint:
            raise RuntimeError("Checkpoint metadata changed during conversion")
        timing["finalize"] = time.monotonic() - phase_started
        timing["write"] = writer.seconds
        report.update(
            status="complete",
            seconds=time.monotonic() - started,
            timing_seconds=timing,
            total_bytes=writer.total_bytes,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            cuda_initialized=torch.cuda.is_initialized(),
            distributed_initialized=torch.distributed.is_initialized(),
        )
        (staging / "conversion-result.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    except BaseException:
        logger.error("Conversion failed; incomplete files retained at %s", staging)
        raise
    logger.info("CPU conversion complete: %s", output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=os.environ.get("CHECKPOINT_PATH"))
    parser.add_argument("--origin-hf-dir", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument(
        "--max-shard-size-gb", type=float, default=2, help="Target shard size in GiB; a single tensor may exceed it"
    )
    parser.add_argument(
        "--memory-budget-gb", type=float, default=64, help="Conservative tensor-buffer budget in GiB, not an RSS limit"
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--read-workers", type=int, default=4, help="Concurrent checkpoint file readers")
    parser.add_argument(
        "--no-prefetch", action="store_true", help="Disable memory-budgeted overlap of loading and writing"
    )
    parser.add_argument(
        "--load-batch-size-gb",
        type=float,
        default=2,
        help="Target input batch size in GiB, capped by the memory budget",
    )
    parser.add_argument(
        "--allow-base-fallback",
        choices=("vision", "ple", "mtp"),
        action="append",
        default=[],
        help="Explicitly copy missing tensors in this group from the exact training base",
    )
    parser.add_argument(
        "--dry-run",
        "--print-plan",
        action="store_true",
        help="Validate metadata/mappings/memory without reading weight values or writing output",
    )
    args = parser.parse_args()
    if any(value is None for value in (args.input_dir, args.origin_hf_dir, args.output_dir)):
        parser.error("Set CHECKPOINT_PATH, MODEL_PATH, OUTPUT_DIR or the equivalent CLI options")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
        parser.error("Run as one CPU process, without torchrun")
    import torch

    torch.set_num_threads(args.cpu_threads)
    export_checkpoint(
        args.input_dir,
        args.origin_hf_dir,
        args.output_dir,
        shard_bytes=int(args.max_shard_size_gb * 1024**3),
        memory_budget=int(args.memory_budget_gb * 1024**3),
        batch_bytes=int(args.load_batch_size_gb * 1024**3),
        read_workers=args.read_workers,
        prefetch=not args.no_prefetch,
        allow_base_fallback=tuple(args.allow_base_fallback),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

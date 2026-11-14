# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Streaming safetensors output and optional PyTorch snapshot input."""

import json
import math
import os
import shutil
import struct
import sys
import tempfile
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from .deadline import checkpoint as check_deadline
from .parallel_io import ordered_map
from .store import (
    DTYPES,
    MAX_MANIFEST_BYTES,
    VersionStore,
    canonical,
    digest,
    load_json,
    sync_directory,
    validate_manifest,
)
from .traffic import file_write


def validate_checkpoint(checkpoint: Path, manifest: dict) -> None:
    """Bind every checkpoint byte using at most eight positional reads."""
    validate_manifest(manifest, (checkpoint / "model.safetensors").stat().st_size)
    config_hash = digest(canonical(load_json((checkpoint / "config.json").read_bytes())))
    if manifest.get("schema_context", {}).get("model_config_sha256") != config_hash:
        raise ValueError("Checkpoint model configuration is not bound to the manifest")
    target = {k: manifest[k] for k in ("stream", "version", "root")}
    if json.loads((checkpoint / "weight-version.json").read_bytes()) != target:
        raise ValueError("Checkpoint belongs to another version")
    with (checkpoint / "model.safetensors").open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated checkpoint")
        size = struct.unpack("<Q", prefix)[0]
        if size > MAX_MANIFEST_BYTES:
            raise ValueError("Checkpoint header too large")
        header = load_json(source.read(size))
        if not isinstance(header, dict) or set(header) != {t["name"] for t in manifest["tensors"]}:
            raise ValueError("Checkpoint tensor coverage differs")
        offset, jobs = 0, []
        for tensor in manifest["tensors"]:
            expected = header[tensor["name"]]
            length = sum(c["size"] for c in tensor["chunks"])
            if expected != {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [offset, offset + length],
            }:
                raise ValueError("Checkpoint schema differs")
            for chunk in tensor["chunks"]:
                jobs.append((8 + size + offset, chunk["size"], chunk["output"]))
                offset += chunk["size"]
        expected_size = 8 + size + offset
        if os.fstat(source.fileno()).st_size != expected_size:
            raise ValueError("Checkpoint size differs from published version")

        def verify(job: tuple[int, int, str]) -> None:
            check_deadline("checkpoint verification")
            start, length, expected_hash = job
            value = os.pread(source.fileno(), length, start)
            if len(value) != length or digest(value) != expected_hash:
                raise ValueError("Checkpoint content differs from published version")
            check_deadline("checkpoint verification")

        for _ in ordered_map(verify, jobs, min(8, max(1, len(jobs)))):
            pass
        if os.fstat(source.fileno()).st_size != expected_size:
            raise ValueError("Checkpoint size changed during verification")


def snapshot_checkpoint(source: str | Path, destination: str | Path, *, max_weight_bytes: int) -> dict:
    """Read safetensors checkpoints without pickle or optional tensor
    libraries."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result, total = {}, 0
    for shard in sorted(source.glob("*.safetensors")):
        with shard.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors file")
            size = struct.unpack("<Q", prefix)[0]
            if size > MAX_MANIFEST_BYTES:
                raise ValueError("Safetensors header exceeds budget")
            header = load_json(handle.read(size))
            position = 0
            entries = [(name, info) for name, info in header.items() if name != "__metadata__"]
            entries.sort(key=lambda item: item[1]["data_offsets"][0])
            for name, info in entries:
                if name in result or info["dtype"] not in DTYPES:
                    raise ValueError("Duplicate tensor or unsupported dtype")
                shape = info["shape"]
                if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                    raise ValueError("Invalid shape")
                start, end = info["data_offsets"]
                expected = math.prod(shape) * DTYPES[info["dtype"]]
                total += expected
                if start != position or end - start != expected or total > max_weight_bytes:
                    raise ValueError("Invalid tensor range or weight budget exceeded")
                path = destination / (digest(name.encode()) + ".bin")
                with path.open("wb") as out:
                    remaining = expected
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("Truncated tensor")
                        out.write(block)
                        remaining -= len(block)
                result[name] = (info["dtype"], shape, path)
                position = end
            if handle.read(1):
                raise ValueError("Unreferenced safetensors bytes")
    if not result:
        raise ValueError("No safetensors weights found")
    return result


def snapshot_tensors(
    tensors: object,
    directory: str | Path,
    *,
    hashes: dict | None = None,
    hash_executor: Executor | None = None,
) -> dict:
    """Snapshot iterable (name, torch.Tensor) at a quiescent update boundary.

    Conversion may synchronize the device; this is an explicit checkpoint
    boundary, not the forward hot path. Caller must freeze optimizer mutation.

    Optional hashes describe the same CPU byte views written to the snapshot;
    they do not certify an independent disk readback. Publish/consumer storage
    checks still verify actual files. The supplied mapping changes only after
    every tensor has been written successfully; the returned table is unchanged.
    A supplied hash_executor is borrowed, never shut down here. This call drains
    its own submitted work even when writing fails, before releasing CPU views.
    """
    import torch

    if sys.byteorder != "little":
        raise ValueError("Only little-endian hosts are supported")
    names = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    written_hashes = {}
    workers = (
        (
            nullcontext(hash_executor)
            if hash_executor is not None
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-snapshot-hash")
        )
        if hashes is not None
        else nullcontext(None)
    )
    with workers as pool:
        for name, tensor in tensors:
            if name in result or tensor.dtype not in names:
                raise ValueError("Duplicate tensor or unsupported dtype")
            path = directory / (digest(name.encode()) + ".bin")
            # Conversion output is owned by this synchronous export until it returns.
            # A second host clone would double the largest tensor's CPU footprint.
            host = tensor.detach().contiguous().cpu()
            value = memoryview(host.reshape(-1).view(torch.uint8).numpy()).toreadonly()
            # One hashing worker overlaps this same view's write. Its result is
            # drained before advancing to another tensor, bounding retained memory.
            future = pool.submit(digest, value) if pool is not None else None
            try:
                with path.open("wb") as out:
                    if out.write(value) != value.nbytes:
                        raise OSError("Incomplete tensor snapshot write")
            except BaseException:
                # The borrowed executor may outlive this call. Drain only our
                # task and preserve the original write error if hashing fails too.
                if future is not None:
                    try:
                        future.result()
                    except BaseException:
                        pass
                raise
            if future is not None:
                written_hashes[name] = future.result()
            file_write(path, host.numel() * host.element_size())
            result[name] = (names[tensor.dtype], list(tensor.shape), path)
    if hashes is not None:
        hashes.update(written_hashes)
    return result


def write_checkpoint(
    store: VersionStore, stream: str, version: int, destination: str | Path, *, config_source: str | Path | None = None
) -> Path:
    """Create a verified HF checkpoint while retaining its source version.

    Destination must not exist. Tokenizer assets stay at the engine's startup
    model path; the checkpoint is used only by update_weights_from_disk.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Collection is allowed to remove unpinned versions. Hold a durable pin
    # from before materialization until every source chunk has been copied and
    # the staged checkpoint has been validated and atomically published.
    owner = f"checkpoint-{uuid.uuid4().hex}"
    store.pin(stream, version, owner)
    staging = None
    try:
        # iter_tensor verifies each chunk while it is copied. A preliminary
        # materialize would read and hash the complete cached model a second
        # time. Missing/corrupt caches are repaired on the first failed read.
        m = store.manifest(stream, version, prefer_full=True)
        config = load_json(Path(config_source).read_bytes()) if config_source else m["schema_context"]["model_config"]
        if m.get("schema_context", {}).get("model_config_sha256") != digest(canonical(config)):
            raise ValueError("Checkpoint config differs from published model schema")
        staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
        header, offset = {}, 0
        for tensor in m["tensors"]:
            size = sum(c["size"] for c in tensor["chunks"])
            header[tensor["name"]] = {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [offset, offset + size],
            }
            offset += size
        encoded = canonical(header)
        encoded += b" " * (-len(encoded) % 8)
        if shutil.disk_usage(staging).free < offset + len(encoded) + 8:
            raise OSError("Insufficient space for checkpoint")
        with (staging / "model.safetensors").open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for tensor in m["tensors"]:
                start = out.tell()
                repaired = False
                blocks = iter(store.iter_tensor(m, tensor))
                try:
                    while True:
                        check_deadline("checkpoint construction")
                        try:
                            block = next(blocks)
                        except StopIteration:
                            break
                        except (FileNotFoundError, ValueError):
                            if repaired:
                                raise
                            store.materialize(stream, version)
                            # A tensor may have yielded a prefix before failure.
                            # Rewrite that tensor from the now verified cache.
                            out.seek(start)
                            out.truncate()
                            blocks = iter(store.iter_tensor(m, tensor))
                            repaired = True
                            continue
                        out.write(block)
                finally:
                    # On writer/deadline failure, drain prefetched reads before
                    # releasing the durable pin in the outer cleanup.
                    close = getattr(blocks, "close", None)
                    if close is not None:
                        close()
            out.flush()
            os.fsync(out.fileno())
        (staging / "config.json").write_bytes(canonical(config))
        (staging / "weight-version.json").write_bytes(
            canonical({"stream": stream, "version": version, "root": m["root"]})
        )
        for name in ("config.json", "weight-version.json"):
            with (staging / name).open("r+b") as handle:
                os.fsync(handle.fileno())
        validate_checkpoint(staging, m)
        sync_directory(staging)
        os.rename(staging, destination)
        staging = None
        sync_directory(destination.parent)
        return destination
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        store.unpin(owner)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild a published weight version as a HF checkpoint")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disk-budget", type=int, required=True)
    parser.add_argument("--max-weight-bytes", type=int, required=True)
    options = parser.parse_args()
    store = VersionStore(options.store, disk_budget=options.disk_budget, max_weight_bytes=options.max_weight_bytes)
    checkpoint = write_checkpoint(store, options.stream, options.version, options.output)
    validate_checkpoint(checkpoint, store.manifest(options.stream, options.version))
    print(checkpoint)

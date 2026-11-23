# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build bidirectional U16 packages from SGLang checkpoints or a captured
Teacher pair."""

import argparse
import hashlib
import json
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from relax.backends.sglang.adp import bitpack
from relax.backends.sglang.adp_teacher_loader import (
    _hash_tensor,
    _runtime_view,
    _tp_context,
)
from relax.utils.replay.bundle import sha256_file


_FRAME_BYTES = 16 * 1024**2
_HEADER_BYTES = 4 * 1024**2


def encode_frame(left, right):
    """Encode exact little-endian 16-bit words; equal frames have no
    payload."""
    tile = bitpack.TILE
    delta = right.view("<u2").astype(np.int32) - left.view("<u2").astype(np.int32)
    delta = ((delta + 32768) & 65535) - 32768
    if not np.any(delta):
        return None
    codes = ((delta << 1) ^ (delta >> 15)).astype("<u2")
    # Pad the final tile to the fixed width expected by the GPU decoder.
    codes = np.pad(codes, (0, -len(codes) % tile)).reshape(-1, tile)
    escapes4 = np.count_nonzero(codes >= 15, axis=1)
    escapes8 = np.count_nonzero(codes >= 255, axis=1)
    sizes = np.stack((tile // 2 + 2 * escapes4, tile + 2 * escapes8, np.full(len(codes), tile * 2)), axis=1)
    modes = np.argmin(sizes, axis=1) + 1
    lengths = np.min(sizes, axis=1)
    zero = ~np.any(codes, axis=1)
    modes[zero] = lengths[zero] = 0
    offsets = np.cumsum(lengths, dtype=np.int64) - lengths
    table = ((offsets << 2) | modes).astype("<u4")
    body = np.zeros((int(lengths.sum()) + 3) & ~3, dtype=np.uint8)
    for mode, escape in ((1, 15), (2, 255), (3, 65535)):
        selected = np.flatnonzero(modes == mode)
        if not selected.size:
            continue
        values = codes[selected]
        if mode == 3:
            base = values.view(np.uint8).reshape(len(selected), tile * 2)
        else:
            literals = np.minimum(values, escape).astype(np.uint8)
            base = literals[:, ::2] | (literals[:, 1::2] << 4) if mode == 1 else literals
        body[offsets[selected, None] + np.arange(base.shape[1])] = base
        if mode != 3:
            rows, cols = np.nonzero(values >= escape)
            counts = (escapes4 if mode == 1 else escapes8)[selected]
            first = np.cumsum(counts) - counts
            positions = offsets[selected[rows]] + base.shape[1] + 2 * (np.arange(len(rows)) - first[rows])
            escaped = values[rows, cols]
            body[positions] = escaped & 255
            body[positions + 1] = escaped >> 8
    return table.tobytes() + body.tobytes()


def _frames(captures, tensors):
    for name, record in tensors.items():
        buffers = []
        for capture, key in zip(captures, ("a_sha256", "b_sha256"), strict=True):
            tensor = capture["params"][name].contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            if (
                list(tensor.shape) != record["shape"]
                or str(tensor.dtype) != record["dtype"]
                or hashlib.sha256(raw).hexdigest() != record[key]
                or len(raw) % 2
            ):
                raise ValueError(f"Capture differs from the published tensor record: {name}")
            buffers.append(raw)
        for start in range(0, len(raw), _FRAME_BYTES):
            yield name, start, *(buf[start : start + _FRAME_BYTES] for buf in buffers)


def _write_package(path, captures, header):
    packets, cursor = [], 0
    with path.open("xb") as stream, ThreadPoolExecutor(max_workers=6) as pool:
        stream.write(b"ADP1\0\0\0\0" + struct.pack("<Q", _HEADER_BYTES))
        stream.seek(16 + _HEADER_BYTES)
        pending = deque()

        def write_next():
            nonlocal cursor
            name, offset, length, future = pending.popleft()
            payload = future.result()
            if payload is not None:
                packets.append(
                    dict(
                        name=name, offset=offset, n=length, size=len(payload), payload_offset=cursor, mode="U16_PACKED"
                    )
                )
                stream.write(payload)
                cursor += len(payload)

        for name, offset, left, right in _frames(captures, header["tensors"]):
            pending.append((name, offset, len(left), pool.submit(encode_frame, left, right)))
            if len(pending) == 8:
                write_next()
        while pending:
            write_next()
        packed = dict(
            format="ADP1",
            version=1,
            policy="size",
            provenance=header["provenance"],
            tensors=header["tensors"],
            packets=packets,
        )
        metadata = json.dumps(packed, separators=(",", ":")).encode()
        if len(metadata) > _HEADER_BYTES:
            raise ValueError("Tensor and packet records exceed the reserved ADP1 header")
        stream.seek(16)
        stream.write(metadata.ljust(_HEADER_BYTES, b" "))


def _publish(output, manifest, pairs):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "bank"
        staged.mkdir()
        edges, integrity, shards = [], {}, {}
        tp_size = manifest["tp_size"]
        for source, target, captures, header in pairs:
            key = (source, target)
            if key not in shards:
                shards[key] = [None] * tp_size
            rank = header["provenance"]["tp_rank"]
            if not 0 <= rank < tp_size or shards[key][rank] is not None:
                raise ValueError("Duplicate or invalid TP rank for pair")
            number = list(shards).index(key)
            filename = "pair.adp" if len(manifest["teachers"]) == 2 else f"pair-{number}.adp"
            if tp_size > 1:
                filename = f"pair-{number}-rank-{rank}.adp"
            packet = staged / filename
            _write_package(packet, captures, header)
            integrity[filename] = {"bytes": packet.stat().st_size, "sha256": sha256_file(packet)}
            shards[key][rank] = filename
        for (source, target), packages in shards.items():
            if None in packages:
                raise ValueError("Missing TP rank package")
            edges.extend(
                (
                    dict(source=source, target=target, packages=packages),
                    dict(source=target, target=source, packages=packages, reverse=True),
                )
            )
        result = dict(
            manifest,
            packet_codec="u16-bitpack-v1",
            edges=edges,
            package_integrity=integrity,
            algorithm_sources={"bitpack": {"sha256": sha256_file(Path(bitpack.__file__))}},
        )
        (staged / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        staged.rename(output)


_CAPTURE_LOADER = "examples.on_policy_distillation.mopd.pack_teacher_pair.capture_teacher"


def capture_teacher(model, named_tensors):
    """Offline SGLang callback.

    Only the capture process selects the output file.
    """
    ((checkpoint, marker),) = named_tensors
    _, _, rank, size = _tp_context()
    if marker.numel() != 0:
        raise ValueError("Capture requires an empty control tensor")
    template = os.environ["RELAX_ADP_CAPTURE_FILE"]
    if size > 1 and "{rank}" not in template:
        raise ValueError("A sharded capture requires a distinct file per TP rank")
    view = _runtime_view(model)
    record = {
        "checkpoint_path": checkpoint,
        "metadata": view.metadata,
        "runtime_config": view.config_json,
        "tp_rank": rank,
        "tp_size": size,
        "params": {k: t.detach().to("cpu").contiguous() for k, t in view.params.items()},
        "auxiliary_hashes": {k: _hash_tensor(t) for k, t in view.auxiliary.items()},
    }
    path = Path(template.format(rank=rank))
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".incomplete")
    try:
        with temp.open("xb") as f:
            torch.save(record, f)
        temp.rename(path)
    finally:
        temp.unlink(missing_ok=True)


def _capture_checkpoint(_rank, checkpoint, destination, tp_size=1):
    from sglang import Engine

    # This process and its SGLang children are exclusive to one checkpoint.
    os.environ["RELAX_ADP_CAPTURE_FILE"] = str(destination)
    engine = Engine(
        model_path=str(checkpoint),
        tp_size=tp_size,
        dtype="bfloat16",
        speculative_algorithm=None,
        context_length=1024,
        max_total_tokens=2048,
        max_running_requests=2,
        chunked_prefill_size=512,
        attention_backend="triton",
        mem_fraction_static=0.75,
        disable_cuda_graph=True,
        skip_server_warmup=True,
        custom_weight_loader=[_CAPTURE_LOADER],
    )
    try:
        get_info = getattr(engine, "get_model_info", None) or engine.get_server_info
        if Path(get_info()["model_path"]).resolve() != checkpoint:
            raise ValueError(f"SGLang did not load {checkpoint}")
        success, message = engine.update_weights_from_tensor(
            named_tensors=[(str(checkpoint), torch.empty(0, dtype=torch.uint8))],
            load_format=_CAPTURE_LOADER,
        )
        if not success:
            raise RuntimeError(f"Capture failed for {checkpoint}: {message}")
    finally:
        engine.shutdown()


def _capture_signature(capture):
    """Normalize effective model metadata; keep every runtime layout/alias
    entry."""

    def canonical(fields):
        for name in ("_name_or_path", "_commit_hash", "transformers_version"):
            fields.pop(name, None)
        for name in ("torch_dtype", "dtype"):
            if fields.get(name) in (None, "bfloat16", "bf16"):
                fields.pop(name, None)
        return fields

    meta = dict(capture["metadata"])
    config = capture["runtime_config"]
    if config is not None:
        if hashlib.sha256(config.encode()).hexdigest() != meta["config_sha256"]:
            raise ValueError("Corrupt runtime config hash in capture")
        meta["config_sha256"] = json.loads(config, object_hook=canonical)
    return meta


def _check_capture_pair(first, second, path, rank):
    a, b = _capture_signature(first), _capture_signature(second)
    if a != b:
        fields = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
        raise ValueError(f"Incompatible SGLang runtime layout: {path}, rank {rank}, fields={fields}")
    left, right = first["auxiliary_hashes"], second["auxiliary_hashes"]
    if left != right:
        changed = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        raise ValueError(f"Changing auxiliary tensors: {path}, rank {rank}, tensors={changed}")


def _rank_pairs(checkpoints, files, rank, size):
    captures, hashes = {}, {}
    first = next(iter(checkpoints))
    for teacher, path in checkpoints.items():
        capture = torch.load(files[teacher][rank], map_location="cpu", mmap=True, weights_only=True)
        if (capture["checkpoint_path"], capture["tp_rank"], capture["tp_size"]) != (str(path), rank, size):
            raise ValueError(f"Incorrect capture provenance: {path}, rank {rank}")
        if set(capture["params"]) != set(capture["metadata"]["layout"]):
            raise ValueError(f"Incomplete parameter coverage: {path}, rank {rank}")
        if captures:
            _check_capture_pair(captures[first], capture, path, rank)
        captures[teacher] = capture
        hashes[teacher] = {name: _hash_tensor(tensor) for name, tensor in capture["params"].items()}

    for left, right in combinations(checkpoints, 2):
        print(f"Packing {left} <-> {right}, rank {rank}/{size}", flush=True)
        header = {
            "provenance": dict(runtime="sglang-dense-v1", tp_rank=rank, tp_size=size, source=left, target=right),
            "tensors": {
                name: dict(
                    shape=list(t.shape), dtype=str(t.dtype), a_sha256=hashes[left][name], b_sha256=hashes[right][name]
                )
                for name, t in captures[left]["params"].items()
            },
        }
        yield left, right, [captures[left], captures[right]], header


def build_rank_captures(checkpoints, captures, output):
    tp_size = len(next(iter(captures.values())))
    if set(captures) != set(checkpoints) or tp_size < 1 or any(len(v) != tp_size for v in captures.values()):
        raise ValueError("Every Teacher requires one capture per tensor-parallel rank")
    manifest = dict(
        version=1,
        runtime="sglang-dense-v1",
        tp_size=tp_size,
        initial_teacher=next(iter(checkpoints)),
        teachers={name: {"checkpoint_path": str(path)} for name, path in checkpoints.items()},
    )
    pairs = (pair for rank in range(tp_size) for pair in _rank_pairs(checkpoints, captures, rank, tp_size))
    _publish(output, manifest, pairs)


def build_checkpoints(paths, output, tp_size=1):
    checkpoints = {f"teacher{i}": path.resolve(strict=True) for i, path in enumerate(paths)}
    if tp_size < 1 or len(checkpoints) < 2 or len(set(checkpoints.values())) != len(checkpoints):
        raise ValueError("Provide positive TP size and at least two distinct checkpoint directories")
    for path in checkpoints.values():
        if not (path / "config.json").is_file():
            raise ValueError(f"Missing checkpoint config: {path}")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        files = {}
        for teacher, path in checkpoints.items():
            template = Path(tmp) / (teacher + "-{rank}.pt")
            print(f"Capturing {path} with TP{tp_size}", flush=True)
            args = (path, template) if tp_size == 1 else (path, template, tp_size)
            torch.multiprocessing.spawn(_capture_checkpoint, args=args, nprocs=1)
            files[teacher] = [Path(str(template).format(rank=rank)) for rank in range(tp_size)]
        build_rank_captures(checkpoints, files, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        required=True,
        nargs="+",
        type=Path,
        help="homogeneous checkpoint directories, initial Teacher first",
    )
    parser.add_argument("--output", required=True, type=Path, help="new bank for --checkpoints")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size for native captures")
    args = parser.parse_args(argv)
    build_checkpoints(args.checkpoints, args.output, args.tp_size)


if __name__ == "__main__":
    main()

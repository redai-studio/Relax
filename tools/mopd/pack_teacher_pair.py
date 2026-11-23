# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build bidirectional, rank-local U16 packages from Teacher checkpoints."""

import argparse
import json
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from relax.backends.sglang.adts import bitpack
from relax.backends.sglang.adts_teacher_loader import _get_target_tensors, _hash_tensor
from relax.utils.replay.bundle import sha256_file


_FRAME_BYTES = 16 * 1024**2
_HEADER_BYTES = 4 * 1024**2


def _iter_weight_slices(captures, tensors):
    left, right = captures
    for name, record in tensors.items():
        if record["a_sha256"] == record["b_sha256"]:
            continue
        a = left["params"][name].reshape(-1).view(torch.uint8).numpy()
        b = right["params"][name].reshape(-1).view(torch.uint8).numpy()
        for start in range(0, len(a), _FRAME_BYTES):
            end = start + _FRAME_BYTES
            yield name, start, a[start:end], b[start:end]


def _write_package(path, captures, header):
    packets = []
    with path.open("xb") as stream, ThreadPoolExecutor(max_workers=6) as pool:
        stream.write(bitpack.MAGIC + struct.pack("<Q", _HEADER_BYTES))
        stream.seek(16 + _HEADER_BYTES)

        # Encode frames in parallel but write them in packet order so the reader can stream them.
        pending = deque()

        def write_next():
            name, offset, length, future = pending.popleft()
            payload = future.result()
            if payload is not None:
                packets.append(dict(name=name, offset=offset, n=length, size=len(payload)))
                stream.write(payload)

        for name, offset, left, right in _iter_weight_slices(captures, header["tensors"]):
            pending.append((name, offset, len(left), pool.submit(bitpack.encode_frame, left, right)))
            if len(pending) == 8:
                write_next()
        while pending:
            write_next()

        metadata = json.dumps({**header, "packets": packets}, separators=(",", ":")).encode()
        if len(metadata) > _HEADER_BYTES:
            raise ValueError("Tensor and packet records exceed the reserved ADTS header")
        stream.seek(16)
        stream.write(metadata.ljust(_HEADER_BYTES, b" "))


_CAPTURE_LOADER = "tools.mopd.pack_teacher_pair.capture_teacher"


def capture_teacher(model, named_tensors):
    """Capture the actual SGLang weight layout in the offline packing
    process."""
    from sglang.srt.distributed import get_tp_group

    ((prefix, _),) = named_tensors
    rank = get_tp_group().rank_in_group
    weights = _get_target_tensors(model)
    record = {
        "params": {name: t.detach().cpu().contiguous() for name, t in weights.items()},
        "auxiliary_hashes": {
            name: _hash_tensor(t) for name, t in model.named_buffers() if f"buffer:{name}" not in weights
        },
    }
    with Path(f"{prefix}-{rank}.pt").open("xb") as stream:
        torch.save(record, stream)


def _capture_checkpoint(_rank, checkpoint, destination, tp_size):
    from sglang import Engine

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
        success, message = engine.update_weights_from_tensor(
            named_tensors=[(str(destination), torch.empty(0, dtype=torch.uint8))],
            load_format=_CAPTURE_LOADER,
        )
        if not success:
            raise RuntimeError(f"Capture failed for {checkpoint}: {message}")
    finally:
        engine.shutdown()


def _checkpoint_pairs(checkpoints, scratch, rank):
    captures, hashes = {}, {}
    base_layout = base_buffers = None
    for teacher, path in checkpoints.items():
        destination = scratch / f"{teacher}-{rank}.pt"
        capture = torch.load(destination, map_location="cpu", mmap=True, weights_only=True)
        layout = {name: (t.shape, t.dtype) for name, t in capture["params"].items()}
        if not captures:
            base_layout, base_buffers = layout, capture["auxiliary_hashes"]
        elif layout != base_layout or capture["auxiliary_hashes"] != base_buffers:
            raise ValueError(f"Teacher layout or fixed buffers differ: {path}")
        captures[teacher] = capture
        hashes[teacher] = {name: _hash_tensor(tensor) for name, tensor in capture["params"].items()}

    for left, right in combinations(checkpoints, 2):
        print(f"Packing {checkpoints[left].name} <-> {checkpoints[right].name}, TP rank {rank}", flush=True)
        header = {
            "provenance": dict(source=left, target=right),
            "tensors": {
                name: dict(
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    a_sha256=hashes[left][name],
                    b_sha256=hashes[right][name],
                )
                for name, tensor in captures[left]["params"].items()
            },
        }
        yield [captures[left], captures[right]], header


def build_teacher_delta_packages(paths, output, tp_size=1):
    checkpoints = {f"teacher{i}": path.resolve(strict=True) for i, path in enumerate(paths)}
    if len(checkpoints) < 2 or len(set(checkpoints.values())) != len(checkpoints):
        raise ValueError("Provide at least two distinct checkpoint directories in initial Teacher order")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output.parent) as tmp:
        scratch = Path(tmp)
        staged = scratch / "bank"
        staged.mkdir()
        integrity = {}
        for teacher, path in checkpoints.items():
            print(f"Capturing {path} with TP{tp_size}", flush=True)
            torch.multiprocessing.spawn(_capture_checkpoint, args=(path, scratch / teacher, tp_size), nprocs=1)

        for rank in range(tp_size):
            for index, (captures, header) in enumerate(_checkpoint_pairs(checkpoints, scratch, rank)):
                packet = staged / f"pair-{index}-rank-{rank}.adts"
                _write_package(packet, captures, header)
                integrity[packet.name] = {"rank": rank, "bytes": packet.stat().st_size, "sha256": sha256_file(packet)}

        manifest = dict(
            tp_size=tp_size,
            initial_teacher=next(iter(checkpoints)),
            teachers={teacher: {"checkpoint_path": str(path)} for teacher, path in checkpoints.items()},
            package_integrity=integrity,
        )
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", required=True, nargs="+", type=Path, help="Teacher checkpoints, initial Teacher first"
    )
    parser.add_argument("--output", required=True, type=Path, help="New package directory")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size of each Teacher engine")
    args = parser.parse_args(argv)
    build_teacher_delta_packages(args.checkpoints, args.output, args.tp_size)


if __name__ == "__main__":
    main()

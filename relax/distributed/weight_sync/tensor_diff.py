# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare safetensors checkpoints by stored tensor bits, one tensor at a
time."""

import argparse
import json
import mmap
import struct
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_CHUNK_BYTES = 1024 * 1024


def _checkpoint(path: Path) -> dict[str, tuple[Path, int, int, str, list[int]]]:
    if path.is_file():
        files = [path]
        weight_map = None
    elif path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Empty or invalid weight_map")
            for name, filename in weight_map.items():
                if not isinstance(name, str) or not isinstance(filename, str) or Path(filename).name != filename:
                    raise ValueError("Invalid index entry")
            files = [path / name for name in sorted(set(weight_map.values()))]
            if {file.name for file in path.glob("*.safetensors")} != {file.name for file in files}:
                raise ValueError("Index omits a safetensors shard")
        else:
            weight_map = None
            files = sorted(path.glob("*.safetensors"))
    else:
        raise ValueError(f"Checkpoint does not exist: {path}")
    if not files:
        raise ValueError("No safetensors files")
    tensors: dict[str, tuple[Path, int, int, str, list[int]]] = {}
    for file in files:
        with file.open("rb") as stream:
            size = file.stat().st_size
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors prefix: {file}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > 100 * 1024 * 1024 or 8 + header_size > size:
                raise ValueError(f"Invalid safetensors header size: {file}")
            header = json.loads(stream.read(header_size))
        ranges = []
        file_tensor_count = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            if weight_map is not None and weight_map.get(name) != file.name:
                raise ValueError(f"Index and shard disagree: {name}")
            dtype, shape = info["dtype"], info["shape"]
            start, end = info["data_offsets"]
            if (
                dtype not in _DTYPE_BYTES
                or not isinstance(shape, list)
                or any(not isinstance(dim, int) or dim < 0 for dim in shape)
            ):
                raise ValueError(f"Invalid dtype or shape: {name}")
            count = 1
            for dim in shape:
                count *= dim
            if not (0 <= start <= end <= size - 8 - header_size) or end - start != count * _DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid tensor extent: {name}")
            ranges.append((start, end))
            tensors[name] = file, 8 + header_size + start, end - start, dtype, shape
            file_tensor_count += 1
        if not isinstance(header, dict) or not file_tensor_count:
            raise ValueError(f"Empty safetensors header: {file}")
        ordered = sorted(ranges)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping tensor extents: {file}")
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError("Index names missing from shards")
    return tensors


def _coordinate(index: int, shape: list[int]) -> list[int]:
    coordinate = []
    for dim in reversed(shape):
        coordinate.append(index % dim)
        index //= dim
    return list(reversed(coordinate))


def compare(left: Path, right: Path) -> dict[str, Any]:
    import numpy as np

    # NumPy is already required by the codec. Work only on bounded raw-byte
    # chunks; no floating-point conversion or whole-model materialization.
    popcount = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
    left_tensors, right_tensors = _checkpoint(left), _checkpoint(right)

    def inodes(tensors: dict) -> set[tuple[int, int]]:
        return {
            (info.st_dev, info.st_ino) for path in {entry[0] for entry in tensors.values()} for info in [path.stat()]
        }

    if inodes(left_tensors) & inodes(right_tensors):
        raise ValueError("Reference and result share a file inode")
    names = set(left_tensors) & set(right_tensors)
    result: dict[str, Any] = {
        "left": str(left),
        "right": str(right),
        "left_tensors": len(left_tensors),
        "right_tensors": len(right_tensors),
        "missing": sorted(set(left_tensors) - set(right_tensors)),
        "extra": sorted(set(right_tensors) - set(left_tensors)),
        "compared_tensors": 0,
        "compared_elements": 0,
        "compared_bytes": 0,
        "mismatch_tensors": 0,
        "mismatch_elements": 0,
        "mismatch_bytes": 0,
        "mismatch_bits": 0,
        "metadata_mismatches": [],
        "first_difference": None,
    }
    for name in sorted(names):
        lf, lo, length, dtype, shape = left_tensors[name]
        rf, ro, rlength, rdtype, rshape = right_tensors[name]
        if (dtype, shape, length) != (rdtype, rshape, rlength):
            result["mismatch_tensors"] += 1
            result["metadata_mismatches"].append(
                {
                    "name": name,
                    "left": {"dtype": dtype, "shape": shape, "bytes": length},
                    "right": {"dtype": rdtype, "shape": rshape, "bytes": rlength},
                }
            )
            continue
        width = _DTYPE_BYTES[dtype]
        result["compared_tensors"] += 1
        result["compared_elements"] += length // width
        result["compared_bytes"] += length
        differs = False
        with ExitStack() as stack:
            left_file = stack.enter_context(lf.open("rb"))
            right_file = stack.enter_context(rf.open("rb"))
            lm = stack.enter_context(mmap.mmap(left_file.fileno(), 0, access=mmap.ACCESS_READ))
            rm = stack.enter_context(mmap.mmap(right_file.fileno(), 0, access=mmap.ACCESS_READ))
            for offset in range(0, length, _CHUNK_BYTES):
                chunk_length = min(_CHUNK_BYTES, length - offset)
                a = lm[lo + offset : lo + offset + chunk_length]
                b = rm[ro + offset : ro + offset + chunk_length]
                if a == b:
                    continue
                differs = True
                xor = np.bitwise_xor(np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8))
                result["mismatch_bytes"] += int(np.count_nonzero(xor))
                result["mismatch_bits"] += int(popcount[xor].sum(dtype=np.uint64))
                result["mismatch_elements"] += int(np.count_nonzero(np.any(xor.reshape(-1, width), axis=1)))
                if result["first_difference"] is None:
                    byte_offset = offset + int(np.argmax(xor != 0))
                    element = byte_offset // width
                    base = element * width
                    result["first_difference"] = {
                        "name": name,
                        "coordinate": _coordinate(element, shape),
                        "element_index": element,
                        "byte_offset": byte_offset,
                        "left_bits": lm[lo + base : lo + base + width].hex(),
                        "right_bits": rm[ro + base : ro + base + width].hex(),
                    }
        if differs:
            result["mismatch_tensors"] += 1
    result["equal"] = not any(
        (result["missing"], result["extra"], result["metadata_mismatches"], result["mismatch_bits"])
    )
    logger.info(
        "tensor diff: equal=%s compared=%s bytes=%s mismatched=%s bytes/%s bits",
        result["equal"],
        result["compared_tensors"],
        result["compared_bytes"],
        result["mismatch_bytes"],
        result["mismatch_bits"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.left, args.right)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = {"equal": False, "error": str(exc), "left": str(args.left), "right": str(args.right)}
        logger.error("tensor diff failed: %s", exc)
    args.result.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

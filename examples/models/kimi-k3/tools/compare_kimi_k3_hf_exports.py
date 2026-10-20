# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare every safetensors value in two exports using bounded CPU memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any


def _header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < length < 100_000_000:
            raise ValueError(f"Invalid safetensors header: {path}")
        return json.loads(handle.read(length)), length + 8


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def compare_shard(reference: str, candidate: str, filename: str) -> dict[str, Any]:
    left, right = Path(reference) / filename, Path(candidate) / filename
    a, a_start = _header(left)
    b, b_start = _header(right)
    a.pop("__metadata__", None)
    b.pop("__metadata__", None)
    if a.keys() != b.keys():
        raise ValueError(f"Key mismatch: {filename}")
    for key in a:
        if a[key]["dtype"] != b[key]["dtype"] or a[key]["shape"] != b[key]["shape"]:
            raise ValueError(f"Tensor schema mismatch: {key}")
    left_digest, right_digest = _digest(left), _digest(right)
    tensor_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in a.values())
    mismatches = []
    if left_digest != right_digest:
        # Header ordering/padding may differ without changing any tensor value.
        with left.open("rb") as x, right.open("rb") as y:
            for key in sorted(a):
                start, end = a[key]["data_offsets"]
                other_start, other_end = b[key]["data_offsets"]
                if end - start != other_end - other_start:
                    raise ValueError(f"Tensor byte length mismatch: {key}")
                x.seek(a_start + start)
                y.seek(b_start + other_start)
                remaining = end - start
                differs = False
                while remaining:
                    size = min(remaining, 8 * 1024 * 1024)
                    one, two = x.read(size), y.read(size)
                    if len(one) != size or len(two) != size:
                        raise ValueError(f"Truncated tensor: {key}")
                    differs |= one != two
                    remaining -= size
                if differs:
                    mismatches.append(key)
    return {
        "file": filename,
        "reference_sha256": left_digest,
        "candidate_sha256": right_digest,
        "tensor_count": len(a),
        "tensor_bytes": tensor_bytes,
        "mismatched_keys": mismatches,
        "exact_file_match": left_digest == right_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--after-job-id")
    parser.add_argument("--job-address")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    if args.after_job_id:
        if not args.job_address:
            parser.error("--after-job-id requires --job-address")
        from convert_kimi_k3_torch_dist_to_hf import _wait_for_job

        _wait_for_job(args.job_address, args.after_job_id)
    left, right = Path(args.reference_dir), Path(args.candidate_dir)
    indexes = [json.loads((d / "model.safetensors.index.json").read_text()) for d in (left, right)]
    if indexes[0]["weight_map"] != indexes[1]["weight_map"]:
        raise ValueError("HF weight indexes differ")
    if (left / "config.json").read_bytes() != (right / "config.json").read_bytes():
        raise ValueError("HF configs differ")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.ray_address)
    nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    if not nodes:
        raise RuntimeError("No cluster nodes available")
    files = sorted(set(indexes[0]["weight_map"].values()))
    started = time.monotonic()
    results = []
    task = ray.remote(num_cpus=1, max_retries=0)(compare_shard)
    try:
        # At most one file pair per node at a time, bounding memory and shared-FS pressure.
        for first in range(0, len(files), len(nodes)):
            futures = [
                task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[i], soft=False)).remote(
                    str(left), str(right), filename
                )
                for i, filename in enumerate(files[first : first + len(nodes)])
            ]
            results.extend(ray.get(futures))
            logger.info("EXPORT_COMPARE_PROGRESS shards=%d/%d", len(results), len(files))
        passed = all(not r["mismatched_keys"] for r in results)
        report = {
            "status": "passed" if passed else "failed",
            "reference": str(left),
            "candidate": str(right),
            "elapsed_seconds": time.monotonic() - started,
            "shards": len(results),
            "tensors": sum(r["tensor_count"] for r in results),
            "tensor_bytes": sum(r["tensor_bytes"] for r in results),
            "files": results,
        }
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            raise ValueError("Export tensor bytes differ; inspect comparison report")
        logger.info("EXPORT_COMPARE_PASSED tensors=%d report=%s", report["tensors"], args.report)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

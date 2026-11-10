# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ordinary FULL GPU loads to check previously captured TCP-delivered models.

Run only after the original audit training job has exited. These loads never
overwrite the tested models before their delivered-state dumps are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any


def check_version(actor: Any, receipt_path: str, output: str) -> dict:
    from relax.distributed.weight_sync.store import atomic_write
    from relax.distributed.weight_sync.tensor_diff import compare
    from task_delta_weight_sync.memory_audit import snapshot_engine

    receipt = json.loads(Path(receipt_path).read_text())
    if receipt["status"] != "CAPTURED_NOT_COMPARED" or len(receipt["captured"]) != 8:
        raise ValueError("Expected all eight delivered models and an independent HF export")
    captured = receipt["captured"]
    if sorted(c["engine_rank"] for c in captured) != list(range(8)):
        raise ValueError("Delivered engine ranks must be exactly 0..7")
    for key in ("actor_id", "directory"):
        values = [c[key] if key != "directory" else str(Path(c[key]).resolve()) for c in captured]
        if not all(values) or len(set(values)) != 8:
            raise ValueError(f"Delivered {key} identities must be unique")
    started = monotonic()
    export = Path(receipt["independent_hf"])
    if {str(p.relative_to(export)) for p in export.rglob("*") if p.is_file()} != set(receipt["hf_files"]):
        raise ValueError("Independent HF file inventory changed")
    for name, expected in receipt["hf_files"].items():
        path = export / name
        if not path.resolve().is_relative_to(export.resolve()):
            raise ValueError("HF manifest path escapes the exported checkpoint")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Independent HF file changed: {name}")
    engine = actor.engine
    engine.pause_generation(timeout=30)
    result = engine._make_request(
        "update_weights_from_disk",
        dict(
            model_path=str(export),
            load_format="auto",
            weight_version=receipt["update"],
            keep_pause=True,
            flush_cache=True,
            recapture_cuda_graph=False,
        ),
        timeout=300,
    )
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(f"Ordinary reference FULL load failed: {result}")
    directory = Path(output) / f"version-{receipt['version']:06d}"
    reference = snapshot_engine(engine, str(directory / "reference"), receipt["update"])
    coverage = json.loads((Path(reference["directory"]) / "coverage.json").read_text())
    pairs = []
    for delivered in captured:
        check = compare(Path(reference["directory"]), Path(delivered["directory"]))
        delivered_coverage = json.loads((Path(delivered["directory"]) / "coverage.json").read_text())
        pair = dict(
            engine_rank=delivered["engine_rank"],
            delivered=delivered["directory"],
            diff=check,
            coverage_equal=delivered_coverage == coverage,
        )
        pairs.append(pair)
        atomic_write(directory / f"engine-{delivered['engine_rank']}.json", json.dumps(pair, indent=2).encode())
    report = dict(
        version=receipt["version"],
        update=receipt["update"],
        stream=receipt["stream"],
        schema=receipt["schema"],
        receipt=receipt_path,
        reference=reference,
        pairs=pairs,
        equal=all(pair["diff"]["equal"] and pair["coverage_equal"] for pair in pairs),
        audit_s=monotonic() - started,
    )
    atomic_write(directory / "result.json", json.dumps(report, indent=2).encode())
    if not report["equal"]:
        raise RuntimeError(f"GPU raw-bit comparison failed at version {receipt['version']}")
    # Preserve failed comparisons before any resume/probe; a mismatched model
    # remains paused. Successful generation is separate from raw-bit equality.
    engine.continue_generation(timeout=30)
    report["probe"] = engine.disk_weight_generation_probe(60)
    report["audit_s"] = monotonic() - started
    atomic_write(directory / "result.json", json.dumps(report, indent=2).encode())
    return report


def main() -> int:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.distributed.weight_sync.store import atomic_write
    from task_delta_weight_sync.gpu_replay import CheckedEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    args = parser.parse_args()
    if not args.audit_root.is_absolute() or not args.output.is_absolute():
        raise ValueError("Use absolute audit and output directories")
    args.audit_root, args.output = args.audit_root.resolve(), args.output.resolve()
    if args.output.is_relative_to(args.audit_root) or args.audit_root.is_relative_to(args.output):
        raise ValueError("Keep oracle output separate from the frozen audit directory")
    receipts = sorted((args.audit_root / "receipts").glob("*.json"))
    records = [json.loads(path.read_text()) for path in receipts]
    if [r["version"] for r in records] != list(range(11)):
        raise ValueError("Expected initialization plus ten actual optimizer updates")
    if len({r["stream"] for r in records}) != 1 or len({r["schema"] for r in records}) != 1:
        raise ValueError("Audit receipts mix different streams or layouts")
    args.output.mkdir(parents=True, exist_ok=False)
    ray.init(address="auto")
    nodes = {n["NodeManagerAddress"]: n for n in ray.nodes() if n["Alive"]}
    node = nodes[args.node_ip]
    if node["Resources"].get("GPU", 0) < 8:
        raise ValueError("Eight reference engines require an eight-GPU node")
    engines, reports = [], []
    status = "FAILED"
    try:
        for rank in range(8):
            engines.append(
                CheckedEngine.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote(rank, records[0]["independent_hf"], str(args.output / "engines"))
            )
        ray.get([engine.pause.remote() for engine in engines], timeout=600)
        for offset in range(0, len(receipts), 8):
            reports.extend(
                ray.get(
                    [
                        engines[index].__ray_call__.remote(check_version, str(path), str(args.output))
                        for index, path in enumerate(receipts[offset : offset + 8])
                    ],
                    timeout=900,
                )
            )
        status = "VERIFIED"
        return 0
    finally:
        cleanup_errors = []
        for engine in engines:
            try:
                ray.get(engine.shutdown.remote(), timeout=30)
            except Exception as exc:
                cleanup_errors.append(f"shutdown: {type(exc).__name__}: {exc}")
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as kill_error:
                    cleanup_errors.append(f"kill: {type(kill_error).__name__}: {kill_error}")
        if cleanup_errors:
            status = "FAILED_CLEANUP"
        atomic_write(
            args.output / "result.json",
            json.dumps(dict(status=status, versions=reports, cleanup_errors=cleanup_errors), indent=2).encode(),
        )
        if cleanup_errors:
            raise RuntimeError(f"Reference engine cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    raise SystemExit(main())

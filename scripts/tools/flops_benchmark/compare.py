# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare complete optimizer shards with predeclared tensor-level
tolerances."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger
from scripts.tools.flops_benchmark.common import digest, file_digest, read_metrics, write_json


logger = get_logger(__name__)


def _parameter_key(entry: dict) -> str:
    return digest({key: entry[key] for key in ("optimizer", "chunk", "name", "shape", "numel", "dp_group_ranks")})


def validate_consumed_samples(manifests: dict[int, dict]) -> None:
    first = manifests[0]
    expected = Counter(digest(row) for row in first["expected_samples"])
    per_dp = {}
    for manifest in manifests.values():
        if manifest["expected_samples"] != first["expected_samples"] or manifest["dp_size"] != first["dp_size"]:
            raise ValueError("Ranks disagree on the frozen batch")
        chunks = manifest["consumed_samples"]
        if len(chunks) != manifest["model_chunks"]:
            raise ValueError("Missing model chunk sample consumption")
        local = Counter(digest(row) for row in chunks[0])
        if any(Counter(digest(row) for row in rows) != local for rows in chunks[1:]):
            raise ValueError("Virtual pipeline chunks consumed different samples")
        dp_rank = manifest["dp_rank"]
        if dp_rank in per_dp and per_dp[dp_rank] != local:
            raise ValueError("Model parallel replicas consumed different samples")
        per_dp[dp_rank] = local
    if set(per_dp) != set(range(first["dp_size"])) or sum(per_dp.values(), Counter()) != expected:
        raise ValueError("Consumed tokens/masks or sample multiplicity differ from the frozen batch")


def load_capture(root: Path) -> tuple[dict[int, dict], dict[str, tuple[Path, dict]]]:
    manifests = {}
    shards = {}
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    expected = {}
    for path in sorted(root.glob("rank-*/manifest.json")):
        manifest = json.loads(path.read_text())
        rank = manifest["rank"]
        if rank in manifests or manifest["prepare_calls"] != 1 or not manifest["update_successful"]:
            raise ValueError("Duplicate rank or unsuccessful optimizer update")
        manifests[rank] = manifest
        for parameter in manifest["expected_parameters"]:
            expected[_parameter_key(parameter)] = parameter
        for shard in manifest["shards"]:
            key = f"{rank}/{shard['file']}"
            if key in shards:
                raise ValueError("Duplicate optimizer shard")
            file = path.parent / shard["file"]
            if file.name != shard["file"] or file_digest(file) != shard["sha256"]:
                raise ValueError("Missing or corrupt optimizer shard")
            shards[key] = (file, shard)
            intervals[_parameter_key(shard)].append((shard["start"], shard["end"]))
    if not manifests or not shards:
        raise ValueError("Incomplete capture: no rank manifests or parameter shards")
    world_sizes = {m["world_size"] for m in manifests.values()}
    if len(world_sizes) != 1 or set(manifests) != set(range(next(iter(world_sizes)))):
        raise ValueError("Incomplete capture: missing ranks")
    if set(expected) != set(intervals):
        raise ValueError("Incomplete capture: missing or unexpected parameters")
    for key, spans in intervals.items():
        end = 0
        for start, stop in sorted(spans):
            if start != end or stop <= start:
                raise ValueError(f"Missing or overlapping shard coverage for {expected[key]['name']}")
            end = stop
        if end != expected[key]["numel"]:
            raise ValueError(f"Incomplete parameter: {expected[key]['name']}")
    validate_consumed_samples(manifests)
    return manifests, shards


def squared_errors(reference: Any, candidate: Any) -> tuple[float, float, int]:
    import torch

    if reference.shape != candidate.shape or reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise ValueError("Numerical exports must have matching FP32 shapes")
    reference, candidate = reference.reshape(-1), candidate.reshape(-1)
    errors, norms = [], []
    for start in range(0, reference.numel(), 1 << 20):
        a, b = reference[start : start + (1 << 20)].double(), candidate[start : start + (1 << 20)].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite numerical comparison")
        errors.append(float((a - b).square().sum()))
        norms.append(float(a.square().sum()))
    return math.fsum(errors), math.fsum(norms), reference.numel()


def compare(reference: Path, candidate: Path, reference_log: Path, candidate_log: Path, tolerances_path: Path) -> dict:
    import torch

    tolerances = json.loads(tolerances_path.read_text())
    before, left = load_capture(reference)
    after, right = load_capture(candidate)
    if set(before) != set(after) or set(left) != set(right):
        raise ValueError("Capture ranks or optimizer layouts differ")
    fields = (
        "world_size",
        "manifest_sha256",
        "model",
        "model_revision",
        "seed",
        "loss_type",
        "per_token",
        "global_batch_size",
        "rollout_id",
        "step_id",
        "tp_rank",
        "pp_rank",
        "cp_rank",
        "ep_rank",
        "dp_rank",
        "dp_size",
        "model_chunks",
        "expected_samples",
        "expected_parameters",
        "batch_source_ids_sha256",
        "initial_state_id",
        "scheduler",
        "rng",
    )
    for rank, manifest in before.items():
        if (
            manifest["tolerances_sha256"] != file_digest(tolerances_path)
            or after[rank]["tolerances_sha256"] != manifest["tolerances_sha256"]
        ):
            raise ValueError("Tolerances changed after numerical capture")
        if any(manifest[key] != after[rank][key] for key in fields):
            raise ValueError(f"Experiment configuration differs on rank {rank}")
    totals: dict[tuple[str, str], list] = {}
    for key, (path, metadata) in left.items():
        other_path, other = right[key]
        if metadata["initial_master_sha256"] != other["initial_master_sha256"]:
            raise ValueError(f"Initial FP32 weights differ for {metadata['name']}")
        if metadata["initial_optimizer_state"] != other["initial_optimizer_state"]:
            raise ValueError(f"Initial optimizer state differs for {metadata['name']}")
        a = torch.load(path, map_location="cpu", weights_only=True)
        b = torch.load(other_path, map_location="cpu", weights_only=True)
        for quantity in ("gradient", "delta"):
            error, norm, count = squared_errors(a[quantity], b[quantity])
            if count != metadata["end"] - metadata["start"]:
                raise ValueError("Tensor shape does not match optimizer shard range")
            total_key = (_parameter_key(metadata), quantity)
            if total_key not in totals:
                totals[total_key] = [0.0, 0.0, 0, metadata["name"]]
            values = totals[total_key]
            values[0] += error
            values[1] += norm
            values[2] += count
        del a, b
    comparisons = []
    for (key, quantity), (error, norm, count, name) in totals.items():
        error_rms, reference_rms = math.sqrt(error / count), math.sqrt(norm / count)
        tolerance = tolerances[quantity]
        limit = tolerance["atol_rms"] + tolerance["rtol_l2"] * reference_rms
        comparisons.append(
            {
                "parameter": name,
                "key": key,
                "quantity": quantity,
                "numel": count,
                "error_rms": error_rms,
                "reference_rms": reference_rms,
                "limit_rms": limit,
                "relative_l2": math.sqrt(error / norm) if norm else None,
                "passed": error_rms <= limit,
            }
        )
    step = before[0]["rollout_id"]
    loss_a = read_metrics(reference_log, "train")[step]["train/loss"]
    loss_b = read_metrics(candidate_log, "train")[step]["train/loss"]
    if not math.isfinite(loss_a) or not math.isfinite(loss_b):
        raise ValueError("Non-finite loss")
    loss_passed = abs(loss_a - loss_b) <= tolerances["loss"]["atol"] + tolerances["loss"]["rtol"] * abs(loss_a)
    return {
        "version": 1,
        "tolerances_sha256": file_digest(tolerances_path),
        "reference": str(reference),
        "candidate": str(candidate),
        "loss": {"reference": loss_a, "candidate": loss_b, "passed": loss_passed},
        "parameters": comparisons,
        "passed": loss_passed and all(row["passed"] for row in comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, default=Path(__file__).with_name("tolerances.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.reference_log, args.candidate_log, args.tolerances)
    write_json(args.output, result)
    logger.info("Numerical comparison passed=%s; report=%s", result["passed"], args.output)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

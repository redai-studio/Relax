# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproducible preference-evaluation artifacts and RFC statistics."""

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PREFERENCE_PROBE_PAIR_COUNT = 512


def preference_eval_chunk_sizes(pair_count: int, global_batch_size: int) -> list[int]:
    """Split every real probe pair into capacity-bounded eval chunks."""
    if pair_count <= 0 or global_batch_size <= 0:
        raise ValueError("preference eval pair count and global batch size must be positive")
    full_chunks, remainder = divmod(pair_count, global_batch_size)
    return [global_batch_size] * full_chunks + ([remainder] if remainder else [])


def preference_eval_local_batch_sizes(pair_count: int, global_batch_size: int, dp_size: int) -> list[int]:
    if dp_size <= 0:
        raise ValueError("preference eval data-parallel size must be positive")
    chunk_sizes = preference_eval_chunk_sizes(pair_count, global_batch_size)
    invalid = [size for size in chunk_sizes if size % dp_size != 0]
    if invalid:
        raise ValueError(
            f"preference eval chunk sizes must be divisible by data-parallel size: chunks={invalid}, dp={dp_size}"
        )
    return [size // dp_size for size in chunk_sizes]


def encoded_pair_id(pair_id: str) -> int:
    return int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:8], "big") >> 1


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def paired_bootstrap(values: Sequence[float], *, seed: int = 1234, num_replicates: int = 10_000) -> dict[str, Any]:
    """Compute the RFC's one-sided percentile bootstrap without dtype drift."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("paired bootstrap requires a non-empty one-dimensional vector")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, array.size, size=(num_replicates, array.size), endpoint=False)
    replicates = array[indices].mean(axis=1)
    if replicates.dtype != np.float64:
        raise RuntimeError(f"bootstrap replicates must remain float64, got {replicates.dtype}")
    lower_95 = np.quantile(replicates, 0.05, method="linear")
    return {
        "numpy_version": np.__version__,
        "seed": seed,
        "num_replicates": num_replicates,
        "point_estimate": float(array.mean()),
        "lower_95": float(lower_95),
        "passes_lower_bound_gt_0_50": bool(lower_95 > 0.50),
        "indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        "replicates_sha256": hashlib.sha256(replicates.astype("<f8", copy=False).tobytes()).hexdigest(),
    }


def artifact_directory(save_path: str | None) -> Path | None:
    return None if not save_path else Path(save_path) / "preference_eval"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_probe_contract(
    save_path: str | None,
    objective: str,
    completed_steps: int,
    pairs,
    *,
    expected_pair_count: int = PREFERENCE_PROBE_PAIR_COUNT,
) -> dict[str, Any] | None:
    if len(pairs) != expected_pair_count:
        raise RuntimeError(f"preference eval requires exactly {expected_pair_count} probe pairs, got {len(pairs)}")
    directory = artifact_directory(save_path)
    if directory is None:
        return None
    rows = [
        {
            "pair_id": pair.pair_id,
            "encoded_pair_id": encoded_pair_id(pair.pair_id),
            "chosen_tokens": pair.chosen_tokens.tolist(),
            "rejected_tokens": pair.rejected_tokens.tolist(),
            "chosen_raw_loss_mask": pair.chosen_loss_mask.tolist(),
            "rejected_raw_loss_mask": pair.rejected_loss_mask.tolist(),
            "chosen_score_position": pair.chosen_score_position,
            "rejected_score_position": pair.rejected_score_position,
        }
        for pair in pairs
    ]
    contract = {
        "objective": objective,
        "pair_count": len(rows),
        "pair_ids": [row["pair_id"] for row in rows],
        "encoded_pair_id_map": {str(row["encoded_pair_id"]): row["pair_id"] for row in rows},
        "probe_sha256": canonical_sha256(rows),
    }
    baseline_path = directory / f"{objective}-probe-contract.json"
    if completed_steps == 0:
        _atomic_write_json(baseline_path, contract)
    else:
        if not baseline_path.is_file():
            raise RuntimeError(f"preference final eval is missing step-0 probe contract: {baseline_path}")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if contract != baseline:
            raise RuntimeError(
                "preference probe preprocessing/order changed between step 0 and final eval: "
                f"baseline={baseline.get('probe_sha256')}, current={contract['probe_sha256']}"
            )
    return contract


def write_pair_artifacts(
    save_path: str | None,
    objective: str,
    completed_steps: int,
    encoded_rows: list[dict[str, Any]],
    batch_plan: list[dict[str, Any]],
) -> dict[str, Any] | None:
    directory = artifact_directory(save_path)
    if directory is None:
        return None
    contract_path = directory / f"{objective}-probe-contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"preference eval is missing probe contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pair_id_map = contract["encoded_pair_id_map"]
    rows = []
    for row in encoded_rows:
        row = dict(row)
        encoded = str(row.pop("encoded_pair_id"))
        if encoded not in pair_id_map:
            raise RuntimeError(f"preference eval produced unknown encoded pair id {encoded}")
        row["pair_id"] = pair_id_map[encoded]
        rows.append(row)
    rows.sort(key=lambda row: contract["pair_ids"].index(row["pair_id"]))
    if len(rows) != contract["pair_count"] or len({row["pair_id"] for row in rows}) != len(rows):
        raise RuntimeError(
            "preference eval pair artifact is incomplete or duplicated: "
            f"expected={contract['pair_count']}, actual={len(rows)}"
        )

    plan_sha256 = canonical_sha256(batch_plan)
    baseline_summary_path = directory / f"{objective}-step-0000000-summary.json"
    if completed_steps != 0:
        if not baseline_summary_path.is_file():
            raise RuntimeError("final preference eval is missing step-0 batch-plan evidence")
        baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        if plan_sha256 != baseline["batch_plan_sha256"]:
            raise RuntimeError(
                "preference eval batch plan changed between step 0 and final: "
                f"baseline={baseline['batch_plan_sha256']}, current={plan_sha256}"
            )

    if objective == "reward_model":
        accuracy_values = [float(row["chosen_score"] > row["rejected_score"]) for row in rows]
    else:
        epsilon = 1e-6
        accuracy_values = [
            1.0 if row["reward_margin"] > epsilon else 0.0 if row["reward_margin"] < -epsilon else 0.5 for row in rows
        ]
    summary = {
        "objective": objective,
        "completed_steps": completed_steps,
        "pair_count": len(rows),
        "probe_sha256": contract["probe_sha256"],
        "batch_plan_sha256": plan_sha256,
        "bootstrap": paired_bootstrap(accuracy_values),
    }
    stem = f"{objective}-step-{completed_steps:07d}"
    rows_path = directory / f"{stem}-pairs.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _atomic_write_json(directory / f"{stem}-batch-plan.json", batch_plan)
    _atomic_write_json(directory / f"{stem}-summary.json", summary)
    return summary


__all__ = [
    "PREFERENCE_PROBE_PAIR_COUNT",
    "canonical_sha256",
    "encoded_pair_id",
    "paired_bootstrap",
    "preference_eval_chunk_sizes",
    "preference_eval_local_batch_sizes",
    "record_probe_contract",
    "write_pair_artifacts",
]

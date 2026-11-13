# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Join collected observability files into a weight-age audit report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from relax.utils.agentic_obs.merge import merge_train_updates
from relax.utils.agentic_obs.weight_audit import audit_sample


def build_weight_report(
    trajectories: Iterable[dict[str, Any]], stages: list[dict[str, Any]], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Count successful optimizer consumption only, retaining unknown
    coverage."""
    exports: dict[tuple[Any, str], Any] = {}
    for trajectory in trajectories:
        key = (trajectory.get("run_id"), str(trajectory.get("sample_export_id")))
        if key in exports and exports[key] != trajectory:
            exports[key] = None
        else:
            exports[key] = trajectory
    updates = merge_train_updates(stages, inputs)
    reports, incomplete, skipped, conflicts = [], [], [], []
    committed = 0
    for update in updates:
        key = (update.get("run_id"), update.get("train_generation_id"), update.get("update_id"))
        if update.get("status") != "committed":
            skipped.append({"update_id": key[2], "status": update.get("status")})
            continue
        committed += 1
        if not update.get("input_complete"):
            incomplete.append(key[2])
        owners = {
            (row.get("dp_rank"), row.get("stage_id"))
            for row in stages
            if (row.get("run_id"), row.get("train_generation_id"), row.get("update_id")) == key
            and row.get("scope") == "update_fragment"
            and row.get("status") == "completed"
        }
        fragments = defaultdict(list)
        for row in inputs:
            if (row.get("run_id"), row.get("train_generation_id"), row.get("update_id")) != key:
                continue
            owner = (row.get("dp_rank"), row.get("stage_id"))
            if owner not in owners or row in fragments[owner]:
                continue
            fragments[owner].append(row)
        for owner, rows in fragments.items():
            if len(rows) != 1:
                conflicts.append({"update_id": key[2], "dp_rank": owner[0], "stage_id": owner[1]})
                continue
            row = rows[0]
            for occurrence, sample in enumerate(row.get("samples", [])):
                trajectory = exports.get((key[0], str(sample.get("sample_export_id"))))
                report = audit_sample(trajectory, sample, row.get("consumption_version"))
                report.update(update_id=key[2], dp_rank=row.get("dp_rank"), occurrence=occurrence)
                reports.append(report)
    total = sum(row["effective_tokens"] or 0 for row in reports)
    version_known = sum(row["known_version_tokens"] for row in reports)
    age_known = sum(row["known_age_tokens"] for row in reports)
    weighted_age_sum = sum(row["weighted_age_sum"] for row in reports)
    missing_masks = sum(row["effective_tokens"] is None for row in reports)
    boundaries_known = all(
        row.get("coordinates_valid") and row["generation_span"] is not None and row["post_generation_lag"] is not None
        for row in reports
        if row["effective_tokens"]
    )
    return {
        "statistics_scope": "successful_optimizer_updates; known final-objective masks",
        "evidence_available": bool(committed),
        "complete": bool(committed)
        and not incomplete
        and not missing_masks
        and not skipped
        and age_known == total
        and boundaries_known,
        "no_consumption_evidence": not committed,
        "incomplete_updates": incomplete,
        "conflicting_input_owners": conflicts,
        "excluded_updates": skipped,
        "unknown_mask_samples": missing_masks,
        "effective_tokens": total,
        "version_missing_rate": 1 - version_known / total if total else None,
        "age_missing_rate": 1 - age_known / total if total else None,
        "token_weighted_age": weighted_age_sum / age_known if age_known else None,
        "samples": reports,
    }


def read_records(root: Path, stream: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / stream).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Collected --dump-agentic-obs run directory")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = build_weight_report(
        read_records(options.root, "trajectories"),
        read_records(options.root, "stages"),
        read_records(options.root, "train"),
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

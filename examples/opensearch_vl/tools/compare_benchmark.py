# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare complete base and Relax runs under the same evaluation protocol."""

import argparse
import json
from pathlib import Path

from common import case_manifest_protocol, validate_evaluation_model, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, help="Compare a specific intermediate checkpoint against Base")
    parser.add_argument("--case-manifest", type=Path, help="Compare both runs on these exact V*Bench question IDs")
    args = parser.parse_args()
    results = {}
    for dataset, expected in [("vstar", 191)]:
        trained_dir = "relax" if args.checkpoint_step is None else f"checkpoint-step{args.checkpoint_step}"
        runs = {
            label: json.loads((args.root / subdir / dataset / "summary.json").read_text())
            for label, subdir in (("base", "base"), ("relax", trained_dir))
        }
        if args.case_manifest:
            protocol = runs["base"]["protocol"]
            cohort = case_manifest_protocol(args.case_manifest, protocol["dataset_revision"], protocol["questions_sha256"])
            for label, subdir in (("base", "base"), ("relax", trained_dir)):
                if any(runs[label]["protocol"].get(k) != v for k, v in cohort.items()):
                    raise ValueError("Benchmark run differs from the requested case manifest")
                with (args.root / subdir / dataset / "trajectories.jsonl").open() as trajectories:
                    actual_ids = [json.loads(line)["id"] for line in trajectories]
                if sorted(actual_ids) != sorted(cohort["case_ids"]):
                    raise ValueError("Incomplete, duplicate or unexpected trajectories for the requested case manifest")
            expected = len(cohort["case_ids"])
        elif any("case_manifest_sha256" in run["protocol"] for run in runs.values()):
            raise ValueError("Pass --case-manifest to compare subset results")
        if any(run["protocol"]["smoke"] or run["expected_cases"] != expected for run in runs.values()):
            raise ValueError("Only full benchmark runs can be compared")
        model_failures = {
            "context_limit",
            "turn_limit",
            "invalid_final_option",
            "unfinished_reasoning",
            "invalid_tool_call",
        }
        if any(set(run.get("errors", {})) - model_failures for run in runs.values()):
            raise ValueError("Resolve infrastructure errors before comparing model scores")
        if runs["base"]["protocol"] != runs["relax"]["protocol"]:
            raise ValueError("Base and Relax used different datasets, templates, tools or deduplication")
        validate_evaluation_model(runs["base"]["model_source"], "base", None)
        validate_evaluation_model(runs["relax"]["model_source"], "relax", args.checkpoint_step)
        if runs["base"]["deduplicated_cases"] != runs["relax"]["deduplicated_cases"]:
            raise ValueError("The deduplicated evaluation sets differ")
        results[dataset] = {
            "runs": runs,
            "pass_at_1_difference": runs["relax"]["pass_at_1"] - runs["base"]["pass_at_1"],
            "deduplicated_pass_at_1_difference": (
                runs["relax"]["deduplicated_pass_at_1"] - runs["base"]["deduplicated_pass_at_1"]
                if runs["base"]["deduplicated_pass_at_1"] is not None
                and runs["relax"]["deduplicated_pass_at_1"] is not None
                else None
            ),
        }
    write_json(args.output, results)


if __name__ == "__main__":
    main()

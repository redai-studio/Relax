# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare complete base and Relax runs under the same evaluation protocol."""

import argparse
import json
from pathlib import Path

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for dataset, expected in [("vstar", 191)]:
        runs = {
            label: json.loads((args.root / label / dataset / "summary.json").read_text())
            for label in ["base", "relax"]
        }
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
        if runs["relax"]["model_source"]["kind"] != "trained-final":
            raise ValueError("Relax results must use a trained final model")
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

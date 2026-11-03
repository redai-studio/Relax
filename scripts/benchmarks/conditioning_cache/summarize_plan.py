#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Read-only conditioning-cache paired summary.

Incomplete plans never pass performance gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

from prepare_experiments import prepare, sha256
from run_experiments import numerical_gate


def variation(values: list[float | None]) -> dict[str, Any]:
    known = [value for value in values if value is not None]
    return {
        "count": len(known),
        "expected_count": len(values),
        "mean": statistics.mean(known) if known else None,
        "sample_sd": statistics.stdev(known) if len(known) > 1 else None,
        "min": min(known) if known else None,
        "max": max(known) if known else None,
    }


def summarize(plan: dict[str, Any], runs_root: Path, public: Any) -> dict[str, Any]:
    declared = {x["run"]["name"]: x for x in prepare(plan)["requests"]}
    result: dict[str, Any] = {
        "runs": {},
        "groups": {},
        "verification": {},
        "excluded_profiles": [],
        "acceptance_complete": False,
        "notes": [
            "Independent models and optimization configurations are not pooled.",
            "This report proves paired throughput only; joint CPU/GPU/lifecycle/resource gates remain separate.",
            "low_reuse is the complete first pass including construction, not a warm-cache window.",
        ],
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in plan["runs"]:
        name = row["name"]
        if row["purpose"] == "profile":
            result["excluded_profiles"].append(name)
            continue
        directory = runs_root / name
        report: dict[str, Any] = {"status": "incomplete", "issues": []}
        try:
            request = json.loads((directory / "request.json").read_text())
            expected = declared[name]
            for field in ("run", "env", "comparison_signature", "implementation_sha256"):
                if request.get(field) != expected[field]:
                    report["issues"].append(f"Actual request differs from plan: {field}")
            value = public.summarize_run(directory, cold_steps=row["cold_steps"])
            report["measurement"] = value
            report["issues"].extend(value["issues"])
            if not value["source_artifact_verified"] or not value["job_complete"]:
                report["issues"].append("Source artifact / successful complete job not verified")
            expected_signature = expected["comparison_signature"]
            if value["comparison_signature"]["world_size"] != expected_signature["training"]["world_size"]:
                report["issues"].append("Observed world size differs")
            if (
                value["comparison_signature"]["global_batch_size"]
                != expected_signature["training"]["global_batch_size"]
            ):
                report["issues"].append("Observed global batch differs")
            if value["comparison_signature"]["model_fingerprint"] != expected_signature["model"]["content_sha256"]:
                report["issues"].append("Observed model fingerprint differs")
            if value["comparison_signature"]["metadata_sha256"] != expected_signature["data"]["metadata_sha256"]:
                report["issues"].append("Observed metadata fingerprint differs")
            if row["purpose"] == "verify":
                proof = json.loads((directory / "conditioning-verification.json").read_text())
                report["numerical"] = numerical_gate(proof)
                result["verification"][name] = report["numerical"]
            after = json.loads((directory / "resource_after.json").read_text())
            if after.get("compute_empty") is not True or after.get("namespace_count") != 0:
                report["issues"].append("Post-job resource release incomplete")
            report["status"] = "complete" if not report["issues"] else "incomplete"
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["issues"].append(f"{type(error).__name__}: {error}")
        result["runs"][name] = report
        if row["purpose"] in {"performance", "low_reuse", "screen"}:
            grouped.setdefault((row["purpose"], row["group"]), []).append(row)
    for (purpose, name), rows in grouped.items():
        complete = all(result["runs"][row["name"]]["status"] == "complete" for row in rows)
        modes = {
            mode: [result["runs"][row["name"]].get("measurement") for row in rows if row["cache"] is enabled]
            for mode, enabled in (("online", False), ("cached", True))
        }
        measurements = {mode: [value for value in values if value is not None] for mode, values in modes.items()}
        comparison = public.compare_runs(measurements["online"], measurements["cached"])
        group = {
            "purpose": purpose,
            "status": "complete" if complete and comparison.get("comparable") else "incomplete",
            "expected_names": [row["name"] for row in rows],
            "comparison": comparison,
            "cold_steps": rows[0]["cold_steps"],
            "steps": rows[0]["steps"],
            "repeated_measurements": len(modes["online"]) >= 3,
        }
        if group["status"] != "complete":
            group["comparison"] = {
                "comparable": False,
                "reasons": ["Some planned run is missing, incompatible or incomplete"],
                "observed_at_least_five_percent": None,
            }
        group["metrics"] = {
            mode: {
                phase: {
                    field: variation([value[phase][field] for value in values])
                    for field in ("seconds", "samples_per_second")
                }
                for phase in ("cold", "steady", "training")
            }
            for mode, values in measurements.items()
        }
        result["groups"][name] = group
    result["all_planned_runs_complete"] = all(x["status"] == "complete" for x in result["runs"].values())
    result["performance_gate"] = any(
        x["purpose"] == "performance"
        and x["status"] == "complete"
        and x["repeated_measurements"]
        and x["comparison"].get("observed_at_least_five_percent") is True
        for x in result["groups"].values()
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    script = a.repo / "scripts/tools/benchmark_condition_cache.py"
    spec = importlib.util.spec_from_file_location("conditioning_benchmark_summary", script)
    if spec is None or spec.loader is None:
        raise ValueError("Public benchmark module missing")
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    result = summarize(json.loads(a.plan.read_text()), a.runs_root, public)
    result.update(plan_sha256=sha256(a.plan.read_bytes()), public_benchmark_sha256=sha256(script.read_bytes()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "all_planned_runs_complete": result["all_planned_runs_complete"],
                "performance_gate": result["performance_gate"],
                "acceptance_complete": False,
            }
        )
    )


if __name__ == "__main__":
    main()

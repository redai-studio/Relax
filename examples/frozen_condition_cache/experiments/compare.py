"""Compare all four recorded trajectories; never select a best run."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.evidence import assess_strict_acceptance, atomic_json, compare_trajectories, numerical_components


def load(directory: Path) -> list[dict]:
    files = sorted(directory.glob("trajectory-rank*.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectory evidence in {directory}")
    result = [torch.load(file, weights_only=False, map_location="cpu") for file in files]
    if [row["rank"] for row in result] != list(range(len(result))):
        raise ValueError("Incomplete or unordered rank evidence")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a1", "a2", "bcold", "bwarm", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: load(getattr(args, name)) for name in ("a1", "a2", "bcold", "bwarm")}
    # Inspect the fixed online repeat before either cached comparison. Its
    # observed exactness never becomes a post-hoc fitted absolute tolerance.
    baseline = compare_trajectories(runs["a1"], runs["a2"])
    result = {"a2": baseline}
    policies = {}
    for name in ("bcold", "bwarm"):
        result[name] = compare_trajectories(runs["a1"], runs[name])
        policies[name] = assess_strict_acceptance(baseline, result[name])
    cache_exact = all(
        row["cache_audit"]["failed"] == 0 and row["cache_audit"]["checked_hits"] > 0 for row in runs["bwarm"]
    )
    passed = (
        all(row["passed"] for row in result.values())
        and all(row["passed"] for row in policies.values())
        and cache_exact
    )
    strict = passed and all(row["strict_acceptance"] for row in policies.values())
    needs_review = passed and any(row["needs_nondeterminism_review"] for row in policies.values())
    payload = {
        "schema": 2,
        "passed": passed,
        "hard_numerical_limits_passed": all(row["passed"] for row in result.values()),
        "strict_acceptance": strict,
        "acceptance_status": "accepted" if strict else "needs_nondeterminism_review" if needs_review else "failed",
        "needs_nondeterminism_review": needs_review,
        "comparisons": result,
        "strict_comparisons": policies,
        "baseline_observed_exact": bool(numerical_components(baseline))
        and all(row.get("bitwise_equal", False) for row in numerical_components(baseline).values()),
        "baseline_nondeterministic_components": sorted(
            name for name, row in numerical_components(baseline).items() if not row.get("bitwise_equal", False)
        ),
        "deterministic_mismatches": {name: row["deterministic_mismatches"] for name, row in policies.items()},
        "acceptance_policy": {
            "baseline": "fixed online A1 versus A2, inspected before cached comparisons",
            "observed_exact_component": "cached cold and warm must both be bitwise exact for that component",
            "nonexact_baseline_component": "retain hard 1%/zero/finite/RNG checks; no automatic strict acceptance",
            "absolute_tolerance": "none inferred or invented; near-zero nondeterministic acceptance needs review",
            "performance_collection": (
                "may continue only when passed=true; strict_acceptance=false cannot meet final technical thresholds"
            ),
        },
        "cache_payload_exact_with_checked_hits": cache_exact,
        "nondeterministic_tolerance_relaxed": False,
    }
    atomic_json(args.output, payload)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

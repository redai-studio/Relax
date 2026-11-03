#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate a model-neutral conditioning-cache plan and render reviewable
requests; never submit.

This preparation stage performs no network or Ray operations. --check-local
also validates the frozen source, launcher and dataset on the execution host.
An execution runner must additionally enforce live ownership/resource gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PURPOSES = {"verify", "smoke", "screen", "performance", "low_reuse", "profile", "cancel"}
PROTECTED = {
    "RELAX",
    "PYTHONPATH",
    "DRIVER_PYTHONPATH",
    "WORKING_DIR",
    "JOB_ID",
    "EXP_NAME",
    "MODEL_PATH",
    "DATA_ROOT",
    "METADATA_PATH",
    "DATASET_SIZE",
    "GPUS",
    "GPUS_PER_NODE",
    "GLOBAL_BATCH_SIZE",
    "MICRO_BATCH_SIZE",
    "SEED",
    "STEPS",
    "SAVE_DIR",
    "SAVE_CHECKPOINTS",
    "RAY_ADDRESS",
    "RAY_DASHBOARD_ADDRESS",
    "RAY_NO_WAIT",
    "RELAX_SKIP_CLUSTER_CLEANUP",
    "RELAX_ENTRYPOINT_MODE",
    "RUNTIME_ENV_JSON",
    "USE_CLEARML",
    "DRY_RUN",
    "RELAX_CONDITION_CACHE",
    "RELAX_CONDITION_PROFILE",
    "RELAX_CONDITION_VERIFY",
    "RELAX_CONDITION_VERIFY_RTOL",
    "RELAX_CONDITION_VERIFY_SNAPSHOT",
    "RELAX_CONDITION_CACHE_DIR",
    "RELAX_CONDITION_CACHE_MAX_BYTES",
    "RELAX_CONDITION_CACHE_RUN_DIR",
    "RELAX_CONDITION_PREFETCH_DEPTH",
    "RELAX_CONDITION_PREFETCH_MAX_BYTES",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_value(value: Any, field: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{field} must be a pinned SHA256"
    )
    return value


def positive(value: Any, field: str, *, zero: bool = False) -> int:
    require(type(value) is int and value >= (0 if zero else 1), f"Invalid {field}")
    return value


def absolute(value: Any, field: str) -> Path:
    require(
        isinstance(value, str) and "\x00" not in value and Path(value).is_absolute(),
        f"{field} must be an absolute path",
    )
    return Path(value)


def environment(value: Any) -> dict[str, str]:
    require(isinstance(value, dict), "env must be an object")
    for key, item in value.items():
        require(
            isinstance(key, str)
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None
            and key not in PROTECTED
            and isinstance(item, str)
            and "\x00" not in item,
            f"Invalid or protected env override: {key}",
        )
    return value


def source_snapshot(repo: Path) -> tuple[str, bytes, str]:
    """Keep the established provenance encoding without importing mutable
    analysis modules."""
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "relax", "scripts"], cwd=repo)
    names = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "relax", "scripts"],
        cwd=repo,
        text=True,
    ).splitlines()
    for name in sorted(names):
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "/dev/null", name], cwd=repo, capture_output=True, check=False
        )
        require(diff.returncode in (0, 1), f"Cannot snapshot {name}")
        patch += diff.stdout
    return base, patch, sha256(base.encode() + b"\n" + patch)


def dataset_rows(path: Path, kind: str) -> int:
    if kind == "json_array":
        rows = json.loads(path.read_text())
        require(isinstance(rows, list), "Metadata must be a JSON array")
        return len(rows)
    require(kind == "jsonl", "Only explicit json_array/jsonl metadata is supported")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "JSONL metadata rows must be objects")
    return len(rows)


def workload(value: dict[str, Any], check_local: bool) -> dict[str, Any]:
    require(isinstance(value, dict), "workload must be an object")
    for key in ("model", "data", "source", "training", "cache"):
        require(isinstance(value.get(key), dict), f"workload.{key} is required")
    model, data, source, training, cache = (value[key] for key in ("model", "data", "source", "training", "cache"))
    require(isinstance(model.get("id"), str) and bool(model["id"]), "Model ID is required")
    absolute(model.get("path"), "model.path")
    digest_value(model.get("content_sha256"), "model.content_sha256")
    absolute(data.get("root"), "data.root")
    metadata = absolute(data.get("metadata"), "data.metadata")
    digest_value(data.get("metadata_sha256"), "data.metadata_sha256")
    count = positive(data.get("rows"), "data.rows")
    require(data.get("format") in {"json_array", "jsonl"}, "Explicit metadata format required")
    require(
        isinstance(data.get("selection_rule"), str) and bool(data["selection_rule"]), "Data selection rule required"
    )
    repo = absolute(source.get("repo"), "source.repo")
    digest_value(source.get("implementation_sha256"), "source.implementation_sha256")
    launcher = value.get("launcher")
    require(
        isinstance(launcher, str)
        and launcher.startswith("scripts/training/")
        and ".." not in Path(launcher).parts
        and launcher.endswith(".sh"),
        "Launcher must be a training script",
    )
    digest_value(value.get("launcher_sha256"), "launcher_sha256")
    world = positive(training.get("world_size"), "world_size")
    batch = positive(training.get("global_batch_size"), "global_batch_size")
    micro = positive(training.get("micro_batch_size"), "micro_batch_size")
    require(micro == 1, "Current public image SFT requires micro_batch_size=1; use GBS for accumulation")
    require(
        batch % (world * micro) == 0, "This draft supports data parallel FSDP; GBS must be divisible by world*micro"
    )
    positive(training.get("seed"), "seed", zero=True)
    require(
        isinstance(training.get("semantic_config"), dict) and bool(training["semantic_config"]),
        "Pin optimizer/dtype/resolution/trainable mode/etc in semantic_config",
    )
    absolute(cache.get("root"), "cache.root")
    for key in ("payload_bytes", "prefetch_bytes"):
        positive(cache.get(key), "cache." + key)
    positive(cache.get("prefetch_depth"), "cache.prefetch_depth", zero=True)
    environment(value.get("env", {}))
    require(
        isinstance(value.get("extra_args", []), list)
        and all(isinstance(item, str) and "\x00" not in item for item in value.get("extra_args", [])),
        "extra_args must be a list of strings",
    )
    if check_local:
        require((repo / "scripts/entrypoint/ray-job.sh").is_file(), "Missing canonical entrypoint")
        require(sha256((repo / launcher).read_bytes()) == value["launcher_sha256"], "Launcher digest mismatch")
        require(source_snapshot(repo)[2] == source["implementation_sha256"], "Implementation digest mismatch")
        require(sha256(metadata.read_bytes()) == data["metadata_sha256"], "Metadata digest mismatch")
        require(dataset_rows(metadata, data["format"]) == count, "Metadata row count mismatch")
        require(Path(model["path"]).is_dir(), "Model directory missing")
    return value


def prepare(plan: dict[str, Any], *, check_local: bool = False) -> dict[str, Any]:
    require(
        plan.get("schema_version") == 1 and plan.get("status") == "ready",
        "Plan is not a frozen ready conditioning-cache plan",
    )
    cluster = plan.get("cluster", {})
    require(
        isinstance(cluster.get("ray_address"), str) and ":" in cluster["ray_address"], "Explicit Ray address required"
    )
    require(
        cluster.get("dashboard") in {"http://127.0.0.1:8265", "http://localhost:8265"},
        "Use the execution-host local dashboard; do not route localhost through proxy",
    )
    capacity = positive(cluster.get("expected_gpus"), "expected_gpus")
    absolute(cluster.get("lock_file"), "cluster.lock_file")
    deployments = cluster.get("serve_deployments")
    require(
        isinstance(deployments, dict)
        and bool(deployments)
        and all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in deployments.items()),
        "Declare exact expected Serve application/deployment names",
    )
    pythonpath = cluster.get("pythonpath")
    require(isinstance(pythonpath, list) and bool(pythonpath), "Explicit dependency pythonpath required")
    for path in pythonpath:
        absolute(path, "cluster.pythonpath")
    runs_root = absolute(plan.get("runs_root"), "runs_root")
    workloads = {name: workload(value, check_local) for name, value in plan.get("workloads", {}).items()}
    require(bool(workloads), "At least one workload required")
    rows = plan.get("runs")
    require(isinstance(rows, list) and bool(rows), "Explicit nonempty runs required")
    requests, seen, paired = [], set(), {}
    for row in rows:
        name = row.get("name")
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None
            and name not in seen,
            "Unique safe run names required",
        )
        seen.add(name)
        require(row.get("workload") in workloads, "Unknown workload")
        w = workloads[row["workload"]]
        t, c = w["training"], w["cache"]
        require(t["world_size"] <= capacity, "Run exceeds cluster capacity")
        purpose = row.get("purpose")
        require(purpose in PURPOSES, "Explicit run purpose required")
        require(type(row.get("cache")) is bool, "cache must be boolean")
        steps = positive(row.get("steps"), "steps")
        cold = positive(row.get("cold_steps"), "cold_steps", zero=True)
        require(cold < steps, "cold_steps must leave a measurement window")
        if purpose == "low_reuse":
            require(cold == 0, "low_reuse measures the complete first pass, not a warm window")
        if purpose == "performance":
            require(cold > 0, "Performance must explicitly identify a cold window")
        if purpose == "verify":
            require(
                row["cache"] is True and row.get("rtol") in (0, 0.0),
                "Initial verification must be exact and exercise cache",
            )
        override = environment(row.get("env", {}))
        recipe_env = w.get("env", {}) | override
        extra = w.get("extra_args", []) + row.get("extra_args", [])
        require(all(isinstance(x, str) and "\x00" not in x for x in extra), "Invalid extra_args")
        signature = {
            "workload": row["workload"],
            "model": w["model"],
            "data": w["data"],
            "source": w["source"],
            "launcher": w["launcher"],
            "launcher_sha256": w["launcher_sha256"],
            "training": t,
            "env": recipe_env,
            "extra_args": extra,
            "steps": steps,
            "cold_steps": cold,
        }
        if purpose in {"performance", "low_reuse", "screen"}:
            group = row.get("group")
            require(isinstance(group, str) and bool(group), "Paired runs require a group")
            repeat = positive(row.get("repeat"), "repeat")
            paired.setdefault((purpose, group), []).append((repeat, row["cache"], signature))
        repo = Path(w["source"]["repo"])
        directory = runs_root / name
        env = recipe_env | {
            "RELAX": str(repo),
            "PYTHONPATH": ":".join([str(repo), *pythonpath]),
            "DRIVER_PYTHONPATH": ":".join([str(repo), *pythonpath]),
            "MODEL_PATH": w["model"]["path"],
            "DATA_ROOT": w["data"]["root"],
            "METADATA_PATH": w["data"]["metadata"],
            "DATASET_SIZE": str(w["data"]["rows"]),
            "GPUS": str(t["world_size"]),
            "GPUS_PER_NODE": str(t["world_size"]),
            "GLOBAL_BATCH_SIZE": str(t["global_batch_size"]),
            "MICRO_BATCH_SIZE": str(t["micro_batch_size"]),
            "SEED": str(t["seed"]),
            "STEPS": str(steps),
            "EXP_NAME": name,
            "JOB_ID": "conditioning-cache-" + name,
            "SAVE_DIR": str(runs_root / "checkpoints"),
            "SAVE_CHECKPOINTS": "0",
            "USE_CLEARML": "0",
            "RAY_ADDRESS": cluster["ray_address"],
            "RAY_DASHBOARD_ADDRESS": cluster["dashboard"],
            "RAY_NO_WAIT": "1",
            "RELAX_SKIP_CLUSTER_CLEANUP": "1",
            "RELAX_CONDITION_CACHE": str(int(row["cache"])),
            "RELAX_CONDITION_PROFILE": "1",
            "RELAX_CONDITION_VERIFY": str(int(purpose == "verify")),
            "RELAX_CONDITION_VERIFY_RTOL": str(row.get("rtol", 0)),
            "RELAX_CONDITION_VERIFY_SNAPSHOT": str(int(purpose == "verify")),
            "RELAX_CONDITION_CACHE_DIR": c["root"],
            "RELAX_CONDITION_CACHE_MAX_BYTES": str(c["payload_bytes"]),
            "RELAX_CONDITION_CACHE_RUN_DIR": str(directory),
            "RELAX_CONDITION_PREFETCH_DEPTH": str(c["prefetch_depth"] if row["cache"] else 0),
            "RELAX_CONDITION_PREFETCH_MAX_BYTES": str(c["prefetch_bytes"]),
        }
        requests.append(
            {
                "run": row,
                "directory": str(directory),
                "job_id": env["JOB_ID"],
                "command": ["bash", str(repo / "scripts/entrypoint/ray-job.sh"), str(repo / w["launcher"]), *extra],
                "env": env,
                "comparison_signature": signature,
                "implementation_sha256": w["source"]["implementation_sha256"],
            }
        )
    for (purpose, group), pairs in paired.items():
        signatures = {json.dumps(x[2], sort_keys=True) for x in pairs}
        require(len(signatures) == 1, f"{group} mixes workload/source/config/window")
        identities = [(repeat, enabled) for repeat, enabled, _ in pairs]
        require(len(set(identities)) == len(identities), f"{group} repeats a mode/repeat")
        repeats = {x[0] for x in identities}
        require(
            set(identities) == {(r, enabled) for r in repeats for enabled in (False, True)},
            f"{group} has an unpaired planned repeat",
        )
        if purpose == "performance":
            require(len(repeats) >= 3, f"{group} final performance requires at least three pairs")
    return {
        "schema_version": 1,
        "status": "prepared_only",
        "execution_supported": False,
        "local_source_and_metadata_checked": check_local,
        "model_content_checked": False,
        "requests": requests,
        "acceptance_passed": False,
        "notes": [
            "No Ray/network/model load/cleanup was performed.",
            "Model fingerprint must be verified by execution preflight and training init telemetry.",
            "Recipe env is declarative until the launcher and observed config are cross-checked.",
            "Use a single node-wide experiment lock, including across different runs roots.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-local", action="store_true")
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    result = prepare(json.loads(raw), check_local=args.check_local)
    result.update(plan_sha256=sha256(raw), preparer_sha256=sha256(Path(__file__).read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Prepared {len(result['requests'])} requests; no jobs submitted: {args.output}")


if __name__ == "__main__":
    main()
